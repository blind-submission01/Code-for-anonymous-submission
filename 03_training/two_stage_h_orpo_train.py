#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-Stage Hierarchical Training Script for Qwen
================================================

训练流程：
Stage 1 (H-SFT): 使用 fine_tuning=SFT 的数据，进行 A+B+C 三种任务的 SFT 训练
Stage 2 (H-ORPO): 使用 fine_tuning=ORPO 的数据，进行 A+B+C 三种任务的 ORPO 训练

任务类型：
- Type A: Root Cause Analysis (issue+slice → rc)
- Type B: Repair Suggestion (issue+slice+rc → rs)  
- Type C: Full CoT (issue+slice → rc+rs)

Loss 公式：
- Stage 1: Loss = (SFT_A + SFT_B + SFT_C) / 3
- Stage 2: Loss = (SFT_A + SFT_B + SFT_C) / 3 + λ * (ORPO_A + ORPO_B + ORPO_C) / 3
"""

import os
import json
import torch
import logging
import argparse
import torch.nn.functional as F
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


# =============================================================================
# 1. Configuration
# =============================================================================

class TrainingStage(Enum):
    STAGE1_H_SFT = "stage1_h_sft"
    STAGE2_H_ORPO = "stage2_h_orpo"
    BOTH = "both"


@dataclass
class TwoStageHierarchicalConfig:
    """两阶段层次化训练配置"""
    model_name: str
    max_seq_length: int = 6144
    
    # Stage 1 SFT 参数
    sft_learning_rate: float = 1e-4
    sft_num_epochs: int = 3
    
    # Stage 2 ORPO 参数
    orpo_learning_rate: float = 1e-6
    orpo_num_epochs: int = 6
    orpo_beta: float = 1.0
    lambda_orpo: float = 1.0
    
    # 通用训练参数
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    save_steps: int = 300
    eval_steps: int = 200
    logging_steps: int = 10
    
    # LoRA配置
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


# =============================================================================
# 2. Prompt Engineering (A/B/C Types)
# =============================================================================

class HierarchicalPromptGenerator:
    """处理 A/B/C 三种任务类型的 Prompt 生成"""
    
    TYPE_A = "A"  # Root Cause
    TYPE_B = "B"  # Repair Suggestion
    TYPE_C = "C"  # End-to-End CoT

    @staticmethod
    def get_task_description(sample_type: str, item: Dict) -> Tuple[str, str]:
        problem_statement = item.get('problem_statement', '')
        buggy_slice = item.get('buggy_slice', '')
        
        if sample_type == HierarchicalPromptGenerator.TYPE_A:
            desc = """Analyze the root cause of the issue based on the given information. Generate a concise but sufficient description of **the issue root cause** only.
    - root cause: the layered explanation of reason for this issue, highlighting exactly why it is incorrect."""
            context = f"# GitHub Issue Description:\n{problem_statement}\n\n# Segments Groups:\n{buggy_slice}"
            
        elif sample_type == HierarchicalPromptGenerator.TYPE_B:
            win_rc = item.get('win_rc', '')
            desc = """Based on the given issue root cause, generate a concise but sufficient **the proposed repair suggestion** only.
    - repair suggestion: actionable and concrete steps in execution order describing how the fix addresses this issue. (Note:1. A couple of small code snippets or inline code may be included, but avoid outputting medium and large blocks of code. 2. Hierarchical and structured format is encouraged."""
            context = f"# GitHub Issue Description:\n{problem_statement}\n\n# Segments Groups:\n{buggy_slice}\n\n# Root Cause:\n{win_rc}"
            
        else:  # TYPE_C
            desc = """Generate Natural Language Description: Generate a concise but sufficient, developer-friendly description including both **the issue root cause** and **the proposed repair suggestion**：
    - root cause: the layered explanation of reason for this issue, highlighting exactly why it is incorrect.
    - repair suggestion: actionable and concrete steps in execution order describing how the fix addresses this issue. (Note:1. A couple of small code snippets or inline code may be included, but avoid outputting medium and large blocks of code. 2. Hierarchical and structured format is encouraged."""
            context = f"# GitHub Issue Description:\n{problem_statement}\n\n# Segments Groups:\n{buggy_slice}"
            
        return desc, context

    @staticmethod
    def format_input(item: Dict, sample_type: str) -> str:
        task_desc, context = HierarchicalPromptGenerator.get_task_description(sample_type, item)
        return f"""<|im_start|>system
You are a code analysis expert specializing in bug repair.
<|im_end|>
<|im_start|>user
You will be given:
- A GitHub issue description
- Several segment groups: each contains a suspicious code slice.

Perform the following Task:
{task_desc}

{context}

<|im_end|>
<|im_start|>assistant
"""

    @staticmethod
    def format_output(item: Dict, sample_type: str, use_lose: bool = False) -> str:
        prefix = 'lose' if use_lose else 'win'
        rc = item.get(f'{prefix}_rc', '').strip()
        rs = item.get(f'{prefix}_rs', '').strip()
        
        if sample_type == HierarchicalPromptGenerator.TYPE_A:
            content = f"# Root Cause:\n{rc}"
        elif sample_type == HierarchicalPromptGenerator.TYPE_B:
            content = f"# Repair Suggestion:\n{rs}"
        else:  # TYPE_C
            content = f"# Root Cause:\n{rc}\n\n# Repair Suggestion:\n{rs}"
            
        return content + "<|im_end|>"


# =============================================================================
# 3. Data Processors
# =============================================================================

class HierarchicalDataProcessor:
    """处理 A/B/C 三种任务的数据编码"""
    
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _tokenize(self, input_text: str, output_text: str) -> Dict:
        full_text = input_text + output_text
        full_enc = self.tokenizer(full_text, truncation=True, max_length=self.max_length, padding=False, return_tensors="pt")
        input_enc = self.tokenizer(input_text, truncation=True, max_length=self.max_length, padding=False, return_tensors="pt")
        return {
            "input_ids": full_enc["input_ids"].squeeze(),
            "attention_mask": full_enc["attention_mask"].squeeze(),
            "input_length": input_enc["input_ids"].size(1)
        }
    
    def encode_pair(self, item: Dict, sample_type: str, is_orpo: bool) -> Dict:
        """编码一个 win/lose 对"""
        win_in = HierarchicalPromptGenerator.format_input(item, sample_type)
        win_out = HierarchicalPromptGenerator.format_output(item, sample_type, use_lose=False)
        winning = self._tokenize(win_in, win_out)
        
        if is_orpo:
            lose_out = HierarchicalPromptGenerator.format_output(item, sample_type, use_lose=True)
            losing = self._tokenize(win_in, lose_out)
        else:
            losing = winning  # SFT 模式下，losing 与 winning 相同
            
        return {
            "winning_input_ids": winning["input_ids"],
            "winning_attention_mask": winning["attention_mask"],
            "losing_input_ids": losing["input_ids"],
            "losing_attention_mask": losing["attention_mask"],
            "input_length": winning["input_length"],
        }

    def process(self, item: Dict, is_orpo: bool) -> Dict:
        """处理一个样本，生成 A/B/C 三种任务的数据"""
        result = {}
        for task_type in ['A', 'B', 'C']:
            pair_data = self.encode_pair(item, task_type, is_orpo)
            result.update({f"{task_type.lower()}_{k}": v for k, v in pair_data.items()})
        return result


# =============================================================================
# 4. Datasets
# =============================================================================

class HierarchicalSFTDataset(Dataset):
    """Stage 1: Hierarchical SFT 数据集 (只使用 SFT 标记的数据)"""
    
    def __init__(self, data_file: str, tokenizer, max_length: int):
        self.processor = HierarchicalDataProcessor(tokenizer, max_length)
        self.data = self._load_data(data_file)
        logging.getLogger(__name__).info(f"[Stage1 H-SFT] Loaded {len(self.data)} SFT samples")
    
    def _load_data(self, data_file: str) -> List[Dict]:
        data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    ft_tag = (item.get('fine_tuning', '') or '').strip().upper()
                    if ft_tag == "SFT" and item.get('win_rc') and item.get('win_rs'):
                        data.append(item)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        processed = self.processor.process(item, is_orpo=False)
        processed["fine_tuning"] = 0  # SFT
        return processed


class HierarchicalORPODataset(Dataset):
    """Stage 2: Hierarchical ORPO 数据集 (只使用 ORPO 标记的数据)"""
    
    def __init__(self, data_file: str, tokenizer, max_length: int):
        self.processor = HierarchicalDataProcessor(tokenizer, max_length)
        self.data = self._load_data(data_file)
        logging.getLogger(__name__).info(f"[Stage2 H-ORPO] Loaded {len(self.data)} ORPO samples")
    
    def _load_data(self, data_file: str) -> List[Dict]:
        data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    ft_tag = (item.get('fine_tuning', '') or '').strip().upper()
                    has_win = item.get('win_rc') and item.get('win_rs')
                    has_lose = item.get('lose_rc') and item.get('lose_rs')
                    if ft_tag == "ORPO" and has_win and has_lose:
                        data.append(item)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        processed = self.processor.process(item, is_orpo=True)
        processed["fine_tuning"] = 1  # ORPO
        return processed


# =============================================================================
# 5. Collate Function
# =============================================================================

def hierarchical_collate_fn(batch, tokenizer):
    """支持 A/B/C 三种任务的 collate 函数"""
    pad_id = tokenizer.pad_token_id
    first_item = batch[0]
    prefixes = sorted(set([k.split('_')[0] + '_' for k in first_item.keys() if '_winning_input_ids' in k]))
    
    result = {}
    
    def _pad_and_stack(items, key, pad_value):
        max_len = max(item[key].size(0) for item in items)
        padded = [F.pad(item[key], (0, max_len - item[key].size(0)), value=pad_value) for item in items]
        return torch.stack(padded)
    
    for prefix in prefixes:
        result[f"{prefix}winning_input_ids"] = _pad_and_stack(batch, f"{prefix}winning_input_ids", pad_id)
        result[f"{prefix}winning_attention_mask"] = _pad_and_stack(batch, f"{prefix}winning_attention_mask", 0)
        result[f"{prefix}losing_input_ids"] = _pad_and_stack(batch, f"{prefix}losing_input_ids", pad_id)
        result[f"{prefix}losing_attention_mask"] = _pad_and_stack(batch, f"{prefix}losing_attention_mask", 0)
        result[f"{prefix}input_length"] = torch.tensor([item[f"{prefix}input_length"] for item in batch], dtype=torch.long)
    
    result["fine_tuning"] = torch.tensor([item["fine_tuning"] for item in batch], dtype=torch.long)
    return result


# =============================================================================
# 6. Utility Functions
# =============================================================================

def log1mexp(x):
    """数值稳定的 log(1 - exp(x)) 计算"""
    limit = -0.69314718056
    x = torch.clamp(x, max=-1e-7)
    return torch.where(x < limit, torch.log1p(-torch.exp(x)), torch.log(-torch.expm1(x)))


# =============================================================================
# 7. Hierarchical Trainer
# =============================================================================

class HierarchicalTrainer(Trainer):
    """
    统一的层次化 Trainer，支持 SFT 和 ORPO 两种模式
    - lambda_reg=0: 纯 SFT 模式 (Stage 1)
    - lambda_reg>0: SFT + ORPO 混合模式 (Stage 2)
    """
    
    def __init__(self, beta: float = 1.0, lambda_reg: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.lambda_reg = lambda_reg
        self.logger = logging.getLogger(__name__)
        self._init_accum_stats()
    
    def _init_accum_stats(self):
        self.accum_stats = {
            'total_samples': 0,
            'metrics': {k: {'sft_sum': 0.0, 'sft_count': 0, 'orpo_sum': 0.0, 'orpo_count': 0} for k in ['A', 'B', 'C']},
            'global': {'chosen_sum': 0.0, 'rejected_sum': 0.0, 'margin_sum': 0.0, 'acc_sum': 0.0, 'log_odds_sum': 0.0}
        }

    def calculate_response_logprobs(self, logits, input_ids, input_length, attention_mask):
        """计算 response 部分的 log 概率"""
        batch_size = input_ids.size(0)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()
        
        if input_length.dim() == 0:
            input_length = input_length.unsqueeze(0).expand(batch_size)
        
        response_mask = torch.zeros_like(shift_mask)
        for i in range(batch_size):
            start = max(0, input_length[i].item() - 1)
            response_mask[i, start:] = shift_mask[i, start:]
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        selected_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        masked_log_probs = selected_log_probs * response_mask
        response_lengths = response_mask.sum(dim=-1).clamp(min=1)
        
        return masked_log_probs.sum(dim=-1) / response_lengths

    def _compute_single_type_loss(self, model, winning_input_ids, winning_attention_mask,
                                   losing_input_ids, losing_attention_mask, input_length, fine_tuning):
        """计算单个任务类型的 SFT + ORPO loss"""
        device = winning_input_ids.device
        batch_size = winning_input_ids.size(0)
        
        if input_length.dim() == 0:
            input_length = input_length.unsqueeze(0).expand(batch_size)
        
        # Forward winning
        winning_outputs = model(input_ids=winning_input_ids, attention_mask=winning_attention_mask, return_dict=True)
        logits = winning_outputs.logits
        
        # SFT Loss
        sft_labels = winning_input_ids.clone()
        sft_labels[winning_attention_mask == 0] = -100
        for i in range(batch_size):
            sft_labels[i, :input_length[i]] = -100
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = sft_labels[..., 1:].contiguous()
        
        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
            ignore_index=-100, reduction='none'
        ).view(batch_size, -1)
        
        valid_mask = (shift_labels != -100).float()
        valid_length = valid_mask.sum(dim=-1).clamp(min=1)
        per_sample_sft_loss = (per_token_loss * valid_mask).sum(dim=-1) / valid_length
        sft_loss_mean = per_sample_sft_loss.mean()
        
        # Stats
        stats = {
            'sft_sum': per_sample_sft_loss.sum().detach().float().item(),
            'sft_count': batch_size,
            'orpo_sum': 0.0, 'orpo_count': 0,
            'chosen_sum': 0.0, 'rejected_sum': 0.0, 'margin_sum': 0.0, 'acc_sum': 0.0, 'log_odds_sum': 0.0
        }
        
        # ORPO Loss (仅当 lambda_reg > 0 且存在 ORPO 样本)
        orpo_mask = fine_tuning == 1
        orpo_loss_mean = torch.tensor(0.0, device=device)
        
        if orpo_mask.any() and self.lambda_reg > 0:
            losing_outputs = model(input_ids=losing_input_ids, attention_mask=losing_attention_mask, return_dict=True)
            
            winning_logprobs = self.calculate_response_logprobs(winning_outputs.logits, winning_input_ids, input_length, winning_attention_mask)
            losing_logprobs = self.calculate_response_logprobs(losing_outputs.logits, losing_input_ids, input_length, losing_attention_mask)
            
            log_odds_w = winning_logprobs - log1mexp(winning_logprobs)
            log_odds_l = losing_logprobs - log1mexp(losing_logprobs)
            log_odds_ratio = log_odds_w - log_odds_l
            
            pref_loss_all = F.softplus(-self.beta * log_odds_ratio).view(-1)
            orpo_loss_mean = pref_loss_all[orpo_mask].sum() / batch_size
            
            stats['orpo_sum'] = pref_loss_all[orpo_mask].sum().detach().float().item()
            stats['orpo_count'] = orpo_mask.sum().item()
            stats['chosen_sum'] = winning_logprobs[orpo_mask].sum().detach().float().item()
            stats['rejected_sum'] = losing_logprobs[orpo_mask].sum().detach().float().item()
            stats['margin_sum'] = (winning_logprobs - losing_logprobs)[orpo_mask].sum().detach().float().item()
            stats['acc_sum'] = (winning_logprobs > losing_logprobs)[orpo_mask].float().sum().detach().float().item()
            stats['log_odds_sum'] = log_odds_ratio[orpo_mask].sum().detach().float().item()
        
        return sft_loss_mean, orpo_loss_mean, stats

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        fine_tuning = inputs.get("fine_tuning", None)
        first_input_key = next(k for k in inputs.keys() if "input_ids" in k)
        device = inputs[first_input_key].device

        if fine_tuning is None:
            fine_tuning = torch.zeros(inputs[first_input_key].size(0), device=device)
        fine_tuning = fine_tuning.to(device)  # [B], 1=ORPO, 0=SFT

        
        # 动态检测任务类型
        available_prefixes = sorted(set([k.split('_')[0] + '_' for k in inputs.keys() if '_winning_input_ids' in k]))
        num_tasks = len(available_prefixes)
        
        total_loss_scalar = 0.0
        
        for prefix in available_prefixes:
            type_name = prefix.rstrip('_').upper()
            
            sft_loss, orpo_loss, stats = self._compute_single_type_loss(
                model=model,
                winning_input_ids=inputs[f"{prefix}winning_input_ids"],
                winning_attention_mask=inputs[f"{prefix}winning_attention_mask"],
                losing_input_ids=inputs[f"{prefix}losing_input_ids"],
                losing_attention_mask=inputs[f"{prefix}losing_attention_mask"],
                input_length=inputs[f"{prefix}input_length"],
                fine_tuning=fine_tuning
            )
            
            self._update_accum_stats(type_name, stats)
            
            task_loss = sft_loss + self.lambda_reg * orpo_loss
            normalized_loss = (task_loss / num_tasks) / self.args.gradient_accumulation_steps
            self.accelerator.backward(normalized_loss)
            total_loss_scalar += normalized_loss.detach().float()
        
        if num_tasks > 0:
            self.accum_stats['total_samples'] += inputs[f"{available_prefixes[0]}winning_input_ids"].size(0)
        
        if self.accelerator.sync_gradients:
            if self.state.global_step % self.args.logging_steps == 0 and self.accelerator.is_main_process:
                self._perform_logging(num_tasks)
                self._init_accum_stats()
        
        return total_loss_scalar

    def _update_accum_stats(self, type_name, stats):
        if type_name not in self.accum_stats['metrics']:
            self.accum_stats['metrics'][type_name] = {'sft_sum': 0.0, 'sft_count': 0, 'orpo_sum': 0.0, 'orpo_count': 0}
        
        self.accum_stats['metrics'][type_name]['sft_sum'] += stats['sft_sum']
        self.accum_stats['metrics'][type_name]['sft_count'] += stats['sft_count']
        self.accum_stats['metrics'][type_name]['orpo_sum'] += stats['orpo_sum']
        self.accum_stats['metrics'][type_name]['orpo_count'] += stats['orpo_count']
        
        for key in ['chosen_sum', 'rejected_sum', 'margin_sum', 'acc_sum', 'log_odds_sum']:
            self.accum_stats['global'][key] += stats[key]

    def _perform_logging(self, num_tasks):
        safe_div = lambda x, y: x / y if y > 0 else 0.0
        
        # 计算各任务平均 loss
        sum_avg_sft, sum_avg_orpo = 0.0, 0.0
        table_rows = []
        
        for t in ['A', 'B', 'C']:
            if t in self.accum_stats['metrics'] and self.accum_stats['metrics'][t]['sft_count'] > 0:
                d = self.accum_stats['metrics'][t]
                avg_sft = safe_div(d['sft_sum'], d['sft_count'])
                avg_orpo = safe_div(d['orpo_sum'], d['sft_count'])
                sum_avg_sft += avg_sft
                sum_avg_orpo += avg_orpo
                table_rows.append(f"{avg_sft:.4f}/{avg_orpo:.4f}")
            else:
                table_rows.append("   -   /   -   ")
        
        global_avg_sft = safe_div(sum_avg_sft, num_tasks)
        global_avg_orpo = safe_div(sum_avg_orpo, num_tasks)
        global_total = global_avg_sft + self.lambda_reg * global_avg_orpo
        
        # ORPO 指标
        total_orpo = sum(self.accum_stats['metrics'][t].get('orpo_count', 0) for t in ['A', 'B', 'C'])
        avg_acc = safe_div(self.accum_stats['global']['acc_sum'], total_orpo)
        avg_margin = safe_div(self.accum_stats['global']['margin_sum'], total_orpo)

        # 【新增】找回 chosen, rejected, log_odds
        avg_chosen = safe_div(self.accum_stats['global']['chosen_sum'], total_orpo)
        avg_rejected = safe_div(self.accum_stats['global']['rejected_sum'], total_orpo)
        avg_log_odds = safe_div(self.accum_stats['global']['log_odds_sum'], total_orpo)
        
        # 3. 构造 TensorBoard 日志 (全面版)
        tb_logs = {
            "train/total_loss": global_total,
            "train/sft_loss": global_avg_sft,
            "train/orpo_loss": global_avg_orpo,
            # 核心指标
            "logits/accuracy": avg_acc,
            "logps/margin": avg_margin,         # log(P_w) - log(P_l)
            "logps/log_odds_ratio": avg_log_odds, # log(Odds_w) - log(Odds_l) [ORPO 核心]
            "logps/chosen": avg_chosen,
            "logps/rejected": avg_rejected,
        }
        self.log(tb_logs)
        
        # 4. 增强版控制台表格
        if (self.state.global_step // self.args.logging_steps) % 10 == 1:
            # 调整表头宽度以容纳 LogOdd
            header = (
                f"+" + "-"*7 + "+" + "-"*35 + "+" + "-"*18 + "+" + "-"*17 + "+" + "-"*17 + "+" + "-"*17 + "+\n"
                f"| Step  |    ORPO Metrics (Acc/Marg/Odd)    | Total (Avg) Loss |   SFT/ORPO (A)  |   SFT/ORPO (B)  |   SFT/ORPO (C)  |\n"
                f"+" + "-"*7 + "+" + "-"*35 + "+" + "-"*18 + "+" + "-"*17 + "+" + "-"*17 + "+" + "-"*17 + "+"
            )
            self.logger.info(header)
        
        # 格式化数据行：增加 LogOdd 显示
        metrics_str = f"{avg_acc:.2f} | {avg_margin:+.2f} | {avg_log_odds:+.2f}"
        row = f"| {self.state.global_step:<5} | {metrics_str:^33} | {global_total:^16.4f} | {table_rows[0]:^15} | {table_rows[1]:^15} | {table_rows[2]:^15} |"
        self.logger.info(row)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """用于 evaluation"""
        fine_tuning = inputs.get("fine_tuning", None)
        first_input_key = next(k for k in inputs.keys() if "input_ids" in k)
        device = inputs[first_input_key].device

        if fine_tuning is None:
            fine_tuning = torch.zeros(inputs[first_input_key].size(0), device=device)
        fine_tuning = fine_tuning.to(device)  # [B], 1=ORPO, 0=SFT
        
        available_prefixes = sorted(set([k.split('_')[0] + '_' for k in inputs.keys() if '_winning_input_ids' in k]))
        num_tasks = len(available_prefixes)
        
        total_sft, total_orpo = 0, 0
        for prefix in available_prefixes:
            sft_loss, orpo_loss, _ = self._compute_single_type_loss(
                model, inputs[f"{prefix}winning_input_ids"], inputs[f"{prefix}winning_attention_mask"],
                inputs[f"{prefix}losing_input_ids"], inputs[f"{prefix}losing_attention_mask"],
                inputs[f"{prefix}input_length"], fine_tuning
            )
            total_sft += sft_loss
            total_orpo += orpo_loss
        
        if num_tasks > 0:
            total_loss = (total_sft / num_tasks) + self.lambda_reg * (total_orpo / num_tasks)
        else:
            total_loss = torch.tensor(0.0, device=device)
        
        return (total_loss, None) if return_outputs else total_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        model.eval()
        with torch.no_grad():
            loss = self.compute_loss(model, inputs, return_outputs=False)
        return (loss.detach() if loss is not None else None, None, None)


# =============================================================================
# 8. Model Creation
# =============================================================================

def create_model_and_tokenizer(model_name: str, config: TwoStageHierarchicalConfig, 
                                sft_model_path: str = None, gpu_id: str = "auto"):
    logger = logging.getLogger(__name__)
    
    # --------------------------------------------------
    # 1. device_map + max_memory（核心：防 GPU0 OOM）
    # --------------------------------------------------
    if gpu_id == "auto":
        device_map = "auto"
        max_memory = {
            0: "20GiB",   # 👈 刻意压低 GPU 0
            1: "38GiB",
            2: "38GiB",
            # 3: "38GiB",
        }
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device_map = {"": f"cuda:{gpu_id}"}
        max_memory = None
    
    logger.info(f"Loading tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if sft_model_path:
        logger.info(f"Loading base model + LoRA from: {sft_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device_map, max_memory=max_memory, trust_remote_code=True
        )

        # ✅【关键】彻底关闭 KV cache（否则 checkpoint 不省显存）
        base_model.config.use_cache = False
        
        model = PeftModel.from_pretrained(base_model, sft_model_path, is_trainable=True)
    else:
        logger.info(f"Loading base model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device_map, max_memory=max_memory, trust_remote_code=True
        )

        # ✅【关键】彻底关闭 KV cache
        model.config.use_cache = False

        if config.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, r=config.lora_r, lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            model = get_peft_model(model, lora_config)
    
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    
    model.print_trainable_parameters()
    return model, tokenizer


# =============================================================================
# 9. Training Functions
# =============================================================================

def run_stage1_h_sft(config, data_file, output_dir, tensorboard_dir, train_ratio=0.85, gpu_id="auto"):
    """Stage 1: Hierarchical SFT"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Stage 1: Hierarchical SFT (A+B+C)")
    logger.info("Loss = (SFT_A + SFT_B + SFT_C) / 3")
    logger.info("=" * 60)
    
    model, tokenizer = create_model_and_tokenizer(config.model_name, config, gpu_id=gpu_id)
    dataset = HierarchicalSFTDataset(data_file, tokenizer, config.max_seq_length)
    
    if len(dataset) == 0:
        logger.error("No valid SFT samples!")
        return None
    
    train_size = int(len(dataset) * train_ratio)
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size], generator=torch.Generator().manual_seed(42)
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=config.sft_num_epochs,
        per_device_train_batch_size=config.batch_size, per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.sft_learning_rate, weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio, logging_steps=config.logging_steps,
        eval_strategy="steps", eval_steps=config.eval_steps, save_steps=config.save_steps,
        save_total_limit=3, report_to="tensorboard", logging_dir=tensorboard_dir,
        dataloader_num_workers=4, remove_unused_columns=False, bf16=True,
        gradient_checkpointing=True, dataloader_pin_memory=True, seed=42,
    )
    
    trainer = HierarchicalTrainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset,
        tokenizer=tokenizer, data_collator=lambda b: hierarchical_collate_fn(b, tokenizer),
        beta=config.orpo_beta, lambda_reg=0.0  # Stage 1: lambda=0
    )
    
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Stage 1 completed! Model saved to: {output_dir}")
    return output_dir


def run_stage2_h_orpo(config, data_file, sft_model_path, output_dir, tensorboard_dir, train_ratio=0.85, gpu_id="auto"):
    """Stage 2: Hierarchical ORPO"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Stage 2: Hierarchical ORPO (A+B+C)")
    logger.info(f"Loss = (SFT_A + SFT_B + SFT_C) / 3 + {config.lambda_orpo} * (ORPO_A + ORPO_B + ORPO_C) / 3")
    logger.info("=" * 60)
    
    model, tokenizer = create_model_and_tokenizer(config.model_name, config, sft_model_path, gpu_id)
    dataset = HierarchicalORPODataset(data_file, tokenizer, config.max_seq_length)
    
    if len(dataset) == 0:
        logger.warning("No valid ORPO samples!")
        return None
    
    train_size = int(len(dataset) * train_ratio)
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size], generator=torch.Generator().manual_seed(42)
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=config.orpo_num_epochs,
        per_device_train_batch_size=config.batch_size, per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.orpo_learning_rate, weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio, logging_steps=config.logging_steps,
        eval_strategy="steps", eval_steps=config.eval_steps, save_steps=config.save_steps,
        save_total_limit=3, report_to="tensorboard", logging_dir=tensorboard_dir,
        dataloader_num_workers=4, remove_unused_columns=False, bf16=True,
        gradient_checkpointing=True, dataloader_pin_memory=True, seed=42,
    )
    
    trainer = HierarchicalTrainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset,
        tokenizer=tokenizer, data_collator=lambda b: hierarchical_collate_fn(b, tokenizer),
        beta=config.orpo_beta, lambda_reg=config.lambda_orpo  # Stage 2: lambda > 0
    )
    
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Stage 2 completed! Model saved to: {output_dir}")
    return output_dir


# =============================================================================
# 10. Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Two-Stage Hierarchical Training (H-SFT → H-ORPO)")
    
    parser.add_argument("--stage", type=str, required=True, choices=["stage1_h_sft", "stage2_h_orpo", "both"])
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./two_stage_h_output")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--sft_model_path", type=str, default="")
    
    parser.add_argument("--sft_learning_rate", type=float, default=1e-4)
    parser.add_argument("--sft_num_epochs", type=int, default=3)
    parser.add_argument("--orpo_learning_rate", type=float, default=1e-6)
    parser.add_argument("--orpo_num_epochs", type=int, default=6)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lambda_orpo", type=float, default=1.0)
    
    parser.add_argument("--max_seq_length", type=int, default=6144)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    
    parser.add_argument("--save_steps", type=int, default=300)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--train_ratio", type=float, default=0.85)
    parser.add_argument("--gpu_id", type=str, default="auto")
    
    args = parser.parse_args()
    logger = setup_logging()
    
    config = TwoStageHierarchicalConfig(
        model_name=args.model_name, max_seq_length=args.max_seq_length,
        sft_learning_rate=args.sft_learning_rate, sft_num_epochs=args.sft_num_epochs,
        orpo_learning_rate=args.orpo_learning_rate, orpo_num_epochs=args.orpo_num_epochs,
        orpo_beta=args.beta, lambda_orpo=args.lambda_orpo,
        batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio, weight_decay=args.weight_decay,
        save_steps=args.save_steps, eval_steps=args.eval_steps, logging_steps=args.logging_steps,
        use_lora=args.use_lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
    )
    
    stage1_dir = os.path.join(args.output_dir, "stage1_h_sft")
    stage2_dir = os.path.join(args.output_dir, "stage2_h_orpo")
    stage1_tb = os.path.join(args.output_dir, "tb_stage1")
    stage2_tb = os.path.join(args.output_dir, "tb_stage2")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    stage = TrainingStage(args.stage)
    
    if stage in [TrainingStage.STAGE1_H_SFT, TrainingStage.BOTH]:
        os.makedirs(stage1_dir, exist_ok=True)
        os.makedirs(stage1_tb, exist_ok=True)
        sft_out = run_stage1_h_sft(config, args.data_file, stage1_dir, stage1_tb, args.train_ratio, args.gpu_id)
        if stage == TrainingStage.BOTH and sft_out:
            args.sft_model_path = sft_out
    
    if stage in [TrainingStage.STAGE2_H_ORPO, TrainingStage.BOTH]:
        if not args.sft_model_path:
            logger.error("Stage 2 requires --sft_model_path!")
            return
        os.makedirs(stage2_dir, exist_ok=True)
        os.makedirs(stage2_tb, exist_ok=True)
        run_stage2_h_orpo(config, args.data_file, args.sft_model_path, stage2_dir, stage2_tb, args.train_ratio, args.gpu_id)
    
    logger.info("=" * 60)
    logger.info("Training Completed!")
    logger.info(f"TensorBoard: tensorboard --logdir {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()