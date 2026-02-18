import argparse
import json
import os
from typing import Any, Dict, List, Tuple
from transformers import AutoTokenizer

# 尝试导入绘图库
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOT_LIB = True
except ImportError:
    HAS_PLOT_LIB = False
    print("Warning: matplotlib or numpy not installed. Distribution plot will be skipped.")

# =========================
# Utils & Core Logic
# =========================

def format_segments(segments: List[Dict[str, Any]], include_supp: bool = False) -> str:
    """
    格式化代码片段。
    根据新需求，默认策略改为只包含 Suspicious 部分 (include_supp=False)。
    """
    lines = []
    if not segments:
        return ""

    for idx, seg in enumerate(segments, start=1):
        susp = seg.get("suspicious", {}) or {}
        supp = seg.get("supplementary", {}) or {}
        
        lines.append(f"--- Segment Group {idx} ---")
        file_path = susp.get('file', '') or ''
        content = susp.get('content', '') or ''
        lines.append(f"[Suspicious] file: {file_path}")
        lines.append("```")
        lines.append(content.rstrip())
        lines.append("```")
        
        # 只有显式要求包含 supplement 时才添加
        if include_supp:
            supp_file = supp.get('file', '') or ''
            supp_content = supp.get('content', '') or ''
            if supp_file or supp_content:
                lines.append(f"[Supplementary] file: {supp_file}")
                lines.append("```")
                lines.append(supp_content.rstrip())
                lines.append("```")
        lines.append("") 
    return "\n".join(lines)

def parse_patch_to_segments(patch: str) -> List[Dict[str, Any]]:
    """
    将 unified diff patch 解析为 segments 格式。
    提取文件路径、'-' 行（suspicious）以及上下文行。
    
    返回格式: [{"suspicious": {"file": str, "content": str}, "supplementary": {...}}, ...]
    """
    if not patch:
        return []
    
    segments = []
    current_file = ""
    current_suspicious_lines = []
    
    for line in patch.splitlines():
        # 提取文件路径 (--- a/path/to/file 或 +++ b/path/to/file)
        if line.startswith("--- a/") or line.startswith("--- "):
            # 保存之前的 segment
            if current_file and current_suspicious_lines:
                segments.append({
                    "suspicious": {
                        "file": current_file,
                        "content": "\n".join(current_suspicious_lines)
                    },
                    "supplementary": {}
                })
            # 开始新文件
            current_file = line[6:] if line.startswith("--- a/") else line[4:]
            current_suspicious_lines = []
        elif line.startswith("+++ "):
            # 跳过 +++ 行，文件路径已从 --- 获取
            continue
        elif line.startswith("@@"):
            # hunk header，跳过
            continue
        elif line.startswith("-") and not line.startswith("---"):
            # suspicious line: 删除的行
            current_suspicious_lines.append(line)
        elif line.startswith("+") and not line.startswith("+++"):
            # 修复行，不加入 suspicious
            continue
        elif line.startswith(" "):
            # 上下文行
            current_suspicious_lines.append(line)
        elif not line.strip():
            # 空行作为上下文
            current_suspicious_lines.append("")
    
    # 保存最后一个 segment
    if current_file and current_suspicious_lines:
        segments.append({
            "suspicious": {
                "file": current_file,
                "content": "\n".join(current_suspicious_lines)
            },
            "supplementary": {}
        })
    
    return segments

def load_swe_live_test_index(path: str) -> Dict[str, Dict[str, Any]]:
    """
    加载 swe-live-test.json 并按 instance_id 建立索引。
    """
    if not os.path.exists(path):
        print(f"Warning: SWE-live-test file not found: {path}")
        return {}
    
    print(f"Loading SWE-live-test index from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    index = {}
    for item in data:
        instance_id = item.get("instance_id", "")
        if instance_id:
            index[instance_id] = item
    print(f"  Loaded {len(index)} instances from swe-live-test.json")
    return index

def process_round12_file(path: str, swe_index: Dict[str, Dict[str, Any]], 
                          fout, log_fout, stats: Dict[str, int],
                          processor: 'SmartProcessor',
                          stats_collector_list: List[Dict],
                          final_totals_list: List[int]):
    """
    处理 round12_rc_rs.jsonl 文件，与 swe-live-test.json 对齐后生成训练数据。
    """
    if not os.path.exists(path):
        print(f"Warning: File not found {path}")
        return
    
    print(f"Processing round12 file: {path}")
    filename = os.path.basename(path)
    
    with open(path, "r", encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            if i % 100 == 0:
                print(f"  Processed {i} lines...", end='\r')
            line = line.strip()
            if not line or line == "null":
                continue
            
            try:
                rec = json.loads(line)
                if rec is None:
                    continue
                
                instance_id = rec.get("instance_id", "")
                if not instance_id or instance_id not in swe_index:
                    stats["bad"] += 1
                    continue
                
                swe_item = swe_index[instance_id]
                problem_statement = swe_item.get("problem_statement", "")
                patch = swe_item.get("patch", "")
                
                # 将 patch 解析为 segments 格式
                segments = parse_patch_to_segments(patch)
                
                # 构造与原有格式对齐的记录
                aligned_rec = {
                    "segments": segments,
                    "problem_statement": problem_statement,
                    "fine_tuning": "ORPO",  # 新数据全部为 ORPO
                    # round2 作为 win (new)，round1 作为 lose (old)
                    "new_root_cause": rec.get("round2_root_cause", ""),
                    "new_repair_suggestion": rec.get("round2_repair_suggestion", ""),
                    "old_root_cause": rec.get("round1_root_cause", ""),
                    "old_repair_suggestion": rec.get("round1_repair_suggestion", ""),
                    "index": i
                }
                
                out, log_info = processor.process_record(aligned_rec)
                
                # 收集绘图所需数据
                stats_collector_list.append(log_info['original_stats_breakdown'])
                final_totals_list.append(log_info['final_total'])
                
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                
                org_stats = log_info['original_stats_breakdown']
                log_line = (
                    f"[{filename}][Idx:{i}][{log_info['type']}] "
                    f"Tok: {log_info['original_total']}->{log_info['final_total']} | "
                    f"Brk: P={org_stats['problem']}, B={org_stats['buggy']}, Out={org_stats['output']} | "
                    f"Acts: {', '.join(log_info['actions'])}\n"
                )
                log_fout.write(log_line)
                
                ft = out["fine_tuning"].upper()
                if ft:
                    stats[ft] = stats.get(ft, 0) + 1
                    
            except Exception as e:
                stats["bad"] += 1
                
    print(f"  Finished file: {path}          ")


class SmartProcessor:
    def __init__(self, tokenizer_path: str, max_tokens: int):
        print(f"Loading tokenizer from: {tokenizer_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, 
            trust_remote_code=True,
            use_fast=True
        )
        self.max_tokens = max_tokens
        # 预留少量 Buffer 避免边界误差，也可以设为 0
        self.safe_limit = max_tokens - 300 #300是模板嵌入大小预留 
        print(f"Token Limit: {self.max_tokens}")

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def truncate_by_count(self, text: str, remove_count: int) -> str:
        """
        从文本末尾移除指定数量的 Token
        """
        if not text or remove_count <= 0:
            return text
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if remove_count >= len(ids):
            return ""
        keep_len = len(ids) - remove_count
        return self.tokenizer.decode(ids[:keep_len], skip_special_tokens=True)

    def process_record(self, rec: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        raw_segments = rec.get("segments")
        problem_statement = rec.get("problem_statement", "")
        fine_tuning = rec.get("fine_tuning", "").strip()
        ft_upper = fine_tuning.upper()
        
        lose_rc = rec.get("old_root_cause", "")
        lose_rs = rec.get("old_repair_suggestion", "")
        win_rc = rec.get("new_root_cause", "")
        win_rs = rec.get("new_repair_suggestion", "")

        if ft_upper == "SFT":
            if not win_rc.strip(): win_rc = lose_rc
            if not win_rs.strip(): win_rs = lose_rs

        # 1. 初始构建：根据新要求，buggy_slice 默认只取 suspicious
        current_buggy_slice = format_segments(raw_segments, include_supp=False)
        current_problem = problem_statement
        current_win_rc = win_rc
        current_win_rs = win_rs
        current_lose_rc = lose_rc
        current_lose_rs = lose_rs
        
        is_orpo = (ft_upper == "ORPO")

        # 辅助函数：计算当前各部分 Token 及总和
        def get_counts(p, b, wrc, wrs, lrc, lrs):
            c_p = self.count_tokens(p)
            c_b = self.count_tokens(b)
            c_w = self.count_tokens(wrc) + self.count_tokens(wrs)
            c_l = (self.count_tokens(lrc) + self.count_tokens(lrs)) if is_orpo else 0
            return c_p, c_b, c_w, c_l, (c_p + c_b + c_w + c_l)

        # 初始计算
        tok_p, tok_b, tok_w, tok_l, total = get_counts(
            current_problem, current_buggy_slice, 
            current_win_rc, current_win_rs, current_lose_rc, current_lose_rs
        )

        # 记录原始分布用于绘图
        original_stats = {
            "total": total,
            "problem": tok_p,
            "buggy": tok_b,
            "output": tok_w + tok_l # Win + Lose (if ORPO)
        }
        
        actions_taken = [] 

        # ==========================
        # 新的截断逻辑
        # ==========================
        
        if total > self.safe_limit:
            overflow = total - self.safe_limit
            
            # --- 阶段 1: 尝试分摊截断 Problem 和 Buggy (Max 30%) ---
            denom = tok_p + tok_b
            if denom > 0:
                ratio_p = tok_p / denom
                # 计算目标削减量
                target_cut_p = int(overflow * ratio_p)
                target_cut_b = int(overflow * (1 - ratio_p))
                
                # 计算 30% 限制
                limit_cut_p = int(tok_p * 0.3)
                limit_cut_b = int(tok_b * 0.3)
                
                # 实际执行削减（取 min）
                actual_cut_p = min(target_cut_p, limit_cut_p)
                actual_cut_b = min(target_cut_b, limit_cut_b)
                
                if actual_cut_p > 0:
                    current_problem = self.truncate_by_count(current_problem, actual_cut_p)
                if actual_cut_b > 0:
                    current_buggy_slice = self.truncate_by_count(current_buggy_slice, actual_cut_b)
                
                actions_taken.append(f"SoftTrunc_PB_30%(-{actual_cut_p+actual_cut_b})")
            
            # 重新计算 Total
            tok_p, tok_b, tok_w, tok_l, total = get_counts(
                current_problem, current_buggy_slice, 
                current_win_rc, current_win_rs, current_lose_rc, current_lose_rs
            )

            # --- 阶段 2: 如果依然超标 ---
            if total > self.safe_limit:
                current_overflow = total - self.safe_limit
                
                if not is_orpo: 
                    # === SFT 策略 ===
                    # "继续截断 current_problem, current_buggy_slice 直到成功"
                    # 这里不再受 30% 限制，直接按比例把剩余的溢出切掉
                    denom = tok_p + tok_b
                    if denom > 0:
                        ratio_p = tok_p / denom
                        final_cut_p = int(current_overflow * ratio_p) + 1 # +1 确保切够
                        final_cut_b = int(current_overflow * (1 - ratio_p)) + 1
                        
                        current_problem = self.truncate_by_count(current_problem, final_cut_p)
                        current_buggy_slice = self.truncate_by_count(current_buggy_slice, final_cut_b)
                        actions_taken.append(f"ForceTrunc_SFT_PB(-{final_cut_p+final_cut_b})")
                
                else:
                    # === ORPO 策略 ===
                    # 1. 先尝试截断 Lose RC/RS (Max 30%)
                    tok_lose_rc = self.count_tokens(current_lose_rc)
                    tok_lose_rs = self.count_tokens(current_lose_rs)
                    denom_lose = tok_lose_rc + tok_lose_rs
                    
                    cut_lose_success = False
                    
                    if denom_lose > 0:
                        ratio_lrc = tok_lose_rc / denom_lose
                        
                        target_cut_l = current_overflow
                        # 30% Limit
                        max_cut_lose = int(denom_lose * 0.3)
                        
                        actual_cut_total_l = min(target_cut_l, max_cut_lose)
                        
                        cut_lrc = int(actual_cut_total_l * ratio_lrc)
                        cut_lrs = actual_cut_total_l - cut_lrc
                        
                        if actual_cut_total_l > 0:
                            current_lose_rc = self.truncate_by_count(current_lose_rc, cut_lrc)
                            current_lose_rs = self.truncate_by_count(current_lose_rs, cut_lrs)
                            actions_taken.append(f"SoftTrunc_ORPO_Lose_30%(-{actual_cut_total_l})")
                    
                    # 重新计算看是否达标
                    tok_p, tok_b, tok_w, tok_l, total = get_counts(
                        current_problem, current_buggy_slice, 
                        current_win_rc, current_win_rs, current_lose_rc, current_lose_rs
                    )
                    
                    # 2. 如果 ORPO 还是超标 -> 强行截断 Problem/Buggy
                    if total > self.safe_limit:
                        final_overflow = total - self.safe_limit
                        denom_pb = tok_p + tok_b
                        if denom_pb > 0:
                            ratio_p = tok_p / denom_pb
                            force_cut_p = int(final_overflow * ratio_p) + 1
                            force_cut_b = int(final_overflow * (1 - ratio_p)) + 1
                            
                            current_problem = self.truncate_by_count(current_problem, force_cut_p)
                            current_buggy_slice = self.truncate_by_count(current_buggy_slice, force_cut_b)
                            actions_taken.append(f"ForceTrunc_ORPO_PB(-{force_cut_p+force_cut_b})")

        # 最终计算 Total
        _, _, _, _, final_total = get_counts(
            current_problem, current_buggy_slice, 
            current_win_rc, current_win_rs, current_lose_rc, current_lose_rs
        )

        if not actions_taken:
            actions_taken.append("None")

        processed_data = {
            "problem_statement": current_problem,
            "buggy_slice": current_buggy_slice,
            "fine_tuning": fine_tuning,
            "win_rc": current_win_rc,
            "lose_rc": current_lose_rc,
            "win_rs": current_win_rs,
            "lose_rs": current_lose_rs,
            "index": rec.get("index") 
        }
        
        log_info = {
            "type": ft_upper,
            "original_total": original_stats["total"],
            "final_total": final_total,
            "actions": actions_taken,
            "original_stats_breakdown": original_stats
        }

        return processed_data, log_info

# =========================
# Analysis & Plotting (Modified)
# =========================

def analyze_and_plot_distribution(stats_list: List[Dict], final_totals: List[int], output_path: str, current_max: int):
    """
    绘制分布图：
    图1：原始数据的组成分布（Total, Problem, Buggy, Output），在同一张图中叠加展示。
    图2：处理后的 Total 分布。
    """
    if not HAS_PLOT_LIB or not stats_list:
        return

    # 提取数据
    orig_totals = [x['total'] for x in stats_list]
    probs = [x['problem'] for x in stats_list]
    buggys = [x['buggy'] for x in stats_list]
    outputs = [x['output'] for x in stats_list]
    
    # 准备绘图
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # === Subplot 1: Original Composition ===
    ax1 = axes[0]
    
    # 设置 Bins
    max_val = max(np.max(orig_totals), current_max)
    # 稍微限制一下绘图上限，避免极个别超长数据拉跨整个图
    display_limit = max(current_max * 1.5, np.percentile(orig_totals, 99))
    bins = np.linspace(0, display_limit, 100)
    
    # 绘制各部分
    # 使用 step 类型或者透明度较高的 bar 来叠加
    ax1.hist(orig_totals, bins=bins, color='gray', alpha=0.3, label='Total (Original)', edgecolor='black')
    ax1.hist(probs, bins=bins, histtype='step', linewidth=2, color='blue', label='Problem Statement')
    ax1.hist(buggys, bins=bins, histtype='step', linewidth=2, color='orange', label='Buggy Slice')
    ax1.hist(outputs, bins=bins, histtype='step', linewidth=2, color='green', label='Output (Win+Lose)')
    
    ax1.axvline(current_max, color='red', linestyle='--', linewidth=2, label=f'Max Limit ({current_max})')
    
    ax1.set_title('Original Token Distribution by Component', fontsize=14)
    ax1.set_xlabel('Token Count', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # === Subplot 2: Final Distribution (After Processing) ===
    ax2 = axes[1]
    
    # 将超过 max 的归并在最后一组（虽然理论上处理后应该都小于等于 max，但画出来确认一下）
    final_data = np.array(final_totals)
    clip_bins = np.linspace(0, current_max + 100, 100) # 稍微多一点空间看是否溢出
    
    n, _, patches_list = ax2.hist(final_data, bins=clip_bins, color='purple', alpha=0.7, edgecolor='white', label='Final Total')
    
    # 标记 P80
    p80 = np.percentile(final_data, 80)
    ax2.axvline(p80, color='gold', linestyle='--', linewidth=2, label=f'P80: {int(p80)}')
    ax2.axvline(current_max, color='red', linestyle='-', linewidth=2, label=f'Limit: {current_max}')

    ax2.set_title(f'Final Token Distribution (After Truncation)', fontsize=14)
    ax2.set_xlabel('Token Count', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 打印简要统计
    print("\n" + "="*50)
    print("DISTRIBUTION STATISTICS")
    print("-" * 30)
    print(f"Original Mean Total: {np.mean(orig_totals):.1f}")
    print(f"Original Max Total : {np.max(orig_totals)}")
    print(f"Original P80 Total : {np.percentile(orig_totals, 80):.1f}")
    print("-" * 30)
    print(f"Final Mean Total   : {np.mean(final_totals):.1f}")
    print(f"Final Max Total    : {np.max(final_totals)}")
    print("="*50)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"[Plot Saved] {output_path}")
    plt.close()

# =========================
# IO processing
# =========================

def process_file(path: str, fout, log_fout, stats: Dict[str, int], 
                 processor: SmartProcessor, 
                 stats_collector_list: List[Dict], 
                 final_totals_list: List[int]):
    if not os.path.exists(path):
        print(f"Warning: File not found {path}")
        return

    print(f"Processing file: {path}")
    filename = os.path.basename(path)

    with open(path, "r", encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            if i % 100 == 0:
                print(f"  Processed {i} lines...", end='\r')
            line = line.strip()
            if not line or line == "null": continue
            try:
                rec = json.loads(line)
                if rec is None: continue
                
                out, log_info = processor.process_record(rec)
                
                # 收集绘图所需数据
                stats_collector_list.append(log_info['original_stats_breakdown'])
                final_totals_list.append(log_info['final_total'])

                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                
                rec_idx = rec.get("index", i)
                org_stats = log_info['original_stats_breakdown']
                
                log_line = (
                    f"[{filename}][Idx:{rec_idx}][{log_info['type']}] "
                    f"Tok: {log_info['original_total']}->{log_info['final_total']} | "
                    f"Brk: P={org_stats['problem']}, B={org_stats['buggy']}, Out={org_stats['output']} | "
                    f"Acts: {', '.join(log_info['actions'])}\n"
                )
                log_fout.write(log_line)
                
                ft = out["fine_tuning"].upper()
                if ft: stats[ft] = stats.get(ft, 0) + 1
            except Exception as e:
                stats["bad"] += 1
    print(f"  Finished file: {path}          ")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_input", default="")
    ap.add_argument("--old_input", default="")
    # 新增两个输入参数
    ap.add_argument("--round12_input", default="")
    ap.add_argument("--swe_live_input", default="")
    # 修改输出文件名
    ap.add_argument("--output", default="")
    ap.add_argument("--log_file", default="")
    ap.add_argument("--plot_file", default="")
    ap.add_argument("--tokenizer_path", default="")
    ap.add_argument("--max_tokens", type=int, default=6144)
    args = ap.parse_args()

    processor = SmartProcessor(args.tokenizer_path, args.max_tokens)
    stats = {"SFT": 0, "ORPO": 0, "bad": 0}
    
    # 用于绘图的数据列表
    stats_collector = [] # 存原始分布字典
    final_totals = []    # 存最终 Token 数

    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.plot_file), exist_ok=True)

    # 预加载 swe-live-test.json 索引
    swe_index = load_swe_live_test_index(args.swe_live_input)

    with open(args.output, "w", encoding="utf-8") as fout, \
         open(args.log_file, "w", encoding="utf-8") as log_fout:
        log_fout.write("File | Index | Type | Tokens (Org->Fin) | Breakdown (Org) | Actions\n")
        log_fout.write("="*120 + "\n")
        
        process_file(args.new_input, fout, log_fout, stats, processor, stats_collector, final_totals)
        process_file(args.old_input, fout, log_fout, stats, processor, stats_collector, final_totals)

        # 处理新增的 round12 文件
        process_round12_file(args.round12_input, swe_index, fout, log_fout, stats, 
                             processor, stats_collector, final_totals)

    print("\n[Processing Done]")
    print(f"SFT={stats.get('SFT',0)}, ORPO={stats.get('ORPO',0)}, bad={stats['bad']}")
    
    # 执行分析与绘图
    analyze_and_plot_distribution(stats_collector, final_totals, args.plot_file, args.max_tokens)

    print(f"[written] {args.output}")
    print(f"[log]     {args.log_file}")
    print(f"[plot]    {args.plot_file}")

if __name__ == "__main__":
    main()