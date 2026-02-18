"""
评测指标计算脚本（高速版 v2）：
- BERTScore 批量计算（10-20x 加速）
- ESS/RAS 多进程并行
- 多模型文件并发处理
- 支持断点续传
"""
import json
import sys
import os
from typing import Any, Dict, List, Tuple
from statistics import mean, variance, stdev
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
import threading

# ========== 文件路径配置 ==========
METRICS_DIR = ""
sys.path.insert(0, METRICS_DIR)
sys.path.insert(0, os.path.join(METRICS_DIR, "BERTScore"))
sys.path.insert(0, os.path.join(METRICS_DIR, "BLEU"))
sys.path.insert(0, os.path.join(METRICS_DIR, "ESS_and_RAS"))

from bleu_score import compute_bleu
from esra_score import compute_ess, compute_ras


REFERENCE_FILE = ""


# 模型输出文件
MODEL_FILES = {
    "only_orpo_8B": "",
    "two_stage_orpo_14B": "",
    "two_stage_h_orpo_14B": "",

}

# 输出文件
OUTPUT_JSONL = ""
OUTPUT_REPORT = ""
CHECKPOINT_FILE = ""

# ========== 并行配置 ==========
NUM_TOTAL_CORES = multiprocessing.cpu_count()
NUM_MODEL_WORKERS = min(len(MODEL_FILES), 3)  # 同时处理的模型文件数（根据内存调整）
NUM_ESS_RAS_WORKERS_PER_MODEL = max(1, (NUM_TOTAL_CORES - 2) // NUM_MODEL_WORKERS)  # 每个模型的 ESS/RAS 进程数

# 线程锁（用于打印和检查点保存）
print_lock = threading.Lock()
checkpoint_lock = threading.Lock()
bertscore_lock = threading.Lock()  # 新增：BERTScore 锁（避免多线程并发加载模型）

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def safe_print(*args, **kwargs):
    """线程安全的打印"""
    with print_lock:
        print(*args, **kwargs)

def batch_bertscore(candidates: List[str], references: List[str]) -> List[Tuple[float, float, float]]:
    """
    批量计算 BERTScore（核心加速）
    注意：需要加锁，避免多线程并发加载模型导致冲突
    """
    from bert_score import score
    
    # 处理空字符串
    valid_indices = []
    valid_cands = []
    valid_refs = []
    
    for i, (c, r) in enumerate(zip(candidates, references)):
        if c and r:
            valid_indices.append(i)
            valid_cands.append(c)
            valid_refs.append(r)
    
    # 初始化结果
    results = [(0.0, 0.0, 0.0)] * len(candidates)
    
    if valid_cands:
        # 使用锁保护 BERTScore 计算（避免多线程并发加载模型）
        with bertscore_lock:
            P, R, F1 = score(
                valid_cands, 
                valid_refs, 
                model_type="bert-base-uncased",
                verbose=False,
                batch_size=32
            )
        
        for idx, (p, r, f1) in zip(valid_indices, zip(P.tolist(), R.tolist(), F1.tolist())):
            results[idx] = (p, r, f1)
    
    return results


def compute_bleu_batch(pairs: List[Tuple[str, str]]) -> List[float]:
    """批量计算 BLEU"""
    results = []
    for pred, ref in pairs:
        if pred and ref:
            results.append(compute_bleu(pred, ref))
        else:
            results.append(0.0)
    return results


# 全局 spaCy 模型（用于多进程初始化）
_nlp = None

def init_worker():
    """进程池初始化：加载 spaCy 模型"""
    global _nlp
    import spacy
    _nlp = spacy.load("en_core_web_sm")


def compute_ess_ras_worker(args):
    """Worker 函数：使用全局 nlp"""
    global _nlp
    idx, pred_rc, ref_rc, pred_rs, ref_rs = args
    
    ess = compute_ess(ref_rc, pred_rc, nlp=_nlp) if pred_rc and ref_rc else 0.0
    ras = compute_ras(ref_rs, pred_rs, nlp=_nlp) if pred_rs and ref_rs else 0.0
    
    return idx, ess, ras


def compute_ess_ras_parallel(data_list: List[Tuple], num_workers: int, model_name: str = "") -> Dict[int, Tuple[float, float]]:
    """
    多进程并行计算 ESS/RAS
    """
    results = {}
    
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
        futures = {executor.submit(compute_ess_ras_worker, item): item[0] for item in data_list}
        
        # 使用 position 参数避免进度条重叠
        for future in tqdm(as_completed(futures), total=len(futures), 
                          desc=f"[{model_name}] ESS/RAS", ncols=80, leave=False):
            idx, ess, ras = future.result()
            results[idx] = (ess, ras)
    
    return results

def process_model_standalone(model_name: str, model_file: str, ref_dict: Dict, 
                             num_ess_workers: int) -> Tuple[str, List[Dict]]:
    """
    处理单个模型的所有数据（独立函数，用于并发调用）
    返回 (model_name, results)
    """
    safe_print(f"\n🚀 [{model_name}] 开始处理...")
    safe_print(f"   [{model_name}] 文件: {model_file}")
    safe_print(f"   [{model_name}] ESS/RAS 进程数: {num_ess_workers}")
    
    model_data = read_jsonl(model_file)
    safe_print(f"   [{model_name}] 模型输出条数: {len(model_data)}")
    
    # 准备数据
    valid_items = []
    for item in model_data:
        instance_id = item["instance_id"]
        ref_item = ref_dict.get(instance_id)
        if ref_item:
            valid_items.append({
                "instance_id": instance_id,
                "pred_rc": item.get("root_cause", ""),
                "pred_rs": item.get("repair_suggestion", ""),
                "ref_rc": ref_item.get("root_cause", ""),
                "ref_rs": ref_item.get("repair_suggestion", ""),
                "status": item.get("status", ""),
            })
    
    safe_print(f"   [{model_name}] 有效数据条数: {len(valid_items)}")
    
    # ========== 阶段1：批量计算 BERTScore ==========
    safe_print(f"   [{model_name}] 📊 阶段1/3: BERTScore...")
    
    rc_cands = [item["pred_rc"] for item in valid_items]
    rc_refs = [item["ref_rc"] for item in valid_items]
    rs_cands = [item["pred_rs"] for item in valid_items]
    rs_refs = [item["ref_rs"] for item in valid_items]
    
    rc_bert_results = batch_bertscore(rc_cands, rc_refs)
    rs_bert_results = batch_bertscore(rs_cands, rs_refs)

    # ========== 阶段2：计算 BLEU ==========
    safe_print(f"   [{model_name}] 📊 阶段2/3: BLEU...")
    rc_bleu_results = []
    rs_bleu_results = []
    
    for item in valid_items:
        rc_bleu_results.append(compute_bleu(item["pred_rc"], item["ref_rc"]) if item["pred_rc"] and item["ref_rc"] else 0.0)
        rs_bleu_results.append(compute_bleu(item["pred_rs"], item["ref_rs"]) if item["pred_rs"] and item["ref_rs"] else 0.0)
    
    # ========== 阶段3：并行计算 ESS/RAS ==========
    safe_print(f"   [{model_name}] 📊 阶段3/3: ESS/RAS...")
    
    ess_ras_inputs = [
        (i, item["pred_rc"], item["ref_rc"], item["pred_rs"], item["ref_rs"])
        for i, item in enumerate(valid_items)
    ]
    
    ess_ras_results = compute_ess_ras_parallel(ess_ras_inputs, num_ess_workers, model_name)
    
    # ========== 汇总结果 ==========
    model_results = []
    
    for i, item in enumerate(valid_items):
        rc_p, rc_r, rc_f1 = rc_bert_results[i]
        rs_p, rs_r, rs_f1 = rs_bert_results[i]
        ess, ras = ess_ras_results[i]
        
        output_item = {
            "instance_id": item["instance_id"],
            "model": model_name,
            "status": item["status"],
            "rc_bertscore_p": rc_p,
            "rc_bertscore_r": rc_r,
            "rc_bertscore_f1": rc_f1,
            "rc_bleu4": rc_bleu_results[i],
            "rc_ess": ess,
            "rs_bertscore_p": rs_p,
            "rs_bertscore_r": rs_r,
            "rs_bertscore_f1": rs_f1,
            "rs_bleu4": rs_bleu_results[i],
            "rs_ras": ras,
            "esra": (ess + ras) / 2.0,
            "metrics": {
                "rc_bertscore_p": rc_p,
                "rc_bertscore_r": rc_r,
                "rc_bertscore_f1": rc_f1,
                "rc_bleu4": rc_bleu_results[i],
                "rc_ess": ess,
                "rs_bertscore_p": rs_p,
                "rs_bertscore_r": rs_r,
                "rs_bertscore_f1": rs_f1,
                "rs_bleu4": rs_bleu_results[i],
                "rs_ras": ras,
                "esra": (ess + ras) / 2.0,
            }
        }
        model_results.append(output_item)

    safe_print(f"   ✅ [{model_name}] 完成，共 {len(model_results)} 条")
    return model_name, model_results

def save_checkpoint(completed_models: List[str], all_output_items: List[Dict]):
    """保存检查点（线程安全）"""
    with checkpoint_lock:
        checkpoint = {
            "completed_models": completed_models,
            "items_count": len(all_output_items)
        }
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(checkpoint, f)
        write_jsonl(OUTPUT_JSONL, all_output_items)


def load_checkpoint() -> Tuple[List[str], List[Dict]]:
    """加载检查点；若无检查点，则从已有输出推断已完成模型"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            checkpoint = json.load(f)
        # 兼容旧版本的 checkpoint（使用 mode）
        completed = checkpoint.get("completed_models", checkpoint.get("completed_modes", []))
        items = read_jsonl(OUTPUT_JSONL) if os.path.exists(OUTPUT_JSONL) else []
        return completed, items

    # 无检查点时，直接从已生成的输出文件中恢复
    items = read_jsonl(OUTPUT_JSONL) if os.path.exists(OUTPUT_JSONL) else []
    completed = []
    for item in items:
        model = item.get("model", item.get("mode"))
        if model and model not in completed:
            completed.append(model)
    return completed, items

def compute_statistics(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"max": 0.0, "min": 0.0, "mean": 0.0, "variance": 0.0, "std": 0.0}
    return {
        "max": max(values),
        "min": min(values),
        "mean": mean(values),
        "variance": variance(values) if len(values) > 1 else 0.0,
        "std": stdev(values) if len(values) > 1 else 0.0,
    }

def generate_markdown_report(all_results: Dict[str, List[Dict]], output_path: str) -> None:
    """生成 Markdown 报告"""
    lines = []
    
    lines.append("# 📊 模型评测指标分析报告")
    lines.append("")
    lines.append("> 本报告对比了多种模型生成的 Root Cause 和 Repair Suggestion 的质量指标。")
    lines.append("")
    
    lines.append("## 📋 指标说明")
    lines.append("")
    lines.append("| 指标 | 说明 | 范围 |")
    lines.append("|------|------|------|")
    lines.append("| **BERTScore F1** | 基于 BERT 嵌入的语义相似度 | 0~1 |")
    lines.append("| **BLEU-4** | 4-gram 词面匹配度 | 0~100 |")
    lines.append("| **ESS** | Error State Score，缺陷概念覆盖度 | 0~1 |")
    lines.append("| **RAS** | Repair Action Score，修复动作覆盖度 | 0~1 |")
    lines.append("| **ESRA** | (ESS + RAS) / 2 综合得分 | 0~1 |")
    lines.append("")
    
    metric_keys = [
        ("rc_bertscore_f1", "RC BERTScore F1"),
        ("rc_bleu4", "RC BLEU-4"),
        ("rc_ess", "RC ESS"),
        ("rs_bertscore_f1", "RS BERTScore F1"),
        ("rs_bleu4", "RS BLEU-4"),
        ("rs_ras", "RS RAS"),
        ("esra", "ESRA"),
    ]
    
    lines.append("---")
    lines.append("")
    lines.append("## 📈 各模型详细分析")
    lines.append("")
    
    for model_name, results in all_results.items():
        resolved = [r for r in results if r.get("status") == "resolved"]
        failed = [r for r in results if r.get("status") == "failed"]
        
        lines.append(f"### 🔹 模型: `{model_name.upper()}`")
        lines.append("")
        lines.append(f"- **样本总数**: {len(results)}")
        lines.append(f"- **Resolved**: {len(resolved)} 条")
        lines.append(f"- **Failed**: {len(failed)} 条")
        lines.append("")
        
        lines.append("#### 📊 全部样本统计")
        lines.append("")
        lines.append("| 指标 | 最大值 | 最小值 | **均值** | 方差 | 标准差 |")
        lines.append("|------|--------|--------|----------|------|--------|")
        
        for key, display_name in metric_keys:
            values = [r["metrics"][key] for r in results if key in r.get("metrics", {})]
            stats = compute_statistics(values)
            lines.append(f"| {display_name} | {stats['max']:.4f} | {stats['min']:.4f} | **{stats['mean']:.4f}** | {stats['variance']:.4f} | {stats['std']:.4f} |")
        
        lines.append("")
        
        lines.append("#### 🔍 Resolved vs Failed 对比")
        lines.append("")
        lines.append("| 指标 | Resolved 均值 | Failed 均值 | 差异 |")
        lines.append("|------|---------------|-------------|------|")
        
        for key, display_name in metric_keys:
            resolved_values = [r["metrics"][key] for r in resolved if key in r.get("metrics", {})]
            failed_values = [r["metrics"][key] for r in failed if key in r.get("metrics", {})]
            
            resolved_mean = mean(resolved_values) if resolved_values else 0.0
            failed_mean = mean(failed_values) if failed_values else 0.0
            diff = resolved_mean - failed_mean
            diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
            
            lines.append(f"| {display_name} | {resolved_mean:.4f} | {failed_mean:.4f} | {diff_str} |")
        
        lines.append("")
        lines.append("---")
        lines.append("")
    
    lines.append("## 🏆 模型间均值对比汇总")
    lines.append("")
    lines.append("> 以下表格展示各模型在所有样本上的**均值**对比，便于横向比较。")
    lines.append("")
    
    model_names = list(all_results.keys())
    header = "| 指标 |"
    separator = "|------|"
    for model in model_names:
        header += f" **{model}** |"
        separator += "--------|"
    
    lines.append(header)
    lines.append(separator)
    
    for key, display_name in metric_keys:
        row = f"| {display_name} |"
        values_per_model = []
        
        for model in model_names:
            results = all_results[model]
            values = [r["metrics"][key] for r in results if key in r.get("metrics", {})]
            avg = mean(values) if values else 0.0
            values_per_model.append(avg)
        
        max_val = max(values_per_model) if values_per_model else 0.0
        
        for avg in values_per_model:
            if avg == max_val and max_val > 0:
                row += f" **{avg:.4f}** 🥇 |"
            else:
                row += f" {avg:.4f} |"
        
        lines.append(row)
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 📝 备注")
    lines.append("")
    lines.append("- 🥇 表示该指标在各模型中的**最高值**")
    lines.append("- BERTScore 使用 `bert-base-uncased` 模型")
    lines.append("- BLEU-4 使用 `sacrebleu` 实现（范围 0~100）")
    lines.append("- ESS/RAS 使用 spaCy 进行依存句法分析")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*报告自动生成*")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"报告已保存到: {output_path}")


def main():
    print("=" * 60)
    print("🚀 评测指标计算（高速版 v2 - 多模型并发）")
    print(f"   CPU 核心数: {NUM_TOTAL_CORES}")
    print(f"   并发模型数: {NUM_MODEL_WORKERS}")
    print(f"   每模型 ESS/RAS 进程数: {NUM_ESS_RAS_WORKERS_PER_MODEL}")
    print("=" * 60)
    
    # 检查断点
    completed_models, all_output_items = load_checkpoint()
    if completed_models:
        if os.path.exists(CHECKPOINT_FILE):
            print(f"📌 发现检查点，已完成模型: {completed_models}")
        else:
            print(f"📌 发现已有输出，已完成模型: {completed_models}")
        print(f"   已有结果: {len(all_output_items)} 条")
    elif all_output_items:
        print(f"📌 发现已有输出，但未识别到已完成模型，已有结果: {len(all_output_items)} 条")
    
    # 读取参考数据
    print(f"\n读取参考数据: {REFERENCE_FILE}")
    reference_data = read_jsonl(REFERENCE_FILE)
    ref_dict = {item["instance_id"]: item for item in reference_data}
    print(f"  参考数据条数: {len(ref_dict)}")
    
    all_results = {}
    
    # 恢复已完成的结果
    for item in all_output_items:
        # 兼容旧版本（使用 mode）
        model = item.get("model", item.get("mode"))
        if model not in all_results:
            all_results[model] = []
        all_results[model].append(item)

    # 筛选待处理的模型
    pending_models = []
    for model_name, model_file in MODEL_FILES.items():
        if model_name in completed_models:
            print(f"\n⏭ 跳过已完成模型: {model_name}")
            continue
        if not os.path.exists(model_file):
            print(f"\n⚠ 文件不存在，跳过: {model_file}")
            continue
        pending_models.append((model_name, model_file))
    
    if not pending_models:
        print("\n所有模型已完成！")
    else:
        print(f"\n🔄 待处理模型数: {len(pending_models)}")
    
        # ========== 多模型并发处理 ==========
        with ThreadPoolExecutor(max_workers=NUM_MODEL_WORKERS) as executor:
            futures = {
                executor.submit(
                    process_model_standalone, 
                    model_name, 
                    model_file, 
                    ref_dict, 
                    NUM_ESS_RAS_WORKERS_PER_MODEL
                ): model_name
                for model_name, model_file in pending_models
            }
            
            for future in as_completed(futures):
                model_name, model_results = future.result()
                all_results[model_name] = model_results
                all_output_items.extend(model_results)
                
                # 保存检查点
                completed_models.append(model_name)
                save_checkpoint(completed_models, all_output_items)
                safe_print(f"💾 [{model_name}] 检查点已保存")
    
    # 生成报告
    print(f"\n生成分析报告: {OUTPUT_REPORT}")
    generate_markdown_report(all_results, OUTPUT_REPORT)
    
    # 清理检查点
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    
    print("\n" + "=" * 60)
    print("✅ 评测完成！")
    print(f"   详细结果: {OUTPUT_JSONL}")
    print(f"   分析报告: {OUTPUT_REPORT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
