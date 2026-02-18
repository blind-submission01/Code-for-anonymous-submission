

"""
基于两组人工标签，比较 8 种 (normalize_mode × build_mode) 组合下：
- 方法 A 的 ROC-AUC：scoreA = mean(new_score1_A, new_score2_A)
- 方法 B 的 ROC-AUC：scoreB = mean(new_score1_B_by_beta[1.0], new_score2_B_by_beta[1.0])

并额外与 baseline 配置 none_legacy 做配对对比：
- 对于每个其它配置、每条样本：
    - 若 label=True：检查 score_other > score_baseline
    - 若 label=False：检查 score_other < score_baseline
  统计 (True 更大 + False 更小) 的比例。

结果文件路径模式：

标签集：
  - HUMAN_LABELS_A
  - HUMAN_LABELS_B
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.model_selection import KFold


# ===== 需要你手动填写的人工标签 =====
# True 表示 “diff 与 repair_suggestion/repair_code 一致”，False 表示不一致。
HUMAN_LABELS_A: Dict[int, bool] = {
    1: True,
    12: False,
    25: True,
    34: True,
    51: True,
    70: False,
    85: True,
    101: False,
    110: False,
    119: True,
    135: True,
    146: True,
    161: True,
    172: True,
    183: True,
    205: True,
    217: True,
    229: True,
    251: True,
    267: True,
    285: True,
    299: False,
    310: False,
    322: True,
    332: True,
    342: True,
    351: True,
    362: True,
    377: False,
    385: True,
    399: False,
    409: True,
    420: True,
    437: True,
    449: False,
    458: True,
    469: False,
    483: False,
    503: True,
    519: False,
    534: True,
    547: True,
    560: True,
    572: True,
    582: False,
    592: True,
    605: True,
    623: True,
    634: True,
    645: True,
    658: False,
    669: True,
    682: True,
    694: True,
    712: True,
    728: True,
    738: True,
    747: False,
    758: True,
    772: False,
    786: False,
    801: False,
    821: True,
    832: False,
    846: True,
    856: True,
    872: True,
    881: True,
    894: True,
    908: False,
    922: False,
    936: False,
    948: False,
    961: True,
    973: False,
    986: False,
    997: False,
    1009: True,
    1019: True,
    1028: False,
    1037: False,
    1050: True,
    1063: True,
    1075: False,
    1093: True,
    1104: True,
    1112: True,
    1126: True,
    1135: True,
    1145: True,
    1160: True,
    1171: True,
    1185: True,
    1194: False,
    1204: True,
    1214: True,
    1228: False,
    1238: True,
    1247: False,
    1260: False,
}

HUMAN_LABELS_B: Dict[int, bool] = {
    658: False,
    669: True,
    682: True,
    694: True,
    712: True,
    728: True,
    738: True,
    747: False,
    758: True,
    772: False,
    786: False,
    801: False,
    821: True,
    832: False,
    846: True,
    856: True,
    872: True,
    881: True,
    894: True,
    908: False,
    922: False,
    936: False,
    948: False,
    961: True,
    973: False,
    986: False,
    997: False,
    1009: True,
    1019: True,
    1028: False,
    1037: False,
    1050: True,
    1063: True,
    1075: False,
    1093: True,
    1104: True,
    1112: True,
    1126: True,
    1135: True,
    1145: True,
    1160: True,
    1171: True,
    1185: True,
    1194: False,
    1204: True,
    1214: True,
    1228: False,
    1238: True,
    1247: False,
    1260: False,
}

DATA_DIR = Path("")

NORMALIZE_MODES = ("none", "noise", "semantic", "full")
BUILD_MODES = ("legacy", "block")

BASELINE_NAME = "none_legacy"  # baseline 配置名

def get_beta_score(d: Any, beta: float = 1.0) -> Optional[float]:
    """从 *_B_by_beta 字典中稳健地取出指定 beta 的分数。"""
    if not isinstance(d, dict):
        return None

    # 常见 key 形式："1.0" / 1.0 / "1"
    # candidates = [beta]
    candidates = [
        str(beta),
        f"{beta:.1f}",
        f"{beta:g}",
        beta,
        int(beta),
    ]
    for k in candidates:
        if k in d:
            try:
                return float(d[k])
            except Exception:
                return None

    return None

def iter_labeled_scores(
    jsonl_path: Path,
    human_labels: Dict[int, bool],
) -> Tuple[List[float], List[float], List[bool], List[int]]:
    """
    从给定 jsonl 文件中，提取 (scoreA, scoreB, label) 列表，只保留 human_labels 中的 index。

    - 方法 A 分数：scoreA = mean(new_score1_A, new_score2_A)
    - 方法 B 分数：scoreB = mean(new_score1_B_by_beta[1.0], new_score2_B_by_beta[1.0])

    只使用同时拥有 A/B 分数的样本（保证比较公平）。
    返回：
        scores_A: List[float]
        scores_B: List[float]
        labels:   List[bool]
        used_indices: 实际成功取到分数的 index 列表
    """
    scores_A: List[float] = []
    scores_B: List[float] = []
    labels: List[bool] = []
    used_indices: List[int] = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            idx = rec.get("index")
            if not isinstance(idx, int):
                continue

            if idx not in human_labels:
                continue  # 不是我们人工标注过的样本，跳过

            # ---- 方法 A：new_score*_A ----
            s1A = rec.get("new_score1_A")
            s2A = rec.get("new_score2_A")
            try:
                s1A = float(s1A)
                s2A = float(s2A)
            except (TypeError, ValueError):
                continue
            scoreA = (s1A + s2A) / 2.0

            # ---- 方法 B：new_score*_B_by_beta[1.0] ----
            s1_dict = rec.get("new_score1_B_by_beta")
            s2_dict = rec.get("new_score2_B_by_beta")
            s1B = get_beta_score(s1_dict, beta=1.0)
            s2B = get_beta_score(s2_dict, beta=1.0)
            if s1B is None or s2B is None:
                continue
            scoreB = (s1B + s2B) / 2.0

            scores_A.append(scoreA)
            scores_B.append(scoreB)
            labels.append(bool(human_labels[idx]))
            used_indices.append(idx)

    return scores_A, scores_B, labels, used_indices

def roc_auc_pairwise(scores: List[float], labels: List[bool]) -> Optional[float]:
    """
    使用“数对比较”的定义计算 ROC-AUC：

    AUC = P(score_pos > score_neg) + 0.5 * P(score_pos == score_neg)
    """
    if not scores or not labels or len(scores) != len(labels):
        return None

    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return None

    wins = 0.0
    ties = 0.0
    for sp in pos:
        for sn in neg:
            if sp > sn:
                wins += 1.0
            elif sp == sn:
                ties += 1.0

    total_pairs = n_pos * n_neg
    auc = (wins + 0.5 * ties) / total_pairs
    return auc

def scan_threshold_min_t_for_precision(
    scores: List[float],
    labels: List[bool],
    target_precision: float = 0.8,
    n_steps: int = 100,
    min_selected: int = 1,
) -> Tuple[Optional[float], Optional[int], Optional[float]]:
    """
    在 [0,1] 上扫描阈值，寻找：
    - precision >= target_precision 的阈值
    - 在这些阈值中，返回 最小的阈值

    返回：
        threshold         : 最小满足条件的阈值（若不存在返回 None）
        n_selected        : 该阈值下被选中的样本数
        actual_precision  : 该阈值下的 precision
    """
    if not scores:
        return None, None, None

    thresholds = np.linspace(0.0, 1.0, n_steps + 1)

    for t in thresholds:
        selected = [(s, y) for s, y in zip(scores, labels) if s >= t]
        if len(selected) < min_selected:
            continue

        tp = sum(1 for _, y in selected if y)
        fp = sum(1 for _, y in selected if not y)
        if tp + fp == 0:
            continue

        precision = tp / (tp + fp)

        if precision >= target_precision:
            return t, len(selected), precision

    return None, None, None

def cv_min_threshold_for_precision(
    scores: List[float],
    labels: List[bool],
    target_precision: float = 0.8,
    n_splits: int = 5,
    n_steps: int = 100,
    min_selected: int = 1,
    agg: str = "mean",  # "max" | "mean"
) -> Tuple[Optional[float], Optional[float], List[float]]:
    """
    K 折交叉验证版本的“最小阈值”选择：

    对每一折：
      - 在训练集上寻找 precision >= target_precision 的最小阈值

    聚合方式：
      - agg="max"  : 取所有折中最大的阈值（最保守，最稳定，强烈推荐）
      - agg="mean" : 取平均阈值（稍激进）

    返回：
        final_threshold       : 聚合后的阈值（若任一折失败则返回 None）
        mean_val_precision    : 使用该阈值在验证集上的平均 precision
        per_fold_thresholds   : 每一折得到的阈值（便于 debug / 输出）
    """
    if len(scores) < n_splits:
        return None, None, []

    thresholds = np.linspace(0.0, 1.0, n_steps + 1)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_thresholds: List[float] = []
    fold_val_precisions: List[float] = []

    for train_idx, val_idx in kf.split(scores):
        train_scores = [scores[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_scores   = [scores[i] for i in val_idx]
        val_labels   = [labels[i] for i in val_idx]

        # === 训练集：找最小满足 precision 的阈值 ===
        found_t = None

        for t in thresholds:
            selected = [(s, y) for s, y in zip(train_scores, train_labels) if s >= t]
            if len(selected) < min_selected:
                continue

            tp = sum(1 for _, y in selected if y)
            fp = sum(1 for _, y in selected if not y)
            if tp + fp == 0:
                continue

            precision = tp / (tp + fp)
            if precision >= target_precision:
                found_t = t
                break

        if found_t is None:
            # 有一折失败，整体认为不稳定
            return None, None, []

        fold_thresholds.append(found_t)

        # === 验证集：评估 precision ===
        sel_val = [(s, y) for s, y in zip(val_scores, val_labels) if s >= found_t]
        if not sel_val:
            fold_val_precisions.append(0.0)
        else:
            tp_val = sum(1 for _, y in sel_val if y)
            fp_val = sum(1 for _, y in sel_val if not y)
            fold_val_precisions.append(
                tp_val / (tp_val + fp_val) if (tp_val + fp_val) > 0 else 0.0
            )

    # === 聚合阈值 ===
    if agg == "max":
        final_t = max(fold_thresholds)
    elif agg == "mean":
        final_t = float(np.mean(fold_thresholds))
    else:
        raise ValueError(f"Unknown agg method: {agg}")

    mean_val_precision = float(np.mean(fold_val_precisions))

    return final_t, mean_val_precision, fold_thresholds

def count_passed_records_with_threshold(
    jsonl_path: Path,
    threshold: float,
    method: str,  # "A" or "B"
) -> Tuple[int, int]:
    """
    读取完整 jsonl 文件，用给定阈值统计：
    - 有多少条记录 score >= threshold

    method:
        "A": score = mean(new_score1_A, new_score2_A)
        "B": score = mean(new_score1_B@1.0, new_score2_B@1.0)

    返回：
        n_passed, n_total
    """
    n_total = 0
    n_passed = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            # ---- score A ----
            if method == "A":
                try:
                    s1 = float(rec.get("new_score1_A"))
                    s2 = float(rec.get("new_score2_A"))
                except (TypeError, ValueError):
                    continue
                score = (s1 + s2) / 2.0

            # ---- score B ----
            elif method == "B":
                s1 = get_beta_score(rec.get("new_score1_B_by_beta"), beta=1.0)
                s2 = get_beta_score(rec.get("new_score2_B_by_beta"), beta=1.0)
                if s1 is None or s2 is None:
                    continue
                score = (s1 + s2) / 2.0

            else:
                raise ValueError(f"Unknown method: {method}")

            n_total += 1
            if score >= threshold:
                n_passed += 1

    return n_passed, n_total


def run_for_label_set(label_name: str, human_labels: Dict[int, bool]) -> None:
    if not human_labels:
        print(f"[标签集 {label_name}] 为空，跳过。")
        return

    print("=" * 80)
    print(f"标签集 {label_name}: 人工标签条数 = {len(human_labels)}")
    print("=" * 80)
    print()

    results_A: List[Tuple[str, float]] = []
    results_B: List[Tuple[str, float]] = []

    # 新增：按“真均值 - 假均值”评估好坏
    mean_gap_A: List[Tuple[str, float, float, float]] = []  # (name, gap, mean_true, mean_false)
    mean_gap_B: List[Tuple[str, float, float, float]] = []  # (name, gap, mean_true, mean_false)

    # 用于后续与 baseline 做配对对比：每个配置下，index -> scoreA / scoreB
    per_config_scores_A: Dict[str, Dict[int, float]] = {}
    per_config_scores_B: Dict[str, Dict[int, float]] = {}

    for norm in NORMALIZE_MODES:
        for build in BUILD_MODES:
            name = f"{norm}_{build}"
            path = DATA_DIR / f"new_result_from2_for3_{norm}_{build}.jsonl"

            if not path.is_file():
                print(f"[{label_name}] [{name}] 文件不存在: {path}")
                continue

            scores_A, scores_B, labels, used_idx = iter_labeled_scores(path, human_labels)
            auc_A = roc_auc_pairwise(scores_A, labels)
            auc_B = roc_auc_pairwise(scores_B, labels)

            # 保存 index->score 映射
            idx_to_A = {idx: s for idx, s in zip(used_idx, scores_A)}
            idx_to_B = {idx: s for idx, s in zip(used_idx, scores_B)}
            per_config_scores_A[name] = idx_to_A
            per_config_scores_B[name] = idx_to_B

            n_used = len(labels)
            n_pos = sum(1 for y in labels if y)
            n_neg = sum(1 for y in labels if not y)
            n_expect = len(human_labels)

            print(f"=== 配置 {name} | 标签集 {label_name} ===")
            print(f"文件: {path}")
            print(f"参与计算的样本数: {n_used} (期望: {n_expect}, 正类: {n_pos}, 负类: {n_neg})")

            # ===== 新增：真/假样本均值差分析 =====
            if n_pos > 0 and n_neg > 0:
                pos_A = [s for s, y in zip(scores_A, labels) if y]
                neg_A = [s for s, y in zip(scores_A, labels) if not y]
                pos_B = [s for s, y in zip(scores_B, labels) if y]
                neg_B = [s for s, y in zip(scores_B, labels) if not y]

                mean_true_A = float(np.mean(pos_A))
                mean_false_A = float(np.mean(neg_A))
                gap_A = mean_true_A - mean_false_A

                mean_true_B = float(np.mean(pos_B))
                mean_false_B = float(np.mean(neg_B))
                gap_B = mean_true_B - mean_false_B

                mean_gap_A.append((name, gap_A, mean_true_A, mean_false_A))
                mean_gap_B.append((name, gap_B, mean_true_B, mean_false_B))

                print("  真/假样本均值分析:")
                print(f"    方法 A: mean_true = {mean_true_A:.4f}, mean_false = {mean_false_A:.4f}, "
                      f"gap = mean_true - mean_false = {gap_A:.4f}")
                print(f"    方法 B: mean_true = {mean_true_B:.4f}, mean_false = {mean_false_B:.4f}, "
                      f"gap = mean_true - mean_false = {gap_B:.4f}")
            else:
                print("  真/假样本均值分析: 无法计算（正类或负类为空）")

            # ===== Precision 阈值分析 =====
            TARGET_PRECISION = 0.8

            t_A, cv_p_A, fold_ts_A = cv_min_threshold_for_precision(
                scores_A,
                labels,
                target_precision=TARGET_PRECISION,
            )

            t_B, cv_p_B, fold_ts_B = cv_min_threshold_for_precision(
                scores_B,
                labels,
                target_precision=TARGET_PRECISION,
            )

            print("  CV 阈值分析（目标 precision ≥ %.2f）:" % TARGET_PRECISION)

            if t_A is not None:
                print(
                    f"    方法 A: final_t = {t_A:.3f}, "
                    f"cv_precision = {cv_p_A:.3f}, "
                    f"fold_ts = {[round(x,3) for x in fold_ts_A]}"
                )
            else:
                print("    方法 A: 在 CV 中无法稳定达到目标 precision")

            if t_B is not None:
                print(
                    f"    方法 B: final_t = {t_B:.3f}, "
                    f"cv_precision = {cv_p_B:.3f}, "
                    f"fold_ts = {[round(x,3) for x in fold_ts_B]}"
                )
            else:
                print("    方法 B: 在 CV 中无法稳定达到目标 precision")

            # ===== 使用该阈值，在全量数据中统计可过滤数量 =====
            if t_A is not None:
                passed_A, total_A = count_passed_records_with_threshold(
                    path, threshold=t_A, method="A"
                )
                ratio_A = passed_A / total_A if total_A > 0 else 0.0
            else:
                passed_A = total_A = ratio_A = 0

            if t_B is not None:
                passed_B, total_B = count_passed_records_with_threshold(
                    path, threshold=t_B, method="B"
                )
                ratio_B = passed_B / total_B if total_B > 0 else 0.0
            else:
                passed_B = total_B = ratio_B = 0

            print("  全量数据过滤能力（使用人工标签得到的最小阈值）:")
            if t_A is not None:
                print(
                    f"    方法 A: t = {t_A:.3f}, "
                    f"passed = {passed_A} / {total_A} "
                    f"({ratio_A:.2%})"
                )
            else:
                print("    方法 A: 无法达到目标 precision，未设置阈值")

            if t_B is not None:
                print(
                    f"    方法 B: t = {t_B:.3f}, "
                    f"passed = {passed_B} / {total_B} "
                    f"({ratio_B:.2%})"
                )
            else:
                print("    方法 B: 无法达到目标 precision，未设置阈值")

            
            if n_used < n_expect:
                missing = sorted(set(human_labels.keys()) - set(used_idx))
                if missing:
                    print(f"  未在该配置文件中成功取到 A/B 分数的 index 有: {missing}")

            if auc_A is None:
                print("  方法 A: 无法计算 ROC-AUC（可能是没有正类或没有负类，或样本为空）")
            else:
                print("  方法 A: ROC-AUC(scoreA = mean(new_score1_A, new_score2_A)) = "
                      f"{auc_A:.6f}")

            if auc_B is None:
                print("  方法 B: 无法计算 ROC-AUC（可能是没有正类或没有负类，或样本为空）")
            else:
                print("  方法 B: ROC-AUC(scoreB = mean(new_score1_B@1.0, new_score2_B@1.0)) = "
                      f"{auc_B:.6f}")
            print()

            if auc_A is not None:
                results_A.append((name, auc_A))
            if auc_B is not None:
                results_B.append((name, auc_B))

    if not results_A and not results_B:
        print(f"[标签集 {label_name}] 没有任何配置成功计算出 ROC-AUC，请检查数据。")
        print()
        return

    # ---- 按方法 A 排名 ----
    if results_A:
        results_A_sorted = sorted(results_A, key=lambda x: x[1], reverse=True)
        print(f"====== 标签集 {label_name} | 方法 A 排名（按 ROC-AUC 从高到低） ======")
        for rank, (name, auc) in enumerate(results_A_sorted, start=1):
            print(f"{rank:2d}. {name:15s}  ROC-AUC(A) = {auc:.6f}")
        best_name_A, best_auc_A = results_A_sorted[0]
        print(f">>> [标签集 {label_name}] 方法 A 最佳配置: {best_name_A}, ROC-AUC(A) = {best_auc_A:.6f}")
        print()

    # ---- 按方法 B 排名 ----
    if results_B:
        results_B_sorted = sorted(results_B, key=lambda x: x[1], reverse=True)
        print(f"====== 标签集 {label_name} | 方法 B 排名（按 ROC-AUC 从高到低） ======")
        for rank, (name, auc) in enumerate(results_B_sorted, start=1):
            print(f"{rank:2d}. {name:15s}  ROC-AUC(B) = {auc:.6f}")
        best_name_B, best_auc_B = results_B_sorted[0]
        print(f">>> [标签集 {label_name}] 方法 B 最佳配置: {best_name_B}, ROC-AUC(B) = {best_auc_B:.6f}")
        print()
    
    # ---- 新增：按“真均值 - 假均值”排名 ----
    if mean_gap_A:
        mean_gap_A_sorted = sorted(mean_gap_A, key=lambda x: x[1], reverse=True)
        print(f"====== 标签集 {label_name} | 方法 A 排名（按 真均值-假均值 从高到低） ======")
        for rank, (name, gap, m_true, m_false) in enumerate(mean_gap_A_sorted, start=1):
            print(f"{rank:2d}. {name:15s}  gap(A) = {gap:.6f}  "
                  f"(mean_true={m_true:.4f}, mean_false={m_false:.4f})")
        best_name_gap_A, best_gap_A, _, _ = mean_gap_A_sorted[0]
        print(f">>> [标签集 {label_name}] 方法 A 按均值差最佳配置: {best_name_gap_A}, gap(A) = {best_gap_A:.6f}")
        print()

    if mean_gap_B:
        mean_gap_B_sorted = sorted(mean_gap_B, key=lambda x: x[1], reverse=True)
        print(f"====== 标签集 {label_name} | 方法 B 排名（按 真均值-假均值 从高到低） ======")
        for rank, (name, gap, m_true, m_false) in enumerate(mean_gap_B_sorted, start=1):
            print(f"{rank:2d}. {name:15s}  gap(B) = {gap:.6f}  "
                  f"(mean_true={m_true:.4f}, mean_false={m_false:.4f})")
        best_name_gap_B, best_gap_B, _, _ = mean_gap_B_sorted[0]
        print(f">>> [标签集 {label_name}] 方法 B 按均值差最佳配置: {best_name_gap_B}, gap(B) = {best_gap_B:.6f}")
        print()

    

    # =====================================================================
    #          新增：与 baseline none_legacy 的“真更大/假更小”配对对比
    # =====================================================================
    if BASELINE_NAME not in per_config_scores_A or BASELINE_NAME not in per_config_scores_B:
        print(f"[标签集 {label_name}] baseline 配置 {BASELINE_NAME} 缺失，无法做配对对比。")
        print()
        return

    base_A = per_config_scores_A[BASELINE_NAME]
    base_B = per_config_scores_B[BASELINE_NAME]
    n_total = len(human_labels)

    compare_A_stats: List[Tuple[str, float, int, int]] = []
    compare_B_stats: List[Tuple[str, float, int, int]] = []

    print(f"====== 标签集 {label_name} | 与 baseline {BASELINE_NAME} 的配对对比（方法 A/B） ======")

    for norm in NORMALIZE_MODES:
        for build in BUILD_MODES:
            name = f"{norm}_{build}"
            if name == BASELINE_NAME:
                continue

            cur_A = per_config_scores_A.get(name)
            cur_B = per_config_scores_B.get(name)
            if cur_A is None or cur_B is None:
                continue

            # 方法 A：scoreA_other vs scoreA_baseline
            success_A = 0
            considered_A = 0

            # 方法 B：scoreB_other vs scoreB_baseline
            success_B = 0
            considered_B = 0

            for idx, label in human_labels.items():
                # ---- 方法 A ----
                if idx in base_A and idx in cur_A:
                    considered_A += 1
                    s_base = base_A[idx]
                    s_cur = cur_A[idx]
                    if label and s_cur > s_base:
                        success_A += 1
                    elif (not label) and s_cur < s_base:
                        success_A += 1

                # ---- 方法 B ----
                if idx in base_B and idx in cur_B:
                    considered_B += 1
                    s_base_b = base_B[idx]
                    s_cur_b = cur_B[idx]
                    if label and s_cur_b > s_base_b:
                        success_B += 1
                    elif (not label) and s_cur_b < s_base_b:
                        success_B += 1

            match_rate_all_A = success_A / n_total if n_total > 0 else 0.0
            match_rate_eff_A = success_A / considered_A if considered_A > 0 else 0.0

            match_rate_all_B = success_B / n_total if n_total > 0 else 0.0
            match_rate_eff_B = success_B / considered_B if considered_B > 0 else 0.0

            compare_A_stats.append((name, match_rate_all_A, success_A, considered_A))
            compare_B_stats.append((name, match_rate_all_B, success_B, considered_B))

            print(f"[{name}] 相对 baseline={BASELINE_NAME}")
            print(f"  方法 A: 成功 {success_A} 条 / 有效对比 {considered_A} 条 / 标签总数 {n_total}")
            print(f"          match_rate_all     = {match_rate_all_A:.4f}  (成功 / 标签总数)")
            print(f"          match_rate_effective = {match_rate_eff_A:.4f}  (成功 / 有效对比数)")
            print(f"  方法 B: 成功 {success_B} 条 / 有效对比 {considered_B} 条 / 标签总数 {n_total}")
            print(f"          match_rate_all     = {match_rate_all_B:.4f}")
            print(f"          match_rate_effective = {match_rate_eff_B:.4f}")
            print()

    # 按 match_rate_all 排名（方法 A）
    if compare_A_stats:
        compare_A_sorted = sorted(compare_A_stats, key=lambda x: x[1], reverse=True)
        print(f"====== 标签集 {label_name} | 方法 A 相对 baseline 的配对对比排名（按 match_rate_all） ======")
        for rank, (name, rate_all, success, considered) in enumerate(compare_A_sorted, start=1):
            print(f"{rank:2d}. {name:15s}  match_rate_all(A) = {rate_all:.4f} "
                  f"(success={success}, considered={considered}, total_labels={n_total})")
        best_name_cmp_A, best_rate_A, _, _ = compare_A_sorted[0]
        print(f">>> [标签集 {label_name}] 方法 A 相对 baseline 最佳配置: {best_name_cmp_A}, "
              f"match_rate_all(A) = {best_rate_A:.4f}")
        print()

    # 按 match_rate_all 排名（方法 B）
    if compare_B_stats:
        compare_B_sorted = sorted(compare_B_stats, key=lambda x: x[1], reverse=True)
        print(f"====== 标签集 {label_name} | 方法 B 相对 baseline 的配对对比排名（按 match_rate_all） ======")
        for rank, (name, rate_all, success, considered) in enumerate(compare_B_sorted, start=1):
            print(f"{rank:2d}. {name:15s}  match_rate_all(B) = {rate_all:.4f} "
                  f"(success={success}, considered={considered}, total_labels={n_total})")
        best_name_cmp_B, best_rate_B, _, _ = compare_B_sorted[0]
        print(f">>> [标签集 {label_name}] 方法 B 相对 baseline 最佳配置: {best_name_cmp_B}, "
              f"match_rate_all(B) = {best_rate_B:.4f}")
        print()

def main() -> None:
    # 依次在 A / B 两套人工标签下评估 8 个配置
    label_sets = [
        ("HUMAN_LABELS_A", HUMAN_LABELS_A),
        ("HUMAN_LABELS_B", HUMAN_LABELS_B),
    ]
    for label_name, labels in label_sets:
        run_for_label_set(label_name, labels)


if __name__ == "__main__":
    main()