import json
import argparse
from typing import Dict, Any

# 全局阈值：score >= THRESHOLD 计作 SFT，否则 ORPO
THRESHOLD: float = 0.75


def load_index_to_record(path: str) -> Dict[int, Dict[str, Any]]:
    """
    读取一个 jsonl 文件，返回 index -> 整条记录 的映射。
    如果同一个 index 出现多次，后面的会覆盖前面的。
    """
    index_to_record: Dict[int, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = obj["index"]
            index_to_record[idx] = obj
    return index_to_record


def process_files(old_path: str, new_path: str, out_path: str, threshold: float) -> None:
    global THRESHOLD
    THRESHOLD = threshold

    # 1. 读取旧文件，构建 index 映射和集合
    old_index_to_record = load_index_to_record(old_path)
    old_indices = set(old_index_to_record.keys())

    # 统计信息
    total_count = 0
    sft_count = 0
    orpo_count = 0
    sum_score_all = 0.0
    sum_score_sft = 0.0
    sum_score_orpo = 0.0

    # 2. 按新文件顺序遍历，每行如果 index 在旧集合，就用旧记录替换
    with open(new_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            idx = record["index"]

            # 如果新文件中的 index 出现在旧文件中，则使用旧记录替换
            if idx in old_indices:
                record = old_index_to_record[idx]

            # 3. 计算新的 score
            # 从 new_score1_B_by_beta / new_score2_B_by_beta 里取 key "1.0"
            score1_1 = record["new_score1_B_by_beta"]["1.0"]
            score2_1 = record["new_score2_B_by_beta"]["1.0"]
            score = (score1_1 + score2_1) / 2.0

            # 4. 按阈值决定 fine_tuning
            fine_tuning = "SFT" if score >= THRESHOLD else "ORPO"

             # 更新统计信息
            total_count += 1
            sum_score_all += score
            if fine_tuning == "SFT":
                sft_count += 1
                sum_score_sft += score
            else:
                orpo_count += 1
                sum_score_orpo += score

            # 5. 构造新的精简 json 结构
            new_record = {
                "index": record["index"],
                "repo": record["repo"],
                "issue_id": record["issue_id"],
                "pr_num": record["pr_num"],
                "base_commit": record["base_commit"],
                "gemini_reasoning": record.get("gemini_reasoning", ""),
                "root_cause": record["root_cause"],
                "repair_suggestion": record["repair_suggestion"],
                "problem_statement": record["problem_statement"],
                "segments": record["segments"],
                "new_diff": record.get("new_diff", ""),
                "score": score,
                "fine_tuning": fine_tuning,
            }

            # 6. 写回新的 jsonl
            json.dump(new_record, fout, ensure_ascii=False)
            fout.write("\n")

    # 7. 打印统计信息
    mean_all = sum_score_all / total_count if total_count > 0 else 0.0
    mean_sft = sum_score_sft / sft_count if sft_count > 0 else 0.0
    mean_orpo = sum_score_orpo / orpo_count if orpo_count > 0 else 0.0

    print(f"总记录数: {total_count}")
    print(f"SFT 数量: {sft_count}, 平均 score: {mean_sft:.6f}")
    print(f"ORPO 数量: {orpo_count}, 平均 score: {mean_orpo:.6f}")
    print(f"整体平均 score: {mean_all:.6f}")
    print(f"阈值 THRESHOLD: {THRESHOLD}")


def main():
    parser = argparse.ArgumentParser(
        description="合并两个 jsonl 文件，并按 score 打标签为 SFT / ORPO。"
    )
    parser.add_argument(
        "--old",
        default="",
        help="旧 jsonl 文件路径（优先使用其中的记录）",
    )
    parser.add_argument(
        "--new",
        default="",
        help="新 jsonl 文件路径（按其顺序输出记录）",
    )
    parser.add_argument(
        "--out",
        default="",
        help="输出 jsonl 文件路径",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.776,
        help="SFT / ORPO 的打分阈值（全局 THRESHOLD）",
    )

    args = parser.parse_args()
    process_files(args.old, args.new, args.out, args.threshold)


if __name__ == "__main__":
    main()
