import json
import os
import math
from typing import Dict, Tuple, List

# 使用非交互后端，便于在无图形界面环境保存图片
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT_FILES = [

]

OUTPUT_FILE = ""

# 新增：collect 源文件路
COLLECT_FILE = ""
# 新增：预置去重文件
PRESEEDED_FILE = ""

def read_jsonl(path):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

# 新增：读取 collect.jsonl，收集 (repo, base_commit) 二元组
def load_collect_pairs(path: str):
    pairs = set()
    for rec in read_jsonl(path) or []:
        try:
            repo = rec.get("repo")
            base_commit = str(rec.get("base_commit"))
            if repo and base_commit:
                pairs.add((repo, base_commit))
        except Exception:
            continue
    return pairs

# 新增：从已有 JSONL 载入已处理键
def load_initial_seen(path: str):
    initial_keys: set[tuple[str, str, str, str]] = set()
    for rec in read_jsonl(path) or []:
        try:
            repo = rec.get("repo")
            issue_id = str(rec.get("issue_id"))
            merged_pr_num = str(rec.get("merged_pr_num"))
            true_base_sha = str(rec.get("true_base_sha"))
            if repo and issue_id and merged_pr_num and true_base_sha:
                initial_keys.add((repo, issue_id, merged_pr_num, true_base_sha))
        except Exception:
            continue
    return initial_keys

def record_ok(r: dict) -> bool:
    try:
        body_length = int(r.get("body_length") or 0)
        star = int(r.get("star") or 0)
        diff_length = int(r.get("diff_length") or 0)
        return (
            100 < body_length and
            # r.get("has_code_example") is True and
            star >= 30
            #r.get("archived") in (False, "false", 0) and
            #r.get("has_code_changes") is True and
            #r.get("if_file_ok") is True
            #diff_length < 1000
        )
    except Exception:
        return False

def filter_python_diff_blocks(diff_content: str) -> str:
    """
    过滤原始 diff_content，仅保留修改目标为 .py 且路径不包含 'test' 的 diff 块。
    块定义：以 "diff" 开头的行至下一个以 "diff" 开头的行之前。
    通过块内的 '--- ' 或 '+++ ' 行提取修改文件路径：
      - 优先使用 '+++ '（新路径），若为 /dev/null 则回退到 '--- '（旧路径）
      - 去除 a/ 或 b/ 前缀
      - 若路径不以 .py 结尾或包含 'test'（大小写不敏感），则丢弃该块
    """
    if not diff_content:
        return ""

    lines = diff_content.splitlines()
    blocks: list[list[str]] = []
    current: list[str] = []

    # 切分 diff 块
    for line in lines:
        if line.startswith("diff"):
            if current:
                blocks.append(current)
            current = [line]
        else:
            if current:
                current.append(line)
            else:
                # 忽略块外零散内容
                continue
    if current:
        blocks.append(current)

    def extract_path_from_block(block: list[str]) -> str | None:
        file_path = None

        for line in block:
            if line.startswith("--- a/") or line.startswith("+++ b/"):
                file_path = line[6:]
                if file_path:
                    file_path = file_path.split("\t")[0].strip()
                    return file_path
        return file_path


    kept_blocks: list[list[str]] = []
    for blk in blocks:
        path = extract_path_from_block(blk)
        if not path:
            continue
        lower = path.lower()
        if not lower.endswith(".py"):
            continue
        if "test" in lower:
            continue
        kept_blocks.append(blk)

    return "\n".join("\n".join(b) for b in kept_blocks)

def main():
    history_seen = load_initial_seen(PRESEEDED_FILE)
    current_seen: set[tuple[str, str, str, str]] = set()
    kept = 0
    total = 0

    collect_pairs = load_collect_pairs(COLLECT_FILE)
    skipped_collect = 0
    skipped_empty_diff = 0
    skipped_history = 0
    skipped_current = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for fp in INPUT_FILES:
            for rec in read_jsonl(fp):
                total += 1

                # 新增：先处理 diff_content，只保留 .py、非 test 的 diff 块
                raw_diff = rec.get("diff_content") or ""
                new_diff = filter_python_diff_blocks(raw_diff)
                if not new_diff or not new_diff.strip():
                    skipped_empty_diff += 1
                    continue

                # 新增：先用 (repo, merged_pr_num) 与 collect 的 (repo, instance_id) 比对
                repo = rec.get("repo")
                true_base_sha = str(rec.get("true_base_sha"))
                if repo and true_base_sha and (repo, true_base_sha) in collect_pairs:
                    skipped_collect += 1
                    continue

                if not record_ok(rec):
                    continue

                key = (repo, str(rec.get("issue_id")), str(rec.get("merged_pr_num")), true_base_sha)

                if key in history_seen:
                    skipped_history += 1
                    continue

                if key in current_seen:
                    skipped_current += 1
                    continue

                current_seen.add(key)
                rec["diff_content"] = new_diff
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
    print(
        f"Done. total={total}, kept={kept}, "
        f"skipped_collect={skipped_collect}, skipped_empty_diff={skipped_empty_diff}, "
        f"skipped_history={skipped_history}, skipped_current={skipped_current}, "
        f"history_keys={len(history_seen)}, current_keys={len(current_seen)}, "
        f"output={OUTPUT_FILE}"
    )
if __name__ == "__main__":
    main()
