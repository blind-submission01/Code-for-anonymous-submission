#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from typing import Iterable, Dict, Any, List, Optional

SECTION_RE = re.compile(
    r"^\s*#\s*(error_patterns|code_elements|problem_behaviors|domain_context)\b",
    re.IGNORECASE,
)

CATEGORIES = ["error_patterns", "code_elements", "problem_behaviors", "domain_context"]


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                continue


def normalize_item(s: str) -> str:
    s = s.strip().strip("`\"' \t")
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".")
    return s


def parse_content_sections(text: str) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {k: [] for k in CATEGORIES}
    if not text:
        return result

    lines = text.splitlines()
    n = len(lines)
    i = 0
    while i < n:
        m = SECTION_RE.match(lines[i])
        if m:
            key = m.group(1).lower()
            # 找下一条非空行作为内容行
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            if j < n:
                items_line = lines[j].strip()
                # 逗号分隔，清洗
                items = [normalize_item(x) for x in items_line.split(",")]
                items = [x for x in items if x]
                result[key] = items
            i = j + 1
            continue
        i += 1
    return result


def extract_message_content(obj: Dict[str, Any]) -> Optional[str]:
    # 兼容 {"response":{"body":{"choices":[{"message":{"content":...}}]}}}
    resp = obj.get("response")
    if isinstance(resp, dict):
        body = resp.get("body")
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except Exception:
                body = None
        if isinstance(body, dict):
            choices = body.get("choices") or []
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message") if isinstance(choices[0], dict) else None
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content
    # 回退：有些行可能直接给 content
    content = obj.get("content")
    if isinstance(content, str):
        return content
    return None


def main():
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "-i", "--input",
        default=os.path.join(os.path.dirname(__file__), ""),
        help="SiliconFlow 批量推理输出 JSONL 文件路径"
    )
    parser.add_argument(
        "-o", "--output",
        default=os.path.join(os.path.dirname(__file__), "k.jsonl"),
        help="输出的 1.py 输入 JSONL 文件路径（每行含 key_words）"
    )
    parser.add_argument(
        "--keep-empty", action="store_true",
        help="当 problem_behaviors 为空时也保留该条（默认跳过）"
    )
    args = parser.parse_args()

    in_count = 0
    out_count = 0
    with open(args.output, "w", encoding="utf-8") as wf:
        for obj in read_jsonl(args.input):
            in_count += 1
            content = extract_message_content(obj)
            if not content:
                continue
            sections = parse_content_sections(content)

            # 至少需要 problem_behaviors 非空
            if not sections.get("problem_behaviors"):
                if not args.keep_empty:
                    continue

            record = {
                "key_words": {
                    "problem_behaviors": sections.get("problem_behaviors", []),
                    "error_patterns": sections.get("error_patterns", []),
                    "code_elements": sections.get("code_elements", []),
                    "domain_context": sections.get("domain_context", []),
                }
            }
            # 可选：携带源 ID，便于追踪
            if "custom_id" in obj:
                record["source_custom_id"] = obj["custom_id"]

            wf.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_count += 1

    print(f"Parsed {in_count} lines, wrote {out_count} lines -> {args.output}")


if __name__ == "__main__":
    main()