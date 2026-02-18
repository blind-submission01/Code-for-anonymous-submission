#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Iterable, Dict, Any, List, Optional

SYSTEM_PROMPT = """You are an expert in analyzing GitHub Issues. Your job is to extract keywords from an issue that can be used to search for similar issues.

Task: Extract 4 categories of keywords from the given GitHub issue to identify issues with similar patterns.

Input: GitHub issue

Output Format:
# error_patterns (error types, error messages, exception patterns, etc.)
word1, word2, ...

# code_elements (class names, method names, parameter names, module names, etc.)
word1, word2, ...

# problem_behaviors (problem description, expected behavior, abnormal behavior, etc.)
word1, word2, ...

# domain_context (technical concepts, technical domains, frameworks, libraries, environments, algorithms, etc.)
word1, word2, ...
"""

def read_json_or_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    # JSON array
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict):
                    yield obj
            return
    except json.JSONDecodeError:
        pass

    # JSONL
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

def build_request_line(
    custom_id: str,
    issue_content: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    top_p: float = 1.0,
    stream: bool = False,
) -> Dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"GitHub issue is {issue_content}" or ""},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": stream,
        },
    }

def main():
    parser = argparse.ArgumentParser(description="Build SiliconFlow batch JSONL from problem_statement of failed items")
    parser.add_argument(
        "-i", "--input",
        default=os.path.join(os.path.dirname(__file__), "..", "Result", "swebench_lite_for_INL_claude.json"),
        help="Input JSON/JSONL file path (contains problem_statement)"
    )
    parser.add_argument(
        "-o", "--output",
        default=os.path.join(os.path.dirname(__file__), "siliconflow_batch.jsonl"),
        help="Output JSONL path for SiliconFlow batch inference"
    )
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V3", help="Model name (same for all lines)")
    parser.add_argument("--max-tokens", type=int, default=1024, help="max_tokens per request")
    parser.add_argument("--temperature", type=float, default=0.0, help="temperature per request (default 0.0)")
    parser.add_argument("--top-p", type=float, default=1.0, help="top_p per request")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of requests (0 means no limit)")
    args = parser.parse_args()

    count_in = 0
    count_out = 0
    with open(args.output, "w", encoding="utf-8") as outf:
        for obj in read_json_or_jsonl(args.input):
            count_in += 1
            try:
                if (obj.get("status") or "").lower() != "failed":
                    continue
                problem_statement = obj.get("problem_statement") or ""
                if not problem_statement.strip():
                    continue

                # Prefer instance_id as custom_id; fallback to ID or index
                custom_id = str(obj.get("instance_id") or obj.get("ID") or f"req-{count_in}")
                line = build_request_line(
                    custom_id=custom_id,
                    issue_content=problem_statement,
                    model=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    top_p=args.top_p,
                    stream=False,
                )
                outf.write(json.dumps(line, ensure_ascii=False) + "\n")
                count_out += 1

                if args.limit and count_out >= args.limit:
                    break
            except Exception:
                continue

    print(f"Scanned: {count_in}, Written: {count_out}, Output: {args.output}")

if __name__ == "__main__":
    main()