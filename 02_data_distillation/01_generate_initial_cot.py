from math import log
import os
import re
import sys
import time
import json
import random
import logging
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import subprocess
import shlex
import shutil
import requests

logger = logging.getLogger("tri_cot")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)


# 提示词层
def format_segments(segments: List[Dict[str, Any]]) -> str:
    """Format segments into a developer-friendly block placed in the USER message."""
    lines = []
    for idx, seg in enumerate(segments, start=1):
        susp = seg.get("suspicious", {})
        supp = seg.get("supplementary", {})
        lines.append(f"--- Segment Group {idx} ---")
        lines.append(f"[Suspicious] file: {susp.get('file','')}")
        lines.append("```")
        lines.append(susp.get("content","").rstrip())
        lines.append("```")
        lines.append(f"[Supplementary] file: {supp.get('file','')}")
        lines.append("```")
        lines.append(supp.get("content","").rstrip())
        lines.append("```")
        lines.append("")
    return "\n".join(lines)

SYSTEM_PROMPT_GPT = """
**IMPORTANT**: Our <task> is solely to analyze the provided text and generate the requested reasoning output. Simply read the following information and generate a response. Do NOT plan sub-tasks, execute commands, inspect files, use tools, or reference the working directory. You must stay in pure text reasoning and generation mode.

You are a code analysis expert specializing in **bug repair**. You will be given:
	- A GitHub issue description
	- Several segment groups: each contains a suspicious code slice and its related supplementary slice.
Perform the following Tasks step by step:

1. Think Step-by-Step:
	- Carefully read the issue description and relevant code slices.
	- Explain why the suspicious code is problematic, referring directly to the issue.
	- Describe the correct behavior or logic, then outline practical and concrete repair solution.
2. Generate Natural Language Description: Summarize your findings into a concise but sufficient, developer-friendly description including both the issue **Root Cause** and the proposed **Repair Suggestion**.
    -**Root Cause**: Provide a layered explanation tying the defect back to the suspicious code, highlighting exactly why it is incorrect.
    -**Repair Suggestion**: Provide concrete and practical repair steps in execution order. A couple of small code snippets or inline code may be included, but avoid outputting medium and large blocks of code.
3. Generate Patches: Based on your reasoning, generate minimal and correct **Code Patches**.
    -**Code Patches**: Include minimal diff-style code necessary to implement the fix.
(Note: Do not add code comments)

Format your answer strictly as follows Output Template:

<root cause>
The Content of Root Cause
</root cause>
<suggestion>
The Content of Repair Suggestion
</suggestion>
<patch>
    <file path>
    The Content of Code Patches

    <file path>
    The Content of Code Patches
</patch>
"""

def format_gpt_user(issue: str, segments: List[Dict[str, Any]]) -> str:
    return f"Issue Description:\n{issue}\n\nSegment Groups:\n{format_segments(segments)}"

# Models (default names; adjust to your tenant
MODEL_COT = os.environ.get("COT_MODEL", "gpt-5.1")

def _find_codex_path() -> str:
    """查找 codex 命令的完整路径"""
    codex_path = shutil.which("codex")
    if codex_path:
        return codex_path
    
    # 常见路径
    common_paths = [
        "/opt/homebrew/bin/codex",
        "/usr/local/bin/codex",
        os.path.expanduser("~/.local/bin/codex"),
    ]
    for path in common_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
    
    raise RuntimeError("找不到 codex 命令，请确认 Codex CLI 已安装并在 PATH 中")

def _build_codex_command(model: Optional[str]) -> List[str]:
    codex_path = _find_codex_path()
    base = f"{codex_path} exec --json --skip-git-repo-check"
    if model:
        base += f" --model {shlex.quote(model)}"
    if os.name == "nt":
        return ["cmd", "/c", base]
    return shlex.split(base)

def call_codex(model: str, system_prompt: str, user_prompt: str,
               timeout: int = 600, max_retries: int = 1,
               retry_delay: float = 5.0) -> str:
    prompt = f"{system_prompt.rstrip()}\n\n{user_prompt}".strip()
    cmd = _build_codex_command(model)
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "====================调用 Codex CLI，模型：%s（第 %d/%d 次尝试）====================",
                model or "默认", attempt, max_retries
            )
            # 确保环境变量包含常见路径
            env = os.environ.copy()
            common_paths = ["/opt/homebrew/bin", "/usr/local/bin", os.path.expanduser("~/.local/bin")]
            current_path = env.get("PATH", "")
            # 修复：先累积所有路径，再一次性设置
            # 将 PATH 拆分为列表，避免子字符串误判
            path_list = current_path.split(":") if current_path else []
            for path in common_paths:
                if path and path not in path_list:
                    path_list.insert(0, path)  # 插入到最前面，优先查找
            env["PATH"] = ":".join(path_list)
            
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout,
                env=env
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Codex exec 执行超时（超过10分钟）") from exc
        except FileNotFoundError as exc:
            raise RuntimeError("找不到 codex 命令，请确认 Codex CLI 已安装并在 PATH 中") from exc
        except Exception as exc:
            last_error = RuntimeError(f"Codex CLI 调用失败: {exc}")
        else:
            if result.returncode != 0:
                stderr = result.stderr.strip()
                stdout = result.stdout.strip()
                # 检测使用限制错误
                for line in stdout.splitlines():
                    try:
                        event = json.loads(line)
                        if event.get("type") == "error":
                            error_msg = event.get("message", "")
                            if "usage limit" in error_msg.lower():
                                logger.error("检测到使用限制错误: %s", error_msg)
                                raise RuntimeError(f"检测到使用限制错误: {error_msg}")
                    except (json.JSONDecodeError, AttributeError):
                        continue
                last_error = RuntimeError(
                    f"Codex exec 执行失败 (返回码: {result.returncode})\nstderr: {stderr}\nstdout: {stdout}"
                )
            else:
                logger.debug("Codex stdout 原始输出:")
                logger.debug(result.stdout.strip())

                assistant_message: Optional[str] = None
                for line in result.stdout.strip().splitlines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    event_type = event.get("type")
                    # 检测使用限制错误
                    if event_type == "error":
                        error_msg = event.get("message", "")
                        if "usage limit" in error_msg.lower():
                            logger.error("检测到使用限制错误: %s", error_msg)
                            raise RuntimeError(f"检测到使用限制错误: {error_msg}")
                    if event_type in ("item.completed", "item.started"):
                        item = event.get("item", {}) or {}
                        item_type = item.get("type") or item.get("item_type")
                        if item_type in ("assistant_message", "agent_message"):
                            content = item.get("text") or item.get("content")
                            if isinstance(content, list):
                                parts: List[str] = []
                                for chunk in content:
                                    if isinstance(chunk, str):
                                        parts.append(chunk)
                                    elif isinstance(chunk, dict):
                                        parts.append(chunk.get("text", ""))
                                content = "".join(parts)
                            if content:
                                assistant_message = content

                if assistant_message:
                    return assistant_message

                stdout_preview = result.stdout.strip()[:500]
                last_error = RuntimeError(
                    f"Codex 输出中未找到 assistant 消息。原始输出: {stdout_preview}"
                )

        if attempt < max_retries:
            delay = retry_delay * attempt
            logger.warning(
                "Codex 调用尝试 %d/%d 失败，将在 %.1f 秒后重试。原因: %s",
                attempt, max_retries, delay, last_error
            )
            time.sleep(delay)
        else:
            break

    raise last_error if last_error else RuntimeError("Codex 调用失败，原因未知")

# 内容提取层
# 提取整个 Natural Language Description 部分


# 提取整个 Patches 部分

# 提取每个 segment 的 patch: 文件路径 + 代码
ROOT_CAUSE_REGEX = re.compile(
    r"<root\s+cause>\s*(.*?)\s*</root\s+cause>",
    re.IGNORECASE | re.DOTALL,
)

REPAIR_SUGGESTION_REGEX = re.compile(
    r"<suggestion>\s*(.*?)\s*</suggestion>",
    re.IGNORECASE | re.DOTALL,
)

PATCH_SECTION_REGEX = re.compile(
    r"<patch>\s*(.*?)\s*</patch>",
    re.IGNORECASE | re.DOTALL,
)

def parse_generation(output_text: str) -> Dict[str, str]:
    """
    新版解析：提取 Root_Cause、Repair_Suggestion、Repair_Code。
    XML 标签内的内容保持原样（仅去除首尾空白），以便后续流程使用。
    """
    output_text = output_text or ""

    rc_match = ROOT_CAUSE_REGEX.search(output_text)
    rs_match = REPAIR_SUGGESTION_REGEX.search(output_text)
    patch_match = PATCH_SECTION_REGEX.search(output_text)

    root_cause = (rc_match.group(1) if rc_match else "").strip()
    repair_suggestion = (rs_match.group(1) if rs_match else "").strip()
    repair_code = (patch_match.group(1) if patch_match else "").strip()

    return {
        "Root_Cause": root_cause,
        "Repair_Suggestion": repair_suggestion,
        "Repair_Code": repair_code,
    }
# 生成层

def model_generate(user_prompt: str,
                   system_prompt: str,
                   model: str,
                   temperature: float = 0.2,
                   record_index: Optional[int] = None) -> Dict[str, Any]:
    """
    通用模型调用与解析封装：
    输入 system_prompt + user_prompt + model，返回解析后的结构。
    """
    #gen_text = call_chatanywhere(model, system_prompt, user_prompt, temperature=temperature)
    logger.info(f"====================第 {record_index} 条记录：开始访问codex模型====================")
    gen_text = call_codex(model, system_prompt, user_prompt)
    logger.info(f"====================第 {record_index} 条记录：生成内容预览:====================")
    logger.info(gen_text)
    parsed = parse_generation(gen_text)
    logger.info(f"====================第 {record_index} 条记录：解析结果预览:====================")
    logger.info(parsed)
    return {
        "gen_text": gen_text,
        "root_cause": parsed.get("Root_Cause", ""),
        "repair_suggestion": parsed.get("Repair_Suggestion", ""),
        "repair_code": parsed.get("Repair_Code", ""),
    }

# io层
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                logger.warning("Skip bad JSONL line: %s", line[:120])
    return out

def write_jsonl(path: str, items: List[Dict[str, Any]], append: bool = False) -> None:
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def has_valid_repair(entry: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(entry, dict):
        return False
    root_cause = entry.get("root_cause") or entry.get("Root_Cause")
    repair_suggestion = entry.get("repair_suggestion") or entry.get("Repair_Suggestion")
    return bool(root_cause and repair_suggestion)

def longest_valid_prefix(results: List[Optional[Dict[str, Any]]]) -> int:
    for idx, entry in enumerate(results):
        if not has_valid_repair(entry):
            return idx
    return len(results)

# 主流程层（并行处理）
from concurrent.futures import ThreadPoolExecutor, as_completed
def process_one_record(global_idx: int, rec: Dict[str, Any]) -> Dict[str, Any]:
    display_idx = global_idx + 1 #global_idx是实际数组索引，从0开始，display_idx是用户看到的记录编号，从1开始，即jsonl文件中的行号
    try:
        logger.info(f"====================开始处理第 {display_idx} 条记录====================")
        issue = rec.get("problem_statement", "") or rec.get("issue", "")
        issue_copy = issue
        # 处理 issue 长度
        if len(issue) < 3000:
            # 长度小于3000，不处理
            pass
        elif 3000 <= len(issue) <= 10000:
            # 长度在3000到10000之间，去掉超过3000部分的50%
            excess = len(issue) - 3000
            cut_amount = int(excess * 0.5)
            issue = issue[:3000 + cut_amount]
        elif len(issue) > 10000:
            # 长度大于10000，去掉超过6000部分的20%
            excess = len(issue) - 6000
            cut_amount = int(excess * 0.2)
            issue = issue[:6000 + cut_amount]
        segments = rec.get("segments", [])
        if not issue or not segments:
            logger.warning("Record %d missing issue/segments. Skipping.", display_idx)
            return rec
        
        raw_diff = rec.get("raw_diff", "")
        issue_id = rec.get("issue_id")
        repo = rec.get("repo") 

        logger.info(f"====================第 {display_idx} 条记录：开始初始{MODEL_COT}生成====================")
        init_user = format_gpt_user(issue, segments)
        init_res = model_generate(
            init_user,
            SYSTEM_PROMPT_GPT,
            MODEL_COT,
            temperature=0.2,
            record_index=display_idx,
        )
        return {
            "index": display_idx,
            "repo": repo,
            "issue_id": issue_id,
            "pr_num": rec.get("pr_num"),
            "is_change": rec.get("is_change"),
            "base_commit": rec.get("base_commit", ""),
            "problem_statement": issue_copy,
            "segments": segments,
            "raw_diff": raw_diff,
            # "issue_id": issue_id,
            # "repo": repo,
            # "raw_diff": raw_diff,
            # "problem_statement": issue_copy,
            # "segments": segments,
            "root_cause": init_res["root_cause"],
            "repair_suggestion": init_res["repair_suggestion"],
            "repair_code": init_res["repair_code"],
            "gpt_gen_text": init_res["gen_text"],
        }
    except Exception as e:
            logger.exception("处理单条记录失败 %d: %s", display_idx, repr(e))
            return rec
    
def process_records(records: List[Dict[str, Any]], max_workers: int = 10, base_index: int = 0) -> List[Dict[str, Any]]:
    results: List[Optional[Dict[str, Any]]] = [None] * len(records)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_local_idx = {}
        for local_idx, rec in enumerate(records):
            actual_idx = base_index + local_idx #actual_idx是全局索引，就像在一个大的数组中处理一样
            future = executor.submit(process_one_record, actual_idx, rec)
            future_to_local_idx[future] = local_idx
        for future in as_completed(future_to_local_idx):
            local_idx = future_to_local_idx[future]
            try:
                results[local_idx] = future.result()
            except Exception as e:
                display_idx = base_index + local_idx + 1 #display_idx是用户看到的记录编号，从1开始，即jsonl文件中的行号
                logger.exception("Unhandled exception in worker for record %d: %s", display_idx, repr(e))
                results[local_idx] = records[local_idx]
    for idx, item in enumerate(results):
        if item is None:
            results[idx] = records[idx]
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tri-Stage CoT Bug-Repair Pipeline (JSONL in/out).")
    parser.add_argument("--input", required=True, help="Path to input JSONL")
    parser.add_argument("--output", required=True, help="Path to output JSONL")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--start-index", type=int, default=1, help="起始记录序号（从 1 开始）")
    parser.add_argument("--count", type=int, default=None, help="本轮希望处理的记录条数")
    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    records = read_jsonl(args.input)
    if not records:
        logger.error("No records loaded from %s", args.input)
        sys.exit(2)

    if args.count is not None and args.count <= 0:
        logger.error("count 必须为正整数。")
        sys.exit(2)

    start_index = max(1, args.start_index) # start_index是从1开始的，记录jsonl文件中的第几条
    if args.start_index < 1:
        logger.warning("start-index 小于 1，已调整为 1。")

    base_index = start_index - 1 # 转换为从0开始的索引，实际数组索引
    total_records = len(records)
    if base_index >= total_records:
        logger.error("Start index %d 超出范围 (总计 %d 条)", start_index, total_records)
        sys.exit(2)

    if args.count is None:
        end_index = total_records
    else:
        end_index = min(total_records, base_index + args.count) #end_index是实际数组索引，取不到，但是代表一共取到多少条

    selected_records = records[base_index:end_index]
    if not selected_records:
        logger.error("选定范围内没有记录可处理。")
        sys.exit(2)

    logger.info("main 开始处理jsonl中第 %d 条起共 %d 条记录 → %s", start_index, len(selected_records), args.output)
    processed = process_records(selected_records, base_index=base_index)

    valid_count = longest_valid_prefix(processed)
    if valid_count:
        valid_results = [item for item in processed[:valid_count] if item]
        write_jsonl(args.output, valid_results, append=True)
        logger.info("已追加写入 %d 条结果（原始编号 %d-%d）→ %s",
                    valid_count, start_index, start_index + valid_count - 1, args.output)
    else:
        logger.warning("本轮在第 %d 条记录即出现失败，无有效结果写入。", start_index)

    next_start = start_index + valid_count
    if valid_count < len(selected_records):
        logger.warning("第 %d 条记录结果缺失 root_cause 或 repair_suggestion，下一轮请从第 %d 条开始。", next_start, next_start)
    else:
        logger.info("本轮全部 %d 条记录处理完成，下一轮可从第 %d 条开始。", len(selected_records), next_start)

    logger.info("main 结束。本轮有效写入 %d 条，下一轮起点：第 %d 条。", valid_count, next_start)


if __name__ == "__main__":
    main()
