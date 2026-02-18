from hmac import new
from math import log
import os
import re
import sys
import time
import json
import random
import logging
from typing import Any, Dict, List, Tuple, Optional
from xml.etree.ElementPath import findtext
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
SYSTEM_PROMPT_GPT_REFINE = """
**IMPORTANT**: Our <task> is solely to analyze the provided text and generate the requested reasoning output. Simply read the following information and generate a response. Do NOT plan sub-tasks, execute commands, inspect files, use tools, or reference the working directory. You must stay in pure text reasoning and generation mode.

You are a code analysis expert specializing in **bug repair**. You will be given:
	- A GitHub issue description.
	- Two additional references:diff_remove(-) and diff_add(+).
        diff_remove(-):It represents the code version where the issue occurred, and was removed by an real commit due to the presence of errors.
        diff_add(+):It represents the code version after the issue was fixed, and was added by a real commit due to the resolution of errors.
    - Root Cause and Repair Suggestion about this GitHub issue.

Perform the following Tasks step by step:
1. Think Step-by-Step:
    - Carefully read the issue description and relevant code slices.
    - Identify the incorrect behavior and consider repair methods referring to diff_remove and diff_add as guidance.
    - (Note: The diff reflects real changes for GitHub issue resolving.It may represent improved best practices and broader considerations, but it may also include human errors.Please evaluate carefully and use with caution.)
2. Refine the Root Cause:
    - Based on your thought, refine the given Root Cause to progressively reveal the root cause of the issue, and express it in a developer-friendly manner.
    - (Note: You may also refer to the reasoning provided by Gemini as an additional reference.)
3. Refine the Repair Suggestion:
    - Based on your thought, refine the given Repair Suggestion to provide a practical, step-by-step solution for resolving the issue, and express it in a developer-friendly manner.
    - (Note:1. A couple of small code snippets or inline code may be included, but avoid outputting medium and large blocks of code. 2. Hierarchical and structured format is encouraged.)

Format your answer strictly as follows Output Template:

<root cause>
The Content of Root Cause   
</root cause>
<suggestion>
The Content of Repair Suggestion
</suggestion>
"""

def format_gpt_refine_user(issue: str, diff_remove: str, diff_add: str, root_cause: str, gemini_reasoning: str, repair_suggestion: str) -> str:
    return (
        f"Issue Description:\n{issue}\n"
        f"diff_remove:\n{diff_remove}\n"
        f"diff_add:\n{ diff_add}\n"
        f"root_cause:\n{root_cause}\n"
        f"gemini_reasoning:\n{gemini_reasoning}\n"
        f"repair_suggestion:\n{repair_suggestion}\n"
    )

# =============== diff 工具函数（参考 softmax_score.py） ===============

def split_diff_blocks(diff_text: str) -> List[List[str]]:
    """
    将原生 git-diff 按块切分：
    - 每个块从以 "diff " 开头的行开始，到下一个 "diff " 行之前（含起始行）。
    - 如果最前面没有 "diff " 行，会把前置行并入第一块。
    """
    if not diff_text:
        return []
    lines = diff_text.splitlines()
    blocks: List[List[str]] = []
    current: List[str] = []
    for line in lines:
        if line.startswith("diff "):
            if current:
                blocks.append(current)
            current = [line]
        else:
            if current:
                current.append(line)
            else:
                current = [line]
    if current:
        blocks.append(current)
    return blocks

def _build_remove_add_from_block(block_text: str) -> Tuple[str, str]:
    """
    从一个 diff block 中构建：
    - diff_remove: file_path + 所有 '-' 和 ' ' 行
    - diff_add:    file_path + 所有 '+' 和 ' ' 行
    若无有效 file_path 或无修改，返回 ("", "")
    """
    lines = block_text.splitlines()
    file_path: Optional[str] = None
    remove_lines: List[str] = []
    add_lines: List[str] = []

    for line in lines:
        if line.startswith("+++ b/"):
            file_path = line[6:].strip()
        elif line.startswith("--- a/"):
            file_path = line[6:].strip()
        elif line.startswith("-") and not line.startswith("---"):
            remove_lines.append(line)
        elif line.startswith("+") and not line.startswith("+++"):
            add_lines.append(line)
        elif line.startswith(" "):
            # 上下文同时给两边
            # remove_lines.append(line)
            # add_lines.append(line)
            continue
        else:
            # 其他前缀忽略
            continue

    if not file_path:
        return "", ""
    if not remove_lines and not add_lines:
        return "", ""

    diff_remove = file_path + "\n" + "\n".join(remove_lines) if remove_lines else ""
    diff_add = file_path + "\n" + "\n".join(add_lines) if add_lines else ""
    return diff_remove, diff_add

def build_remove_add_from_diff(diff_text: str) -> Tuple[str, str]:
    """
    对整个 unified diff：
    - 先按 block 切分
    - 每个 block 提取 diff_remove/diff_add
    - 用空行拼接所有 block 的结果
    """
    if not diff_text:
        return "", ""

    blocks = split_diff_blocks(diff_text)
    remove_chunks: List[str] = []
    add_chunks: List[str] = []

    for blk in blocks:
        blk_text = "\n".join(blk)
        r, a = _build_remove_add_from_block(blk_text)
        if r:
            remove_chunks.append(r)
        if a:
            add_chunks.append(a)

    diff_remove = "\n\n".join(remove_chunks) if remove_chunks else ""
    diff_add = "\n\n".join(add_chunks) if add_chunks else ""
    return diff_remove, diff_add

# Models (default names; adjust to your tenant
MODEL_COT_REFINE = os.environ.get("REFINE_MODEL", "gpt-5.1")

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
               timeout: int = 180, max_retries: int = 3,
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
            last_error = RuntimeError("Codex exec 执行超时（超过3分钟）")
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

ROOT_CAUSE_REGEX = re.compile(
    r"<root\s+cause>\s*(.*?)\s*</root\s+cause>",
    re.IGNORECASE | re.DOTALL,
)

REPAIR_SUGGESTION_REGEX = re.compile(
    r"<suggestion>\s*(.*?)\s*</suggestion>",
    re.IGNORECASE | re.DOTALL,
)

def _extract_suggestion_block(text: str) -> str:
    """
    优先使用完整的 <suggestion>...</suggestion>；
    若没有闭合标签，但存在 <suggestion>，则从该标签之后一直截到文本末尾。
    """
    if not text:
        return ""

    # 1）优先：完整闭合标签
    m = REPAIR_SUGGESTION_REGEX.search(text)
    if m:
        return m.group(1).strip()

    # 2）兜底：只有起始 <suggestion>，没有 </suggestion>
    m_open = re.search(r"<suggestion\b[^>]*>", text, re.IGNORECASE)
    if not m_open:
        return ""

    start = m_open.end()
    return text[start:].strip()

def parse_generation(output_text: str) -> Dict[str, str]:
    """
    新版解析：提取 Root_Cause、Repair_Suggestion。
    - root_cause：仍然要求成对的 <root cause>...</root cause>
    - repair_suggestion：优先成对 <suggestion>...</suggestion>，否则从 <suggestion> 起截到末尾
    """
    output_text = output_text or ""

    # root_cause 仍用老办法
    rc_match = ROOT_CAUSE_REGEX.search(output_text)
    root_cause = (rc_match.group(1) if rc_match else "").strip()

    # repair_suggestion 用更鲁棒的方式
    repair_suggestion = _extract_suggestion_block(output_text)

    return {
        "Refined_Root_Cause": root_cause,
        "Refined_Repair_Suggestion": repair_suggestion,
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
        "refined_gen_text": gen_text,
        "refined_root_cause": parsed.get("Refined_Root_Cause", ""),
        "refined_repair_suggestion": parsed.get("Refined_Repair_Suggestion", ""),
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

    fine_tuning = entry.get("fine_tuning", "")

    # SFT 分支：不需要新生成内容，一律视为有效
    if fine_tuning == "SFT":
        return True

    # ORPO 分支：要求 new_root_cause / new_repair_suggestion 都非空
    if fine_tuning == "ORPO":
        new_root_cause = entry.get("new_root_cause") or ""
        new_repair_suggestion = entry.get("new_repair_suggestion") or ""
        return bool(new_root_cause and new_repair_suggestion)

def longest_valid_prefix(results: List[Optional[Dict[str, Any]]]) -> int:
    for idx, entry in enumerate(results):
        if not has_valid_repair(entry):
            return idx
    return len(results)

# 主流程层（并行处理）
from concurrent.futures import ThreadPoolExecutor, as_completed
def process_one_record(global_idx: int, rec: Dict[str, Any]) -> Dict[str, Any]:
    # display_idx = global_idx + 1 #global_idx是实际数组索引，从0开始，display_idx是用户看到的记录编号，从1开始，即jsonl文件中的行号
    # 优先使用输入记录中的 index（1-based），否则回退
    rec_index_raw = rec.get("index", None)

    try:
        display_idx = int(rec_index_raw) if rec_index_raw is not None else (global_idx + 1)
    except (ValueError, TypeError):
        display_idx = global_idx + 1
   
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
         
        new_diff = rec.get("new_diff", "")
        old_root_cause = rec.get("root_cause", "")
        old_repair_suggestion = rec.get("repair_suggestion", "")
        gemini_reasoning = rec.get("gemini_reasoning", "")
        fine_tuning = rec.get("fine_tuning", "")

        # SFT默认不生成新的 root_cause / repair_suggestion
        new_root_cause = ""
        new_repair_suggestion = ""
        refined_gen_text = ""


        # 如果 fine_tuning 为 SFT，则直接返回（不调用 Codex）
        if fine_tuning == "SFT":
            logger.info("第 %d 条记录 fine_tuning=SFT，跳过 refine，直接返回。", display_idx)
        else:
            # 非 SFT（例如 ORPO）→ 使用 new_diff 提取 diff_remove / diff_add，然后调用 Codex refinement
            diff_source = new_diff
            diff_remove, diff_add = build_remove_add_from_diff(diff_source)

            logger.info("第 %d 条记录：开始 refinement（fine_tuning=%s）", display_idx, fine_tuning or "N/A")
            init_user = format_gpt_refine_user(
                issue=issue,
                diff_remove=diff_remove,
                diff_add=diff_add,
                root_cause=old_root_cause,
                gemini_reasoning=gemini_reasoning,
                repair_suggestion=old_repair_suggestion,
            )
            init_res = model_generate(
                init_user,
                SYSTEM_PROMPT_GPT_REFINE,
                MODEL_COT_REFINE,
                temperature=0.2,
                record_index=display_idx,
            )

            # 兼容大小写两种 key
            new_root_cause = (
                init_res.get("refined_root_cause")
                or init_res.get("Refined_Root_Cause")
                or ""
            )
            new_repair_suggestion = (
                init_res.get("refined_repair_suggestion")
                or init_res.get("Refined_Repair_Suggestion")
                or ""
            )
            refined_gen_text = init_res.get("refined_gen_text", "")

        return {
            "index": display_idx, # 使用输入记录的 index（或回退值）
            "repo": rec.get("repo", ""),
            "issue_id": rec.get("issue_id", ""),
            "pr_num": rec.get("pr_num"),
            "is_change": rec.get("is_change"),
            "base_commit": rec.get("base_commit", ""),
            "problem_statement": issue_copy,
            "segments": rec.get("segments", []),
            "new_diff": new_diff,
            "score": rec.get("score", 0.0),
            "fine_tuning": fine_tuning,
            "old_root_cause": old_root_cause,
            "new_root_cause": new_root_cause,
            "old_repair_suggestion": old_repair_suggestion,
            "new_repair_suggestion": new_repair_suggestion,
            "gemini_reasoning": gemini_reasoning,
            "refined_gen_text": refined_gen_text,
        }
    except Exception as e:
            logger.exception("处理单条记录失败 %d: %s", display_idx, repr(e))
            return rec
    
def process_records(records: List[Dict[str, Any]], max_workers: int = 20, base_index: int = 0) -> List[Dict[str, Any]]:
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

    # 统计本轮 fine_tuning 为 ORPO 的记录数
    orpo_processed = sum(
        1
        for item in processed
        if isinstance(item, dict) and item.get("fine_tuning") == "ORPO"
    )
    logger.info("本轮 fine_tuning 为 ORPO 的记录数: %d", orpo_processed)

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
