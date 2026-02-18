#!/usr/bin/env python3
# coding: utf-8
"""
重新计算 repair_suggestion / repair_code 的相似度：
1) 对 raw_diff 先做注释/print/logger 清洗（本文件实现）。
2) 经过 better_diff.semantic_clean_unified_diff 做语义精简。
3) 经过 genCot_phrase_2 提供的提取/归一化与嵌入打分逻辑，得到新的 score1/score2。
4) 把 new_diff（清洗+精简后的 diff）与新分数写回 JSONL。
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import difflib
import re
import textwrap
import time
import requests
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from collections import defaultdict

# =========================
# 日志初始化
# =========================

logger = logging.getLogger("after_phrase2_new_score")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# ============================================================
#           better_diff.py 内嵌逻辑（语义精简 + 去噪）
# ============================================================

def canonical_unit(unit: str) -> str:
    """
    对一个语句单元做“语义归一化”：
    1. 优先用 ast.parse 解析，dump 成不带位置信息的 AST 字符串
    2. 如果解析失败（不完整或语法错），则退化为“去掉所有空白”的字符串
    """
    code = textwrap.dedent(unit)
    try:
        return re.sub(r"\s+", "", code)
    except SyntaxError:
        # 解析失败就退化为“去空白”的文本
        return re.sub(r"\s+", "", code)

def group_units_from_entries(entries):
    """
    将若干 (line_idx, code_line) 组成“语句级单元”。
    - entries: [(原始行号, 行内容不含+-前缀), ...]
    - 逻辑类似之前的 logical_units，只是带上了每个单元对应的行号列表。

    返回: [(unit_text, [line_idx1, line_idx2, ...]), ...]
    """
    units = []
    buf_lines = []
    buf_idxs = []
    depth = 0

    for idx, line in entries:
        # 跳过开头连续的空行
        if not buf_lines and not line.strip():
            continue

        buf_lines.append(line)
        buf_idxs.append(idx)

        # 简单括号深度统计（多行 list comp / 调用会合并成一个单元）
        depth += sum(line.count(ch) for ch in "([{") \
                 - sum(line.count(ch) for ch in ")]}")

        # 括号闭合且不以反斜杠续行 → 结束一个单元
        if depth <= 0 and not line.rstrip().endswith("\\"):
            units.append(("\n".join(buf_lines), list(buf_idxs)))
            buf_lines, buf_idxs = [], []
            depth = 0

    if buf_lines:
        units.append(("\n".join(buf_lines), list(buf_idxs)))

    return units

def semantic_filter_units(old_units, new_units):
    """
    对两个“语句单元列表”做语义 diff，决定哪些单元需要保留。
    - old_units / new_units: 纯文本单元列表（不含行号）

    返回:
    - old_keep: [bool, ...] 旧单元是否保留
    - new_keep: [bool, ...] 新单元是否保留

    逻辑基本沿用你之前的 semantic_diff：
    - 使用 canonical_unit 做 AST 归一化
    - SequenceMatcher 对齐
    - 把仅仅是“移动”的单元识别出来并忽略
    """
    old_can = [canonical_unit(u) for u in old_units]
    new_can = [canonical_unit(u) for u in new_units]

    matcher = difflib.SequenceMatcher(None, old_can, new_can, autojunk=False)

    # 第一次遍历：统计哪些 canonical 同时出现在 delete 和 insert 中 → 视为“移动”
    deleted_counts = {}
    inserted_counts = {}

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("delete", "replace"):
            for idx in range(i1, i2):
                c = old_can[idx]
                deleted_counts[c] = deleted_counts.get(c, 0) + 1
        if tag in ("insert", "replace"):
            for idx in range(j1, j2):
                c = new_can[idx]
                inserted_counts[c] = inserted_counts.get(c, 0) + 1

    move_counts = {
        c: min(deleted_counts.get(c, 0), inserted_counts.get(c, 0))
        for c in deleted_counts
        if c in inserted_counts
    }
    deleted_remaining = dict(move_counts)
    inserted_remaining = dict(move_counts)

    # 第二次遍历：根据 equal / delete / insert / replace + move 信息，决定保留哪些单元
    old_keep = [False] * len(old_units)
    new_keep = [False] * len(new_units)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            # 完全相同的语句单元 → 不需要出现在 diff 中
            continue

        if tag in ("delete", "replace"):
            for idx in range(i1, i2):
                c = old_can[idx]
                # 如果这是一个“移动”的一端，则跳过
                if deleted_remaining.get(c, 0) > 0:
                    deleted_remaining[c] -= 1
                    continue
                old_keep[idx] = True

        if tag in ("insert", "replace"):
            for idx in range(j1, j2):
                c = new_can[idx]
                if inserted_remaining.get(c, 0) > 0:
                    inserted_remaining[c] -= 1
                    continue
                new_keep[idx] = True

    return old_keep, new_keep

def semantic_clean_unified_diff(diff_block: str, context_window: int = 5) -> str:
    """
    入口：单一 diff 文本（一个 hunk 或一段 diff 块）
    要求：
    - ' ' 上下文行原样保留（不参与对齐、不参与删除）
    - 只对 '+' 和 '-' 行做“语义去重”
    - 删除语义完全一致 / 移动 / 纯格式变化的部分
    - 输出 diff 文本，行顺序与原始 diff 一致
    """
    lines = diff_block.splitlines()

    # 收集删除和新增行（带原始行号）
    removed_entries = []  # [(line_idx, content_without_minus), ...]
    added_entries = []    # [(line_idx, content_without_plus), ...]

    for i, raw in enumerate(lines):
        if not raw:
            continue
        prefix = raw[0]
        content = raw[1:] if len(raw) > 0 else ""
        if prefix == '-' and not raw.startswith('---'):
            removed_entries.append((i, content))
        elif prefix == '+' and not raw.startswith('+++'):
            added_entries.append((i, content))
        # ' ' 作为上下文，我们后面直接原样保留

    # 将删除/新增行分别合并成“语句级单元”
    old_units_data = group_units_from_entries(removed_entries)
    new_units_data = group_units_from_entries(added_entries)

    old_units = [u for u, idxs in old_units_data]
    new_units = [u for u, idxs in new_units_data]

    # 根据语义决定哪些单元（旧/新）需要保留
    old_keep, new_keep = semantic_filter_units(old_units, new_units)

    # 把需要保留的单元对应的“原始行号”收集起来
    keep_removed_line_idxs = set()
    for keep_flag, (_, idxs) in zip(old_keep, old_units_data):
        if keep_flag:
            keep_removed_line_idxs.update(idxs)

    keep_added_line_idxs = set()
    for keep_flag, (_, idxs) in zip(new_keep, new_units_data):
        if keep_flag:
            keep_added_line_idxs.update(idxs)

    # 最终保留下来的修改行行号集合
    change_line_idxs = keep_removed_line_idxs | keep_added_line_idxs

    # 按原始行顺序重新组装 diff：
    # - 上下文 ' ' 行始终保留
    # - '-' 行：只有行号在 keep_removed_line_idxs 中才保留
    # - '+' 行：只有行号在 keep_added_line_idxs 中才保留

    n = len(lines)

    # 先按“连续的 ' ' 行”划分上下文块
    keep_context_line_idxs = set()
    i = 0
    while i < n:
        raw = lines[i]
        if raw and raw[0] == ' ':
            # 进入一个上下文块
            start = i
            j = i
            while j + 1 < n and lines[j + 1] and lines[j + 1][0] == ' ':
                j += 1
            end = j

            # 判断这个块是否需要保留
            keep_block = False
            if change_line_idxs:
                # 如果有任意修改行 c 落在 [start - window, end + window]，则保留整块
                lower = start - context_window
                upper = end + context_window
                for c in change_line_idxs:
                    if lower <= c <= upper:
                        keep_block = True
                        break

            if keep_block:
                for k in range(start, end + 1):
                    keep_context_line_idxs.add(k)

            i = end + 1
        else:
            i += 1

    # 第二遍：按原始行顺序重新组装 diff
    out_lines = []
    for i, raw in enumerate(lines):
        if not raw:
            # 空行保持原样（你也可以按需求改成只在有修改时保留）
            continue
            out_lines.append(raw)
            continue

        prefix = raw[0]

        if prefix == ' ':
            if i in keep_context_line_idxs:
                out_lines.append(raw)

        elif prefix == '-':
            if i in keep_removed_line_idxs:
                out_lines.append(raw)

        elif prefix == '+':
            if i in keep_added_line_idxs:
                out_lines.append(raw)

        else:
            # 其他前缀（例如 @@、diff --git 等）直接保留
            out_lines.append(raw)

    return "\n".join(out_lines)

TRIPLE_QUOTE_RE = re.compile(r'("""|\'\'\')')
# 更宽松的日志/打印匹配：支持 log.*，大小写不敏感，允许行内出现（用 \b 边界）
LOG_OR_PRINT_START_RE = re.compile(
    r"\b(print|logger\.\w+|logging\.\w+|log\.\w+)\s*\(",
    re.IGNORECASE,
)

def clean_diff_noise(diff_text: str) -> str:
    """
    输入：git-diff 格式字符串
    输出：移除注释 / 打印 / 日志后的 diff 字符串
    改动：
    1) 使用更宽松的 LOG/PRINT 正则（含 log.*，忽略大小写）。
    2) 按 diff 块（从一行以 "diff " 开头到下一个 "diff " 之间）分段处理，
       防止因截断导致的多行注释误匹配把后续块也删掉。
    """
    if not diff_text:
        return ""

    lines = diff_text.splitlines()

    # 先按 diff 块切分
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

    def _process_block(block_lines: List[str]) -> List[str]:
        out: List[str] = []
        in_log_or_print = False
        paren_depth = 0

        # ======================================================
        # 预扫描：在“本 block 内”配对多行注释区间，只删能配对上的
        # ======================================================
        skip_line_idxs = set()

        i = 0
        n = len(block_lines)
        while i < n:
            raw = block_lines[i]
            if not raw:
                i += 1
                continue

            # 只在 git diff 的代码行里识别三引号（' ', '+', '-'）
            prefix = raw[0]
            if prefix not in (" ", "+", "-"):
                i += 1
                continue

            code = raw[1:]
            body = code.lstrip()

            # 必须是“行首（忽略缩进）出现三引号”，才认为可能是多行注释边界
            m = re.match(r'^[ \t]*("""|\'\'\')', code)
            if not m:
                i += 1
                continue

            # 单行 """ ... """：直接认为是注释块，丢弃这一行即可
            if len(TRIPLE_QUOTE_RE.findall(code)) >= 2:
                skip_line_idxs.add(i)
                i += 1
                continue

            # 否则：尝试向后找到配对的结束三引号
            j = i + 1
            found_end = False
            while j < n:
                rawj = block_lines[j]
                if rawj and rawj[0] in (" ", "+", "-"):
                    codej = rawj[1:]
                    bodyj = codej.lstrip()
                    if TRIPLE_QUOTE_RE.search(bodyj):
                        found_end = True
                        break
                j += 1

            if found_end:
                # i..j 是一个可配对的多行注释区间 → 全部跳过（但结构行仍会在主循环中保留）
                for k in (range(i, j + 1)):
                    skip_line_idxs.add(k)
                i = j + 1
            else:
                # 没找到结束 → 认为是 diff 截断残片，不做任何删除
                i += 1

        # ======================================================
        # 主循环：按原逻辑处理，但遇到 skip_line_idxs 就跳过
        # ======================================================
        for idx, raw in enumerate(block_lines):
            # 空行保留
            if raw == "":
                out.append(raw)
                continue

            # diff 结构行直接保留
            if raw.startswith(("diff ", "index ", "@@ ", "\\ No newline at end of file")):
                out.append(raw)
                continue
            if raw.startswith("--- ") or raw.startswith("+++ "):
                out.append(raw)
                continue

            # git diff 行前缀：可能是 ' ', '+', '-'
            prefix = raw[0]
            if prefix not in (" ", "+", "-"):
                out.append(raw)
                continue

            # ✅ 如果属于“可配对”的多行注释区间 → 跳过
            if idx in skip_line_idxs:
                continue

            code = raw[1:]
            body = code.lstrip()

            # 2) 单行 # 注释（整行以 # 开头）
            if body.startswith("#"):
                continue

            # 3) 多行 print / logger / logging / log 调用处理
            if in_log_or_print:
                paren_depth += code.count("(") - code.count(")")
                if paren_depth <= 0:
                    in_log_or_print = False
                continue

            if LOG_OR_PRINT_START_RE.search(body):
                in_log_or_print = True
                paren_depth = code.count("(") - code.count(")")
                if paren_depth <= 0:
                    in_log_or_print = False
                continue  # 丢弃触发行

            # 4) 其他正常代码行保留
            out.append(raw)

        return out

    # 分块处理再拼回
    cleaned: List[str] = []
    for blk in blocks:
        cleaned.extend(_process_block(blk))

    return "\n".join(cleaned)

# 这个切分diff块的函数在多个地方用到，提取出来复用
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

def normalize_diff(diff_text: str, mode: str) -> str:
    """
    mode:
        'none'     -> return raw diff
        'noise'    -> clean_noise(block)
        'semantic' -> semantic_clean(block)
        'full'     -> clean_noise(block) -> semantic_clean(block)
    """
    if not diff_text:
        return ""

    if mode == "none":
        return diff_text

    # 定义块级别处理函数
    def process_block(block_text: str) -> str:
        if mode == "noise":
            return clean_diff_noise(block_text)
        elif mode == "semantic":
            return semantic_clean_unified_diff(block_text)
        elif mode == "full":
            return semantic_clean_unified_diff(clean_diff_noise(block_text))
        else:
            raise ValueError(f"未知的 normalize-mode: {mode}")

    # 公共逻辑：切块 → 逐块处理
    new_blocks = []
    for block in split_diff_blocks(diff_text):
        block_text = "\n".join(block)
        new_blocks.append(process_block(block_text))

    return "\n".join(new_blocks)

# ============================================================
#           genCot_phrase_2.py 内嵌逻辑（提取 + 嵌入）
# ============================================================
### 原始规范化diff块、patch相关内容的逻辑
def normalize_repair_code(patch: str) -> str:
    """
    由于在第一阶段中，patch后面的内容使用了缩进4个空格的格式
    观察log发现有些patch前面多了4个空格，有些没有；有些是直接的+/-，有些是+    /-
    这里做一个简单的规范化，去掉多余的缩进，统一成直接的+/-格式
    这样可以和raw_diff更好地对齐，
    """
    if not patch:
        return patch
    cleaned_lines: List[str] = []
    for line in patch.splitlines():
        if line.startswith("    ---") or line.startswith("    +++") or line.startswith("+++") or line.startswith("---") or line.startswith("***") or line.startswith("    ***"):
            continue
        if line.startswith("+    "):
            cleaned_lines.append("+" + line[5:])
            continue
        if line.startswith("-    "):
            cleaned_lines.append("-" + line[5:])
            continue
        if line.startswith("    "):
            cleaned_lines.append(line[4:])
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

### 规范化diff块、patch相关内容
def build_rs_chunk(block_text: str) -> str:
    """
    将一个 diff-block 转换成用于 repair_suggestion 的文本格式：
    
    file_path
    ### Before Change
    <removed lines>
    ### After Change
    <added lines>

    若此块没有任何修改（无 + 或 -），返回 ""（表示丢弃此块）
    """

    lines = block_text.splitlines()
    file_path = None
    removed = []
    added = []

    for line in lines:
        if line.startswith("+++ b/"):
            file_path = line[6:].strip()
        elif line.startswith("--- a/"):
            file_path = line[6:].strip()
        elif line.startswith("@@"):
            continue  # hunk header 不需要
        elif line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:])
        else:
            continue  # 上下文行忽略

    if not file_path:
        return ""

    # 如果没有任何修改内容，则跳过此块
    if not removed and not added:
        return ""
    
    out = [file_path, "### Before Change"]
    if removed:
        out.extend(removed)
    out.append("### After Change")
    if added:
        out.extend(added)

    return "\n".join(out).strip()

def build_rc_chunk(block_text: str) -> str:
    """
    将一个 diff-block 转换成 repair_code 用的格式：
    
    file_path
    @@ ...
    <context / + / - lines>
    
    若无修改返回 ""（丢弃）
    """

    lines = block_text.splitlines()
    file_path = None
    out = []
    has_change = False

    for line in lines:
        if line.startswith("+++ b/"):
            file_path = line[6:].strip()
        elif line.startswith("--- a/"):
            file_path = line[6:].strip()
        elif line.startswith("@@"):
            out.append(line)
        elif line.startswith("+") and not line.startswith("+++"):
            out.append(line)
            has_change = True
        elif line.startswith("-") and not line.startswith("---"):
            out.append(line)
            has_change = True
        elif line.startswith(" "):
            out.append(line)

    if not file_path:
        return ""
    if not has_change:
        return ""

    # 文件路径作为第一行
    return file_path + "\n" + "\n".join(out)

### 统一的 block-aware 截断函数
def truncate_chunks_by_total_len(
    chunks: List[str],
    max_len: int,
) -> List[str]:
    """
    对 chunk 列表做“整体预算”的尾部截断：
    - 按顺序累计字符数
    - 超过 max_len 后：
        - 当前 chunk 截断到刚好剩余字符数
        - 后续 chunk 丢弃
    """
    if not chunks:
        return []

    out = []
    used = 0

    for chunk in chunks:
        if used >= max_len:
            break

        remain = max_len - used
        if len(chunk) <= remain:
            out.append(chunk)
            used += len(chunk)
        else:
            # 当前块截断
            out.append(chunk[:remain])
            used += remain
            break

    return out
### 统一的构造嵌入列表
def build_diff_inputs(
    new_diff: str,
    build_mode: str,      # "legacy" | "block"
    max_len: int = 16000,
) -> Tuple[List[str], List[str]]:
    """
    返回:
        diff_for_rs_chunks: List[str]
        diff_for_rc_chunks: List[str]

    语义保证：
    - 不论 split_blocks=True/False，RS/RC 的“块内容组织方式”一致
    - 唯一差异是是否把块 join 成一个大 chunk !!
    - 截断采用“总预算顺序截断”，保证 split/non-split 一致
    """
    diff_text = new_diff
    blocks = split_diff_blocks(diff_text)

    rs_chunks: List[str] = []
    rc_chunks: List[str] = []

    for blk in blocks:
        blk_text = "\n".join(blk)

        rs = build_rs_chunk(blk_text)
        if rs.strip():
            rs_chunks.append(rs)

        rc = build_rc_chunk(blk_text)
        if rc.strip():
            rc_chunks.append(rc)

    # 🔑 对“分块结果”做整体预算截断
    rs_chunks = truncate_chunks_by_total_len(rs_chunks, max_len)
    rc_chunks = truncate_chunks_by_total_len(rc_chunks, max_len)

    if build_mode == "legacy":
        # 旧版：合并成单个大 chunk
        rs_chunks = ["\n\n".join(rs_chunks)] if rs_chunks else []
        rc_chunks = ["\n\n".join(rc_chunks)] if rc_chunks else []

    return rs_chunks, rc_chunks

### 嵌入模型调用
def call_siliconflow_embedding(
    inputs,
    model: str = "Qwen/Qwen3-Embedding-8B",
    encoding_format: str = "float",
    dimensions: Optional[int] = 4096,
    timeout: int = 120,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> Optional[List[List[float]]]:
    """
    调用硅基流动 /v1/embeddings 接口生成文本向量。
    :param inputs: str 或 List[str]
    :param model: 具体嵌入模型名称
    :param encoding_format: 'float' 或 'base64'
    :param dimensions: 可选维度 (仅 Qwen/Qwen3-Embedding 系列支持)
    :param timeout: 请求超时秒数
    :param max_retries: 最大重试次数
    :param retry_delay: 重试基础间隔秒数
    :return: List[List[float]] 或 None
    """
    if isinstance(inputs, str):
        input_payload = [inputs]
    elif isinstance(inputs, list):
        input_payload = inputs
    else:
        raise TypeError("inputs 必须是 str 或 List[str]")

    api_key = ""
    if not api_key:
        logger.error("缺少环境变量 SILICONFLOW_API_KEY")
        return None

    url = "https://api.siliconflow.cn/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    body = {
        "model": model,
        "input": input_payload,
        "encoding_format": encoding_format,
    }
    if dimensions is not None:
        body["dimensions"] = dimensions

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            time.sleep(0.5)  # 避免过快请求
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            emb_list = []
            for item in data.get("data", []):
                emb = item.get("embedding")
                if emb is None:
                    logger.warning("缺少 embedding 字段: %s", item)
                    continue
                emb_list.append(emb)
            return emb_list
        except requests.exceptions.HTTPError as e:
            resp_payload = e.response.text if e.response is not None else repr(e)
            logger.error("嵌入请求 HTTP 错误 %s: %s", getattr(e.response, "status_code", "?"), resp_payload[:300])
            last_err = e
        except requests.exceptions.RequestException as e:
            logger.error("嵌入请求网络错误: %s", repr(e))
            last_err = e
        except Exception as e:
            logger.error("嵌入请求未知错误: %s", repr(e))
            last_err = e

        sleep_sec = retry_delay * (attempt + 1)
        logger.info("嵌入调用第 %d/%d 次失败，%.1f 秒后重试……", attempt + 1, max_retries, sleep_sec)
        time.sleep(sleep_sec)
    
    logger.error("嵌入请求多次失败，放弃: %s", repr(last_err))
    return None

### 相似度计算部分
def cosine_similarity_np(vec1: List[float], vec2: List[float]) -> float:
    """
    使用 numpy 计算余弦相似度。维度不一致时按最短维度对齐。
    """
    try:
        v1 = np.asarray(vec1, dtype=float)
        v2 = np.asarray(vec2, dtype=float)
        if v1.size == 0 or v2.size == 0:
            return 0.0
        n = min(v1.size, v2.size)
        v1 = v1[:n]
        v2 = v2[:n]
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom == 0.0:
            return 0.0
        return float(np.dot(v1, v2) / denom)
    except Exception as e:
        logger.warning("cosine_similarity_np error: %s", repr(e))
        return 0.0

def chunk_embedding_similarity(query_emb, chunk_emb_list):
    sims = []
    for emb in chunk_emb_list:
        sims.append(cosine_similarity_np(query_emb, emb))
    return sims

def compute_beta_from_range(scores: List[float]) -> float:
    """
    根据极差动态计算 beta:
        beta = 1 - 0.5 * range
    确保 beta 在 [0.5, 1.0]。
    """
    if not scores:
        return 1.0
    
    smin = min(scores)
    smax = max(scores)
    r = smax - smin

    beta = 1.0 - 0.5 * r
    beta = max(0.5, min(1.0, beta))  # 安全边界
    return beta

def softmax(x: List[float], beta: float = 1.0) -> np.ndarray:
    x = np.array(x, dtype=float) / beta
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()

BETA_LIST = [0.5, 0.75, 1.0, 1.5, 2.0] # 备用 beta 列表，方便纵向对比
BASELINE_BETA = 1.0
def compute_scores(
    repair_suggestion: str,
    diff_for_repair_suggestion_chunks: List[str],
    repair_code: str,
    diff_for_repair_code_chunks: List[str],
    display_idx: int,
) -> Tuple[
    float,                     # score1_A
    float,                     # score2_A
    Dict[float, float],        # score1_B_dict
    Dict[float, float],        # score2_B_dict
    Dict[str, Any],            # emb_info
]:
    """
    使用 Softmax 加权的 diff-block 相似度。
    同时返回方法 A / 方法 B 的分数，以及本条记录的 embedding 信息。
    
    统一返回结构（无论成功 / 失败）：
        - score1_A: float
        - score2_A: float
        - score1_B_dict: {beta: float}
        - score2_B_dict: {beta: float}
        - emb_info: Dict
    """
    # ---------- 0. 预构造空返回 ----------
    zero_B_dict = {beta: 0.0 for beta in BETA_LIST}

    empty_emb_info: Dict[str, Any] = {
        "index": display_idx,
        "emb_rs": None,
        "emb_rs_chunks": [],
        "emb_rc": None,
        "emb_rc_chunks": [],
    }

    # ===== Step 1：组织输入 =====
    # 调用硅基流动时需要合并在一个列表中
    inputs = (
        [repair_suggestion]
        + diff_for_repair_suggestion_chunks
        + [repair_code]
        + diff_for_repair_code_chunks
    )

    # ===== Step 2：嵌入 =====
    emb_list = call_siliconflow_embedding(inputs)
    if not emb_list or len(emb_list) != len(inputs):
        logger.warning(f"[{display_idx}] 计算嵌入失败，返回 0 分")
        empty_emb_info: Dict[str, Any] = {
            "index": display_idx,
            "emb_rs": None,
            "emb_rs_chunks": [],
            "emb_rc": None,
            "emb_rc_chunks": [],
            "diff_for_rs_chunks": diff_for_repair_suggestion_chunks,
            "diff_for_rc_chunks": diff_for_repair_code_chunks,
        }
        return 0.0, 0.0, zero_B_dict, zero_B_dict, empty_emb_info

    # ===== Step 3：切分向量 =====
    n_rs = len(diff_for_repair_suggestion_chunks)
    n_rc = len(diff_for_repair_code_chunks)

    emb_rs = emb_list[0]                        # repair_suggestion 的向量
    emb_rs_chunks = emb_list[1 : 1 + n_rs]      # 每个 diff-block 的向量

    emb_rc = emb_list[1 + n_rs]                 # repair_code 的向量
    emb_rc_chunks = emb_list[2 + n_rs : 2 + n_rs + n_rc]

    # ===== Step 4：块相似度 =====
    sims_rs = chunk_embedding_similarity(emb_rs, emb_rs_chunks)
    sims_rc = chunk_embedding_similarity(emb_rc, emb_rc_chunks)

    # ===== Step 5：动态计算beta =====
    # beta_rs = compute_beta_from_range(sims_rs)
    # beta_rc = compute_beta_from_range(sims_rc)

    # ===== Step 6：Softmax 权重 =====
    w_rs_dict = {}
    w_rc_dict = {}

    for beta in BETA_LIST:
        w_rs_dict[beta] = softmax(sims_rs, beta) if sims_rs else np.array([])
        w_rc_dict[beta] = softmax(sims_rc, beta) if sims_rc else np.array([])

    # =============================================================
    #                   方法 A（你的原始计算方式）
    # =============================================================

    if BASELINE_BETA not in w_rs_dict:
        raise ValueError(f"BASELINE_BETA={BASELINE_BETA} 不在 BETA_LIST 中")
    
    score1_A = float(np.sum(w_rs_dict[BASELINE_BETA] * np.array(sims_rs))) if sims_rs else 0.0
    score2_A = float(np.sum(w_rc_dict[BASELINE_BETA] * np.array(sims_rc))) if sims_rc else 0.0

    # =============================================================
    #                   方法 B：向量加权再算相似度（多 beta）
    # =============================================================

    score1_B_dict = {}
    score2_B_dict = {}

    # 修复：若没有 chunk，则返回 0
    for beta in BETA_LIST:
        w_rs = w_rs_dict[beta]
        w_rc = w_rc_dict[beta]

        # score1_B
        if emb_rs_chunks and len(w_rs) == len(emb_rs_chunks):
            V_rs = np.sum(w_rs.reshape(-1, 1) * np.asarray(emb_rs_chunks), axis=0)
            score1_B_dict[beta] = cosine_similarity_np(emb_rs, V_rs)
        else:
            score1_B_dict[beta] = 0.0

        # score2_B
        if emb_rc_chunks and len(w_rc) == len(emb_rc_chunks):
            V_rc = np.sum(w_rc.reshape(-1, 1) * np.asarray(emb_rc_chunks), axis=0)
            score2_B_dict[beta] = cosine_similarity_np(emb_rc, V_rc)
        else:
            score2_B_dict[beta] = 0.0


    # ===== Step 7：日志输出 =====
    logger.info(
        "[%s] A(score1=%.4f, score2=%.4f) | "
        "B(beta=%.2f)(score1=%.4f, score2=%.4f)",
        display_idx,
        score1_A,
        score2_A,
        BASELINE_BETA,
        score1_B_dict[BASELINE_BETA],
        score2_B_dict[BASELINE_BETA],
    )
    
    # ===== Step 8：准备 embedding 信息（写到单独文件用） =====
    emb_info: Dict[str, Any] = {
        "index": display_idx,
        "emb_rs": emb_rs,
        "emb_rs_chunks": emb_rs_chunks,
        "emb_rc": emb_rc,
        "emb_rc_chunks": emb_rc_chunks,
    }

    return score1_A, score2_A, score1_B_dict, score2_B_dict, emb_info

# ============================================================
#           after_phrase2_new_score.py 原有主体逻辑
# ============================================================
def compute_scores_with_retry(
    repair_suggestion: str,
    diff_for_repair_suggestion_chunks: List[str],
    normalized_repair_code: str,
    diff_for_repair_code_chunks: List[str],
    display_idx: int,
    max_retries: int = 3,
) -> Tuple[
    float,
    float,
    Dict[float, float],
    Dict[float, float],
    int,
    Dict[str, Any],
]:
    """
    调用 compute_scores，若所有分数均为 0 则重试。

    返回（始终结构一致）：
        score1_A: float
        score2_A: float
        score1_B_dict: Dict[beta, float]
        score2_B_dict: Dict[beta, float]
        attempts_used: int
        emb_info: Dict
    """
    zero_B_dict = {beta: 0.0 for beta in BETA_LIST}
    last_score1_A = 0.0
    last_score2_A = 0.0
    last_score1_B_dict = zero_B_dict
    last_score2_B_dict = zero_B_dict
    last_emb_info: Dict[str, Any] = {
        "index": display_idx,
        "emb_rs": None,
        "emb_rs_chunks": [],
        "emb_rc": None,
        "emb_rc_chunks": [],
    }

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            (
                score1_A,
                score2_A,
                score1_B_dict,
                score2_B_dict,
                emb_info,
            ) = compute_scores(
                repair_suggestion,
                diff_for_repair_suggestion_chunks,
                normalized_repair_code,
                diff_for_repair_code_chunks,
                display_idx,
            )
            last_score1_A = score1_A
            last_score2_A = score2_A
            last_score1_B_dict = score1_B_dict
            last_score2_B_dict = score2_B_dict
            last_emb_info = emb_info

            # ===== 成功判定：不是“全 0” =====
            all_zero = (
                score1_A == 0.0
                and score2_A == 0.0
                and all(v == 0.0 for v in score1_B_dict.values())
                and all(v == 0.0 for v in score2_B_dict.values())
            )

            if not all_zero:
                return (
                    score1_A,
                    score2_A,
                    score1_B_dict,
                    score2_B_dict,
                    attempt,
                    emb_info,
                )
            
        except Exception as e:
            last_err = e
            logger.exception("记录 %d 第 %d 次计算相似度失败：%s", display_idx, attempt, repr(e))
            last_scores = (0.0, 0.0, 0.0, 0.0)
    if last_err:
        logger.warning("记录 %d 重试 %d 次后仍为 0，最后异常：%s", display_idx, max_retries, repr(last_err))
    else:
        logger.warning("记录 %d 重试 %d 次后仍为 0", display_idx, max_retries)

    return (
        last_score1_A,
        last_score2_A,
        last_score1_B_dict,
        last_score2_B_dict,
        max_retries,
        last_emb_info,
    )

### JSONL 读写
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def write_jsonl(path: str, items: List[Dict[str, Any]], append: bool = False) -> None:
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")

### 绘图
def plot_hist(values, title, save_path, bins=200):
    """绘制直方图（bins 默认 200）"""
    if not values:
        logger.warning(f"plot_hist: 无数据可绘图 {title}")
        return
    
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, color="steelblue", edgecolor="black", alpha=0.7)
    plt.title(title)
    plt.xlabel("score")
    plt.ylabel("count")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"直方图已保存到：{save_path}")

def plot_hist_overlay(
    values_a,
    values_b,
    label_a,
    label_b,
    title,
    save_path,
    bins=200,
):
    if not values_a or not values_b:
        logger.warning(f"plot_hist_overlay: 无数据可绘图 {title}")
        return

    plt.figure(figsize=(8, 5))

    plt.hist(
        values_a,
        bins=bins,
        alpha=0.5,
        label=label_a,
        color="steelblue",
        edgecolor="black",
    )
    plt.hist(
        values_b,
        bins=bins,
        alpha=0.5,
        label=label_b,
        color="orange",
        edgecolor="black",
    )

    plt.title(title)
    plt.xlabel("score")
    plt.ylabel("count")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logger.info(f"叠加直方图已保存到：{save_path}")

def plot_multi_beta_hist(beta_to_values, title, save_path, bins=200):
    if not beta_to_values:
        logger.warning(f"plot_multi_beta_hist: 无数据 {title}")
        return

    plt.figure(figsize=(9, 6))

    colors = {
        0.5: "#1f77b4",
        0.75: "#2ca02c",
        1.0: "#ff7f0e",
        1.5: "#9467bd",
        2.0: "#d62728",
    }

    for beta, values in sorted(beta_to_values.items()):
        if not values:
            continue
        plt.hist(
            values,
            bins=bins,
            alpha=0.4,
            label=f"β={beta}",
            color=colors.get(beta),
            edgecolor="black",
        )

    plt.title(title)
    plt.xlabel("score")
    plt.ylabel("count")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logger.info(f"多 beta 分布图已保存到：{save_path}")

def generate_pdf_report(image_paths, output_pdf):
    doc = SimpleDocTemplate(output_pdf, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Softmax Score Analysis Report", styles["Title"]))

    for img in image_paths:
        story.append(Image(img, width=500, height=400))
        story.append(Paragraph(img, styles["Normal"]))

    doc.build(story)

### 处理单条记录
def process_one_record(
    rec: Dict[str, Any],
    normalize_mode: str,
    build_mode: str,
) -> Dict[str, Any]:
    """
    处理单条记录。

    返回字段语义：
        - new_diff
        - new_score1_A, new_score2_A
        - new_score1_B_by_beta, new_score2_B_by_beta
        - score1_B_vs_A_improved (beta=1.0)
        - score2_B_vs_A_improved (beta=1.0)
        - attempts_used
        - embeddings
    """
    display_idx = rec.get("index") 
    raw_diff = rec.get("raw_diff", "")

    # 1) 使用 normalize_diff 进行清洗+语义精简
    new_diff = normalize_diff(raw_diff, normalize_mode)

    # 2) 生成打分所需字段（统一入口）
    repair_suggestion = rec.get("repair_suggestion") or rec.get("Repair_Suggestion", "")
    repair_code = rec.get("repair_code") or rec.get("Repair_Code", "")
    normalized_repair_code = normalize_repair_code(repair_code)

    diff_for_rs_chunks, diff_for_rc_chunks = build_diff_inputs(
        new_diff=new_diff,
        build_mode=build_mode,   # "legacy" | "block"
        max_len=16000,
    )

    # 3) 相似度打分（含重试）
    (
        score1_A,
        score2_A,
        score1_B_dict,
        score2_B_dict,
        attempts_used,
        emb_info,
    ) = compute_scores_with_retry(
        repair_suggestion,
        diff_for_rs_chunks,
        normalized_repair_code,
        diff_for_rc_chunks,
        display_idx,
    )

    score1_B_baseline = score1_B_dict.get(BASELINE_BETA, 0.0)
    score2_B_baseline = score2_B_dict.get(BASELINE_BETA, 0.0)

    # B 相比 A 是否提升
    score1_B_vs_A_improved = score1_B_baseline > score1_A
    score2_B_vs_A_improved = score2_B_baseline > score2_A
    
    logger.info(
        "记录 %d 完成 | "
        "A(score1=%.4f, score2=%.4f) | "
        "B(beta=%.1f)(score1=%.4f, score2=%.4f) | "
        "B>A: (score1=%s, score2=%s) | "
        "attempts=%d",
        display_idx,
        score1_A,
        score2_A,
        BASELINE_BETA,
        score1_B_baseline,
        score2_B_baseline,
        score1_B_vs_A_improved,
        score2_B_vs_A_improved,
        attempts_used,
    )
    
    res = dict(rec)  # 不改动原对象引用
    res.update({
        "new_diff": new_diff,

        # A 方法（标量）
        "new_score1_A": score1_A,
        "new_score2_A": score2_A,

        # B 方法（按 beta）
        "new_score1_B_by_beta": score1_B_dict,
        "new_score2_B_by_beta": score2_B_dict,

        # 横向对比（beta=1.0）
        "score1_B_vs_A_improved": score1_B_vs_A_improved,
        "score2_B_vs_A_improved": score2_B_vs_A_improved,

        # 调试信息
        "attempts_used": attempts_used,

        # 嵌入向量存储
        "embeddings": emb_info
    })
    return res

### 主函数入口
def main():
    parser = argparse.ArgumentParser(description="重新计算 score1/score2 (A/B方法) 并写回 JSONL")
    parser.add_argument("--input", required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--output", required=True, help="输出 JSONL 文件路径")
    parser.add_argument(
        "--normalize-mode",
        choices=["none", "noise", "semantic", "full"],
        default="full",
        help="diff 预处理模式：none / noise / semantic / full"
    )
    parser.add_argument(
        "--build-mode",
        choices=["legacy", "block"],
        default="block",
        help="diff 构造方式：legacy=整块旧逻辑，block=按 diff 块新逻辑"
    )
    parser.add_argument(
        "--embed-output",
        default=None,
        help="若指定，则输出 embedding 信息到此 JSONL"
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="若指定，则在该目录下输出分数分布/增量的直方图",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="线程池大小",
    )
    args = parser.parse_args()

    logger.info(
    "=== normalize-mode = %s | build-mode = %s ===",
    args.normalize_mode,
    args.build_mode
)

    # ===== 读取数据 =====
    records = read_jsonl(args.input)

    total = len(records)
    logger.info("读取输入文件 %s，记录数: %d", args.input, total)

    # 先清空输出文件
    write_jsonl(args.output, [], append=False)
    if args.embed_output:
        write_jsonl(args.embed_output, [], append=False)

    # ===== 统计容器 =====
    # A 方法（标量）
    score1_A_by_grp = {"all": [], "true": [], "false": []}
    score2_A_by_grp = {"all": [], "true": [], "false": []}

    # B 方法（按 beta）
    score1_B_by_beta = {
        "all": defaultdict(list),
        "true": defaultdict(list),
        "false": defaultdict(list),
    }
    score2_B_by_beta = {
        "all": defaultdict(list),
        "true": defaultdict(list),
        "false": defaultdict(list),
    }

    # B - A（仅 beta=1.0）
    diff1_BA_by_grp = {"all": [], "true": [], "false": []}
    diff2_BA_by_grp = {"all": [], "true": [], "false": []}

    # 计数
    processed = 0
    score1_BA_improved_cnt = {"all": 0, "true": 0, "false": 0}
    score2_BA_improved_cnt = {"all": 0, "true": 0, "false": 0}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx = {}
        for rec in records:
            rec_idx = rec.get("index")
            future = executor.submit(process_one_record, rec, args.normalize_mode, args.build_mode)
            future_to_idx[future] = rec_idx

        for future in as_completed(future_to_idx):
            rec_idx = future_to_idx[future]
            try:
                new_rec = future.result() # process获得的数据
            except Exception as e:
                logger.exception("子线程处理异常（index=%s）：%s", rec_idx, repr(e))
                continue

            if not isinstance(new_rec, dict):
                logger.warning("子线程返回非字典结果（index=%s），跳过：%s", rec_idx, type(new_rec))
                continue

            processed += 1

            # 立即写出主结果 这边还需要改一改
            emb_info = new_rec.pop("embeddings", None)
            write_jsonl(args.output, [new_rec], append=True)

            # 写出 embedding 信息
            if emb_info is not None and args.embed_output:
                write_jsonl(args.embed_output, [emb_info], append=True)

            # ===== 统计处理 =====

            grp = "true" if new_rec.get("gemini_judgement") is True else "false"

            s1A = new_rec["new_score1_A"]
            s2A = new_rec["new_score2_A"]
            s1B_by_beta = new_rec["new_score1_B_by_beta"]
            s2B_by_beta = new_rec["new_score2_B_by_beta"]

            # === A 方法 ===
            for g in ("all", grp):
                score1_A_by_grp[g].append(s1A)
                score2_A_by_grp[g].append(s2A)
            # === B 方法（所有 beta）===
            for beta, v in s1B_by_beta.items():
                for g in ("all", grp):
                    score1_B_by_beta[g][beta].append(v)

            for beta, v in s2B_by_beta.items():
                for g in ("all", grp):
                    score2_B_by_beta[g][beta].append(v)

            # === B - A（beta=BASELINE_BETA）===
            s1B_baseline = s1B_by_beta.get(BASELINE_BETA, 0.0)
            s2B_baseline = s2B_by_beta.get(BASELINE_BETA, 0.0)

            for g in ("all", grp):
                diff1_BA_by_grp[g].append(s1B_baseline - s1A)
                diff2_BA_by_grp[g].append(s2B_baseline - s2A)

            if s1B_baseline > s1A:
                for g in ("all", grp):
                    score1_BA_improved_cnt[g] += 1
            if s2B_baseline > s2A:
                for g in ("all", grp):
                    score2_BA_improved_cnt[g] += 1


    # ===== 汇总 =====
    def mean_safe(xs):
        return float(np.mean(xs)) if xs else 0.0

    logger.info("======= 汇总统计 =======")
    logger.info("处理条数: %d / %d", processed, total)

    for g in ("all", "true", "false"):
        denom = max(len(score1_A_by_grp[g]), 1)
        logger.info(
            "[%s] mean scores | "
            "A(score1=%.4f, score2=%.4f) | "
            "B(beta=%.1f)(score1=%.4f, score2=%.4f) | "
            "B>A rate(score1=%.2f%%, score2=%.2f%%)",
            g,
            mean_safe(score1_A_by_grp[g]),
            mean_safe(score2_A_by_grp[g]),
            BASELINE_BETA,
            mean_safe(score1_B_by_beta[g][BASELINE_BETA]),
            mean_safe(score2_B_by_beta[g][BASELINE_BETA]),
            score1_BA_improved_cnt[g] / denom * 100,
            score2_BA_improved_cnt[g] / denom * 100,
        )

    # ===== 可选绘图 =====
    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)

        beta_tag = f"beta={BASELINE_BETA}"

        for grp in ["all", "true", "false"]:
            grp_suffix = f"gemini={grp}"

            # ---------- A vs B (baseline beta) ----------
            plot_hist_overlay(
                score1_A_by_grp[grp],
                score1_B_by_beta[grp][BASELINE_BETA],
                label_a="Method A",
                label_b=f"Method B ({beta_tag})",
                title=f"Score1 Distribution (A vs B @ {beta_tag}) | {grp_suffix}",
                save_path=f"{args.plot_dir}/score1_A_vs_B_{beta_tag}_{grp}.png",
            )

            plot_hist_overlay(
                score2_A_by_grp[grp],
                score2_B_by_beta[grp][BASELINE_BETA],
                label_a="Method A",
                label_b=f"Method B ({beta_tag})",
                title=f"Score2 Distribution (A vs B @ {beta_tag}) | {grp_suffix}",
                save_path=f"{args.plot_dir}/score2_A_vs_B_{beta_tag}_{grp}.png",
            )

            # ---------- B - A 差值 (baseline beta) ----------
            plot_hist(
                diff1_BA_by_grp[grp],
                f"Score1 Difference (B - A @ {beta_tag}) | {grp_suffix}",
                f"{args.plot_dir}/diff_score1_BA_{beta_tag}_{grp}.png",
            )

            plot_hist(
                diff2_BA_by_grp[grp],
                f"Score2 Difference (B - A @ {beta_tag}) | {grp_suffix}",
                f"{args.plot_dir}/diff_score2_BA_{beta_tag}_{grp}.png",
            )

            # ---------- Method B: multi-beta ----------
            plot_multi_beta_hist(
                score1_B_by_beta[grp],
                f"Score1 Method-B (multi beta) | {grp_suffix}",
                f"{args.plot_dir}/score1_B_multi_beta_{grp}.png",
            )

            plot_multi_beta_hist(
                score2_B_by_beta[grp],
                f"Score2 Method-B (multi beta) | {grp_suffix}",
                f"{args.plot_dir}/score2_B_multi_beta_{grp}.png",
            )

        logger.info(
            "已在 %s 中输出 all / true / false 三种视角下的 A/B、B-A、multi-beta 分布图（baseline=%s）",
            args.plot_dir,
            BASELINE_BETA,
        )



if __name__ == "__main__":
    main()
