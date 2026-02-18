from curses import raw
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
import requests


logger = logging.getLogger("tri_cot")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)

ZERO_SCORE_INDICES: List[Any] = []
INVALID_INDICES: List[Any] = []

SYSTEM_PROMPT_GEMINI = """
You are a software issue analysis expert. You will be given:
    - A GitHub issue description.
    - A Root Cause analysis of this GitHub issue.
    - Two additional references: diff_remove(-) and diff_add(+).
        diff_remove(-): It represents the code version where the issue occurred, and was removed by a real commit due to the presence of errors.
        diff_add(+): It represents the code version after the issue was fixed, and was added by a real commit due to the resolution of errors.

Perform the following Tasks step by step:
1.Thoroughly Consider
    - Carefully read the issue description and relevant diff_remove (Because the diff_remove reflects the erroneous code that caused the issue).
    - Learning the cause of this issue errors from the comparisons from diff_remove to diff_add.
2.Judgment
    - Determine whether the Root Cause given reveals the actual cause of this issue based only on its correctness.
    (Note: Sometimes diff code may be insufficient and serve only as a reference. Therefore, carefully analyze the issue description itself to determine whether the provided Root Cause analysis is correct.)

Format your answer strictly as follows:
    - When you think the root cause is correct, Output:
        <Yes>
        (No reason or additional output is required.)
    - When you think the Root Cause is not correct, Output:
        <No>(, and then give your precise reason.)
"""

# 先不使用看看效果
OTHER = """
You are a software issue analysis expert. You will be given:
    - A GitHub issue description.
    - A Root Cause analysis of this GitHub issue.
    - Two additional references: diff remove (real commit removed content) and diff add (real commit added content).

Perform the following Tasks step by step:

1.Thoroughly Consider
    - Carefully read the issue description and relevant diff_remove (Because the diff_remove reflects the erroneous code that caused the issue).
    - Learning the cause of this issue errors from the comparisons from diff_remove to diff_add.
2.Judgment
    - Determine whether the Root Cause given reveals the actual cause of this issue based only on its correctness and sufficiency.
    - Output **Yes**, when you think the root cause is correct.  
    - Output **No**, when you think the Root Cause is not correct, and then give your **precise reason**.

Format your answer strictly as follows Output Template:

<judgement>
Yes or No
</judgement>
<reasoning>
The Content of precise reason when judgement is No, otherwise empty.
</reasoning>
"""

def format_gemini_user(issue: str, root_cause: str, diff_remove: str, diff_add: str) -> str:
    return (
        f"Issue Description:\n{issue}\n\n"
        f"Root Cause:\n{root_cause}\n\n"
        f"diff_remove:\n{diff_remove}\n\n"
        f"diff_add:\n{ diff_add}\n"
    )

### Claude 调用
MODEL_CLAUDE = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
MODEL_GEMINI = os.environ.get("GEMINI_MODEL", "gemini-3-pro-preview")

def call_claude(
    model: str, 
    system_prompt: str,  
    user_prompt: str,
    max_retries: int = 3,
    # max_tokens: int = 512
) -> str:

    api_key =""
    base_url ="https://cc.585dg.com"
    url = f"{base_url}/v1/messages"

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    
    payload = {
    "model": model,
    # "max_tokens": max_tokens,
    "system": [
        {"type": "text", "text": system_prompt}
    ],
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt}
            ],
        }
    ],
}
    
    last_err = None
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=90)
            if response.status_code in (401, 403, 429, 500, 503):
                logger.error("Claude API returned %s: %s", response.status_code, response.text[:200])
                time.sleep(1.5 * (attempt + 1))
                continue
            elif response.status_code == 200:
                try:
                    data = response.json()
                    if not isinstance(data, dict):
                        logger.error("API 返回的不是字典,而是: %s", type(data).__name__)
                        logger.error("响应内容: %s", str(data)[:200])
                        time.sleep(1.5 * (attempt + 1))
                        continue

                    parts: List[str] = []
                    for block in data.get("content", []) or []:
                        if block.get("type") == "text" and isinstance(block.get("text"), str):
                            parts.append(block["text"])
                    text_response = "".join(parts) if parts else json.dumps(data, ensure_ascii=False, indent=2)           
                    # 提取 token 用量
                    usage = data.get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
                    logger.info("Claude API 调用成功，获取了用量")
                    return text_response, {"input_tokens": input_tokens, "output_tokens": output_tokens}
                except json.JSONDecodeError as je:
                    logger.error("JSON 解析失败: %s", repr(je))
                    time.sleep(1.5 * (attempt + 1))
                    continue
            else:
                response.raise_for_status()
        except Exception as e:
            last_err = e
            logger.warning("Claude request failed (attempt %d/%d): %s", attempt+1, max_retries, repr(e))
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"All retries failed. Last error: {repr(last_err)}")

def call_gemini(
    model: str, 
    system_prompt: str,  
    user_prompt: str,
    max_retries: int = 1,
    # max_tokens: int = 512
) -> Tuple[str, Dict[str, int]]:
    # 使用与 call_claude 相同的 API Key
    api_key = ""
    base_url = "https://xinghuapi.com"
    
    # 构造 URL (Gemini 标准 REST API 格式)
    url = f"{base_url}/v1beta/models/{model}:generateContent"

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }
    
    # 构造请求体
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}]
            }
        ],
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "generationConfig": {
            "temperature": 1.0,
            #"maxOutputTokens": max_tokens
        }
    }

    last_err = None
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=500)
            
            if response.status_code != 200:
                logger.error("Gemini API returned %s: %s", response.status_code, response.text[:200])
                time.sleep(1.5 * (attempt + 1))
                continue
            
            data = response.json()

            # 解析响应内容
            # Gemini 的响应结构: candidates[0].content.parts[0].text
            candidates = data.get("candidates", [])
            text_parts = []
            thought_parts = []
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                for p in parts:
                    if p.get("thought", False) is True:
                        thought_parts.append(p.get("text", ""))
                    else:
                        text_parts.append(p.get("text", ""))
            else:
                logger.warning("Gemini response has no candidates")

            text_response = "".join(text_parts)
            thought_response = "".join(thought_parts)

            usage_metadata = data.get("usageMetadata", {})
            promptTokenCount = usage_metadata.get("promptTokenCount", 0)
            candidatesTokenCount = usage_metadata.get("candidatesTokenCount", 0)
            totalTokenCount = usage_metadata.get("totalTokenCount", 0)
            thoughtsTokenCount = usage_metadata.get("thoughtsTokenCount", 0)
            return data, thought_response, text_response, {"prompt_tokens": promptTokenCount, "candidates_tokens": candidatesTokenCount, "total_tokens": totalTokenCount, "thoughts_tokens": thoughtsTokenCount}

        except Exception as e:
            last_err = e
            logger.warning("Gemini request failed (attempt %d/%d): %s", attempt+1, max_retries, repr(e))
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"All retries failed. Last error: {repr(last_err)}")

### 嵌入模型调用
def call_siliconflow_embedding(
    inputs,
    model: str = "Qwen/Qwen3-Embedding-8B",
    encoding_format: str = "float",
    dimensions: Optional[int] = 4096,
    timeout: int = 120,
    max_retries: int = 1,
    retry_delay: float = 1.5,
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

def compute_scores(repair_suggestion: str,
                   diff_for_repair_suggestion: str,
                   repair_code: str,
                   diff_for_repair_code: str,
                   display_idx: int) -> Tuple[float, float]:

    # 增加截断逻辑，防止 token 超出 embedding 模型限制 (Qwen-Embedding 通常支持 32k，但为了安全截断到 16k 字符)
    MAX_LEN = 16000
    def truncate(s: str) -> str:
        return s[:MAX_LEN] if s and len(s) > MAX_LEN else s

    inputs = [
        truncate(repair_suggestion), 
        truncate(diff_for_repair_suggestion), 
        truncate(repair_code), 
        truncate(diff_for_repair_code)
    ]
    emb_list = call_siliconflow_embedding(inputs)
    if not (emb_list and isinstance(emb_list, list) and len(emb_list) == 4):
        logger.warning(f"第{display_idx}条数据 compute_scores 无法获取完整嵌入，返回 0.0 分数。")
        return 0.0, 0.0
    emb_repair_suggestion, emb_diff_for_repair_suggestion, emb_repair_code, emb_diff_for_repair_code = emb_list
    score1 = cosine_similarity_np(emb_repair_suggestion, emb_diff_for_repair_suggestion)
    score2 = cosine_similarity_np(emb_repair_code, emb_diff_for_repair_code)
    logger.info("第 %d 条记录计算得分: score1=%.4f, score2=%.4f",display_idx, score1, score2)
    return score1, score2

### 解析claude生成内容 我写的暂时用不上
JUDGEMENT_REGEX = re.compile(r"<judgement>\s*(.*?)\s*</judgement>", re.IGNORECASE | re.DOTALL)
REASONING_REGEX = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.IGNORECASE | re.DOTALL)

def parse_claude_other(response_text: str) -> Tuple[Optional[bool], str]:
    text = (response_text or "").strip()
    if not text:
        return None, ""
    judgement_match = JUDGEMENT_REGEX.search(text)
    reasoning_match = REASONING_REGEX.search(text)

    judgement_raw = (judgement_match.group(1).strip().lower() if judgement_match else "")
    reasoning = (reasoning_match.group(1).strip() if reasoning_match else "")

    if judgement_raw == "yes":
        return True, ""
    if judgement_raw == "no":
        return False, reasoning
    return None, text

### 目前解析claude生成内容 实际使用的
def parse_gemini_verdict(response_text: str) -> Tuple[Optional[bool], str]:
    text = (response_text or "").strip()
    if not text:
        return None, ""
    
    lowered = text.lower()
    yes_match = re.search(r"<\s*yes\s*>", lowered)
    no_match = re.search(r"<\s*no\s*>", lowered)

    # 情况1: 同时匹配到 <yes> 和 <no>
    if yes_match and no_match:
        return False, f"<BOTH_YES_NO>\n{text}"

    # 情况2: 只匹配到 <yes>
    if yes_match:
        # 移除 <yes> 标签，剩下的作为 reason
        reason = re.sub(r"<\s*yes\s*>", "", text, flags=re.IGNORECASE).strip()
        return True, reason

    # 情况3: 只匹配到 <no>
    if no_match:
        # 移除 <no> 标签，剩下的作为 reason
        reason = re.sub(r"<\s*no\s*>", "", text, flags=re.IGNORECASE).strip()
        return False, reason

    # 情况4: 都未匹配到
    return None, text

### 规范化diff、patch相关内容
def extract_diff_add(raw_diff: str, for_rs: bool = False) -> str:
    """
    标志位 for_rs 表示是否用于 repair_suggestion 部分
    如果是false，则给 root cause 用，保留上下文行
    如果是true，则给 repair_suggestion 用，忽略上下文行，并且需要转化成md格式
    """
    diff_text = str(raw_diff or "")
    if not diff_text.strip():
        return ""

    output_lines = []

    for line in diff_text.splitlines():
        if line.startswith("diff ") or line.startswith("index ") or line.startswith("--- a/")  or line.startswith("@@") or line.startswith("-"): 
            continue
        elif line.startswith("+++ b/"):
            file_path = line[6:].strip()
            output_lines.append(file_path)
            if for_rs:
                output_lines.append("### After Change")
        elif line.startswith("+") and not line.startswith("+++"):
            # 给root cause用整行，给rs用去掉+
            output_lines.append(line if not for_rs else line[1:])
        else:
            # 给root cause上下文行，给rs忽略
            if not for_rs:
                output_lines.append(line)

    return "\n".join(output_lines)

def extract_diff_remove(raw_diff: str, for_rs: bool = False) -> str:
    """
    标志位 for_rs 表示是否用于 repair_suggestion 部分
    如果是false，则给 root cause 用，保留上下文行
    如果是true，则给 repair_suggestion 用，忽略上下文行，并且需要转化成md格式
    """
    diff_text = str(raw_diff or "")
    if not diff_text.strip():
        return ""

    output_lines = []

    for line in diff_text.splitlines():
        if line.startswith("diff ") or line.startswith("index ") or line.startswith("+++ b/")  or line.startswith("@@") or line.startswith("+"): 
            continue
        elif line.startswith("--- a/"):
            file_path = line[6:].strip()
            output_lines.append(file_path)
            if for_rs:
                output_lines.append("### Before Change")
        elif line.startswith("-") and not line.startswith("---"):
            output_lines.append(line if not for_rs else line[1:])
        else:
            if not for_rs:
                output_lines.append(line)

    return "\n".join(output_lines)

def normalize_raw_diff_for_repair_code(raw_diff: str) -> str:
    """
    由于raw_diff中包含了diff头信息和文件路径等内容
    这里做一个简单的规范化，去掉diff头信息，只保留文件路径和具体的+/-行
    这样可以和repair_code更好地对齐
    """
    diff_text = str(raw_diff or "")
    if not diff_text.strip():
        return ""

    output_lines = []
    current_file = None

    for line in diff_text.splitlines():
        if line.startswith("diff") or line.startswith("index") or line.startswith("+++ b/"): 
            continue
        elif line.startswith("--- a/"):  # 一个diff块记一下 file path
            current_file = line[6:].strip()
            output_lines.append(current_file)
        else:
            output_lines.append(line) # 保留所有行，这样就和repair_code对齐了,带上+/-

    return "\n".join(output_lines)

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

### 模型生成层，返回解析后的内容
def model_generate(user_prompt: str,
                   system_prompt: str,
                   model: str,
                   temperature: float = 0.2,
                   record_index: Optional[int] = None) -> Dict[str, Any]:
    label = f"第 {record_index} 条记录：" if record_index is not None else ""
    logger.info("====================%s开始访问 %s 模型====================", label, model)
    
    thought = ""
    if "gemini" in model.lower():
        data, thought, response, usage = call_gemini(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_retries=1,
            # max_tokens=512,
        )
    else:
        response, usage = call_claude(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_retries=3,
            max_tokens=512,
        )

    logger.debug("====================%s模型返回内容: %s====================", label, json.dumps(data, ensure_ascii=False, indent=2))
    judgement, reasoning = parse_gemini_verdict(response)
    logger.info("====================%s判定结果: %s====================", label, judgement)
    if reasoning:
        logger.info("====================%s判定理由: %s====================", label, reasoning)
    logger.info("====================%s模型用量: %s====================", label, usage)
    return {
        "judgement": judgement,
        "reasoning": reasoning,
        "gen_text": response,
        "thought": thought,
        "usage": usage,
    }

### io层
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

### 最终结果验证
def has_valid_repair(entry: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(entry, dict):
        return False
    judgement = entry.get("gemini_judgement")
    score1 = entry.get("score1")
    score2 = entry.get("score2")
    index = entry.get("index")
    if not isinstance(judgement, bool):
        INVALID_INDICES.append(index)
        return False
    if not (isinstance(score1, float) and 0.0 <= score1 <= 1.0):
        INVALID_INDICES.append(index)
        return False
    if not (isinstance(score2, float) and 0.0 <= score2 <= 1.0):
        INVALID_INDICES.append(index)
        return False
    if score1 == 0.0 and score2 == 0.0:
        logger.warning(f"第 {index} 条记录 score1 和 score2 均为 0.0，视为无效结果。")
        ZERO_SCORE_INDICES.append(index)
        return False
    return True

### 不再需要
def longest_valid_prefix(results: List[Optional[Dict[str, Any]]]) -> int:
    for idx, entry in enumerate(results):
        if not has_valid_repair(entry):
            return idx
    return len(results)

### 主流程层（并行处理）
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_one_record(global_idx: int, rec: Dict[str, Any]) -> Dict[str, Any]:
    display_idx = global_idx + 1 #global_idx是实际数组索引，从0开始，display_idx是用户看到的记录编号，从1开始，即jsonl文件中的行号
    try:
        logger.info(f"====================开始处理第 {display_idx} 条记录====================")

        ### 预处理 issue 长度
        issue = rec.get("problem_statement", "") or rec.get("issue", "")
        issue_copy = issue
        if len(issue) < 3000:
            # 长度小于3000，不处理
            pass
        elif 3000 <= len(issue) <= 10000:
            # 长度在3000到10000之间，保留超过3000部分的50%
            excess = len(issue) - 3000
            cut_amount = int(excess * 0.5)
            issue = issue[:3000 + cut_amount]
        elif len(issue) > 10000:
            # 长度大于10000，保留超过6000部分的20%
            excess = len(issue) - 6000
            cut_amount = int(excess * 0.2)
            issue = issue[:6000 + cut_amount]
        
        ### 先拿出raw_diff
        raw_diff = rec.get("raw_diff", "")

        ### 考虑claude需要的内容
        root_cause = rec.get("root_cause") or rec.get("Root_Cause", "")
        gemini_diff_remove = extract_diff_remove(raw_diff,False)
        gemini_diff_add = extract_diff_add(raw_diff,False)
        gemini_user = format_gemini_user(issue, root_cause, gemini_diff_remove, gemini_diff_add)
        gemini_res = model_generate(
            gemini_user,
            SYSTEM_PROMPT_GEMINI,
            MODEL_GEMINI,
            temperature=1.0,
            record_index=display_idx,
        )
        gemini_judgement = gemini_res.get("judgement")
        gemini_reasoning = gemini_res.get("reasoning", "")
        gemini_gen_text = gemini_res.get("gen_text", "")
        gemini_thought = gemini_res.get("thought", "")
        gemini_usage = gemini_res.get("usage", {})
        if gemini_judgement is None:
            logger.warning("Record %d Gemini 判定结果为空，默认设为 False。", display_idx)
            gemini_judgement = False
        
        ### 考虑repair_suggestion相关内容
        repair_suggestion = rec.get("repair_suggestion") or rec.get("Repair_Suggestion", "")
        rs_diff_remove = extract_diff_remove(raw_diff,True)
        rs_diff_add = extract_diff_add(raw_diff,True)
        diff_for_repair_suggestion = f"{rs_diff_remove}\n{rs_diff_add}"
        
        ### 考虑repair_code相关内容
        repair_code = rec.get("repair_code") or rec.get("Repair_Code", "")
        repair_code_copy = repair_code
        repair_code = normalize_repair_code(repair_code)
        diff_for_repair_code = normalize_raw_diff_for_repair_code(raw_diff)

        score1, score2 = compute_scores(
            repair_suggestion,
            diff_for_repair_suggestion,
            repair_code,
            diff_for_repair_code,
            display_idx,
        )
              
        ### 接收其他字段
        segments = rec.get("segments", [])
        issue_id = rec.get("issue_id")
        repo = rec.get("repo") 
        
        return {
            "index": display_idx,
            "repo": repo,
            "issue_id": issue_id,
            "pr_num": rec.get("pr_num"),
            "is_change": rec.get("is_change"),
            "base_commit": rec.get("base_commit", ""),
            "gemini_judgement": gemini_judgement,
            "gemini_reasoning": gemini_reasoning,
            "gemini_gen_text": gemini_gen_text,
            "gemini_thought": gemini_thought,
            "root_cause": root_cause,
            "repair_suggestion": repair_suggestion,
            "score1": score1,
            "repair_code": repair_code_copy,
            "score2": score2,
            "problem_statement": issue_copy,
            "segments": segments,
            "raw_diff": raw_diff,
            "gemini_usage": gemini_usage,
        }
    except Exception as e:
            logger.exception("处理单条记录失败 %d: %s", display_idx, repr(e))
            return rec
    
def process_records(records: List[Dict[str, Any]],
                    output_path: str,
                    max_workers: int = 6,
                    base_index: int = 0,
                    global_indices: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """
    并行处理记录，并实时写入有效结果到 output_path。
    返回成功写入的记录数量。
    """
    # 统计本轮有效写入数
    valid_count_in_batch = 0
    results: List[Optional[Dict[str, Any]]] = [None] * len(records)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_meta = {} 
        for local_idx, rec in enumerate(records):
            actual_idx = (global_indices[local_idx]
                          if (global_indices is not None and local_idx < len(global_indices))
                          else (base_index + local_idx)) #actual_idx是全局索引，就像在一个大的数组中处理一样
            future = executor.submit(process_one_record, actual_idx, rec)
            # 记录 local_idx 与 actual_idx，后续用于日志与回填
            future_to_meta[future] = (local_idx, actual_idx)

        for future in as_completed(future_to_meta):
            local_idx, actual_idx = future_to_meta[future]
            display_idx = actual_idx + 1
            try:
                res = future.result()
                results[local_idx] = res

                if res:
                    # 验证有效性（仅用于日志记录，不阻止写入）
                    if not has_valid_repair(res):
                        logger.warning(f"第 {display_idx} 条记录验证未通过（可能缺少字段或分数异常），但依然写入。")
                    else:
                        valid_count_in_batch += 1
                    
                    write_jsonl(output_path, [res], append=True)
                    logger.info(f"第 {display_idx} 条记录已即时写入。")
                else:
                    logger.error(f"第 {display_idx} 条记录返回为空，跳过写入。")

            except Exception as e:
                display_idx = base_index + local_idx + 1 #display_idx是用户看到的记录编号，从1开始，即jsonl文件中的行号
                logger.exception("Unhandled exception in worker for record %d: %s", display_idx, repr(e))
                results[local_idx] = records[local_idx]
    logger.info(f"本批次处理完成，共即时写入 {valid_count_in_batch} 条有效记录（总处理 {len(records)} 条）。")
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

    target_indices = [694]
    target_indices = sorted(set(target_indices))
    logger.info("本次将按 index 精确选取 %d 条记录：%s", len(target_indices), target_indices)
    
    # 在 selected_records 这里改为按 index 选取（index 为 1-based，数组为 0-based）
    # 构造 selected_records 与对应的 global_indices（0-based），保证一一对应
    selected_records: List[Dict[str, Any]] = []
    global_indices: List[int] = []
    missing_indices: List[int] = []
    for idx in target_indices:
        pos = idx - 1
        if 0 <= pos < total_records:
            selected_records.append(records[pos])
            global_indices.append(pos)  # 与 selected_records 对齐
        else:
            missing_indices.append(idx)
    
    if missing_indices:
        logger.warning("以下 index 超出范围或不存在，已跳过：%s", missing_indices)
    # selected_records = records[base_index:end_index]
    if not selected_records:
        logger.error("选定范围内没有记录可处理。")
        sys.exit(2)

    logger.info("main 开始处理jsonl中第 %d 条起共 %d 条记录 → %s", start_index, len(selected_records), args.output)
    processed = process_records(
        selected_records,
        output_path=args.output,
        base_index=0,                 # 此时 base_index 不再使用
        global_indices=global_indices # 由它决定“第 N 条”以及 entry['index']
    )

    logger.warning("score1 和 score2 均为 0.0 的记录索引列表（可能无效结果）：%s", ZERO_SCORE_INDICES)
    logger.warning("验证未通过的记录索引列表（可能缺少字段或分数异常）：%s", INVALID_INDICES)


if __name__ == "__main__":
    main()