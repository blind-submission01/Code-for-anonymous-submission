"""
评测脚本：
1. 读取 prompt 文件，调用模型生成
2. 解析模型输出，提取 root_cause 和 repair_suggestion
3. 与测试数据对齐，输出结果到 jsonl 文件
4. 使用线程池并发处理，最大 8 线程
5. 对解析结果进行校验，不合法则重试
"""
import json
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI


# ========== 模型调用配置 ==========
MODEL_NAME = "sft_v2"
MAX_WORKERS = 70
MAX_PARSE_RETRIES = 3  # 解析失败最大重试次数
PORT = 6006
client = OpenAI(
    base_url=f"",
    api_key="EMPTY"
)

# 线程安全的计数器和锁
progress_lock = threading.Lock()
completed_count = 0


def call_model(prompt: str, max_retries: int = 3, retry_delay: float = 2.0) -> str:
    """调用模型获取响应"""
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    raise RuntimeError(f"模型调用失败: {last_error}")


def remove_think_tags(text: str) -> str:
    """移除 <think>...</think> 标签及其内容"""
    pattern = r"<think>.*?</think>"
    return re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def parse_model_output(raw_output: str) -> Tuple[str, str]:
    """
    解析模型输出，提取 root_cause 和 repair_suggestion
    先移除思维链，再匹配 # Root Cause: 和 # Repair Suggestion: 格式
    """
    text = remove_think_tags(raw_output)
    
    rc_pattern = r"#\s*Root\s*Cause:\s*(.*?)(?=#\s*Repair\s*Suggestion:|$)"
    rs_pattern = r"#\s*Repair\s*Suggestion:\s*(.*?)$"
    
    rc_match = re.search(rc_pattern, text, re.DOTALL | re.IGNORECASE)
    rs_match = re.search(rs_pattern, text, re.DOTALL | re.IGNORECASE)
    
    root_cause = rc_match.group(1).strip() if rc_match else ""
    repair_suggestion = rs_match.group(1).strip() if rs_match else ""
    
    return root_cause, repair_suggestion


def is_valid_output(root_cause: str, repair_suggestion: str, min_length: int = 20) -> bool:
    """
    检查解析结果是否合法
    - root_cause 和 repair_suggestion 都不能为空
    - 长度需要达到最小阈值（避免无意义的短输出）
    """
    if not root_cause or not repair_suggestion:
        return False
    if len(root_cause) < min_length or len(repair_suggestion) < min_length:
        return False
    return True


def process_single_item(item: Dict[str, Any], idx: int, total: int) -> Dict[str, Any]:
    """处理单条数据（线程任务），包含解析校验和重试逻辑"""
    global completed_count
    
    instance_id = item["instance_id"]
    prompt = item["prompt"]
    status = item.get("status", "")
    
    result = {
        "instance_id": instance_id,
        "status": status,
        "raw_output": "",
        "root_cause": "",
        "repair_suggestion": "",
        "error": None,
        "retry_count": 0
    }
    
    try:
        # 重试循环：如果解析结果不合法，重新生成
        for attempt in range(MAX_PARSE_RETRIES):
            raw_output = call_model(prompt)
            root_cause, repair_suggestion = parse_model_output(raw_output)
            
            if is_valid_output(root_cause, repair_suggestion):
                # 解析成功
                result["raw_output"] = raw_output
                result["root_cause"] = root_cause
                result["repair_suggestion"] = repair_suggestion
                result["retry_count"] = attempt
                break
            else:
                # 解析失败，记录并重试
                if attempt < MAX_PARSE_RETRIES - 1:
                    with progress_lock:
                        print(f"  [{instance_id}] 解析结果不合法，重试 ({attempt + 1}/{MAX_PARSE_RETRIES})...")
                else:
                    # 最后一次尝试仍失败，保存当前结果
                    result["raw_output"] = raw_output
                    result["root_cause"] = root_cause
                    result["repair_suggestion"] = repair_suggestion
                    result["retry_count"] = attempt
                    result["error"] = "解析结果不完整（重试后仍失败）"
        
        with progress_lock:
            completed_count += 1
            retry_info = f" (重试{result['retry_count']}次)" if result['retry_count'] > 0 else ""
            if result["error"]:
                print(f"[{completed_count}/{total}] ⚠ {instance_id}{retry_info}: {result['error']}")
            else:
                print(f"[{completed_count}/{total}] ✓ {instance_id}{retry_info}")
            
    except Exception as e:
        result["error"] = str(e)
        with progress_lock:
            completed_count += 1
            print(f"[{completed_count}/{total}] ✗ {instance_id}: {e}")
    
    return result


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """读取 jsonl 文件"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    """写入 jsonl 文件"""
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    global completed_count
    global MODEL_NAME
    
    # 文件路径
    prompt_file = ""
    output_file = f""
    log_file = f""
    
    # 读取数据
    prompt_data = read_jsonl(prompt_file)
    total = len(prompt_data)
    
    print(f"开始处理 {total} 条数据，使用 {MAX_WORKERS} 个线程...")
    print(f"解析失败最大重试次数: {MAX_PARSE_RETRIES}")
    start_time = time.time()
    
    # 使用线程池并发处理
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_item = {
            executor.submit(process_single_item, item, idx, total): item
            for idx, item in enumerate(prompt_data, start=1)
        }
        
        # 收集结果
        for future in as_completed(future_to_item):
            result = future.result()
            results.append(result)
    
    # 按 instance_id 排序以保持顺序
    results.sort(key=lambda x: x["instance_id"])
    
    # 统计
    success_count = sum(1 for r in results if not r["error"])
    retry_count = sum(1 for r in results if r["retry_count"] > 0)
    fail_count = sum(1 for r in results if r["error"])
    
    # 生成最终输出（不含 raw_output）
    final_results = []
    log_lines = []
    
    for r in results:
        # 输出文件
        final_results.append({
            "instance_id": r["instance_id"],
            "root_cause": r["root_cause"],
            "repair_suggestion": r["repair_suggestion"],
            "status": r["status"]
        })
        
        # 日志文件
        log_lines.append("=" * 80)
        log_lines.append(f"Instance ID: {r['instance_id']} (重试次数: {r['retry_count']})")
        log_lines.append("=" * 80)
        log_lines.append(f"Raw Output:\n{r['raw_output']}")
        log_lines.append("-" * 40)
        log_lines.append(f"Parsed Root Cause:\n{r['root_cause']}")
        log_lines.append("-" * 40)
        log_lines.append(f"Parsed Repair Suggestion:\n{r['repair_suggestion']}")
        if r["error"]:
            log_lines.append(f"Error: {r['error']}")
        log_lines.append("\n")
    
    # 保存结果
    write_jsonl(output_file, final_results)
    
    # 保存日志
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    
    elapsed = time.time() - start_time
    print(f"\n{'=' * 50}")
    print(f"完成！共处理 {len(results)} 条记录，耗时 {elapsed:.1f} 秒")
    print(f"  成功: {success_count} 条")
    print(f"  重试后成功: {retry_count} 条")
    print(f"  失败: {fail_count} 条")
    print(f"结果输出到: {output_file}")
    print(f"日志输出到: {log_file}")


if __name__ == "__main__":
    main()