"""
数据格式转换脚本：
1. 读取 swebench_lite_for_INL_claude.json
2. 提取 instance_id、problem_statement、golden_patch、status
3. 对 golden_patch 进行预处理，生成 buggy_slice
4. 构建 prompt（Type C 任务）
5. 写入 jsonl 文件
"""
from __future__ import annotations

import json
from typing import Any, Dict, List


def extract_diff_remove_for_buggy_slice(raw_diff: str) -> str:
    """
    从 diff 中提取文件路径、移除内容(带-号)、上下文信息。
    参考 genCot_phrase_2.py 中的 extract_diff_remove 逻辑，
    这里 for_rs=False，保留上下文行。
    """
    diff_text = str(raw_diff or "")
    if not diff_text.strip():
        return ""

    output_lines = []

    for line in diff_text.splitlines():
        # 跳过 diff 头信息、+++ 行、@@ 行、+ 开头的行
        if (line.startswith("diff ") or 
            line.startswith("index ") or 
            line.startswith("+++ b/") or 
            line.startswith("@@") or 
            line.startswith("+")):
            continue
        elif line.startswith("--- a/"):
            # 提取文件路径
            file_path = line[6:].strip()
            output_lines.append(file_path)
        elif line.startswith("-") and not line.startswith("---"):
            # 保留移除行（带 - 号）
            output_lines.append(line)
        else:
            # 保留上下文行
            if line.strip():
                output_lines.append(line)

    return "\n".join(output_lines)


def format_buggy_slice(processed_diff: str) -> str:
    """
    参考 format_segments 的逻辑，将处理后的 diff 格式化为 buggy_slice。
    将每个文件路径及其内容组合成 segment 格式。
    """
    if not processed_diff.strip():
        return ""

    lines = processed_diff.splitlines()
    segments = []
    current_file = None
    current_content = []

    for line in lines:
        # 判断是否是文件路径行（不以 - 或空格开头，且包含文件扩展名特征）
        if not line.startswith("-") and not line.startswith(" ") and "/" in line:
            # 保存之前的 segment
            if current_file is not None and current_content:
                segments.append({
                    "file": current_file,
                    "content": "\n".join(current_content)
                })
            current_file = line
            current_content = []
        else:
            current_content.append(line)

    # 保存最后一个 segment
    if current_file is not None and current_content:
        segments.append({
            "file": current_file,
            "content": "\n".join(current_content)
        })

    # 格式化为类似 format_segments 的输出
    output_lines = []
    for idx, seg in enumerate(segments, start=1):
        output_lines.append(f"--- Segment Group {idx} ---")
        output_lines.append(f"[Suspicious] file: {seg['file']}")
        output_lines.append("```")
        output_lines.append(seg['content'].rstrip())
        output_lines.append("```")
        output_lines.append("")

    return "\n".join(output_lines)


def build_prompt_type_c(problem_statement: str, buggy_slice: str) -> str:
    """
    参考 rlhf.py 中的 prompt 构建逻辑，构建 Type C 任务的 prompt。
    不带 <|im_start|> 符号，末尾增加模板指示。
    """
    # 任务描述 (Type C)
    task_desc = """Generate Natural Language Description: Generate a concise but sufficient, developer-friendly description including both **the issue root cause** and **the proposed repair suggestion**：
    - root cause: the layered explanation of reason for this issue, highlighting exactly why it is incorrect.
    - repair suggestion: actionable and condrete steps in execution order describing how the fix addresses this issue. (Note:1. A couple of small code snippets or inline code may be included, but avoid outputting medium and large blocks of code. 2. Hierarchical and structured format is encouraged. 
    """
    
    # 上下文
    context = f"# GitHub Issue Description:\n{problem_statement}\n\n# Segments Groups:\n{buggy_slice}"
    
    # 格式化输出模板指示
    output_template = """Format your answer strictly as follows Output Template:

# Root Cause:
The Content of Root Cause

# Repair Suggestion:
The Content of Repair Suggestion
"""

    # 构建 prompt（不带 <|im_start|> 符号）
    prompt = f"""
You are a code analysis expert specializing in bug repair.

You will be given:
- A GitHub issue description
- Several segment groups: each contains a suspicious code slice.

Perform the following Task:
{task_desc}

{context}

{output_template}
"""
    
    return prompt


def convert_data(input_path: str, output_path: str, log_path: str = None) -> None:
    """
    主函数：读取 JSON，转换数据，写入 JSONL。
    可选：将 instance_id 和 prompt 写入 log 文件供查看。
    """
    # 读取输入文件
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = []
    log_lines = []
    
    for item in data:
        instance_id = item.get("instance_id", "")
        problem_statement = item.get("problem_statement", "")
        golden_patch = item.get("golden_patch", "")
        status = item.get("status", "")
        
        # 处理 golden_patch 生成 buggy_slice
        processed_diff = extract_diff_remove_for_buggy_slice(golden_patch)
        buggy_slice = format_buggy_slice(processed_diff)
        
        # 构建 prompt
        prompt = build_prompt_type_c(problem_statement, buggy_slice)
        
        # 构建输出对象
        result = {
            "instance_id": instance_id,
            "prompt": prompt,
            "status": status
        }
        results.append(result)
        
        # 构建 log 内容
        log_lines.append("=" * 80)
        log_lines.append(f"Instance ID: {instance_id}")
        log_lines.append("=" * 80)
        log_lines.append(prompt)
        log_lines.append("\n")

    # 写入 JSONL 文件
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # 写入 log 文件
    if log_path:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))
        print(f"日志已写入 {log_path}")
    
    print(f"转换完成！共处理 {len(results)} 条记录，输出到 {output_path}")

if __name__ == "__main__":
    input_file = ""
    output_file = ""
    log_file = ""
    convert_data(input_file, output_file, log_file)