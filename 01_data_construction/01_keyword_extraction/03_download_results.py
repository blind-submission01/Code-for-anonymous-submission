#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from typing import Optional, Tuple, Dict, Any

import requests

API_BASE = "https://api.siliconflow.cn"
FILES_URL = f"{API_BASE}/v1/files"
BATCH_URL = f"{API_BASE}/v1/batches"

TERMINAL_STATUSES = {
    "completed",
    "failed",
    "cancelled",
    "expired",
}

def get_headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}

def get_batch(batch_id: str, api_key: str) -> Dict[str, Any]:
    url = f"{BATCH_URL}/{batch_id}"
    resp = requests.get(url, headers=get_headers(api_key), timeout=30)
    resp.raise_for_status()
    return resp.json()

def wait_batch(batch_id: str, api_key: str, interval: int = 10, timeout: int = 3600) -> Dict[str, Any]:
    start = time.time()
    while True:
        info = get_batch(batch_id, api_key)
        status = info.get("status")
        print(f"[batch {batch_id}] status={status}", flush=True)
        if status in TERMINAL_STATUSES:
            return info
        if time.time() - start > timeout:
            raise TimeoutError(f"Wait batch timeout after {timeout}s, last status={status}")
        time.sleep(interval)

def download_file_content(file_id: str, api_key: str, out_path: Optional[str] = None) -> str:
    # 如果是完整 URL，直接下载
    if isinstance(file_id, str) and (file_id.startswith("http://") or file_id.startswith("https://")):
        resp = requests.get(file_id, stream=True, timeout=120)
        resp.raise_for_status()
        if out_path:
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return out_path
        return resp.text

    # 否则按 file_id 调 SiliconFlow /content 接口
    url = f"{FILES_URL}/{file_id}/content"
    resp = requests.get(url, headers=get_headers(api_key), stream=True, timeout=120)
    resp.raise_for_status()
    if out_path:
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return out_path
    return resp.text

def parse_output_preview(jsonl_path: str, max_lines: int = 3) -> None:
    # 简单预览：打印每行 custom_id 与 choices[0].message.content（若存在）
    print(f"\nPreview first {max_lines} lines from {jsonl_path}:")
    shown = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                print(f"- raw: {line[:200]}...")
                shown += 1
                if shown >= max_lines:
                    break
                continue

            custom_id = obj.get("custom_id") or obj.get("id") or "unknown"
            # 兼容常见输出结构：response.body.choices[0].message.content
            content = None
            body = (obj.get("response") or {}).get("body") if isinstance(obj.get("response"), dict) else None
            if isinstance(body, dict):
                choices = body.get("choices") or []
                if choices and isinstance(choices[0], dict):
                    msg = choices[0].get("message") or {}
                    content = msg.get("content")

            print(f"- custom_id={custom_id}")
            if content:
                print(f"  content: {content[:200]}...")
            else:
                # 没匹配到标准结构就打印一小段原始行
                print(f"  raw: {json.dumps(obj, ensure_ascii=False)[:200]}...")
            shown += 1
            if shown >= max_lines:
                break

def main():
    parser = argparse.ArgumentParser(description="获取 SiliconFlow 批量推理的生成结果（轮询 batch 并下载输出文件）")
    parser.add_argument("--api-key", default=os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("SFB_API_KEY"),
                        help="SiliconFlow API Key（建议用环境变量传入）")
    parser.add_argument("--batch-id", required=True, help="目标批处理任务的 batch_id")
    parser.add_argument("--interval", type=int, default=10, help="轮询间隔秒数")
    parser.add_argument("--timeout", type=int, default=3600, help="轮询超时秒数")
    parser.add_argument("--out", default=None, help="输出文件保存路径（默认：自动命名为 batch_<id>_output.jsonl）")
    parser.add_argument("--no-wait", action="store_true", help="不轮询，直接查询一次 batch 状态（若已完成则下载）")
    parser.add_argument("--preview", action="store_true", help="下载后打印前几行内容预览")
    args = parser.parse_args()

    API_key = args.api_key
    if not API_key:
        print("请提供 API Key", file=sys.stderr)
        sys.exit(2)

    info = get_batch(args.batch_id, API_key) if args.no_wait else wait_batch(
        args.batch_id, API_key, interval=args.interval, timeout=args.timeout
    )

    status = info.get("status")
    print(json.dumps(info, ensure_ascii=False, indent=2))

    if status != "completed":
        print(f"批处理未完成（status={status}），无法下载输出。", file=sys.stderr)
        sys.exit(1)

    output_file_id = info.get("output_file_id")
    error_file_id = info.get("error_file_id")
    if not output_file_id:
        print("没有 output_file_id。请检查错误文件或重试。", file=sys.stderr)
        if error_file_id:
            print(f"error_file_id: {error_file_id}")
        sys.exit(1)

    out_path = args.out or os.path.join(os.getcwd(), f"batch_{args.batch_id}_output.jsonl")
    print(f"Downloading output file: {output_file_id} -> {out_path}")
    download_file_content(output_file_id, API_key, out_path=out_path)
    print(f"Saved to {out_path}")

    if error_file_id:
        err_path = os.path.join(os.getcwd(), f"batch_{args.batch_id}_error.jsonl")
        try:
            print(f"Downloading error file: {error_file_id} -> {err_path}")
            download_file_content(error_file_id, API_key, out_path=err_path)
            print(f"Saved error file to {err_path}")
        except Exception as e:
            print(f"下载错误文件失败：{e}", file=sys.stderr)

    if args.preview:
        parse_output_preview(out_path, max_lines=3)

if __name__ == "__main__":
    main()