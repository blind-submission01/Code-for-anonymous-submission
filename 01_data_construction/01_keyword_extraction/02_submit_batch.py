#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Optional, Tuple

import requests


API_BASE = "https://api.siliconflow.cn"
UPLOAD_URL = f"{API_BASE}/v1/files"
BATCH_URL = f"{API_BASE}/v1/batches"


def upload_file(file_path: str, api_key: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"purpose": "batch"}
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "application/jsonl")}
        resp = requests.post(UPLOAD_URL, data=data, files=files, headers=headers, timeout=120)
    try:
        payload = resp.json()
    except Exception:
        resp.raise_for_status()
        raise

    if resp.status_code != 200 or not payload.get("status"):
        raise RuntimeError(f"Upload failed: HTTP {resp.status_code}, {json.dumps(payload, ensure_ascii=False)}")

    file_id = payload.get("data", {}).get("id")
    if not file_id:
        raise RuntimeError(f"Upload ok but missing file id: {json.dumps(payload, ensure_ascii=False)}")
    return file_id


def create_batch(file_id: str, api_key: str, endpoint: str = "/v1/chat/completions", window: str = "24h") -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "input_file_id": file_id,
        "endpoint": endpoint,
        "completion_window": window,
    }
    resp = requests.post(BATCH_URL, json=body, headers=headers, timeout=60)
    try:
        payload = resp.json()
    except Exception:
        resp.raise_for_status()
        raise

    if resp.status_code != 200 or not payload.get("status"):
        raise RuntimeError(f"Create batch failed: HTTP {resp.status_code}, {json.dumps(payload, ensure_ascii=False)}")
    return payload


def main():
    parser = argparse.ArgumentParser(description="SiliconFlow 批量推理：上传 JSONL 并创建 batch 任务")
    parser.add_argument(
        "-f", "--file",
        default=os.path.join(os.path.dirname(__file__), "siliconflow_batch.jsonl"),
        help="批量推理输入 JSONL 文件路径"
    )
    parser.add_argument("--api-key", default=os.environ.get("SFB_API_KEY") or os.environ.get("SILICONFLOW_API_KEY"),
                        help="SiliconFlow API Key（建议用环境变量 SFB_API_KEY）")
    parser.add_argument("--endpoint", default="/v1/chat/completions", help="Batch 使用的推理 endpoint")
    parser.add_argument("--window", default="24h", help="completion_window，例如 24h")
    args = parser.parse_args()

    API_key = args.api_key
    if not API_key:
        print("请提供 API Key", file=sys.stderr)
        sys.exit(2)

    if not os.path.isfile(args.file):
        print(f"找不到输入文件: {args.file}", file=sys.stderr)
        sys.exit(2)

    # 可选：简单检查文件大小与行数（不超 1G / 5000 行）
    size = os.path.getsize(args.file)
    with open(args.file, "r", encoding="utf-8") as rf:
        lines = sum(1 for _ in rf)
    if size > 1_000_000_000 or lines > 5000:
        print(f"文件过大或行数超限: size={size} bytes, lines={lines}", file=sys.stderr)
        sys.exit(2)

    print(f"Uploading file: {args.file} ({lines} lines, {size} bytes)")
    file_id = upload_file(args.file, API_key)
    print(f"Uploaded. file_id = {file_id}")

    print("Creating batch...")
    batch_info = create_batch(file_id, API_key, endpoint=args.endpoint, window=args.window)
    print(json.dumps(batch_info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()