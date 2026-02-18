#!/usr/bin/env python3
"""
Main entry point for the diff retrieval project.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List
import json  # 新增
from src.core.processor import DiffRetrievalProcessor
from src.utils.file_utils import FileUtils
from src.core.config import DEFAULT_WORKERS


def main():
    """Main function to execute file search and bug localization."""
    parser = argparse.ArgumentParser(description="Execute file search and bug localization")
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--instance-id", help="Process specific instance ID (optional)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, 
                       help=f"Number of worker threads (default: {DEFAULT_WORKERS})")
    
    args = parser.parse_args()
    
    # Initialize components
    processor = DiffRetrievalProcessor()
    file_utils = FileUtils()
    
    # Load instances
    instances = file_utils.load_instances_from_jsonl(args.input)
    print(f"Total instances loaded: {len(instances)}")

    # Filter instances if specific instance ID is provided
    if args.instance_id:
        instances = file_utils.filter_instances_by_id(instances, args.instance_id)
        print(f"Filtered to {len(instances)} instances with ID: {args.instance_id}")
    
    # Process instances with multi-threading
    # 并发处理，结果固定索引存放，结束后一次性写出
    results: List[Dict] = [None] * len(instances)
    completed_count = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(processor.process_single_instance, instance): idx
            for idx, instance in enumerate(instances, start=0)  # 1-based 索引，便于对齐日志
        }
        
        # Process completed tasks
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            instance = instances[idx]
            try:
                result = future.result()
                # Convert result to dictionary for JSON serialization
                result_dict = {
                    "issue_id": result.issue_id,
                    "repo": result.repo,
                    "base_commit": result.base_commit,
                    "raw_diff": result.raw_diff,
                    "url": result.url,
                    "title": result.title,
                    "problem_statement": result.problem_statement,
                    "file_search_response": result.file_search_result.response,
                    #"suspicious_files": result.file_search_result.suspicious_files,
                    "related_files": result.file_search_result.related_files,
                    "bug_localization_response": result.bug_localization_result.response,
                    "segments": result.bug_localization_result.segments,
                    "if_ok": result.if_ok,
                    "usage_tokens": result.usage_tokens
                }
                results[idx] = result_dict
                completed_count += 1
                
                print(f"Completed {completed_count}/{len(instances)} instances (idx={idx + 1})")
                
            except Exception as e:
                print(f"Failed to process instance {instance.get('issue_id', 'unknown')}: {e}")
                continue

        # 一次性写入 JSONL
    with open(args.output, "w", encoding="utf-8") as f:
        for obj in results:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"\nProcessing completed. Results saved to {args.output}")

    # 统计
    total_usage = sum(int(r.get("usage_tokens", 0) or 0) for r in results if isinstance(r, dict))
    ok_count = sum(1 for r in results if isinstance(r, dict) and r.get("if_ok"))
    failed_indices = [i for i, r in enumerate(results) if not (isinstance(r, dict) and r.get("if_ok"))]  # 0-based

    print(f"\nResults saved to {args.output}")
    print(f"Total usage_tokens: {total_usage}")
    print(f"if_ok True count: {ok_count} / {len(results)}")
    print(f"if_ok False count: {len(failed_indices)}")
    print(f"if_ok False indices (0-based): {failed_indices}")

if __name__ == "__main__":
    main()
