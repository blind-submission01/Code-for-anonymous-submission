#!/usr/bin/env python3
import argparse
import json
import logging
import os
import threading
import time
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple
from datetime import datetime
import requests

GITHUB_API = "https://api.github.com"

logger = logging.getLogger("flatten_prs")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 简单缓存，避免重复拉取 PR
PR_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}

class CoreRateLimitManager:
    def __init__(self):
        self.remaining = 5000
        self._lock = threading.Lock()

    def update_from_response(self, response: requests.Response):
        with self._lock:
            rem = response.headers.get("X-RateLimit-Remaining")
            if rem is not None:
                try:
                    self.remaining = int(rem)
                    logger.info(f"update_from_response Rate limit remaining: {self.remaining}")
                except Exception:
                    pass

    def get_delay(self) -> float:
        with self._lock:
            # 基础延迟 + 抖动，尽量温和
            if self.remaining > 10:
                base = 0.1 #感觉可以改成0.08
            else:
                base = 3200
        return base

core_rlm = CoreRateLimitManager()

def gh_headers(token: str, accept: Optional[str] = None) -> Dict[str, str]:
    h = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "apr-issue-miner/flatten/1.0",
    }
    if accept:
        h["Accept"] = accept
    return h

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def write_jsonl_line(path: str, obj: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def has_code_example(text: str) -> bool:
    if not text:
        return False
    if "```" in text:
        return True
    if re.search(r"\n\s{4,}\S", text):
        return True
    return False

def has_error_message(text: str) -> bool:
    if not text:
        return False
    return bool(
        re.search(r"(Exception|Error|Traceback|TypeError|ValueError|AssertionError|IndexError|KeyError)", text, re.I)
        or re.search(r"bug|error|issue|problem|bug report|error report|issue report", text, re.I)
    )

def fetch_repo(owner: str, repo: str, token: str) -> Optional[Dict[str, Any]]:
    logger.info(f"Fetching repo {owner}/{repo}")
    url = f"{GITHUB_API}/repos/{owner}/{repo}"
    time.sleep(core_rlm.get_delay())
    resp = requests.get(url, headers=gh_headers(token), timeout=30)
    core_rlm.update_from_response(resp)
    if resp.status_code != 200:
        return None
    return resp.json()

def fetch_issue(owner: str, repo: str, issue_number: int, token: str) -> Optional[Dict[str, Any]]:
    logger.info(f"Fetching issue {owner}/{repo}#{issue_number}")
    url = f"{GITHUB_API}/repos/{owner}/{repo}/issues/{issue_number}"
    time.sleep(core_rlm.get_delay())
    resp = requests.get(url, headers=gh_headers(token), timeout=30)
    core_rlm.update_from_response(resp)
    if resp.status_code != 200:
        return None
    return resp.json()

def fetch_issue_timeline_event(owner: str, repo: str, issue_number: int, token: str) -> List[Dict[str, Any]]:
    logger.info(f"Fetching timeline for issue {owner}/{repo}#{issue_number}")
    url = f"{GITHUB_API}/repos/{owner}/{repo}/issues/{issue_number}/timeline"
    headers = gh_headers(token, accept="application/vnd.github+json, application/vnd.github.mockingbird-preview+json")
    events: List[Dict[str, Any]] = []
    page = 1
    max_pages = 10  # 避免无限翻页

    while page <= max_pages:
        time.sleep(core_rlm.get_delay())
        resp = requests.get(url, headers=headers, params={"per_page": 100, "page": page}, timeout=30)
        core_rlm.update_from_response(resp)

        if resp.status_code != 200:
            logger.info(f"Timeline fetch failed {resp.status_code} on page {page}")
            break

        batch = resp.json() or []
        if not batch:
            break

        events.extend(batch)
        if len(batch) < 100:
            break
        page += 1

    return events

# 新增：从时间线中抽取关联 PR 编号
def extract_linked_pr_numbers_from_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    linked_prs: List[Dict[str, Any]] = []
    seen = set()
    for ev in events:
        if ev.get("event") != "cross-referenced":
            continue
        issue = (ev.get("source") or {}).get("issue") or {}
        if not issue.get("pull_request"):
            continue
        num = issue.get("number")
        if not isinstance(num, int) or num in seen:
            continue
        seen.add(num)
        linked_prs.append(
            {
                "number": num,
                "body": issue.get("body") or "",
                "merged_at": (issue.get("pull_request") or {}).get("merged_at"),
            }
        )
    return linked_prs

def parse_github_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except ValueError:
        return None

def select_relevant_prs(
    entries: List[Dict[str, Any]],
    issue_id: Any,
    closed_at: Optional[str]
) -> List[Dict[str, Any]]:
    if not entries:
        return []
    if closed_at:
        for entry in entries:
            if entry.get("merged_at") == closed_at:
                return [entry]

    pattern = None
    if issue_id is not None:
        pattern = re.compile(
            rf"(?i)\b(close[sd]?|fix(?:e[sd])?|resolve[sd]?)\b[^\S\r\n]*#\s*{re.escape(str(issue_id))}\b"
        )

    if pattern:
        matched = [entry for entry in entries if pattern.search(entry.get("body") or "")]
        if matched:
            return matched

    closed_dt = parse_github_ts(closed_at)
    if closed_dt:
        before_closed = []
        for entry in entries:
            merged_dt = parse_github_ts(entry.get("merged_at"))
            if merged_dt and merged_dt <= closed_dt:
                before_closed.append((merged_dt, entry))
        if before_closed:
            before_closed.sort(key=lambda pair: pair[0], reverse=True)
            return [before_closed[0][1]]

    return [entries[0]]

def fetch_pr(owner: str, repo: str, pr_number: int, token: str) -> Optional[Dict[str, Any]]:
    logger.info(f"Fetching PR {owner}/{repo}#{pr_number}")
    cache_key = f"{owner}/{repo}/pulls/{pr_number}".lower()
    if cache_key in PR_CACHE:
        return PR_CACHE[cache_key]

    url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}"
    backoff = 3
    for _ in range(3):
        time.sleep(core_rlm.get_delay())
        resp = requests.get(url, headers=gh_headers(token), timeout=30)
        core_rlm.update_from_response(resp)

        if resp.status_code == 403 and 'secondary rate limit' in (resp.text or '').lower():
            retry_after = int(resp.headers.get('Retry-After', backoff))
            logger.warning(f"PR secondary rate limit, sleep {retry_after}s")
            time.sleep(retry_after)
            backoff = min(backoff * 2, 60)
            continue

        if resp.status_code != 200:
            PR_CACHE[cache_key] = None
            return None

        data = resp.json()
        PR_CACHE[cache_key] = data
        return data

    PR_CACHE[cache_key] = None
    return None

def compare_shas(owner: str, repo: str, base_sha: str, head_sha: str, token: str, diff: bool) -> Optional[Dict[str, Any] | str]:
    logger.info(f"Comparing SHAs {owner}/{repo}: {base_sha}...{head_sha} (diff={diff})")
    url = f"{GITHUB_API}/repos/{owner}/{repo}/compare/{base_sha}...{head_sha}"
    accept = "application/vnd.github.v3.diff" if diff else "application/vnd.github+json"
    backoff = 3
    for _ in range(3):
        time.sleep(core_rlm.get_delay())
        resp = requests.get(url, headers=gh_headers(token, accept), timeout=30)
        core_rlm.update_from_response(resp)

        if resp.status_code == 403 and 'secondary rate limit' in (resp.text or '').lower():
            retry_after = int(resp.headers.get('Retry-After', backoff))
            logger.warning(f"Compare secondary rate limit, sleep {retry_after}s")
            time.sleep(retry_after)
            backoff = min(backoff * 2, 60)
            continue

        if resp.status_code != 200:
            return None

        return resp.text if diff else resp.json()
    return None

def fetch_pr_files(owner: str, repo: str, pr_number: int, token: str) -> List[Dict[str, Any]]:
    logger.info(f"Fetching PR files {owner}/{repo}#{pr_number}/files")
    url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}/files"
    files: List[Dict[str, Any]] = []
    page = 1
    max_pages = 5
    while page <= max_pages:
        time.sleep(core_rlm.get_delay())
        resp = requests.get(url, headers=gh_headers(token), params={"per_page": 100, "page": page}, timeout=30)
        core_rlm.update_from_response(resp)
        if resp.status_code != 200:
            break
        batch = resp.json() or []
        if not batch:
            break
        files.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return files

def is_docs_only(files: List[Dict[str, Any]]) -> bool:
    if not files:
        return True
    code_like = 0
    for f in files:
        filename = (f.get("filename") or "").lower()
        if filename.endswith(('.md', '.rst', '.txt', '.adoc', '.docx', '.doc', '.pdf')):
            continue
        if filename.startswith('docs/') or '/docs/' in filename:
            continue
        code_like += 1
        if code_like > 0:
            return False
    return True

def if_files_ok(files: List[Dict[str, Any]]) -> bool:
    """
    遍历 PR 文件列表，统计满足：
      1) filename 以 .py 结尾
      2) additions >= 5
      3) deletions >= 3
    的文件数量，数量 >= 2 返回 True，否则 False。
    """
    if not files:
        return False
    hit = 0
    for f in files:
        name = (f.get("filename") or "").lower()
        if not name.endswith(".py"):
            continue
        additions = int(f.get("additions") or 0)
        deletions = int(f.get("deletions") or 0)
        if additions >= 5 and deletions >= 3:
            hit += 1
            if hit >= 2:
                return True
    return False

def parse_owner_repo(repo_full: str) -> Tuple[str, str]:
    owner, repo = (repo_full or "").split("/", 1)
    return owner, repo

def main():
    parser = argparse.ArgumentParser(description="Flatten PR info from issues jsonl")
    parser.add_argument("--input", required=True, help="Path to 6.jsonl")
    parser.add_argument("--output", required=True, help="Path to output jsonl")
    parser.add_argument("--token", type=str, default="", help="GitHub API token")
    args = parser.parse_args()

    token = args.token
    if not token:
        raise SystemExit("Please export GITHUB_TOKEN (or GH_TOKEN)")

    total_written = 0
    total_prs_seen = 0
    skipped_base_mismatch = 0

    inputs = list(read_jsonl(args.input))

    for idx, rec in enumerate(inputs):
        try:
            start_time = time.time()
            logger.info(f"====================Processing record #{idx+1}/{len(inputs)}====================")
            repo_full = (rec.get("repo") or "").strip()
            if not repo_full or "/" not in repo_full:
                continue
            owner, repo = parse_owner_repo(repo_full)
            issue_id = rec.get("issue_id")
            issue_url = rec.get("url", "")
            title = rec.get("title", "")
            body = rec.get("body", "")

            created_at = rec.get("created_at", "")
            instance_time = rec.get("instance_time", "")

            # 1) 仓库质量与元信息
            repo_info = fetch_repo(owner, repo, token)
            archived = bool(repo_info.get("archived")) if repo_info else False
            star = int(repo_info.get("stargazers_count") or 0) if repo_info else 0

            # 2) issue 文本质量
            body_length = len(body or "")
            body_has_code = has_code_example(body)
            body_has_err = has_error_message(body)      

            # 获取最新 issue 信息（含 closed_at）
            issue_detail = fetch_issue(owner, repo, int(issue_id), token)
            closed_at = (issue_detail or {}).get("closed_at")

            # 新逻辑：通过时间线获取 linked_prs
            try:
                logger.info(f"开始处理 {owner}/{repo}#{issue_id} fetch_issue_timeline_event")
                events = fetch_issue_timeline_event(owner, repo, int(issue_id), token)
                linked_entries = extract_linked_pr_numbers_from_events(events)
                selected_entries = select_relevant_prs(linked_entries, issue_id, closed_at)
                if not selected_entries:
                    logger.info(f"没有找到可用的关联 PR，处理失败 {owner}/{repo}#{issue_id}")
                    continue
                logger.info(f"查看:{selected_entries}")
            except Exception as e:
                logger.info(f"Skip record timeline {owner}/{repo}#{issue_id}: {e}")
                continue

            for pr_entry in selected_entries:
                total_prs_seen += 1
                pr_num = pr_entry.get("number")
                try:
                    logger.info(f"开始处理 {owner}/{repo}#{pr_num} fetch_pr")
                    pr_info = fetch_pr(owner, repo, pr_num, token)
                    if not pr_info or not pr_info.get("merged"):
                        logger.info(f"没有merged_pr，处理失败 {owner}/{repo}#{pr_num}: fetch_pr")
                        continue

                    logger.info(f"开始处理 {owner}/{repo}#{pr_num} 三条sha")
                    base_sha = pr_info.get("base", {}).get("sha", "")
                    head_sha = pr_info.get("head", {}).get("sha", "")
                    merged_sha = pr_info.get("merge_commit_sha", "")
                    if not base_sha or not head_sha or not merged_sha:
                        logger.info(f"处理失败 {owner}/{repo}#{pr_num}: 三条sha")
                        continue

                    # 先拿 JSON，获取 true_base_sha
                    logger.info(f"开始处理 {owner}/{repo}#{pr_num} compare_shas")
                    compare_json = compare_shas(owner, repo, base_sha, head_sha, token, diff=False)
                    if not compare_json or not compare_json.get("merge_base_commit"):
                        logger.info(f"处理失败 {owner}/{repo}#{pr_num}: compare_shas")
                        continue

                    logger.info(f"开始处理 {owner}/{repo}#{pr_num} true_base_sha")
                    true_base_sha = (compare_json["merge_base_commit"] or {}).get("sha", "")
                    if not true_base_sha:
                        logger.info(f"处理失败 {owner}/{repo}#{pr_num}: true_base_sha")
                        continue

                    # 再拿 diff 文本
                    logger.info(f"开始处理 {owner}/{repo}#{pr_num} diff_content")
                    diff_content = compare_shas(owner, repo, base_sha, head_sha, token, diff=True)
                    if not diff_content:
                        logger.info(f"处理失败 {owner}/{repo}#{pr_num}: diff_content")
                        continue
                    diff_length = len(diff_content or "")

                    # 变更文件与“是否仅文档改动”
                    files = fetch_pr_files(owner, repo, int(pr_num), token)
                    has_code_changes = not is_docs_only(files)
                    validate_ok = if_files_ok(files)

                    flat_obj = {
                        "repo": f"{owner}/{repo}",
                        "url": issue_url,
                        "issue_id": issue_id,
                        "body_length": body_length,
                        "has_code_example": body_has_code,
                        "has_error_message": body_has_err,
                        "star": star,
                        "archived": archived,   
                        "merged_pr_num": str(pr_num),
                        "has_code_changes": has_code_changes,
                        "if_file_ok": validate_ok,
                        "true_base_sha": true_base_sha,
                        "created_at": created_at,
                        "instance_time": instance_time,
                        "title": title,
                        "body": body,
                        "base_sha": base_sha,
                        "head_sha": head_sha,
                        "merged_sha": merged_sha,
                        "diff_content": diff_content,
                        "diff_length": diff_length
                    }
                    write_jsonl_line(args.output, flat_obj)
                    total_written += 1
                    logger.info(f"处理成功 {owner}/{repo}#{pr_num}")
                except Exception as e:
                    logger.info(f"Skip PR {owner}/{repo}#{pr_num}: {e}")
                    continue
                end_time = time.time()
                logger.info(f"====================Finished record #{idx+1} in {end_time - start_time:.2f}s, total_written={total_written}, total_prs_seen={total_prs_seen}, skipped_base_mismatch={skipped_base_mismatch}====================")
        except Exception as e:
            logger.info(f"Skip record: {e}")
            continue

    logger.info(f"Done. Wrote {total_written} lines to {args.output}")

if __name__ == "__main__":
    main()