#!/usr/bin/env python3
import argparse
from datetime import datetime
import json
import logging
import os
import random
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests


GITHUB_API = "https://api.github.com"
SEARCH_ENDPOINT = f"{GITHUB_API}/search/issues"

logger = logging.getLogger("mine_similar_github_issues")

# 全局缓存
PR_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}

# Rate limit管理（线程安全）
class SearchRateLimitManager:
    def __init__(self):
        self.remaining = 30  # search默认值
        self._lock = threading.Lock()  # 线程锁
    
    def update_from_response(self, response: requests.Response, name: str):
        """从响应头更新rate limit信息（线程安全）"""
        with self._lock:
            if 'X-RateLimit-Remaining' in response.headers:
                self.remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
                logger.info(f"{name} update_from_response remaining: {self.remaining}")
    
    def get_delay(self, name: str) -> float:
        """获取智能延迟时间（线程安全）"""
        with self._lock:
            if self.remaining > 10:
                return 2.6  # 多线程时缩短延迟
            elif self.remaining > 2:
                return 5.6
            else:
                logger.warning(f"{name} Low remaining rate limit: {self.remaining}, delay 30s")
                return 30
    
    def should_continue(self, name: str) -> bool:
        """检查是否应该继续API调用（线程安全）"""
        with self._lock:
            rem = self.remaining
        logger.info(f"{name} should_continue remaining={rem}")
        if rem <= 0:
            logger.warning(f"{name} 配额为0，暂停60秒后继续")
            time.sleep(60)
            return True
        return True

class CoreRateLimitManager(SearchRateLimitManager):
    def __init__(self):
            self.remaining = 5000  # search默认值
            self._lock = threading.Lock()  # 线程锁
        
    def update_from_response(self, response: requests.Response):
        """从响应头更新rate limit信息（线程安全）"""
        with self._lock:
            if 'X-RateLimit-Remaining' in response.headers:
                self.remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
                logger.info(f"update_from_response: remaining: {self.remaining}")
    
    def get_delay(self) -> float:
        """获取智能延迟时间（线程安全）"""
        with self._lock:
            if self.remaining > 1000:
                return 1  # 多线程时缩短延迟
            elif self.remaining > 100:
                return 2
            elif self.remaining > 10:
                return 10
            else:
                return 60
    
    def should_continue(self) -> bool:
        """检查是否应该继续API调用（线程安全）"""
        with self._lock:
            logger.info(f"should_continue: remaining={self.remaining}")
            return self.remaining > 0  # 多线程时保留更多配额

search_rate_limit_manager = SearchRateLimitManager()
core_rate_limit_manager = CoreRateLimitManager()

QUALITY_FILTERS = {
    # Repository quality
    # "min_stars": 100,
    # "is_archived": False,
    # "language": "Python",
    # Issue quality
    #"state": "closed",
    # Solution quality – will verify via PRs
    #"linked_pr_merged": True,
    # Content quality
    "has_code_example": True,
    "min_body_length": 100,
    "max_body_length": 5000,
    "has_error_message": True,
}

# 读取jsonl文件获取生成器
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

# 传入迭代对象增量写
def write_jsonl(records: Iterable[Dict[str, Any]], path: str, mode: str = "a") -> None:
    with open(path, mode, encoding="utf-8") as f:
        for rec in records:
            logger.info(f"Writing record: {rec}")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# 从jsonl路径得到生成器拿去迭代，去重获得(repo, issue_id)集合
def load_present_keys(path: Optional[str]) -> Set[Tuple[str, int]]:
    present: Set[Tuple[str, int]] = set()
    if not path or not os.path.exists(path):
        return present
    for obj in read_jsonl(path):
        try:
            repo = str(obj.get("repo") or "").strip().lower()
            issue_id = obj.get("issue_id")
            if repo and isinstance(issue_id, int):
                present.add((repo, issue_id))
        except Exception:
            continue
    return present

# 传入关键字字典，生成多组关键词组合
def generate_keyword_combinations(key_words: Dict[str, List[str]], max_combinations: int = 10) -> List[List[str]]:
    """
    生成尽可能多的关键词组合，用于提高搜索覆盖率
    """
    pb_list = list(key_words.get("problem_behaviors") or [])
    error_patterns = list(key_words.get("error_patterns") or [])
    code_elements = list(key_words.get("code_elements") or [])
    domain_context = list(key_words.get("domain_context") or [])
    
    if not pb_list:
        return []
    
    combinations = []
    used_signatures = set()
    
    # 策略1：主问题 + 2个其他类别的关键词
    for pb in pb_list[:3]:  # 限制pb数量
        for error in error_patterns[:4]:  # 限制error数量
            for code in code_elements[:3]:  # 限制code数量
                combo = [pb, error, code]
                sig = "|".join(sorted([k.lower() for k in combo if k]))
                if sig not in used_signatures:
                    used_signatures.add(sig)
                    combinations.append(combo)
                    if len(combinations) >= max_combinations:
                        return combinations
    
    # 策略2：主问题 + 领域上下文 + 错误模式
    for pb in pb_list[:3]:
        for domain in domain_context[:4]:
            for error in error_patterns[:3]:
                combo = [pb, domain, error]
                sig = "|".join(sorted([k.lower() for k in combo if k]))
                if sig not in used_signatures:
                    used_signatures.add(sig)
                    combinations.append(combo)
                    if len(combinations) >= max_combinations:
                        return combinations
    
    # 策略3：主问题 + 代码元素 + 领域上下文
    for pb in pb_list[:3]:
        for code in code_elements[:4]:
            for domain in domain_context[:3]:
                combo = [pb, code, domain]
                sig = "|".join(sorted([k.lower() for k in combo if k]))
                if sig not in used_signatures:
                    used_signatures.add(sig)
                    combinations.append(combo)
                    if len(combinations) >= max_combinations:
                        return combinations
    
    # 策略4：双关键词组合
    all_others = error_patterns + code_elements + domain_context
    for pb in pb_list[:4]:
        for other in all_others[:8]:
            combo = [pb, other]
            sig = "|".join(sorted([k.lower() for k in combo if k]))
            if sig not in used_signatures:
                used_signatures.add(sig)
                combinations.append(combo)
                if len(combinations) >= max_combinations:
                    return combinations
    
    # 策略5：单个主关键词（最后备选）
    for pb in pb_list[:5]:
        combo = [pb]
        sig = pb.lower()
        if sig not in used_signatures:
            used_signatures.add(sig)
            combinations.append(combo)
            if len(combinations) >= max_combinations:
                return combinations
    
    return combinations

# 由原始关键词构造具体单个查询，发生在生成多种组合之后
def build_query(kw_triple: List[str], use_or: bool = False, until_date: Optional[str] = None) -> str:
    """
    构建搜索查询
    Args:
        kw_triple: 关键词列表
        use_or: 如果为True使用OR连接，否则使用AND连接
        until_date: 截止日期（YYYY-MM-DD），根据此日期向前5年构造 created: 起止区间
    """
    # 计算 created 过滤器：until_date 向前 5 年到 until_date 当天
    def _created_filter(date_str: Optional[str]) -> str:
        if not date_str:
            return "created:>2020-01-01"
        try:
            end = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
            # 安全减 5 年（处理闰年 2/29）
            try:
                start = end.replace(year=end.year - 5)
            except ValueError:
                # 若为 2/29，则回退到 2/28
                start = end.replace(year=end.year - 5, month=2, day=28)
            return f"created:{start.isoformat()}..{end.isoformat()}"
        except Exception:
            return "created:>2020-01-01"

    created_clause = _created_filter(until_date)

    quoted = [f'{k.strip()}' for k in kw_triple if k and k.strip()]
    base = f"is:issue is:closed linked:pr language:Python"
    # base = f"is:issue is:closed linked:pr language:Python {created_clause} stars:>=1 archived:false"
    if not quoted:
        return base
    
    connector = " OR " if use_or else " AND "
    joined = connector.join(quoted)
    return f"{base} {joined}"

# 构造请求头，需要diff的时候可以改
def gh_headers(token: str, accept: Optional[str] = None) -> Dict[str, str]:
    h = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "apr-issue-miner/1.0",
    }
    if accept:
        h["Accept"] = accept
    return h

# 通过构造的combination中的单个query，搜索issues返回，每次最多6项
def search_issues(q: str, token: str, per_page: int = 6) -> List[Dict[str, Any]]:
    if not search_rate_limit_manager.should_continue(name = "search_issues"):
        logger.warning(f"search_issues Rate limit too low ({search_rate_limit_manager.remaining}), skipping search")
        return []
    
    params = {"q": q, "per_page": per_page, "sort": "comments", "order": "desc"}

    for attempt in range(3):  # Retry up to 3 times
        delay = search_rate_limit_manager.get_delay(name = "search_issues")
        time.sleep(delay)
        resp = requests.get(SEARCH_ENDPOINT, headers=gh_headers(token), params=params, timeout=30)
        # 更新rate limit信息
        search_rate_limit_manager.update_from_response(resp, name = "search_issues")
        logger.info(f"search_issues remaining: {search_rate_limit_manager.remaining}")
        
        if resp.status_code == 403 and 'secondary rate limit' in resp.text.lower():
            retry_after = int(resp.headers.get('Retry-After', 50))
            logger.warning(f"search_issues Hit secondary rate limit, waiting {retry_after} seconds")
            time.sleep(retry_after)
            continue
        if resp.status_code != 200:
            logger.info(f"search_issues failed {resp.status_code}: {resp.text[:200]}")
            return []
        
        data = resp.json() or {}
        items = list(data.get("items") or [])
        logger.info(f"search_issues returned {len(items)} items for query: {q}")
        return items
    
    logger.error(f"Failed to search_issues after {attempt + 1} attempts: {q}")
    return []

# 通过原路径获取owner/repo
def parse_repo_full_name(url_or_full_name: str) -> Tuple[str, str]:
    if "/repos/" in url_or_full_name:
        owner_repo = url_or_full_name.rsplit("/repos/", 1)[-1]
    else:
        owner_repo = url_or_full_name
    owner, repo = owner_repo.split("/", 1)
    return owner, repo

# ？ 本意通过{GITHUB_API}/repos/{owner}/{repo}/issues/{issue_number}/timeline查询issue时间线，返回事件列表
def fetch_issue_timeline_event(owner: str, repo: str, issue_number: int, token: str) -> List[Dict[str, Any]]:
    logger.info(f"Fetching timeline for issue {owner}/{repo}#{issue_number}")
    
    url = f"{GITHUB_API}/repos/{owner}/{repo}/issues/{issue_number}/timeline"
    headers = gh_headers(token, accept="application/vnd.github+json, application/vnd.github.mockingbird-preview+json")
    events: List[Dict[str, Any]] = []
    page = 1
    max_pages = 5  # 限制最大页数避免无限循环
    
    while page <= max_pages:
        # if not rate_limit_manager.should_continue():
        #     logger.warning(f"Rate limit exhausted during timeline fetch, stopping at page {page}")
        #     break
        # delay = rate_limit_manager.get_delay()
        # time.sleep(delay)
        resp = requests.get(url, headers=headers, params={"per_page": 100, "page": page}, timeout=30)
        # rate_limit_manager.update_from_response(resp)
        
        if resp.status_code != 200:
            break
        batch = resp.json() or []
        if not batch:
            break

        events.extend(batch)
        if len(batch) < 100:
            break
        page += 1
        
    return events

# ？ 本意通过时间线查询事件，提取关联的PR编号列表
def extract_linked_pr_numbers_from_events(events: List[Dict[str, Any]]) -> List[int]:
    linked_prs: List[int] = []
    seen = set()
    for ev in events:
        if ev.get("event") == "cross-referenced":
            src = ev.get("source") or {}
            pr = src.get("issue") or {}
            if pr.get("pull_request"):
                num = pr.get("number")
                if isinstance(num, int) and num not in seen:
                    seen.add(num)
                    linked_prs.append(num)
    return linked_prs

# ？ 本意通过{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}查询PR信息，加入缓存
def fetch_pr(owner: str, repo: str, pr_number: int, token: str) -> Optional[Dict[str, Any]]:
    cache_key = f"{owner}/{repo}/pulls/{pr_number}".lower()
    if cache_key in PR_CACHE:
        logger.info(f"Using cached PR info for {owner}/{repo}#{pr_number}")
        return PR_CACHE[cache_key]
    
    # if not rate_limit_manager.should_continue():
    #     logger.warning(f"Rate limit too low, skipping PR fetch for {owner}/{repo}#{pr_number}")
    #     return None
    
    url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}"

    # delay = rate_limit_manager.get_delay()
    # time.sleep(delay)
    resp = requests.get(url, headers=gh_headers(token), timeout=30)
    # rate_limit_manager.update_from_response(resp)
    
    if resp.status_code != 200:
        PR_CACHE[cache_key] = None
        return None
    
    result = resp.json()
    PR_CACHE[cache_key] = result

    return result

def compare_shas(base_sha: str, head_sha: str, token: str, diff: bool) -> Optional[Dict[str, Any]] | None:
    
    url = f"{GITHUB_API}/repos/compare/{base_sha}...{head_sha}"
    
    if diff == "true":
        accept = "application/vnd.github.v3.diff"
        resp = requests.get(url, headers=gh_headers(token, accept), timeout=30)
        if resp.status_code != 200:
            return None
        result = resp.text or ""
    else:
        accept = "application/vnd.github+json"
        resp = requests.get(url, headers=gh_headers(token, accept), timeout=30)
        if resp.status_code != 200:
            return None
        result = resp.json()

    return result

# ？ 感觉不大合理
def has_code_example(text: str) -> bool:
    if not text:
        return False
    if "```" in text:
        return True
    if re.search(r"\n\s{4,}\S", text):
        return True
    return False

# ？ 这个还好
def has_error_message(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r"(Exception|Error|Traceback|TypeError|ValueError|AssertionError|IndexError|KeyError)", text, re.I) or re.search(r"bug|error|issue|problem|bug report|error report|issue report", text, re.I))

# 查看单个issue的body的质量，是否有
def issue_body_quality_ok(body: str) -> bool:
    """
    检查issue质量
    Args:
        issue: issue信息
        skip_state_check: 是否跳过状态检查（因为query中已经过滤了is:closed）
    """
    if not body:
        logger.info("Issue body is empty")
        return False
    
    if QUALITY_FILTERS["min_body_length"] and len(body) < QUALITY_FILTERS["min_body_length"]:
        logger.info(f"Issue body too short: {len(body)}")
        return False
    # if QUALITY_FILTERS["max_body_length"] and len(body) > QUALITY_FILTERS["max_body_length"]:
    #     logger.info(f"Issue body too long: {len(body)}")
    #     return False
    if QUALITY_FILTERS["has_code_example"] and not has_code_example(body):
        logger.info("Issue body missing code example")
        return False
    # if QUALITY_FILTERS["has_error_message"] and not has_error_message(body):
    #     logger.info("Issue body missing error message")
    #     return False
        
    return True

def execute_single_query(
    kws: List[str],
    token: str,
    output_file,
    present_keys: Set[Tuple[str, int]],
    write_lock: threading.Lock,
    until_date: Optional[str] = None,  # 新增参数：用于构造 created 时间范围
) -> Dict[str, Any]:
    """
    执行单个查询，返回精简数据+merged PRs
    """
    thread_name = threading.current_thread().name
    
    if not search_rate_limit_manager.should_continue(name = thread_name):
        logger.warning(f"{thread_name}: Rate limit too low ({search_rate_limit_manager.remaining}), skipping query")
        return {"keywords": kws, "found_count": 0, "written_count": 0, "query_used": None, "status": "skipped_rate_limit"}
    
    try:
        # 首先尝试AND查询
        q_and = build_query(kws, use_or=False, until_date=until_date)
        logger.info(f"{thread_name}: Executing query: {q_and}")
        issues = search_issues(q_and, token, per_page=6)
        used_query = q_and
        query_type = "AND"
        
        # 如果无结果且有多个关键词，尝试OR fallback
        if not issues and len([k for k in kws if k and k.strip()]) > 1:
            q_or = build_query(kws, use_or=True, until_date=until_date)
            logger.info(f"{thread_name}: AND failed, trying OR: {q_or}")
            issues = search_issues(q_or, token, per_page=6)
            used_query = q_or
            query_type = "OR_fallback"
        
        # 最多从combination的一个记录中获得6条issue信息
        found_count = len(issues)
        written_count = 0
        
        if issues:
            logger.info(f"{thread_name}: Found {found_count} issues")
            
            # 处理每个issue（精简数据）
            for item in issues:
                try:
                    repository_url = item.get("repository_url") or ""
                    owner, repo = parse_repo_full_name(repository_url)
                    issue_id = item.get("number")
                    issue_url = item.get("html_url", "")

                    # 检查重复
                    issue_key = (f"{owner}/{repo}".lower(), int(issue_id or 0))
                    with write_lock:
                        if issue_key in present_keys:
                            logger.info(f"{thread_name}: Skipped duplicate issue {owner}/{repo}#{issue_id}")
                            continue
                        present_keys.add(issue_key)
                    
                    title = item.get("title", "")
                    body = item.get("body", "")
                    if not(issue_body_quality_ok(body)):
                        logger.info(f"{thread_name}: Issue body quality fail {owner}/{repo}#{issue_id}")
                        continue
                    
                    # # 通过issue_id，查询时间线事件
                    # events = fetch_issue_timeline_event(owner, repo, issue_id, token)
                    # if not events:
                    #     logger.info(f"{thread_name}: No timeline events for {owner}/{repo}#{issue_id}")
                    #     continue

                    # linked_prs = extract_linked_pr_numbers_from_events(events)
                    # if not linked_prs:
                    #     logger.info(f"{thread_name}: No linked PRs for {owner}/{repo}#{issue_id}")
                    #     continue
                    
                    # merged_prs = List[Dict[str, Dict[str, str]]]()

                    # # 遍历linked_prs，每个pr使用{repo}/pulls/{pull_number}查询PR信息
                    # for pr_num in linked_prs:
                    #     pr_info = fetch_pr(owner, repo, pr_num, token)
                    #     if not pr_info or not pr_info.get('merged'):
                    #         logger.info(f"{thread_name}: Linked PR not merged {owner}/{repo}#{pr_num}")
                    #         continue
                    #     # 注意，我们越过检查fetch_pr_files->is_docs_only/validate_issue_solution
                    #     # 
                    #     base_sha = pr_info.get("base", {}).get("sha", "")
                    #     head_sha = pr_info.get("head", {}).get("sha", "")
                    #     merged_sha = pr_info.get("merge_commit_sha", "")

                    #     compare_date = compare_shas(base_sha, head_sha, token, diff="false")
                    #     if not compare_date:
                    #         logger.info(f"{thread_name}: Failed to compare SHAs for PR {owner}/{repo}#{pr_num}")
                    #         continue
                    #     true_base_sha = compare_date["merge_base_commit"]["sha"]
                    #     if not true_base_sha:
                    #         logger.info(f"{thread_name}: Base SHA mismatch for PR {owner}/{repo}#{pr_num}")
                    #         continue
                    #     diff_content = compare_shas(true_base_sha, head_sha, token, diff="true")
                    #     if not diff_content:
                    #         logger.info(f"{thread_name}: Failed to get diff for PR {owner}/{repo}#{pr_num}")
                    #         continue

                    #     # 将本条 PR 信息写入 merged_prs
                    #     merged_prs.append({
                    #         str(pr_num): {
                    #             "base_sha": base_sha,
                    #             "head_sha": head_sha,
                    #             "merged_sha": merged_sha,
                    #             "true_base_sha": true_base_sha,
                    #             "diff_content": diff_content
                    #         }
                    #     })

                    # 构建精简结果
                    result = {
                        "repo": f"{owner}/{repo}",
                        "issue_id": issue_id,
                        "created_at": item.get("created_at", ""),
                        "instance_time": until_date,
                        "url": issue_url,
                        "title": title,
                        "body": body,
                        #"linked_prs":linked_prs,
                        "comments_count": item.get("comments", 0),
                        "query_used": used_query,
                        "query_type": query_type,
                        "keywords_used": kws,
                        "found_in_query_count": found_count,
                    }
                    
                    # 线程安全写入
                    with write_lock:
                        try:
                            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                            output_file.flush()
                            written_count += 1
                            logger.info(f"{thread_name}: Wrote issue {owner}/{repo}#{issue_id}")
                        except Exception as write_exc:
                            logger.error(f"{thread_name}: Failed to write result: {write_exc}")
                    
                except Exception as e:
                    logger.info(f"{thread_name}: Failed to process issue: {e}")
                    continue
        
        else:
            logger.info(f"{thread_name}: No results for keywords: {kws}")
        
        return {
            "keywords": kws, 
            "found_count": found_count, 
            "written_count": written_count, 
            "query_used": used_query,
            "query_type": query_type,
            "status": "completed"
        }
        
    except Exception as exc:
        logger.warning(f"{thread_name}: Query failed for keywords {kws}: {exc}")
        return {"keywords": kws, "found_count": 0, "written_count": 0, "query_used": None, "status": "error", "error": str(exc)}

def process_record(
    record: Dict[str, Any], 
    token: str, output_file, 
    present_keys: Set[Tuple[str, int]], 
    write_lock: threading.Lock, 
    target_issues: int = 30, 
    max_workers: int = 3
) -> Dict[str, Any]:
    """
    处理单个记录，目标收集30个issue，全面遍历关键词组合
    """
    key_words = record.get("key_words") or {}
    until_date: Optional[str] = (record.get("created_at") or "").strip() or None

    # 1. 生成所有可能的关键词组合
    all_combinations = generate_keyword_combinations(key_words, max_combinations=15)
    
    if not all_combinations:
        logger.warning("No valid keyword combinations generated")
        return {"total_queries": 0, "total_found": 0, "total_written": 0, "queries_detail": [], "target_reached": False}
    
    logger.info(f"Generated {len(all_combinations)} keyword combinations, targeting {target_issues} issues")
    
    query_results = []
    total_written = 0
    combination_index = 0
    
    # 2. 循环执行组合，直到达到目标或用尽组合
    while total_written < target_issues and combination_index < len(all_combinations) and search_rate_limit_manager.should_continue(name = "process_record"):
        # 每次取一批组合并行处理
        batch_size = min(max_workers, len(all_combinations) - combination_index, max(1, (target_issues - total_written) // 6))  # 估算需要的查询数
        
        current_batch = all_combinations[combination_index:combination_index + batch_size]
        combination_index += batch_size
        
        logger.info(f"Processing batch {len(current_batch)} combinations (written so far: {total_written}/{target_issues})")
        
        # 并行执行当前批次
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            logger.info(f"use {max_workers} threads for query execution")
            future_to_kws = {
                executor.submit(execute_single_query, kws, token, output_file, present_keys, write_lock, until_date): kws 
                for kws in current_batch
            }
            
            # 收集结果
            for future in as_completed(future_to_kws):
                kws = future_to_kws[future]
                try:
                    query_stat = future.result()
                    query_results.append(query_stat)
                    
                    if query_stat["status"] == "completed":
                        batch_written = query_stat['written_count']
                        total_written += batch_written
                        logger.info(f"Query [{', '.join(kws)[:20]}...]: found {query_stat['found_count']}, wrote {batch_written} (total: {total_written}/{target_issues})")
                        
                        # 如果已经达到目标，可以提前结束
                        if total_written >= target_issues:
                            logger.info(f"Target {target_issues} issues reached, stopping early")
                            break
                            
                    elif query_stat["status"] == "error":
                        logger.warning(f"Query [{', '.join(kws)[:20]}...]: failed - {query_stat.get('error', 'unknown error')}")
                    elif query_stat["status"] == "skipped_rate_limit":
                        logger.warning(f"Query [{', '.join(kws)[:20]}...]: skipped due to rate limit")
                        
                except Exception as exc:
                    logger.warning(f"Query for {kws} generated an exception: {exc}")
                    query_results.append({"keywords": kws, "found_count": 0, "written_count": 0, "status": "exception", "error": str(exc)})
        
        # 检查是否已经达到目标
        if total_written >= target_issues:
            break
    
    # 统计汇总
    total_found = sum(q.get("found_count", 0) for q in query_results)
    target_reached = total_written >= target_issues
    
    summary = {
        "total_queries": len(query_results),
        "total_found": total_found,
        "total_written": total_written,
        "target_issues": target_issues,
        "target_reached": target_reached,
        "combinations_used": combination_index,
        "combinations_available": len(all_combinations),
        "queries_detail": query_results
    }
    
    status_msg = f"Record complete: {len(query_results)} queries, {total_found} found, {total_written} written ({target_issues} target, {'✓' if target_reached else '✗'})"
    logger.info(status_msg)
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description="Mine similar GitHub issues from keyworded issues.jsonl")
    parser.add_argument("--input", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--token", type=str, default="", help="GitHub API token")
    parser.add_argument("--batch-start", type=int, default=0, help="Number of input records to process")
    parser.add_argument("--batch-end", type=int, default=200, help="")
    parser.add_argument("--per-issue-queries", type=int, default=5, help="Base multiplier for target issues (5*6=30 target issues per record)")
    parser.add_argument("--skip-existing", default=None, help="Skip records already present in this jsonl by (repo, issue_id)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--test", action="store_true", help="Process only the first input record and print one result if any")
    parser.add_argument("--max-query-threads", type=int, default=3, help="Maximum threads for parallel query execution (default: 3)")
    parser.add_argument("--max-record-threads", type=int, default=3, help="Maximum threads for parallel record processing (default: 3)")
    parser.add_argument("--no-threading", action="store_true", help="Disable multithreading (useful for debugging)")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s [%(levelname)s] %(message)s")

    token = args.token
    if not token:
        raise SystemExit("Please export GITHUB_TOKEN (or GH_TOKEN)")

    inputs = list(read_jsonl(args.input))
    if not inputs:
        raise SystemExit("No input records")
    if (args.batch_start + 1) and args.batch_end:
        inputs = inputs[args.batch_start : args.batch_end]

    # 返回已存在的(repo, issue_id)集合
    present_keys = load_present_keys(args.skip_existing or args.output)
    logger.info(f"Loaded {len(present_keys)} existing keys from {args.skip_existing or args.output}")

    # 初始化rate limit检查
    logger.info(f"main Starting with rate limit estimate: {search_rate_limit_manager.remaining}")
    
    # 创建线程安全写入锁
    write_lock = threading.Lock()
    
    if args.test:
        logger.info("Running in test mode - processing first record only")
        try:
            # 为测试创建临时文件
            test_output = args.output + ".test"
            logger.info(f"Test mode: writing to {test_output}")
            
            with open(test_output, "w", encoding="utf-8") as outf:
                max_query_workers = 1 if args.no_threading else args.max_query_threads
                result_stats = process_record(
                    inputs[0], token, outf, present_keys, write_lock, 
                    target_issues=args.per_issue_queries * 6, max_workers=max_query_workers
                )
                
                logger.info(f"Test completed: {result_stats}")
                print(f"Test Results Summary:")
                print(f"  Total queries executed: {result_stats['total_queries']}")
                print(f"  Total issues found: {result_stats['total_found']}")
                print(f"  Total issues written: {result_stats['total_written']}")
                print(f"  Results written to: {test_output}")
                
                # 显示每个query的结果
                for i, query_detail in enumerate(result_stats['queries_detail'], 1):
                    print(f"  Query {i}: {', '.join(query_detail['keywords'])[:50]}... -> found {query_detail['found_count']}, wrote {query_detail['written_count']}")
                    
        except Exception as exc:
            logger.error(f"Test failed: {exc}")
            raise
        return
    
    total_written = 0
    errors = 0
    skipped_low_rate_limit = 0
    
    logger.info(f"Processing {len(inputs)} input records")
    
    try:
        with open(args.output, "a", encoding="utf-8") as outf:
            # 决定是否使用多线程处理records
            if len(inputs) > 1 and search_rate_limit_manager.remaining > 100 and not args.no_threading:
                # 多线程处理多个records
                max_record_workers = min(args.max_record_threads, len(inputs))
                logger.info(f"Using {max_record_workers} threads to process records")
                
                def process_single_record(idx_rec_tuple):
                    idx, rec = idx_rec_tuple
                    if not search_rate_limit_manager.should_continue():
                        return idx, {"total_queries": 0, "total_found": 0, "total_written": 0, "status": "skipped_rate_limit"}
                    try:
                        logger.info(f"Thread processing record #{idx+1}/{len(inputs)}")
                        max_query_workers = 1 if args.no_threading else min(2, args.max_query_threads)
                        result_stats = process_record(
                            rec, token, outf, present_keys, write_lock,
                            target_issues=30, max_workers=max_query_workers
                        )
                        return idx, result_stats
                    except Exception as exc:
                        logger.warning(f"Record #{idx+1} failed: {exc}")
                        return idx, {"total_queries": 0, "total_found": 0, "total_written": 0, "status": "error", "error": str(exc)}
                
                with ThreadPoolExecutor(max_workers=max_record_workers) as record_executor:
                    record_futures = {record_executor.submit(process_single_record, (idx, rec)): idx for idx, rec in enumerate(inputs)}
                    
                    for future in as_completed(record_futures):
                        idx, result_stats = future.result()
                        
                        if result_stats.get("status") == "error":
                            errors += 1
                            continue
                        elif result_stats.get("status") == "skipped_rate_limit":
                            skipped_low_rate_limit += 1
                            continue
                        
                        # 统计结果（已经增量写入，无需再次写入）
                        written_this_record = result_stats.get("total_written", 0)
                        total_written += written_this_record
                        
                        logger.info(f"Record #{idx+1}: {result_stats['total_queries']} queries, {result_stats['total_found']} found, {written_this_record} written; total written: {total_written}")
            
            else:
                # 单线程处理（rate limit低或只有一个record）
                logger.info("Using single-threaded processing")
                for idx, rec in enumerate(inputs):
                    start_time = time.time()
                    # 检查rate limit
                    if not search_rate_limit_manager.should_continue(name = "main_loop"):
                        logger.warning(f"main_loop Rate limit too low ({search_rate_limit_manager.remaining}), stopping processing")
                        skipped_low_rate_limit = len(inputs) - idx
                        break
                    
                    try:
                        logger.info(f"Processing record #{idx+1+args.batch_start}/{len(inputs)}")
                        max_query_workers = args.max_query_threads
                        result_stats = process_record(
                            rec, token, outf, present_keys, write_lock,
                            target_issues=30, max_workers=max_query_workers
                        )
                        
                        # 统计结果（已经增量写入，无需再次写入）
                        written_this_record = result_stats.get("total_written", 0)
                        total_written += written_this_record
                        
                        logger.info(f"Record #{idx+1}: {result_stats['total_queries']} queries, {result_stats['total_found']} found, {written_this_record} written; total written: {total_written}")
                        end_time = time.time()   
                        logger.info(f"================Record #{idx+1+args.batch_start} processed in {end_time - start_time:.2f} seconds=================")
                    except KeyboardInterrupt:
                        logger.info("Interrupted by user")
                        break
                    except Exception as exc:
                        logger.warning(f"Record #{idx+1} failed: {exc}")
                        errors += 1
                        continue
    
    except Exception as exc:
        logger.error(f"Fatal error during processing: {exc}")
        raise
    
    finally:
        # 统计报告
        logger.info(f"\n=== Processing Summary ===")
        logger.info(f"Total records processed: {len(inputs) - skipped_low_rate_limit}")
        logger.info(f"Total results written: {total_written}")
        logger.info(f"Errors encountered: {errors}")
        if skipped_low_rate_limit > 0:
            logger.info(f"Records skipped due to low rate limit: {skipped_low_rate_limit}")
        logger.info(f"Final rate limit remaining: {search_rate_limit_manager.remaining}")
        logger.info(f"=== End Summary ===")

if __name__ == "__main__":
    main()