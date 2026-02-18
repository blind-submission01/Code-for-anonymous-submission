import os
import subprocess
import json
import threading
import time
import shutil
from typing import Tuple, List, Dict

# ------------ 配置 ------------
jsonl_file = ""
base_dir = ""

MAX_THREADS = 5                 # 并发上限（可按带宽调节）
CLONE_TIMEOUT = 900            # 单次 clone 超时（秒）
RETRIES = 3                     # 最大重试次数
RETRY_BASE_DELAY = 5            # 初始退避秒数
USE_PARTIAL_CLONE = False       # 先尝试部分克隆
GIT_ENV = {
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_HTTPS_LOW_SPEED_LIMIT": "1",
    "GIT_HTTPS_LOW_SPEED_TIME": "30"
}

# ------------ 统计数据（线程安全） ------------
lock = threading.Lock()

already_cloned_count = 0
new_cloned_count = 0

already_cloned_ids: List[str] = []
new_cloned_ids: List[str] = []
clone_failures: List[str] = []    # 最终仍失败的 target_dir_name

# ------------ 工具函数 ------------

def run(cmd, cwd=None, timeout=None) -> Tuple[int, str, str]:
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, **GIT_ENV}
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        proc.kill()
        return 124, "", "TIMEOUT"

RETRY_KEYWORDS = (
    "gnutls recv error",
    "early eof",
    "connection timed out",
    "invalid index-pack",
    "unexpected disconnect while reading sideband packet",
    "error in the pull function"
)

def is_retryable(stderr: str) -> bool:
    s = stderr.lower()
    return any(k in s for k in RETRY_KEYWORDS) or "exit status 128" in s

def safe_rmtree(path: str):
    if os.path.isdir(path):
        try:
            shutil.rmtree(path)
        except Exception:
            pass

def is_complete_git_repo(path: str) -> bool:
    return os.path.isdir(os.path.join(path, ".git", "objects"))

def ensure_dir():
    os.makedirs(base_dir, exist_ok=True)

def partial_clone(repo_url: str, target_path: str, timeout: int) -> Tuple[bool, str]:
    # --filter=blob:none 减少主体数据; --no-checkout 避免立即检出
    cmd = ["git", "clone", "--filter=blob:none", "--progress", repo_url, target_path]
    code, out, err = run(cmd, timeout=timeout)
    success = (code == 0)
    return success, err if err else out

def full_clone(repo_url: str, target_path: str, timeout: int) -> Tuple[bool, str]:
    cmd = ["git", "clone", "--progress", repo_url, target_path]
    code, out, err = run(cmd, timeout=timeout)
    success = (code == 0)
    return success, err if err else out

def prepare_repo_simple(repo_name: str, repo_url: str, target_path: str) -> bool:
    """
    仅负责将仓库克隆到指定目录（不检出 commit），带重试与部分克隆策略。
    """
    for attempt in range(1, RETRIES + 1):
        if os.path.exists(target_path) and not is_complete_git_repo(target_path):
            # 不完整仓库清理
            safe_rmtree(target_path)

        if os.path.exists(target_path) and is_complete_git_repo(target_path):
            # 已存在完整仓库，跳过
            return True

        print(f"[clone] ({attempt}/{RETRIES}) {repo_name} -> {target_path}")
        if USE_PARTIAL_CLONE:
            ok, msg = partial_clone(repo_url, target_path, CLONE_TIMEOUT)
            if not ok:
                retryable = is_retryable(msg)
                if not retryable and "repository not found" in msg.lower():
                    print(f"[clone-fatal] 仓库不存在: {repo_url}")
                    return False
                print(f"[clone-partial-fail] {repo_name} retryable={retryable} msg_tail={msg.strip()[-200:]}")
                # 尝试直接改为 full clone
                safe_rmtree(target_path)
                ok2, msg2 = full_clone(repo_url, target_path, CLONE_TIMEOUT)
                if ok2:
                    print(f"[fallback-full-ok] {repo_name}")
                    return True
                else:
                    print(f"[fallback-full-fail] {repo_name} msg_tail={msg2.strip()[-200:]}")
                if attempt < RETRIES:
                    time.sleep(RETRY_BASE_DELAY * attempt)
                    continue
                return False
            else:
                return True
        else:
            ok, msg = full_clone(repo_url, target_path, CLONE_TIMEOUT)
            if ok:
                return True
            retryable = is_retryable(msg)
            print(f"[clone-fail] {repo_name} retryable={retryable} msg_tail={msg.strip()[-200:]}")
            if not retryable:
                return False
            if attempt < RETRIES:
                time.sleep(RETRY_BASE_DELAY * attempt)
                continue
            return False
    return False

def load_jsonl(path: str) -> List[Dict]:
    data: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                continue
    return data

def make_target_dir_name(repo_name: str, issue_id, merged_pr_num, true_base_sha) -> str:
    """
    组合目录名: owner_repo_issueId_mergedPrNum_trueBaseSha
    所有字段转字符串，避免 None 进入
    """
    owner, repo = repo_name.split("/", 1)
    return f"{owner}_{repo}_{issue_id}_{merged_pr_num}_{true_base_sha}"

# ------------ 主流程 ------------

sem = threading.Semaphore(MAX_THREADS)

def clone_one(target_dir_name: str, repo_name: str):
    global already_cloned_count, new_cloned_count
    repo_url = f"https://github.com/{repo_name}.git"
    target_dir = os.path.join(base_dir, target_dir_name)

    with sem:
        if os.path.isdir(target_dir) and is_complete_git_repo(target_dir):
            with lock:
                already_cloned_count += 1
                already_cloned_ids.append(target_dir_name)
            print(f"[exists-ok] {target_dir_name}")
            return

        ok = prepare_repo_simple(repo_name, repo_url, target_dir)
        if ok:
            with lock:
                new_cloned_count += 1
                new_cloned_ids.append(target_dir_name)
            print(f"[clone-new] {target_dir_name}")
        else:
            with lock:
                clone_failures.append(target_dir_name)
            print(f"[clone-final-fail] {target_dir_name}")

def main():
    ensure_dir()
    items = load_jsonl(jsonl_file)

    # 去重（相同 repo + merged_pr_num 只克隆一次）
    tasks = {}
    for it in items:
        repo_name = it.get("repo")
        issue_id = it.get("issue_id")
        merged_pr_num = it.get("merged_pr_num")
        true_base_sha = it.get("true_base_sha")
        # 校验
        if not repo_name or issue_id is None or merged_pr_num is None or not true_base_sha:
            print(f"[skip-invalid] 缺字段: {it}")
            continue
        target_dir_name = make_target_dir_name(
            repo_name,
            str(issue_id),
            str(merged_pr_num),
            str(true_base_sha)
        )
        if target_dir_name not in tasks:
            tasks[target_dir_name] = repo_name

    print(f"[plan] 待克隆唯一仓库数: {len(tasks)} -> {base_dir}")
    # [plan] 待克隆唯一仓库数: 1867 -> /home/jiazixiao/INL_study/GenDNLrepo

    threads = []
    for target_dir_name, repo_name in tasks.items():
        t = threading.Thread(target=clone_one, args=(target_dir_name, repo_name), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("\n=== 统计信息 ===")
    print(f"已存在的仓库数量: {already_cloned_count}")
    print(f"新克隆的仓库数量: {new_cloned_count}")
    if clone_failures:
        print(f"克隆失败的目录({len(clone_failures)}): {clone_failures}")

if __name__ == "__main__":
    main()