#!/usr/bin/env python3
"""
aider_bridge.py - Token-Sparse DeepSeek Aider Session Orchestrator
Handles turn execution, .env loading, stop-word ('QUESTION:') interception,
live account balance checking, diff verification, read-only references,
and session cleanup for Antigravity-driven code generation.
"""

import os
import re
import sys
import time
import json
import argparse
import shutil
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from contextlib import contextmanager


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
THINKING_BLOCK_RE = re.compile(r"► \*\*THINKING\*\*.*?► \*\*ANSWER\*\*", re.DOTALL)


def strip_thinking_block(text: str) -> str:
    """Strip internal chain-of-thought blocks to preserve orchestrator context tokens."""
    return THINKING_BLOCK_RE.sub("", text or "").strip()


def filter_turn_output(output: str, verbose: bool = False) -> str:
    """
    Remove noisy startup banners, dumb terminal notices, and internal thinking traces
    to minimize orchestrator token consumption while preserving surgical diffs, questions,
    and metrics.
    """
    if verbose:
        return output
    clean = strip_thinking_block(output)
    filtered_lines = []
    ignore_prefixes = (
        "Detected dumb terminal",
        "Cost estimates may be inaccurate",
        "Scraping https://",
        "Aider v",
    )
    for line in clean.splitlines():
        trimmed = line.strip()
        if any(trimmed.startswith(prefix) for prefix in ignore_prefixes):
            continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines).strip()


def compact_failure_trace(trace: str, max_lines: int = 40) -> str:
    """Strip ANSI escape codes and return the last max_lines non-empty lines."""
    if max_lines <= 0:
        return ""
    clean = ANSI_ESCAPE_RE.sub("", trace or "")
    lines = [line.rstrip() for line in clean.splitlines() if line.strip()]
    return "\n".join(lines[-max_lines:])


def fetch_account_balance(api_key: str):
    """Query official DeepSeek API for remaining account balance and currency."""
    if not api_key:
        return None
    url = "https://api.deepseek.com/user/balance"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            if data.get("is_available") and data.get("balance_infos"):
                info = data["balance_infos"][0]
                return {
                    "currency": info.get("currency", "USD"),
                    "total_balance": info.get("total_balance", "unknown"),
                    "granted_balance": info.get("granted_balance", "0.00"),
                    "topped_up_balance": info.get("topped_up_balance", "0.00")
                }
    except Exception as e:
        return {"error": str(e)}
    return None


def load_project_env(cwd: Path):
    """Load DEEPSEEK_API_KEY from project .env file only."""
    env_file = cwd / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k not in os.environ:
                    os.environ[k] = v


def parse_cost_and_tokens(output: str):
    """Extract token count (including cache writes/hits) and cost from Aider output."""
    tokens_match = re.search(
        r"Tokens:\s*([0-9.]+[kKmM]?)\s*sent"
        r"(?:,\s*([0-9.]+[kKmM]?)\s*cache write)?"
        r"(?:,\s*([0-9.]+[kKmM]?)\s*cache hit)?"
        r",\s*([0-9.]+[kKmM]?)\s*received",
        output
    )
    cost_match = re.search(r"Cost:\s*\$([0-9.]+)\s*message,\s*\$([0-9.]+)\s*session", output)
    
    tokens_sent = tokens_match.group(1) if tokens_match else "unknown"
    cache_write = tokens_match.group(2) if (tokens_match and tokens_match.group(2)) else "0"
    cache_hit = tokens_match.group(3) if (tokens_match and tokens_match.group(3)) else "0"
    tokens_rcvd = tokens_match.group(4) if tokens_match else "unknown"
    cost_msg = cost_match.group(1) if cost_match else "unknown"
    cost_session = cost_match.group(2) if cost_match else "unknown"
    return tokens_sent, cache_write, cache_hit, tokens_rcvd, cost_msg, cost_session


def extract_questions(output: str):
    """
    Extract questions while preventing false positives from code diffs or commit messages.
    Filters out lines that are part of diff additions, deletions, or unified headers.
    """
    non_diff_lines = []
    for line in output.splitlines():
        trimmed = line.strip()
        # Skip git / unified diff lines
        if trimmed.startswith(("+", "-", "@@", "diff ", "---", "+++", "* ")):
            continue
        if trimmed.startswith(("<<<<<<<", "=======", ">>>>>>>")):
            continue
        non_diff_lines.append(line.rstrip())
        
    filtered_text = "\n".join(non_diff_lines)
    
    # Match QUESTION: followed by text until an empty line or end of string
    matches = re.findall(r"(?:^|\n)\s*QUESTION:\s*(.+?)(?=\n\s*\n|\Z)", filtered_text, re.DOTALL)
    return [m.strip() for m in matches if m.strip()]


def check_diff_applied_successfully(output: str):
    """Detect if Aider encountered a search/replace diff failure."""
    # Strip thinking blocks to avoid matching internal reasoning traces
    clean_output = re.sub(r"► \*\*THINKING\*\*.*?► \*\*ANSWER\*\*", "", output, flags=re.DOTALL)
    failure_patterns = [
        r"Failed to apply edit to",
        r"The search block did not match",
        r"did not find the search block"
    ]
    for pattern in failure_patterns:
        if re.search(pattern, clean_output, re.IGNORECASE):
            return False
    return True


def create_ephemeral_aiderignore(cwd: Path):
    """Create an ephemeral .aiderignore shield covering secrets and cache artifacts."""
    ignore_path = cwd / ".aiderignore.ephemeral"
    patterns = [
        ".env*",
        "*.pem",
        "*.key",
        "__pycache__/",
        "*.pyc",
        ".pytest_cache/",
        ".cache/",
        "node_modules/",
    ]
    ignore_path.write_text("\n".join(patterns) + "\n")
    return ignore_path


def remove_ephemeral_file(file_path: Path):
    """Delete an ephemeral file, ignoring cleanup errors."""
    try:
        if file_path and file_path.exists():
            file_path.unlink()
    except OSError:
        pass


def remove_ephemeral_aiderignore(ignore_path: Path):
    """Delete the ephemeral .aiderignore file, ignoring cleanup errors."""
    remove_ephemeral_file(ignore_path)


def create_ephemeral_model_settings(cwd: Path, model: str):
    """
    Ensure deepseek models (e.g. deepseek/deepseek-flash) have explicit
    caches_by_default=True and edit_format=diff configured in Aider settings.
    """
    if not ("deepseek" in model.lower()):
        return None
    settings_path = cwd / f".aider.model.settings.{os.getpid()}.yml"
    settings_yaml = (
        f"- name: {model}\n"
        f"  edit_format: diff\n"
        f"  caches_by_default: true\n"
    )
    settings_path.write_text(settings_yaml)
    return settings_path


@contextmanager
def with_git_lock(cwd: Path, timeout: float = 60.0, retry_interval: float = 0.5):
    """
    Acquire an atomic directory lock (.git/aider_merge.lock.d) to serialize
    git worktree merges across concurrent sub-agents without index.lock collisions.
    """
    git_dir = cwd / ".git"
    if git_dir.is_file():
        # Inside a worktree, .git is a file pointing to gitdir: ...
        try:
            content = git_dir.read_text().strip()
            if content.startswith("gitdir:"):
                git_dir = Path(content.split("gitdir:", 1)[1].strip()).resolve()
                if "worktrees" in str(git_dir):
                    # Traverse up to common .git root
                    git_dir = git_dir.parent.parent
        except Exception:
            git_dir = cwd / ".git"

    lock_dir = git_dir / "aider_merge.lock.d" if git_dir.exists() else cwd / ".aider_merge.lock.d"
    start_time = time.time()
    acquired = False

    while not acquired:
        try:
            lock_dir.mkdir(parents=True, exist_ok=False)
            acquired = True
            print(f"[MUTEX] Acquired git merge lock: {lock_dir.name}")
        except FileExistsError:
            if time.time() - start_time >= timeout:
                raise TimeoutError(f"Timed out waiting {timeout}s for git merge lock: {lock_dir}")
            time.sleep(retry_interval)

    try:
        yield lock_dir
    finally:
        try:
            if lock_dir.exists():
                lock_dir.rmdir()
                print(f"[MUTEX] Released git merge lock: {lock_dir.name}")
        except Exception as e:
            print(f"[MUTEX] Error releasing git merge lock: {e}", file=sys.stderr)


def auto_rebase_branch_if_needed(branch_name: str, cwd: Path, base_branch: str = "main"):
    """
    Check if base branch has advanced ahead of task branch. If so, rebase task
    branch onto the updated base branch to maintain clean linear history.
    """
    # Detect current base branch if not provided
    head_check = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=cwd, capture_output=True, text=True
    )
    current_head = head_check.stdout.strip() if head_check.returncode == 0 else base_branch
    target_base = current_head if current_head and current_head != "HEAD" else base_branch

    # Find merge base between base branch and task branch
    merge_base = subprocess.run(
        ["git", "merge-base", target_base, branch_name],
        cwd=cwd, capture_output=True, text=True
    )
    base_rev = subprocess.run(
        ["git", "rev-parse", target_base],
        cwd=cwd, capture_output=True, text=True
    )
    if merge_base.returncode == 0 and base_rev.returncode == 0:
        if merge_base.stdout.strip() != base_rev.stdout.strip():
            print(f"[REBASE] Base branch {target_base} has advanced. Rebasing {branch_name} onto {target_base}...")
            rebase = subprocess.run(
                ["git", "rebase", target_base, branch_name],
                cwd=cwd, capture_output=True, text=True
            )
            if rebase.returncode != 0:
                print(f"[REBASE] Warning: Auto-rebase failed: {rebase.stderr.strip()}", file=sys.stderr)
                subprocess.run(["git", "rebase", "--abort"], cwd=cwd, capture_output=True)
            else:
                print(f"[REBASE] Successfully rebased {branch_name} onto {target_base}")


def create_task_worktree(task_id: str, cwd: Path, base_branch: str = None) -> tuple[Path, str]:
    """Create (or reuse) the isolated worktree for a task on branch 'task/<task_id>'."""
    worktree_path = cwd / ".worktrees" / task_id
    branch_name = f"task/{task_id}"
    if worktree_path.exists():
        print(f"[WORKTREE] Reusing existing worktree: {worktree_path}")
        return worktree_path, branch_name
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    # Check if branch already exists
    branch_check = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch_name}"],
        cwd=cwd
    )
    if branch_check.returncode == 0:
        cmd = ["git", "worktree", "add", str(worktree_path), branch_name]
    else:
        cmd = ["git", "worktree", "add", "-b", branch_name, str(worktree_path)]
        if base_branch:
            cmd.append(base_branch)
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create worktree: {result.stderr.strip()}")
    print(f"[WORKTREE] Attached branch {branch_name} at {worktree_path}")
    return worktree_path, branch_name


def merge_and_remove_worktree(task_id: str, worktree_path: Path, branch_name: str,
                              cwd: Path, auto_merge: bool = True):
    """Remove the worktree; if auto_merge, lock git, rebase if needed, merge, then delete branch."""
    remove = subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree_path)],
        cwd=cwd, capture_output=True, text=True
    )
    if remove.returncode != 0:
        print(f"[WORKTREE] Failed to remove worktree: {remove.stderr.strip()}", file=sys.stderr)
    else:
        print(f"[WORKTREE] Removed worktree: {worktree_path}")
    if not auto_merge:
        return

    with with_git_lock(cwd):
        auto_rebase_branch_if_needed(branch_name, cwd)
        merge = subprocess.run(
            ["git", "merge", "--no-edit", branch_name],
            cwd=cwd, capture_output=True, text=True
        )
        if merge.returncode != 0:
            print(f"[WORKTREE] Merge failed: {merge.stderr.strip()}", file=sys.stderr)
            print("[STATUS] MERGE_FAILED", file=sys.stderr)
            sys.exit(15)
        print(f"[WORKTREE] Merged {branch_name} into current branch")
        delete = subprocess.run(
            ["git", "branch", "-D", branch_name],
            cwd=cwd, capture_output=True, text=True
        )
        if delete.returncode != 0:
            print(f"[WORKTREE] Failed to delete branch {branch_name}: {delete.stderr.strip()}", file=sys.stderr)


def merge_worktree_task(task_id: str, cwd: Path):
    """Explicitly merge and cleanup an isolated task worktree under mutex protection."""
    worktree_path = cwd / ".worktrees" / task_id
    branch_name = f"task/{task_id}"
    if worktree_path.exists():
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree_path)],
            cwd=cwd, capture_output=True, text=True
        )
    with with_git_lock(cwd):
        auto_rebase_branch_if_needed(branch_name, cwd)
        merge = subprocess.run(
            ["git", "merge", "--no-edit", branch_name],
            cwd=cwd, capture_output=True, text=True
        )
        if merge.returncode != 0:
            print(f"[ERROR] Merge failed: {merge.stderr.strip()}", file=sys.stderr)
            sys.exit(15)
        print(f"[WORKTREE] Successfully merged {branch_name} into current branch")
        subprocess.run(
            ["git", "branch", "-D", branch_name],
            cwd=cwd, capture_output=True, text=True
        )
        print(f"[WORKTREE] Deleted branch {branch_name}")


def detect_lint_cmd(files: list[str], user_lint_cmd: str = None):
    """Resolve the Aider lint command; fall back to py_compile for Python targets."""
    if user_lint_cmd:
        return user_lint_cmd
    if not any(str(f).endswith(".py") for f in (files or [])):
        return None
    try:
        import importlib.util as _ilu
        if _ilu.find_spec("py_compile") is None:
            return None
    except Exception:
        return None
    return "python:python -m py_compile"


def rollback_target_files(files: list[str], cwd: Path):
    """Roll back target files inside a git repo: restore tracked, delete untracked."""
    if not (cwd / ".git").exists():
        return
    for f in (files or []):
        try:
            tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", f],
                cwd=cwd, capture_output=True, text=True
            )
            if tracked.returncode == 0:
                subprocess.run(
                    ["git", "checkout", "--", f],
                    cwd=cwd, capture_output=True, text=True
                )
                print(f"[ROLLBACK] Restored tracked file: {f}")
            elif (cwd / f).exists():
                (cwd / f).unlink()
                print(f"[ROLLBACK] Deleted untracked file: {f}")
        except Exception as e:
            print(f"[ROLLBACK] Failed for {f}: {e}", file=sys.stderr)


def run_preflight_checks(read_files: list[str], cwd: Path):
    """
    Warn on uncommitted git changes; fail (exit 14) if any read-only file is missing.
    Automatically ensures cwd / src is exported to PYTHONPATH if it exists.
    """
    src_dir = cwd / "src"
    if src_dir.is_dir():
        current_pp = os.environ.get("PYTHONPATH", "")
        src_str = str(src_dir.resolve())
        if src_str not in current_pp.split(os.pathsep):
            os.environ["PYTHONPATH"] = f"{src_str}{os.pathsep}{current_pp}".rstrip(os.pathsep)
            print(f"[PREFLIGHT] Injected PYTHONPATH: {src_str}")

    if (cwd / ".git").exists():
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd, capture_output=True, text=True
            )
            if status.returncode == 0 and status.stdout.strip():
                print("[WARN] Uncommitted git changes detected in working tree.", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] Could not check git status: {e}", file=sys.stderr)
    missing = [f for f in (read_files or []) if not (cwd / f).exists()]
    if missing:
        for f in missing:
            print(f"[ERROR] Read-only reference file missing: {f}", file=sys.stderr)
        print("[STATUS] PREFLIGHT_FAILED")
        sys.exit(14)


def run_turn(task_id: str, message: str, files: list[str], cwd: Path, 
             read_files: list[str] = None, restore: bool = False, 
             model: str = "deepseek/deepseek-flash", lint_cmd: str = None,
             worktree: bool = False, auto_merge: bool = True,
             verbose: bool = False):
    if worktree:
        worktree_path, branch_name = create_task_worktree(task_id, cwd)
        env_src = cwd / ".env"
        if env_src.exists():
            shutil.copy(env_src, worktree_path / ".env")
        try:
            run_turn(task_id, message, files, worktree_path,
                     read_files=read_files, restore=restore, model=model,
                     lint_cmd=lint_cmd, worktree=False, verbose=verbose)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
            if code == 0:
                if auto_merge:
                    merge_and_remove_worktree(task_id, worktree_path, branch_name, cwd)
                else:
                    print(f"[WORKTREE] Turn passed. Kept worktree without merging: {worktree_path} (branch {branch_name})")
            elif code == 10:
                print(f"[WORKTREE] Kept for resumption: {worktree_path} (branch {branch_name})")
            else:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(worktree_path)],
                    cwd=cwd, capture_output=True, text=True
                )
                print(f"[WORKTREE] Removed worktree after failure; branch {branch_name} kept")
            raise
    # Path containment validation first (security guard)
    for f in (files or []) + (read_files or []):
        resolved = (cwd / f).resolve()
        if not str(resolved).startswith(str(cwd.resolve())):
            print(f"[ERROR] Security violation: path '{f}' is outside project root!", file=sys.stderr)
            sys.exit(12)
            
    run_preflight_checks(read_files, cwd)
    load_project_env(cwd)
    
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("[ERROR] DEEPSEEK_API_KEY not found in project .env file!", file=sys.stderr)
        sys.exit(1)
        
    session_file = cwd / f".aider.session_{task_id}.md"
    
    # Create parent directories for any new target files
    for f in (files or []):
        target_path = cwd / f
        target_path.parent.mkdir(parents=True, exist_ok=True)
    
    system_rules = (
        "IMPORTANT RULES: Be extremely concise. Output only surgical unified diffs. "
        "Do NOT write conversational filler or explanations. "
        "If any specification, signature, type, or requirement is missing or ambiguous, "
        "DO NOT GUESS and DO NOT modify code. Immediately output 'QUESTION: <your concise question>' and stop."
    )
    
    full_message = f"{system_rules}\n\nTask: {message}"
    
    cmd = [
        "uvx", "--from", "aider-chat", "aider",
        "--model", model,
        "--edit-format", "diff",
        "--cache-prompts",
        "--cache-keepalive-pings", "6",
        "--map-tokens", "0",
        "--no-analytics",
        "--no-check-update",
        "--no-show-release-notes",
        "--no-detect-urls",
        "--no-gitignore",
        "--chat-history-file", str(session_file),
        "--yes-always",
        "--no-auto-commits",
        "--message", full_message
    ]

    # Dynamic .aiderignore shield for secrets and cache artifacts
    ignore_path = create_ephemeral_aiderignore(cwd)
    cmd.extend(["--aiderignore", str(ignore_path)])

    # Ephemeral model settings for explicit prompt caching and diff format
    model_settings_path = create_ephemeral_model_settings(cwd, model)
    if model_settings_path:
        cmd.extend(["--model-settings-file", str(model_settings_path)])

    # Auto-lint / syntax check support
    resolved_lint_cmd = detect_lint_cmd(files, lint_cmd)
    if resolved_lint_cmd:
        cmd.extend(["--auto-lint", "--lint-cmd", resolved_lint_cmd])
    
    # Fallback if no git repo
    if not (cwd / ".git").exists():
        cmd.append("--no-git")
    
    if restore and session_file.exists():
        cmd.append("--restore-chat-history")
        
    # Read-only files passed with --read
    for rf in (read_files or []):
        cmd.extend(["--read", rf])
        
    # Target editable files passed as positional arguments
    for f in (files or []):
        cmd.append(f)
        
    print(f"--- Running Aider Turn [Task: {task_id}] (restore={restore}) ---")
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True
        )
    except Exception as e:
        rollback_target_files(files, cwd)
        print(f"[ERROR] Aider invocation failed: {e}", file=sys.stderr)
        print("[STATUS] AIDER_INVOCATION_FAILED")
        sys.exit(13)
    finally:
        remove_ephemeral_aiderignore(ignore_path)
        remove_ephemeral_file(model_settings_path)
    
    combined_output = proc.stdout + "\n" + proc.stderr
    print(filter_turn_output(combined_output, verbose=verbose))
    
    tokens_sent, cache_write, cache_hit, tokens_rcvd, cost_msg, cost_session = parse_cost_and_tokens(combined_output)
    
    # Query live DeepSeek balance API
    balance_info = fetch_account_balance(os.environ.get("DEEPSEEK_API_KEY"))
    balance_str = "N/A"
    if balance_info:
        if "error" in balance_info:
            balance_str = f"Error fetching ({balance_info['error']})"
        else:
            balance_str = f"{balance_info['total_balance']} {balance_info['currency']}"

    print("=" * 60)
    print(f"📊 [CALL METRICS & LIVE BALANCE]")
    print(f" • Tokens Sent     : {tokens_sent}")
    print(f" • Cache Write/Hit : {cache_write} write / {cache_hit} hit")
    print(f" • Tokens Received : {tokens_rcvd}")
    print(f" • Turn Cost (USD) : ${cost_msg}")
    print(f" • Session Total   : ${cost_session}")
    print(f" • Live API Balance: {balance_str}")
    print("=" * 60)
    
    # Check for QUESTION: stop-word outside code diffs
    questions = extract_questions(combined_output)
    if questions:
        print(f"\n[INTERCEPT] Sub-agent paused with {len(questions)} question(s):")
        for idx, q in enumerate(questions, 1):
            print(f" {idx}. {q}")
        print(f"[STATUS] INTERACTIVE_QUESTION")
        sys.exit(10)
        
    # Check if diff failed to apply
    if not check_diff_applied_successfully(combined_output):
        rollback_target_files(files, cwd)
        print("[ERROR] Sub-agent search/replace edit failed to apply cleanly to disk!", file=sys.stderr)
        print("[STATUS] DIFF_APPLY_FAILED")
        sys.exit(11)
        
    if proc.returncode != 0:
        rollback_target_files(files, cwd)
        print(f"[ERROR] Aider exited with code {proc.returncode}", file=sys.stderr)
        sys.exit(proc.returncode)
        
    print("[STATUS] COMPLETED")
    sys.exit(0)


def cleanup_session(task_id: str, cwd: Path):
    session_file = cwd / f".aider.session_{task_id}.md"
    if session_file.exists():
        session_file.unlink()
        print(f"[CLEANUP] Deleted session file: {session_file.name}")
    else:
        print(f"[CLEANUP] No session file found to delete for task {task_id}.")


def main():
    parser = argparse.ArgumentParser(description="Token-Sparse Aider Sub-Agent Orchestrator")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    turn_parser = subparsers.add_parser("turn", help="Run a turn in the task session")
    turn_parser.add_argument("--task-id", required=True, help="Unique identifier for the task session")
    turn_parser.add_argument("--message", required=True, help="Prompt/specification for the sub-agent")
    turn_parser.add_argument("--files", nargs="*", default=[], help="Editable target files")
    turn_parser.add_argument("--read-files", nargs="*", default=[], help="Read-only reference files")
    turn_parser.add_argument("--restore", action="store_true", help="Restore prior turn session history for prefix cache")
    turn_parser.add_argument("--model", default="deepseek/deepseek-flash", help="Model name (default: deepseek/deepseek-flash)")
    turn_parser.add_argument("--lint-cmd", default=None, help="Custom lint command passed to Aider (enables --auto-lint)")
    turn_parser.add_argument("--worktree", action="store_true", help="Run the turn inside an isolated git worktree and branch")
    turn_parser.add_argument("--no-auto-merge", action="store_true", help="Do not auto-merge the worktree on turn completion")
    turn_parser.add_argument("-v", "--verbose", action="store_true", help="Include full thinking traces and raw startup banners in output")
    turn_parser.add_argument("--cwd", default=".", help="Project working directory")
    
    clean_parser = subparsers.add_parser("cleanup", help="Clean up session history file")
    clean_parser.add_argument("--task-id", required=True, help="Task identifier to clean up")
    clean_parser.add_argument("--cwd", default=".", help="Project working directory")

    merge_parser = subparsers.add_parser("merge-worktree", help="Merge isolated task worktree branch and clean up worktree")
    merge_parser.add_argument("--task-id", required=True, help="Task identifier whose worktree should be merged")
    merge_parser.add_argument("--cwd", default=".", help="Project working directory")

    log_parser = subparsers.add_parser("compact-log", help="Strip ANSI codes and print the tail of a log")
    log_parser.add_argument("--log-file", required=True, help="Path to the log file to compact")
    log_parser.add_argument("--max-lines", type=int, default=40, help="Trailing non-empty lines to keep (default: 40)")
    
    args = parser.parse_args()
    cwd_path = Path(getattr(args, "cwd", ".")).resolve()
    
    if args.command == "turn":
        run_turn(args.task_id, args.message, args.files, cwd_path, 
                 read_files=args.read_files, restore=args.restore, model=args.model,
                 lint_cmd=args.lint_cmd, worktree=args.worktree,
                 auto_merge=not args.no_auto_merge, verbose=args.verbose)
    elif args.command == "cleanup":
        cleanup_session(args.task_id, cwd_path)
    elif args.command == "merge-worktree":
        merge_worktree_task(args.task_id, cwd_path)
    elif args.command == "compact-log":
        log_path = Path(args.log_file)
        if not log_path.exists():
            print(f"[ERROR] Log file not found: {log_path}", file=sys.stderr)
            sys.exit(1)
        print(compact_failure_trace(log_path.read_text(), args.max_lines))


if __name__ == "__main__":
    main()
