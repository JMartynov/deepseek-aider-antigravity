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
import json
import argparse
import subprocess
import urllib.request
import urllib.error
from pathlib import Path


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
    failure_patterns = [
        r"Failed to apply edit to",
        r"The search block did not match",
        r"did not find the search block"
    ]
    for pattern in failure_patterns:
        if re.search(pattern, output, re.IGNORECASE):
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


def remove_ephemeral_aiderignore(ignore_path: Path):
    """Delete the ephemeral .aiderignore file, ignoring cleanup errors."""
    try:
        if ignore_path and ignore_path.exists():
            ignore_path.unlink()
    except OSError:
        pass


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


def run_turn(task_id: str, message: str, files: list[str], cwd: Path, 
             read_files: list[str] = None, restore: bool = False, 
             model: str = "deepseek/deepseek-flash", lint_cmd: str = None):
    # Path containment validation first (security guard)
    for f in (files or []) + (read_files or []):
        resolved = (cwd / f).resolve()
        if not str(resolved).startswith(str(cwd.resolve())):
            print(f"[ERROR] Security violation: path '{f}' is outside project root!", file=sys.stderr)
            sys.exit(12)
            
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
        "--no-gitignore",
        "--chat-history-file", str(session_file),
        "--yes-always",
        "--no-auto-commits",
        "--message", full_message
    ]

    # Dynamic .aiderignore shield for secrets and cache artifacts
    ignore_path = create_ephemeral_aiderignore(cwd)
    cmd.extend(["--aiderignore", str(ignore_path)])

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
    
    combined_output = proc.stdout + "\n" + proc.stderr
    print(combined_output)
    
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
    turn_parser.add_argument("--cwd", default=".", help="Project working directory")
    
    clean_parser = subparsers.add_parser("cleanup", help="Clean up session history file")
    clean_parser.add_argument("--task-id", required=True, help="Task identifier to clean up")
    clean_parser.add_argument("--cwd", default=".", help="Project working directory")
    
    args = parser.parse_args()
    cwd_path = Path(args.cwd).resolve()
    
    if args.command == "turn":
        run_turn(args.task_id, args.message, args.files, cwd_path, 
                 read_files=args.read_files, restore=args.restore, model=args.model,
                 lint_cmd=args.lint_cmd)
    elif args.command == "cleanup":
        cleanup_session(args.task_id, cwd_path)


if __name__ == "__main__":
    main()
