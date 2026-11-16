#!/usr/bin/env python3
"""
aider_bridge.py - Token-Sparse DeepSeek Aider Session Orchestrator
Handles turn execution, .env loading, stop-word ('QUESTION:') interception,
and clean-up for Antigravity-driven code generation.
"""

import os
import re
import sys
import argparse
import subprocess
from pathlib import Path

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
    """Extract token count and cost from Aider execution output."""
    tokens_match = re.search(r"Tokens:\s*([0-9.]+[kKmM]?)\s*sent,\s*([0-9.]+[kKmM]?)\s*received", output)
    cost_match = re.search(r"Cost:\s*\$([0-9.]+)\s*message,\s*\$([0-9.]+)\s*session", output)
    
    tokens_sent = tokens_match.group(1) if tokens_match else "unknown"
    tokens_rcvd = tokens_match.group(2) if tokens_match else "unknown"
    cost_msg = cost_match.group(1) if cost_match else "unknown"
    cost_session = cost_match.group(2) if cost_match else "unknown"
    return tokens_sent, tokens_rcvd, cost_msg, cost_session

def run_turn(task_id: str, message: str, files: list[str], cwd: Path, restore: bool = False, model: str = "deepseek/deepseek-chat"):
    load_project_env(cwd)
    
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("[ERROR] DEEPSEEK_API_KEY not found in project .env file!", file=sys.stderr)
        sys.exit(1)
        
    session_file = cwd / f".aider.session_{task_id}.md"
    
    # System directive to enforce token sparsity and the question protocol
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
        "--chat-history-file", str(session_file),
        "--yes-always",
        "--no-auto-commits",
        "--message", full_message
    ]
    
    if restore and session_file.exists():
        cmd.append("--restore-chat-history")
        
    for f in files:
        cmd.append(f)
        
    print(f"--- Running Aider Turn [Task: {task_id}] (restore={restore}) ---")
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    
    combined_output = proc.stdout + "\n" + proc.stderr
    print(combined_output)
    
    tokens_sent, tokens_rcvd, cost_msg, cost_session = parse_cost_and_tokens(combined_output)
    print(f"\n[METRICS] Tokens Sent: {tokens_sent} | Received: {tokens_rcvd} | Cost Message: ${cost_msg} | Cost Session: ${cost_session}")
    
    # Check for QUESTION: stop-word
    question_match = re.search(r"QUESTION:\s*(.+)", combined_output)
    if question_match:
        question_text = question_match.group(1).strip()
        print(f"\n[INTERCEPT] Sub-agent paused with QUESTION: {question_text}")
        print(f"[STATUS] INTERACTIVE_QUESTION")
        sys.exit(10)
        
    if proc.returncode != 0:
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
    turn_parser.add_argument("--files", nargs="+", required=True, help="Explicit list of target files")
    turn_parser.add_argument("--restore", action="store_true", help="Restore prior turn session history for prefix cache")
    turn_parser.add_argument("--model", default="deepseek/deepseek-chat", help="Model name (default: deepseek/deepseek-chat)")
    turn_parser.add_argument("--cwd", default=".", help="Project working directory")
    
    clean_parser = subparsers.add_parser("cleanup", help="Clean up session history file")
    clean_parser.add_argument("--task-id", required=True, help="Task identifier to clean up")
    clean_parser.add_argument("--cwd", default=".", help="Project working directory")
    
    args = parser.parse_args()
    cwd_path = Path(args.cwd).resolve()
    
    if args.command == "turn":
        run_turn(args.task_id, args.message, args.files, cwd_path, restore=args.restore, model=args.model)
    elif args.command == "cleanup":
        cleanup_session(args.task_id, cwd_path)

if __name__ == "__main__":
    main()
