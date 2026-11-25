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


def _normalize_search_payload(data):
    """Normalize a chat-completions response into {'results': [...]}."""
    if not isinstance(data, dict):
        return {"results": []}
    if isinstance(data.get("results"), list):
        return data
    for ch in data.get("choices") or []:
        msg = (ch or {}).get("message") or {}
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
                    return parsed
                if isinstance(parsed, list):
                    return {"results": parsed}
            except (json.JSONDecodeError, ValueError):
                pass
        for tc in msg.get("tool_calls") or []:
            args = ((tc or {}).get("function") or {}).get("arguments")
            if isinstance(args, str) and args.strip():
                try:
                    parsed = json.loads(args)
                    if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
                        return parsed
                    if isinstance(parsed, list):
                        return {"results": parsed}
                except (json.JSONDecodeError, ValueError):
                    pass
    return {"results": []}


def _search_via_chat_completions(query, api_key, count, need_content, timeout):
    """Fallback search path via chat-completions + deepseek_web_search tool."""
    url = "https://api.deepseek.com/v1/chat/completions"
    body = {
        "model": "deepseek-flash",
        "tools": [{"type": "deepseek_web_search", "name": "deepseek_web_search"}],
        "messages": [{
            "role": "user",
            "content": f"Search the web for: {query}. Return up to {count} results."
        }],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return _normalize_search_payload(json.loads(resp.read().decode("utf-8")))


def search_web_deepseek(query: str, api_key: str, count: int = 5,
                        need_content: bool = False, timeout: float = 15.0) -> dict:
    """POST to DeepSeek search tool endpoint; fall back to chat-completions on 404/405."""
    url = "https://api.deepseek.com/v1/tools/search"
    body = {"query": query, "count": count, "need_content": need_content}
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if getattr(e, "code", None) in (404, 405):
            return _search_via_chat_completions(query, api_key, count, need_content, timeout)
        raise


def format_search_results(data: dict, need_content: bool = False) -> str:
    """Render search results as numbered, human-readable text."""
    results = (data or {}).get("results") or []
    if not results:
        return "No search results found."
    lines = []
    for i, r in enumerate(results, 1):
        r = r or {}
        title = r.get("title") or "(untitled)"
        url = r.get("url") or "(no url)"
        summary = r.get("summary") or r.get("snippet") or ""
        lines.append(f"{i}. {title}")
        lines.append(f"   URL: {url}")
        if summary:
            lines.append(f"   Summary: {summary}")
        if need_content:
            content = r.get("content") or ""
            if content:
                lines.append(f"   Content: {content}")
    return "\n".join(lines)


SUPPORTED_TIERS = ("patch", "surgical", "architect", "research")

RESEARCH_KEYWORDS = (
    "research", "search", "lookup", "look up", "pep", "cve", "advisory",
    "spec", "what is", "how does", "documentation",
)
ARCHITECT_KEYWORDS = (
    "architect", "refactor", "ast", "visitor", "sqlglot", "security guard",
    "security_guard", "concurrency", "mutex", "lock", "cross-module",
    "cross-repo", "redesign", "rewrite", "migrate",
)
PATCH_KEYWORDS = (
    "typo", "lint", "linter", "bump", "version", "docstring", "comment",
    "readme", "format", "rename",
)
DOC_CONFIG_EXTENSIONS = (
    ".md", ".txt", ".rst", ".toml", ".yaml", ".yml", ".json", ".ini", ".cfg",
)


def _resolve_tier_params(tier: str, test_cmd: str = None, message: str = "") -> dict:
    """Build the resolved Aider/search parameter dict for a given precision tier."""
    if tier == "patch":
        return {
            "tier_num": 1,
            "tier_label": "PATCH",
            "model": "deepseek/deepseek-flash",
            "thinking": "off",
            "reasoning_effort": None,
            "edit_format": "diff",
            "map_tokens": 0,
            "architect": False,
            "editor_model": None,
            "test_cmd": test_cmd,
        }
    if tier == "surgical":
        return {
            "tier_num": 2,
            "tier_label": "SURGICAL",
            "model": "deepseek/deepseek-flash",
            "thinking": "auto",
            "reasoning_effort": "low",
            "edit_format": "diff",
            "map_tokens": 512,
            "architect": False,
            "editor_model": None,
            "test_cmd": test_cmd,
        }
    if tier == "architect":
        return {
            "tier_num": 3,
            "tier_label": "ARCHITECT",
            "model": "deepseek/deepseek-reasoner",
            "thinking": "on",
            "reasoning_effort": "high",
            "edit_format": "diff",
            "map_tokens": 512,
            "architect": True,
            "editor_model": "deepseek/deepseek-flash",
            "test_cmd": test_cmd,
        }
    return {
        "tier_num": 4,
        "tier_label": "RESEARCH",
        "command": "search",
        "query": (message or "").strip()[:128],
        "model": None,
        "thinking": "off",
        "reasoning_effort": None,
        "edit_format": None,
        "map_tokens": 0,
        "architect": False,
        "editor_model": None,
        "test_cmd": None,
    }


def _has_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    """Return True if any keyword matches as a whole word/phrase in text."""
    return any(re.search(rf"\b{re.escape(kw)}\b", text) for kw in keywords)


def classify_task(message: str, files: list[str] = None,
                  read_files: list[str] = None, test_cmd: str = None,
                  forced_tier: str = "auto") -> tuple[str, str, dict]:
    """
    Rule-engine that classifies a task into a precision tier and returns
    (tier_name, reasoning, resolved_params).
    """
    files = list(files or [])
    read_files = list(read_files or [])
    msg = (message or "").lower()

    if forced_tier and forced_tier.lower() != "auto":
        tier = forced_tier.lower()
        if tier not in SUPPORTED_TIERS:
            raise ValueError(
                f"Unsupported tier {forced_tier!r}; expected one of {SUPPORTED_TIERS} or 'auto'."
            )
        reason = f"manual override (--tier {tier})"
        return tier, reason, _resolve_tier_params(tier, test_cmd=test_cmd, message=message)

    # Tier 4: Research (no editable files + research intent or missing test contract)
    if not files and (_has_keyword(msg, RESEARCH_KEYWORDS) or not test_cmd):
        reason = "no editable files with research/lookup intent or no --test-cmd"
        return "research", reason, _resolve_tier_params("research", message=message)

    # Tier 3: Architect (many files or complex/AST/concurrency/security keywords)
    if len(files) > 3 or _has_keyword(msg, ARCHITECT_KEYWORDS):
        reason = f"{len(files)} target file(s) or complex/AST/architecture keywords"
        return "architect", reason, _resolve_tier_params("architect", test_cmd=test_cmd, message=message)

    # Tier 1: Quick patch (single doc/config file or mechanical keywords, no tests)
    if len(files) == 1 and not test_cmd and (
        str(files[0]).lower().endswith(DOC_CONFIG_EXTENSIONS)
        or _has_keyword(msg, PATCH_KEYWORDS)
    ):
        reason = "single doc/config file or mechanical patch keywords"
        return "patch", reason, _resolve_tier_params("patch", test_cmd=test_cmd, message=message)

    # Tier 2: Surgical (default for 1-3 domain files / localized work / active tests)
    reason = f"{len(files)} domain file(s)"
    if test_cmd:
        reason += " with active --test-cmd"
    return "surgical", reason, _resolve_tier_params("surgical", test_cmd=test_cmd, message=message)


def run_dispatch(task_id: str, message: str, files: list[str], cwd: Path,
                 read_files: list[str] = None, test_cmd: str = None,
                 tier: str = "auto", thinking: str = "auto",
                 reasoning_effort: str = None, dry_run: bool = False,
                 worktree: bool = False, auto_merge: bool = True,
                 verbose: bool = False):
    """
    Classify a task into its optimal precision tier and dispatch it to the
    matching model/thinking/edit-format configuration (or DeepSeek search).
    """
    tier_name, reasoning, resolved = classify_task(
        message, files=files, read_files=read_files,
        test_cmd=test_cmd, forced_tier=tier,
    )

    # Explicit CLI overrides win over tier defaults
    if thinking and thinking != "auto":
        resolved["thinking"] = thinking
    if reasoning_effort:
        resolved["reasoning_effort"] = reasoning_effort

    print(
        f"[DISPATCHER] Classified as TIER {resolved['tier_num']}: "
        f"{resolved['tier_label']} (reason: {reasoning})"
    )

    if dry_run:
        print(f"[DISPATCHER] Tier key: {tier_name}")
        print("[DISPATCHER] Resolved parameters:")
        for key in ("model", "thinking", "reasoning_effort", "edit_format",
                    "map_tokens", "architect", "editor_model", "test_cmd",
                    "command", "query"):
            if key in resolved:
                print(f"  {key}: {resolved[key]}")
        print("[STATUS] DRY_RUN")
        sys.exit(0)

    if tier_name == "research":
        run_search(resolved.get("query") or (message or ""), cwd)
        return

    run_turn(
        task_id, message, files, cwd,
        read_files=read_files, restore=False,
        model=resolved.get("model") or "deepseek/deepseek-flash",
        lint_cmd=None,
        test_cmd=resolved.get("test_cmd"),
        dry_run=False,
        architect=resolved.get("architect", False),
        editor_model=resolved.get("editor_model"),
        worktree=worktree, auto_merge=auto_merge, verbose=verbose,
        thinking=resolved.get("thinking", "auto"),
        reasoning_effort=resolved.get("reasoning_effort"),
        map_tokens=resolved.get("map_tokens", 0),
    )


def run_search(query: str, cwd: Path, count: int = 5,
               need_content: bool = False, as_json: bool = False):
    """CLI entry: validate, dispatch to DeepSeek search, map errors to exit codes."""
    load_project_env(cwd)
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("[ERROR] DEEPSEEK_API_KEY missing from project .env and environment.", file=sys.stderr)
        sys.exit(1)

    cleaned = (query or "").strip()
    if not (1 <= len(cleaned) <= 128):
        print(f"[ERROR] Query must be 1-128 characters after strip (got {len(cleaned)}).", file=sys.stderr)
        sys.exit(2)
    try:
        count_int = int(count)
    except (TypeError, ValueError):
        print(f"[ERROR] --count must be an integer between 1 and 50 (got {count!r}).", file=sys.stderr)
        sys.exit(2)
    if not (1 <= count_int <= 50):
        print(f"[ERROR] --count must be between 1 and 50 (got {count_int}).", file=sys.stderr)
        sys.exit(2)

    try:
        data = search_web_deepseek(cleaned, api_key, count=count_int, need_content=need_content)
    except urllib.error.HTTPError as e:
        code = getattr(e, "code", "?")
        if code == 429:
            print("[ERROR] DeepSeek search API rate limit reached (HTTP 429). Retry later.", file=sys.stderr)
            sys.exit(16)
        print(f"[ERROR] DeepSeek search API error (HTTP {code}).", file=sys.stderr)
        sys.exit(17)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"[ERROR] Network error contacting DeepSeek search API: {e}", file=sys.stderr)
        sys.exit(18)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[ERROR] Invalid JSON response from DeepSeek search API: {e}", file=sys.stderr)
        sys.exit(19)

    if as_json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(format_search_results(data, need_content=need_content))


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


def create_ephemeral_model_settings(cwd: Path, model: str, editor_model: str = None,
                                    architect: bool = False, thinking: str = "auto",
                                    reasoning_effort: str = None):
    """
    Ensure deepseek models (e.g. deepseek/deepseek-flash) have explicit
    caches_by_default=True and edit_format=diff configured in Aider settings.
    Optionally emit an editor-model entry with editor-diff and thinking disabled.
    """
    model_is_deepseek = bool(model) and "deepseek" in model.lower()
    editor_is_deepseek = (
        bool(editor_model)
        and "deepseek" in editor_model.lower()
        and editor_model != model
    )
    if not model_is_deepseek and not editor_is_deepseek:
        return None

    lines = []
    if model_is_deepseek:
        lines.append(f"- name: {model}")
        lines.append("  edit_format: diff")
        lines.append("  caches_by_default: true")
        lines.append("  examples_as_sys_msg: true")
        extra = []
        if thinking == "off":
            extra.append("    thinking:\n      type: disabled")
        elif thinking == "on":
            extra.append("    thinking:\n      type: enabled")
        if reasoning_effort:
            extra.append(f"    reasoning_effort: {reasoning_effort}")
        if extra:
            lines.append("  extra_params:")
            lines.extend(extra)

    if editor_is_deepseek:
        lines.append(f"- name: {editor_model}")
        lines.append("  edit_format: editor-diff")
        lines.append("  caches_by_default: true")
        lines.append("  examples_as_sys_msg: true")
        lines.append("  extra_params:")
        lines.append("    thinking:")
        lines.append("      type: disabled")

    settings_path = cwd / f".aider.model.settings.{os.getpid()}.yml"
    settings_path.write_text("\n".join(lines) + "\n")
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
             test_cmd: str = None, dry_run: bool = False,
             architect: bool = False, editor_model: str = None,
             worktree: bool = False, auto_merge: bool = True,
             verbose: bool = False, thinking: str = "auto",
             reasoning_effort: str = None, map_tokens: int = 0):
    if worktree:
        worktree_path, branch_name = create_task_worktree(task_id, cwd)
        env_src = cwd / ".env"
        if env_src.exists():
            shutil.copy(env_src, worktree_path / ".env")
        try:
            run_turn(task_id, message, files, worktree_path,
                     read_files=read_files, restore=restore, model=model,
                     lint_cmd=lint_cmd, test_cmd=test_cmd, dry_run=dry_run,
                     architect=architect, editor_model=editor_model,
                     worktree=False, verbose=verbose,
                     thinking=thinking, reasoning_effort=reasoning_effort,
                     map_tokens=map_tokens)
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
        "--map-tokens", str(map_tokens),
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

    # Resolve the architect editor model early so settings can include it
    resolved_editor_model = editor_model
    if architect and not resolved_editor_model and model == "deepseek/deepseek-reasoner":
        resolved_editor_model = "deepseek/deepseek-flash"

    # Ephemeral model settings for explicit prompt caching and diff format
    model_settings_path = create_ephemeral_model_settings(
        cwd, model, editor_model=resolved_editor_model,
        architect=architect, thinking=thinking, reasoning_effort=reasoning_effort
    )
    if model_settings_path:
        cmd.extend(["--model-settings-file", str(model_settings_path)])

    # Auto-lint / syntax check support
    resolved_lint_cmd = detect_lint_cmd(files, lint_cmd)
    if resolved_lint_cmd:
        cmd.extend(["--auto-lint", "--lint-cmd", resolved_lint_cmd])

    # Closed-loop TDD: let Aider run tests and self-heal failing assertions
    if test_cmd:
        cmd.extend(["--auto-test", "--test-cmd", test_cmd])

    # DeepSeek dynamic thinking / reasoning controls
    if thinking == "off":
        cmd.extend(["--thinking-tokens", "0"])
    if reasoning_effort:
        cmd.extend(["--reasoning-effort", reasoning_effort])

    # Dry-run: propose diffs without writing to disk
    if dry_run:
        cmd.append("--dry-run")

    # Architect mode (two-model: architect + editor)
    if architect:
        cmd.append("--architect")
    if resolved_editor_model:
        cmd.extend(["--editor-model", resolved_editor_model])
    
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
    turn_parser.add_argument("--test-cmd", default=None, help="Test command (enables --auto-test; Aider self-heals failing tests)")
    turn_parser.add_argument("--dry-run", action="store_true", help="Run Aider in dry-run mode (proposes diffs without writing)")
    turn_parser.add_argument("--architect", action="store_true", help="Enable two-model architect mode (--architect)")
    turn_parser.add_argument("--editor-model", default=None, help="Editor model for architect mode (defaults to deepseek/deepseek-flash for reasoner)")
    turn_parser.add_argument("--worktree", action="store_true", help="Run the turn inside an isolated git worktree and branch")
    turn_parser.add_argument("--no-auto-merge", action="store_true", help="Do not auto-merge the worktree on turn completion")
    turn_parser.add_argument("-v", "--verbose", action="store_true", help="Include full thinking traces and raw startup banners in output")
    turn_parser.add_argument("--thinking", default="auto", choices=["auto", "on", "off"], help="DeepSeek thinking mode (default: auto)")
    turn_parser.add_argument("--reasoning-effort", default=None, choices=["low", "high", "max"], help="DeepSeek reasoning effort level")
    turn_parser.add_argument("--map-tokens", type=int, default=0, help="Aider repo-map token budget (default: 0)")
    turn_parser.add_argument("--cwd", default=".", help="Project working directory")

    dispatch_parser = subparsers.add_parser("dispatch", help="Auto-classify and dispatch a task to the optimal tier")
    dispatch_parser.add_argument("--task-id", required=True, help="Unique identifier for the task session")
    dispatch_parser.add_argument("--message", required=True, help="Task specification / research query")
    dispatch_parser.add_argument("--files", nargs="*", default=[], help="Editable target files")
    dispatch_parser.add_argument("--read-files", nargs="*", default=[], help="Read-only reference files")
    dispatch_parser.add_argument("--test-cmd", default=None, help="Test command enabling closed-loop TDD")
    dispatch_parser.add_argument("--tier", default="auto", choices=["auto", "patch", "surgical", "architect", "research"], help="Force a tier (default: auto)")
    dispatch_parser.add_argument("--thinking", default="auto", choices=["auto", "on", "off"], help="Override thinking mode")
    dispatch_parser.add_argument("--reasoning-effort", default=None, choices=["low", "high", "max"], help="Override reasoning effort")
    dispatch_parser.add_argument("--dry-run", action="store_true", help="Print classification + resolved params without executing")
    dispatch_parser.add_argument("--worktree", action="store_true", help="Run inside an isolated git worktree")
    dispatch_parser.add_argument("--no-auto-merge", action="store_true", help="Do not auto-merge the worktree on completion")
    dispatch_parser.add_argument("-v", "--verbose", action="store_true", help="Include full thinking traces in output")
    dispatch_parser.add_argument("--cwd", default=".", help="Project working directory")
    
    clean_parser = subparsers.add_parser("cleanup", help="Clean up session history file")
    clean_parser.add_argument("--task-id", required=True, help="Task identifier to clean up")
    clean_parser.add_argument("--cwd", default=".", help="Project working directory")

    merge_parser = subparsers.add_parser("merge-worktree", help="Merge isolated task worktree branch and clean up worktree")
    merge_parser.add_argument("--task-id", required=True, help="Task identifier whose worktree should be merged")
    merge_parser.add_argument("--cwd", default=".", help="Project working directory")

    log_parser = subparsers.add_parser("compact-log", help="Strip ANSI codes and print the tail of a log")
    log_parser.add_argument("--log-file", required=True, help="Path to the log file to compact")
    log_parser.add_argument("--max-lines", type=int, default=40, help="Trailing non-empty lines to keep (default: 40)")
    
    search_parser = subparsers.add_parser("search", help="Run a web search via DeepSeek API")
    search_parser.add_argument("query", help="Search query (1-128 characters after strip)")
    search_parser.add_argument("--count", type=int, default=5, help="Number of results (1-50, default 5)")
    search_parser.add_argument("--need-content", action="store_true",
                               help="Include full page content in results")
    search_parser.add_argument("--json", action="store_true",
                               help="Emit raw JSON response instead of formatted text")
    search_parser.add_argument("--cwd", default=".", help="Project working directory")

    args = parser.parse_args()
    cwd_path = Path(getattr(args, "cwd", ".")).resolve()
    
    if args.command == "turn":
        run_turn(args.task_id, args.message, args.files, cwd_path, 
                 read_files=args.read_files, restore=args.restore, model=args.model,
                 lint_cmd=args.lint_cmd, test_cmd=args.test_cmd,
                 dry_run=args.dry_run, architect=args.architect,
                 editor_model=args.editor_model, worktree=args.worktree,
                 auto_merge=not args.no_auto_merge, verbose=args.verbose,
                 thinking=args.thinking, reasoning_effort=args.reasoning_effort,
                 map_tokens=args.map_tokens)
    elif args.command == "dispatch":
        run_dispatch(args.task_id, args.message, args.files, cwd_path,
                     read_files=args.read_files, test_cmd=args.test_cmd,
                     tier=args.tier, thinking=args.thinking,
                     reasoning_effort=args.reasoning_effort,
                     dry_run=args.dry_run, worktree=args.worktree,
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
    elif args.command == "search":
        run_search(args.query, cwd_path, count=args.count,
                   need_content=args.need_content, as_json=args.json)


if __name__ == "__main__":
    main()
