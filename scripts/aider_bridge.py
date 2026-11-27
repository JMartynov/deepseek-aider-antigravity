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
import http.client
import shutil
import subprocess
import tempfile
import urllib.request
import urllib.error
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from contextlib import contextmanager


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
THINKING_BLOCK_RE = re.compile(r"► \*\*THINKING\*\*.*?► \*\*ANSWER\*\*", re.DOTALL)
_TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_BEARER_RE = re.compile(r"(Bearer\s+)\S+", re.IGNORECASE)
_SK_TOKEN_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")


def validate_task_id(task_id: str) -> str:
    """Validate that task_id is a safe single-segment identifier for paths and git branch names."""
    if not isinstance(task_id, str) or not _TASK_ID_RE.match(task_id):
        raise ValueError(
            f"Invalid task_id {task_id!r}: must match ^[A-Za-z0-9][A-Za-z0-9._-]{{0,127}}$"
        )
    if ".." in task_id or task_id.endswith(".lock") or task_id.endswith("."):
        raise ValueError(f"Invalid task_id {task_id!r}: unsafe path or git ref sequence.")
    return task_id


def _sanitize_secret(text: str, secret: str = None) -> str:
    """Redact API keys and bearer tokens from diagnostic or error strings."""
    if not text:
        return ""
    out = str(text)
    for tok in (secret, os.environ.get("DEEPSEEK_API_KEY")):
        if tok:
            out = out.replace(tok, "[REDACTED]")
    out = _BEARER_RE.sub(r"\1[REDACTED]", out)
    out = _SK_TOKEN_RE.sub("[REDACTED]", out)
    return out


def _write_state_atomic(state_file: Path, state: dict):
    """Write session state.json atomically via temp file + os.replace."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state, indent=2)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(state_file.parent), prefix=".state.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
        os.replace(tmp_path, state_file)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


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
        return {"error": _sanitize_secret(str(e), api_key)}
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
    if timeout is not None and timeout <= 0:
        raise ValueError(f"timeout must be positive (got {timeout!r})")
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
            if hasattr(e, "close"):
                e.close()
            return _search_via_chat_completions(query, api_key, count, need_content, timeout)
        if getattr(e, "msg", None):
            e.msg = _sanitize_secret(str(e.msg), api_key)
        raise
    except urllib.error.URLError as e:
        if getattr(e, "reason", None):
            e.reason = _sanitize_secret(str(e.reason), api_key)
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
ARCHITECT_AUTO_EDITOR_MODELS = frozenset({
    "deepseek/deepseek-reasoner",
    "deepseek/deepseek-v4-pro",
})


def _resolve_editor_model(model: str, architect: bool = False,
                          editor_model: str = None):
    """Resolve the architect editor model, defaulting to flash for high-reasoning models."""
    if editor_model:
        return editor_model
    if architect and model in ARCHITECT_AUTO_EDITOR_MODELS:
        return "deepseek/deepseek-flash"
    return None

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

AUTO_SUPPRESS_PHRASES = (
    "should i proceed",
    "would you like me to",
    "should i format docstrings",
    "shall i proceed",
    "do you want me to",
    "should i continue",
    "ready to proceed",
    "may i proceed",
)

ESCALATE_CATEGORIES = frozenset({
    "ARCHITECTURE", "SCHEMA", "API_CONTRACT", "CONCURRENCY", "TRADEOFF",
    "BREAKING_CHANGE", "SECURITY", "UNRESOLVABLE_FAILURE",
    "UNRESOLVED_FAILURE", "AMBIGUITY",
})


def _resolve_tier_params(tier: str, test_cmd: str = None, message: str = "",
                         files: list[str] = None, auto_scope: bool = False) -> dict:
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
            "plan_align": False,
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
            "map_tokens": 0 if (files and not auto_scope) else 512,
            "architect": False,
            "editor_model": None,
            "plan_align": False,
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
            "plan_align": False,
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
        "plan_align": False,
        "test_cmd": None,
    }


def _has_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    """Return True if any keyword matches as a whole word/phrase in text."""
    return any(re.search(rf"\b{re.escape(kw)}\b", text) for kw in keywords)


def classify_task(message: str, files: list[str] = None,
                  read_files: list[str] = None, test_cmd: str = None,
                  forced_tier: str = "auto", auto_scope: bool = False) -> tuple[str, str, dict]:
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
        return tier, reason, _resolve_tier_params(tier, test_cmd=test_cmd, message=message, files=files, auto_scope=auto_scope)

    # Tier 4: Research (no editable files + research intent or missing test contract)
    if not files and (_has_keyword(msg, RESEARCH_KEYWORDS) or not test_cmd):
        reason = "no editable files with research/lookup intent or no --test-cmd"
        return "research", reason, _resolve_tier_params("research", message=message, files=files, auto_scope=auto_scope)

    # Doc/config-only edits route to quick patch before architectural escalation
    if files and not test_cmd and all(Path(f).suffix.lower() in DOC_CONFIG_EXTENSIONS for f in files):
        reason = f"{len(files)} doc/config file(s) with no --test-cmd"
        return "patch", reason, _resolve_tier_params("patch", test_cmd=test_cmd, message=message, files=files, auto_scope=auto_scope)

    # Tier 3: Architect (many files or complex/AST/concurrency/security keywords)
    if len(files) > 5 or _has_keyword(msg, ARCHITECT_KEYWORDS):
        reason = f"{len(files)} target file(s) or complex/AST/architecture keywords"
        return "architect", reason, _resolve_tier_params("architect", test_cmd=test_cmd, message=message, files=files, auto_scope=auto_scope)

    # Tier 1: Quick patch (single file with mechanical keywords, no tests)
    if len(files) == 1 and not test_cmd and _has_keyword(msg, PATCH_KEYWORDS):
        reason = "single file with mechanical patch keywords"
        return "patch", reason, _resolve_tier_params("patch", test_cmd=test_cmd, message=message, files=files, auto_scope=auto_scope)

    # Tier 2: Surgical (default for 1-5 domain files / localized work / active tests)
    reason = f"{len(files)} domain file(s)"
    if test_cmd:
        reason += " with active --test-cmd"
    return "surgical", reason, _resolve_tier_params(
        "surgical", test_cmd=test_cmd, message=message,
        files=files, auto_scope=auto_scope,
    )


def run_dispatch(task_id: str, message: str, files: list[str], cwd: Path,
                 read_files: list[str] = None, test_cmd: str = None,
                 lint_cmd: str = None, model: str = None,
                 editor_model: str = None, map_tokens: int = None,
                 architect: bool = False,
                 tier: str = "auto", thinking: str = "auto",
                 reasoning_effort: str = None, dry_run: bool = False,
                 worktree: bool = False, auto_merge: bool = True,
                 verbose: bool = False, plan_align: bool = False,
                 restore: bool = False, count: int = 5,
                 need_content: bool = False, as_json: bool = False,
                 auto_scope: bool = False,
                 max_context_kb: int = 128,
                 allow_large_context: bool = False):
    """
    Classify a task into its optimal precision tier and dispatch it to the
    matching model/thinking/edit-format configuration (or DeepSeek search).
    """
    try:
        validate_task_id(task_id)
    except ValueError as e:
        print(f"[ERROR] Security violation: {e}", file=sys.stderr)
        sys.exit(12)

    auto_scoped = bool(auto_scope and not files)
    if auto_scope and not files:
        scope_result = run_scope(message=message, cwd=cwd)
        if (scope_result or {}).get("confidence") == "low" or not (scope_result or {}).get("editable_files"):
            print(
                "[DISPATCHER] Auto-scope returned low confidence (no editable files); "
                "escalate to Antigravity research subagent."
            )
            sys.exit(14)
        files = list(scope_result["editable_files"])
        merged_read = list(read_files or [])
        for rf in (scope_result.get("read_files") or []):
            if rf not in files and rf not in merged_read:
                merged_read.append(rf)
        read_files = merged_read
        if not test_cmd and scope_result.get("suggested_test_cmd"):
            test_cmd = scope_result["suggested_test_cmd"]

    tier_name, reasoning, resolved = classify_task(
        message, files=files, read_files=read_files,
        test_cmd=test_cmd, forced_tier=tier, auto_scope=auto_scoped,
    )

    # Explicit CLI overrides win over tier defaults
    if model:
        resolved["model"] = model
    if architect:
        resolved["architect"] = True
    if editor_model:
        resolved["editor_model"] = editor_model
    if resolved.get("architect") and not resolved.get("editor_model"):
        auto_ed = _resolve_editor_model(
            resolved.get("model"),
            architect=True,
            editor_model=None,
        )
        if auto_ed:
            resolved["editor_model"] = auto_ed
    if map_tokens is not None:
        resolved["map_tokens"] = map_tokens
    if lint_cmd is not None:
        resolved["lint_cmd"] = lint_cmd
    else:
        resolved.setdefault("lint_cmd", None)
    if thinking and thinking != "auto":
        resolved["thinking"] = thinking
    if reasoning_effort:
        resolved["reasoning_effort"] = reasoning_effort
    effective_plan_align = plan_align or (bool(resolved.get("plan_align", False)) and not restore)
    resolved["plan_align"] = effective_plan_align
    if tier_name == "research":
        resolved["count"] = count
        resolved["need_content"] = need_content
        resolved["as_json"] = as_json

    print(
        f"[DISPATCHER] Classified as TIER {resolved['tier_num']}: "
        f"{resolved['tier_label']} (reason: {reasoning})"
    )

    if dry_run:
        print(f"[DISPATCHER] Tier key: {tier_name}")
        print("[DISPATCHER] Resolved parameters:")
        for key in ("model", "thinking", "reasoning_effort", "edit_format",
                    "map_tokens", "architect", "editor_model", "plan_align",
                    "test_cmd", "lint_cmd", "command", "query",
                    "count", "need_content", "as_json"):
            if key in resolved:
                print(f"  {key}: {resolved[key]}")
        print("[STATUS] DRY_RUN")
        sys.exit(0)

    if tier_name == "research":
        run_search(
            resolved.get("query") or (message or ""),
            cwd,
            count=count,
            need_content=need_content,
            as_json=as_json,
        )
        return

    run_turn(
        task_id, message, files, cwd,
        read_files=read_files, restore=restore,
        model=resolved.get("model") or "deepseek/deepseek-flash",
        lint_cmd=resolved.get("lint_cmd"),
        test_cmd=resolved.get("test_cmd"),
        dry_run=False,
        architect=resolved.get("architect", False),
        editor_model=resolved.get("editor_model"),
        worktree=worktree, auto_merge=auto_merge, verbose=verbose,
        thinking=resolved.get("thinking", "auto"),
        reasoning_effort=resolved.get("reasoning_effort"),
        map_tokens=resolved.get("map_tokens", 0),
        plan_align=effective_plan_align,
        tier=tier_name,
        max_context_kb=max_context_kb,
        allow_large_context=allow_large_context,
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
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as e:
        print(
            f"[ERROR] Network error contacting DeepSeek search API: {_sanitize_secret(str(e), api_key)}",
            file=sys.stderr,
        )
        sys.exit(18)
    except (json.JSONDecodeError, ValueError) as e:
        print(
            f"[ERROR] Invalid JSON response from DeepSeek search API: {_sanitize_secret(str(e), api_key)}",
            file=sys.stderr,
        )
        sys.exit(19)

    if as_json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(format_search_results(data, need_content=need_content))


def get_session_dir(task_id: str, cwd: Path) -> Path:
    """Return the structured session directory for a task."""
    validate_task_id(task_id)
    return cwd / ".aider-sessions" / task_id


def _parse_env_value(raw: str) -> str:
    """Parse a single .env value, handling matching quotes and trailing comments."""
    val = raw.strip()
    if not val:
        return ""
    if val[0] in ("'", '"'):
        quote = val[0]
        i = 1
        while i < len(val):
            if quote == '"' and val[i] == "\\" and i + 1 < len(val):
                i += 2
                continue
            if val[i] == quote:
                inner = val[1:i]
                if quote == '"':
                    inner = inner.replace('\\"', '"').replace("\\\\", "\\")
                return inner
            i += 1
        return val[1:]
    return re.sub(r"\s+#.*$", "", val).strip()


def load_project_env(cwd: Path):
    """Load environment variables from project .env file (without overwriting existing env)."""
    env_file = cwd / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = re.sub(r"^export\s+", "", k.strip()).strip()
                if not k or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k):
                    continue
                parsed_v = _parse_env_value(v)
                if k not in os.environ:
                    os.environ[k] = parsed_v


def _parse_token_count(raw: str) -> float:
    """Convert an Aider token count string (e.g. '2.5k', '150', '1.2M') to float."""
    if not raw:
        return 0.0
    s = raw.strip()
    mult = 1.0
    if s[-1] in ("k", "K"):
        mult = 1000.0
        s = s[:-1]
    elif s[-1] in ("m", "M"):
        mult = 1_000_000.0
        s = s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return 0.0


def _format_token_count(val: float) -> str:
    """Format a summed token count using Aider-style suffixes."""
    if val >= 1_000_000:
        return f"{val / 1_000_000:.1f}M"
    if val >= 1000:
        return f"{val / 1000:.1f}k"
    return str(int(round(val)))


def parse_cost_and_tokens(output: str):
    """Extract token count (including cache writes/hits) and cost from Aider output.

    In multi-call turns (e.g. --architect mode), sums per-message costs and tokens
    and uses the final cumulative session cost.
    """
    from decimal import Decimal, InvalidOperation

    token_matches = re.findall(
        r"Tokens:\s*([0-9.]+[kKmM]?)\s*sent"
        r"(?:,\s*([0-9.]+[kKmM]?)\s*cache write)?"
        r"(?:,\s*([0-9.]+[kKmM]?)\s*cache hit)?"
        r",\s*([0-9.]+[kKmM]?)\s*received",
        output,
    )
    cost_matches = re.findall(r"Cost:\s*\$([0-9.]+)\s*message,\s*\$([0-9.]+)\s*session", output)

    if not token_matches:
        tokens_sent = tokens_rcvd = "unknown"
        cache_write = cache_hit = "0"
    elif len(token_matches) == 1:
        m = token_matches[0]
        tokens_sent = m[0]
        cache_write = m[1] or "0"
        cache_hit = m[2] or "0"
        tokens_rcvd = m[3]
    else:
        tokens_sent = _format_token_count(sum(_parse_token_count(m[0]) for m in token_matches))
        cw_sum = sum(_parse_token_count(m[1]) for m in token_matches if m[1])
        ch_sum = sum(_parse_token_count(m[2]) for m in token_matches if m[2])
        cache_write = _format_token_count(cw_sum) if cw_sum > 0 else "0"
        cache_hit = _format_token_count(ch_sum) if ch_sum > 0 else "0"
        tokens_rcvd = _format_token_count(sum(_parse_token_count(m[3]) for m in token_matches))

    if not cost_matches:
        cost_msg = cost_session = "unknown"
    elif len(cost_matches) == 1:
        cost_msg, cost_session = cost_matches[0]
    else:
        try:
            total_msg = sum(Decimal(m[0]) for m in cost_matches)
            cost_msg = format(total_msg, "f")
        except InvalidOperation:
            cost_msg = cost_matches[-1][0]
        cost_session = cost_matches[-1][1]

    return tokens_sent, cache_write, cache_hit, tokens_rcvd, cost_msg, cost_session


_CONTINUATION_TAG_RE = re.compile(
    r"^\[(?:OPTIONS?|RECOMMENDATION|STAGE|CATEGORY|IMPACT):", re.IGNORECASE
)
_AIDER_FOOTER_RE = re.compile(r"^(?:Tokens:\s*[0-9]|Cost:\s*\$|Applied edit to )")
_OPTION_LINE_RE = re.compile(r"^(?:[A-Z]\)\s+|Option\s+[A-Z]\b)", re.IGNORECASE)


def _should_continue_after_blank(lines: list[str], idx: int, q_indent: int, in_options: bool) -> bool:
    """Return True if a blank line inside a QUESTION block should not terminate it."""
    for j in range(idx + 1, len(lines)):
        nxt = lines[j]
        nxt_trimmed = nxt.strip()
        if not nxt_trimmed:
            continue
        if re.match(r"^\s*QUESTION:\s*", nxt):
            return False
        if _AIDER_FOOTER_RE.match(nxt_trimmed):
            return False
        if nxt_trimmed.startswith(("@@", "diff ", "<<<<<<<", "=======", ">>>>>>>")):
            return False
        if in_options:
            return True
        if _CONTINUATION_TAG_RE.match(nxt_trimmed):
            return True
        if nxt_trimmed.startswith(("- ", "* ")):
            return True
        if _OPTION_LINE_RE.match(nxt_trimmed):
            return True
        nxt_indent = len(nxt) - len(nxt.lstrip())
        if nxt_indent > q_indent:
            return True
        return False
    return False


def extract_questions(output: str):
    """
    Extract questions while preventing false positives from code diffs or commit messages.
    Preserves multi-line structured QUESTION blocks (blank lines, markdown bullets,
    [OPTIONS: ...], [RECOMMENDATION: ...]) while ignoring standalone diffs and fences.
    """
    output = strip_thinking_block(output)
    lines = output.splitlines()
    questions = []
    seen_questions = set()
    current_q = None
    q_indent = 0
    fence_len = None
    options_bracket_depth = 0

    def _update_options_state(trimmed_line: str):
        nonlocal options_bracket_depth
        upper = trimmed_line.upper()
        opt_pos = upper.find("[OPTIONS:")
        if opt_pos == -1:
            opt_pos = upper.find("[OPTION:")
        if opt_pos != -1:
            depth = 0
            for ch in trimmed_line[opt_pos:]:
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth = max(0, depth - 1)
            options_bracket_depth = depth
        elif options_bracket_depth > 0:
            for ch in trimmed_line:
                if ch == "[":
                    options_bracket_depth += 1
                elif ch == "]":
                    options_bracket_depth -= 1
                    if options_bracket_depth <= 0:
                        options_bracket_depth = 0
                        break

    def _flush_question():
        nonlocal current_q, options_bracket_depth
        if current_q is not None:
            joined = "\n".join(current_q).strip()
            if joined:
                norm = re.sub(r"\s+", " ", joined).strip().lower()
                if norm not in seen_questions:
                    seen_questions.add(norm)
                    questions.append(joined)
        current_q = None
        options_bracket_depth = 0

    for i, line in enumerate(lines):
        trimmed = line.strip()
        fence_match = re.match(r"^(`{3,})", trimmed)

        if current_q is None:
            if fence_match:
                ticks = fence_match.group(1)
                if fence_len is None:
                    fence_len = len(ticks)
                elif len(ticks) >= fence_len and trimmed == ticks:
                    fence_len = None
                continue
            if fence_len is not None:
                continue
            if trimmed.startswith(("+", "-", "@@", "diff ", "---", "+++", "* ", "<<<<<<<", "=======", ">>>>>>>")):
                continue
            q_match = re.match(r"^(\s*)QUESTION:\s*(.*)$", line)
            if q_match:
                q_indent = len(q_match.group(1))
                first_part = q_match.group(2).rstrip()
                current_q = [first_part] if first_part else []
                _update_options_state(trimmed)
            continue

        # Inside an active QUESTION block
        if fence_match:
            ticks = fence_match.group(1)
            if fence_len is None:
                fence_len = len(ticks)
            elif len(ticks) >= fence_len and trimmed == ticks:
                fence_len = None
            current_q.append(line.rstrip())
            continue

        if fence_len is not None:
            current_q.append(line.rstrip())
            continue

        q_match = re.match(r"^(\s*)QUESTION:\s*(.*)$", line)
        if q_match:
            _flush_question()
            q_indent = len(q_match.group(1))
            first_part = q_match.group(2).rstrip()
            current_q = [first_part] if first_part else []
            _update_options_state(trimmed)
            continue

        if trimmed.startswith(("@@", "diff ", "<<<<<<<", "=======", ">>>>>>>")) or _AIDER_FOOTER_RE.match(trimmed):
            _flush_question()
            continue

        if not trimmed:
            if _should_continue_after_blank(lines, i, q_indent, options_bracket_depth > 0):
                current_q.append("")
            else:
                _flush_question()
            continue

        current_q.append(line.rstrip())
        _update_options_state(trimmed)

    _flush_question()
    return questions


def _find_bracket_tag(text: str, tag_pattern: str):
    """Find '[TAG: ...]' with balanced inner square brackets (e.g., dict[str, int])."""
    m = re.search(rf"\[(?:{tag_pattern}):", text, re.IGNORECASE)
    if not m:
        return None
    start = m.start()
    content_start = m.end()
    depth = 1
    for pos in range(content_start, len(text)):
        ch = text[pos]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return start, pos + 1, text[content_start:pos]
    return None


def parse_structured_question(raw_question: str, seq: int = 1) -> dict:
    """Parse a structured QUESTION block into a structured event dict."""
    cleaned = (raw_question or "").strip()

    stage_tag = _find_bracket_tag(cleaned, r"STAGE")
    stage = stage_tag[2].strip().upper() if stage_tag else "IMPLEMENTATION"

    category_tag = _find_bracket_tag(cleaned, r"CATEGORY")
    category = category_tag[2].strip().upper() if category_tag else None

    impact_tag = _find_bracket_tag(cleaned, r"IMPACT")
    impact = impact_tag[2].strip().upper() if impact_tag else None

    rec_tag = _find_bracket_tag(cleaned, r"RECOMMENDATION")
    recommendation = rec_tag[2].strip() if rec_tag else None

    options = []
    opt_tag = _find_bracket_tag(cleaned, r"OPTIONS?")
    if opt_tag:
        for m in re.finditer(
            r"([A-Z])\)\s*(.+?)(?=(?:\s+[A-Z]\)\s|\n|\Z))",
            opt_tag[2],
        ):
            options.append({"key": m.group(1), "summary": m.group(2).strip()})

    body = cleaned
    for tag in (r"OPTIONS?", r"STAGE", r"CATEGORY", r"IMPACT", r"RECOMMENDATION"):
        while True:
            found = _find_bracket_tag(body, tag)
            if not found:
                break
            body = body[:found[0]] + body[found[1]:]
    body = body.strip()

    return {
        "seq": seq,
        "qid": f"q_{uuid.uuid4().hex[:8]}",
        "type": "question",
        "stage": stage,
        "category": category,
        "impact": impact,
        "blocking": True,
        "question": body or cleaned,
        "options": options,
        "recommendation": recommendation,
        "suppressed": False,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def evaluate_escalation_threshold(question_event) -> str:
    """Return 'AUTO_SUPPRESS' or 'ESCALATE_PRINCIPAL' for a parsed question event."""
    if isinstance(question_event, str):
        text = question_event
        stage = category = impact = None
    else:
        text = question_event.get("question") or ""
        stage = (question_event.get("stage") or "").upper() or None
        category = (question_event.get("category") or "").upper() or None
        impact = (question_event.get("impact") or "").upper() or None

    lower = text.lower()

    if stage == "PLANNING":
        return "ESCALATE_PRINCIPAL"
    if category and category in ESCALATE_CATEGORIES:
        return "ESCALATE_PRINCIPAL"
    if impact == "HIGH":
        return "ESCALATE_PRINCIPAL"

    if impact == "LOW":
        return "AUTO_SUPPRESS"
    for phrase in AUTO_SUPPRESS_PHRASES:
        if phrase in lower:
            return "AUTO_SUPPRESS"

    if text.strip():
        return "ESCALATE_PRINCIPAL"
    return "AUTO_SUPPRESS"


def check_diff_applied_successfully(output: str):
    """Detect if Aider encountered a search/replace diff failure."""
    # Strip thinking blocks to avoid matching internal reasoning traces
    clean_output = re.sub(r"► \*\*THINKING\*\*.*?► \*\*ANSWER\*\*", "", output, flags=re.DOTALL)
    failure_patterns = [
        r"Failed to apply edit to",
        r"The search block did not match",
        r"did not find the search block"
    ]
    failure_patterns.extend([
        r'Only 3 reflections allowed',
        r"codec can't decode byte",
    ])
    for pattern in failure_patterns:
        if re.search(pattern, clean_output, re.IGNORECASE):
            if 'reflections' in pattern and re.search(r'Applied edit to\s+', clean_output):
                continue
            return False
    return True


def create_ephemeral_aiderignore(cwd: Path, scope_mode: bool = False,
                                 explicit_files: Optional[List[str]] = None,
                                 max_unrequested_kb: int = 100):
    """
    Create an ephemeral .aiderignore shield.

    Covers secrets and cache artifacts, hides data/CSV/Parquet/JSON payloads and
    lockfiles, and re-exempts explicitly requested files via sorted '!' negations.
    A dynamic shield additionally hides unrequested files larger than
    ``max_unrequested_kb`` KB to keep the prompt context (and cost) bounded.
    """
    ignore_path = cwd / ".aiderignore.ephemeral"
    cwd_resolved = Path(cwd).resolve()

    # Normalize explicitly requested files to POSIX repo-relative paths.
    explicit_set = set()
    for ef in (explicit_files or []):
        if not ef:
            continue
        raw = str(ef)
        try:
            rel = (cwd_resolved / raw).resolve().relative_to(cwd_resolved).as_posix()
        except (ValueError, OSError):
            rel = _clean_candidate_path(raw)
        if rel:
            explicit_set.add(rel)

    patterns = [
        ".env*",
        "*.pem",
        "*.key",
        "*.gz",
        "*.zip",
        "*.tar",
        "*.tgz",
        "*.pickle",
        "*.pkl",
        "*.bin",
        "__pycache__/",
        "*.pyc",
        ".pytest_cache/",
        ".cache/",
        "node_modules/",
        ".aider-sessions/",
        ".worktrees/",
        "artifacts/",
        "passports/",
        "advisories/",
    ]

    # Withhold bulk data dirs only when no explicitly requested file lives there.
    if not any(p == "docs/data" or p.startswith("docs/data/") for p in explicit_set):
        patterns.append("docs/data/")
    if not any(p == "data" or p.startswith("data/") for p in explicit_set):
        patterns.append("data/")

    for lock in ("uv.lock", "package-lock.json", "pnpm-lock.yaml",
                 "yarn.lock", "web/catalog.json"):
        if lock not in explicit_set:
            patterns.append(lock)

    patterns.extend(["*.csv", "*.parquet", "*.json"])

    # Explicit exemptions must land after the wildcards that would swallow them.
    negations = sorted(
        f"!{rel}" for rel in explicit_set
        if rel.lower().endswith((".csv", ".parquet", ".json"))
    )
    patterns.extend(negations)

    # Dynamic large-file shield for unrequested payloads.
    seen_dynamic = set()
    walk_skip = {".git", ".venv", ".worktrees", "node_modules"}
    for root, dirnames, filenames in os.walk(str(cwd_resolved)):
        dirnames[:] = [d for d in dirnames if d not in walk_skip]
        for fname in filenames:
            fpath = Path(root) / fname
            try:
                if fpath.stat().st_size <= max_unrequested_kb * 1024:
                    continue
                rel = fpath.resolve().relative_to(cwd_resolved).as_posix()
            except (OSError, ValueError):
                continue
            if rel in explicit_set or rel in seen_dynamic:
                continue
            seen_dynamic.add(rel)
            patterns.append(rel)

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
                                    reasoning_effort: str = None, edit_format: str = "diff"):
    """
    Ensure deepseek models (e.g. deepseek/deepseek-flash) have explicit
    caches_by_default=True and edit_format=diff configured in Aider settings.
    Optionally emit an editor-model entry with editor-diff and thinking disabled.
    """
    resolved_editor = _resolve_editor_model(model, architect=architect, editor_model=editor_model)
    model_is_deepseek = bool(model) and "deepseek" in model.lower()
    editor_is_deepseek = (
        bool(resolved_editor)
        and "deepseek" in resolved_editor.lower()
        and resolved_editor != model
    )
    if not model_is_deepseek and not editor_is_deepseek:
        return None

    lines = []
    if model_is_deepseek:
        lines.append(f"- name: {model}")
        if edit_format is not None:
            lines.append(f"  edit_format: {edit_format}")
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
        lines.append(f"- name: {resolved_editor}")
        lines.append("  edit_format: editor-diff")
        lines.append("  caches_by_default: true")
        lines.append("  examples_as_sys_msg: true")
        lines.append("  extra_params:")
        lines.append("    thinking:")
        lines.append("      type: disabled")

    settings_path = cwd / f".aider.model.settings.{os.getpid()}.yml"
    settings_path.write_text("\n".join(lines) + "\n")
    return settings_path


_CONTEXT_ALL_FILES_RE = re.compile(r"^\s*#*\s*ALL\s+files?\b", re.IGNORECASE)
_CONTEXT_OTHER_FILES_RE = re.compile(
    r"^\s*#*\s*Relevant\s+symbols\s+from\s+OTHER\s+files\b", re.IGNORECASE
)
_PATH_TOKEN_RE = re.compile(r"(?:\./)?[A-Za-z0-9_\-./]+\.[A-Za-z0-9]+")
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")
_IDENT_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*(?:\s*\([^)]*\))?"
)
_BACKTICK_SYMBOL_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
)
_SYMBOL_FILE_EXTS = (
    ".py", ".md", ".json", ".toml", ".yaml", ".yml", ".txt", ".rst", ".ini",
    ".cfg", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".c", ".cpp",
    ".h", ".sh", ".sql",
)
_CONTEXT_STOPWORDS = frozenset({
    "a", "an", "and", "or", "the", "is", "are", "was", "were", "be", "to",
    "of", "in", "on", "for", "with", "by", "as", "at", "from", "that", "this",
    "these", "those", "it", "its", "if", "then", "else", "not", "no", "yes",
    "method", "methods", "class", "classes", "function", "functions", "def",
    "return", "import", "file", "files", "symbol", "symbols", "relevant",
    "other", "all", "use", "used", "uses", "using", "call", "called", "calls",
    "see", "also", "may", "can", "will", "should", "would", "could", "do",
    "does", "did", "has", "have", "had", "here", "there", "where", "when",
    "which", "who", "what", "how", "why", "example", "examples", "note",
    "notes", "test", "tests", "add", "adds", "update", "updates", "run",
    "runs", "new", "old", "some", "any", "one", "two",
})

_DISCOVERY_SKIP_DIRS = frozenset({
    ".git", ".venv", "venv", ".worktrees", "node_modules", "__pycache__",
    ".pytest_cache", ".cache", ".aider-sessions", "artifacts", "passports",
    "advisories", "dist", "build",
})
_DISCOVERY_SCAN_EXTS = (
    ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java",
    ".c", ".cpp", ".h", ".md", ".toml", ".yaml", ".yml",
)
_MAX_SCAN_BYTES = 256 * 1024


def _clean_candidate_path(raw: str) -> str:
    """Normalize a candidate path token: strip quotes/backticks, leading './', trailing punctuation."""
    cand = (raw or "").strip().strip("`'\"")
    cand = cand.strip()
    while cand.startswith("./"):
        cand = cand[2:]
    cand = cand.rstrip(":,.;) ").strip()
    return cand


def parse_context_output(output: str, cwd: Path) -> dict:
    """Parse Aider --chat-mode context output into editable files and relevant symbols."""
    clean = ANSI_ESCAPE_RE.sub("", strip_thinking_block(output or ""))
    cwd_resolved = cwd.resolve()
    editable_files = []
    seen_files = set()
    relevant_symbols = []
    seen_symbols = set()

    def _add_file(candidate: str):
        cand = _clean_candidate_path(candidate)
        if not cand:
            return
        if os.path.isabs(cand) or ".." in Path(cand).parts:
            return
        resolved = (cwd / cand).resolve()
        if resolved == cwd_resolved:
            return
        try:
            if not resolved.is_relative_to(cwd_resolved):
                return
        except (ValueError, AttributeError):
            return
        if not resolved.is_file():
            return
        rel = resolved.relative_to(cwd_resolved).as_posix()
        if rel in seen_files:
            return
        seen_files.add(rel)
        editable_files.append(rel)

    def _add_symbol(name: str):
        sym = (name or "").strip()
        if not sym or sym in seen_symbols:
            return
        seen_symbols.add(sym)
        relevant_symbols.append(sym)

    in_other = False
    for line in clean.splitlines():
        if _CONTEXT_OTHER_FILES_RE.match(line):
            in_other = True
            continue
        if _CONTEXT_ALL_FILES_RE.match(line):
            in_other = False
            continue

        for bt in _BACKTICK_RE.findall(line):
            candidate = bt.strip()
            if not candidate:
                continue
            if "/" in candidate or candidate.lower().endswith(_SYMBOL_FILE_EXTS):
                if not in_other:
                    _add_file(candidate)
                continue
            sym = candidate
            for suffix in ("(...)", "()"):
                if sym.endswith(suffix):
                    sym = sym[:-len(suffix)]
                    break
            sym = sym.strip()
            if _BACKTICK_SYMBOL_RE.match(sym):
                _add_symbol(sym)

        if not in_other:
            for tok in _PATH_TOKEN_RE.findall(line):
                _add_file(tok)

        if "`" in line:
            continue
        bullet = _BULLET_RE.match(line)
        if not bullet:
            continue
        body = bullet.group(1)
        for m in _IDENT_TOKEN_RE.finditer(body):
            tok = m.group(0)
            start, end = m.span()
            if (end < len(body) and body[end] == "/") or (start > 0 and body[start - 1] == "/"):
                continue
            has_parens = "(" in tok
            name = tok.split("(")[0].strip()
            if not name:
                continue
            if "/" in name or name.lower().endswith(_SYMBOL_FILE_EXTS):
                continue
            if not has_parens and "." not in name:
                low = name.lower()
                if low in _CONTEXT_STOPWORDS or len(name) < 3:
                    continue
            _add_symbol(name)

    return {"editable_files": editable_files, "relevant_symbols": relevant_symbols}


def discover_read_files(editable_files: list[str], relevant_symbols: list[str], cwd: Path,
                        max_scan_files: int = 500, max_symbol_files: int = 10) -> tuple[list[str], list[str]]:
    """Resolve read-only references (matching tests + symbol-bearing files) for a scope."""
    editable_list = list(editable_files or [])
    editable_set = set(editable_list)
    cwd_resolved = cwd.resolve()
    read_files = []
    read_seen = set()
    matching_test_files = []
    test_seen = set()

    def _add_read(p: str):
        if p in editable_set or p in read_seen:
            return
        read_seen.add(p)
        read_files.append(p)

    def _add_test(p: str):
        if p in test_seen:
            return
        test_seen.add(p)
        matching_test_files.append(p)

    for ef in editable_list:
        name = Path(ef).name
        if (name.startswith("test_") or name.endswith("_test.py")) and (cwd / ef).is_file():
            _add_test(ef)
        stem = Path(ef).stem
        for cand in (f"tests/test_{stem}.py", f"tests/unit/test_{stem}.py"):
            if (cwd / cand).is_file():
                _add_test(cand)
                _add_read(cand)

    sym_patterns = []
    for sym in (relevant_symbols or []):
        if not sym:
            continue
        base = sym.split(".")[0]
        token = base if len(base) >= 3 else sym
        if not token:
            continue
        try:
            sym_patterns.append(re.compile(rf"\b{re.escape(token)}\b"))
        except re.error:
            continue

    if sym_patterns:
        scanned = 0
        added = 0
        done = False
        for root, dirnames, filenames in os.walk(str(cwd_resolved)):
            dirnames[:] = [d for d in dirnames if d not in _DISCOVERY_SKIP_DIRS]
            for fname in filenames:
                if scanned >= max_scan_files or added >= max_symbol_files:
                    done = True
                    break
                if not fname.endswith(_DISCOVERY_SCAN_EXTS):
                    continue
                fpath = Path(root) / fname
                try:
                    if fpath.stat().st_size > _MAX_SCAN_BYTES:
                        continue
                    content = fpath.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                scanned += 1
                if not any(p.search(content) for p in sym_patterns):
                    continue
                try:
                    rel = fpath.resolve().relative_to(cwd_resolved).as_posix()
                except (ValueError, OSError):
                    continue
                if rel in editable_set or rel in read_seen:
                    continue
                read_seen.add(rel)
                read_files.append(rel)
                added += 1
            if done:
                break

    return read_files, matching_test_files


def run_scope(message: str, cwd: Path, map_tokens: int = 1536, as_json: bool = False,
              model: str = "deepseek/deepseek-flash", thinking: str = "off",
              verbose: bool = False) -> dict:
    """Run the Smart Pre-Task Repo Inspection Gateway via Aider --chat-mode context."""
    load_project_env(cwd)
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("[ERROR] DEEPSEEK_API_KEY not found in project .env file!", file=sys.stderr)
        sys.exit(1)

    ignore_path = create_ephemeral_aiderignore(cwd, scope_mode=True)
    model_settings_path = create_ephemeral_model_settings(
        cwd, model, thinking=thinking, edit_format=None
    )

    cmd = [
        "uvx", "--from", "aider-chat", "aider",
        "--model", model,
        "--chat-mode", "context",
        "--map-tokens", str(map_tokens),
        "--map-multiplier-no-files", "1",
        "--no-auto-commits",
        "--yes-always",
        "--no-detect-urls",
        "--no-analytics",
        "--no-check-update",
        "--no-show-release-notes",
        "--no-gitignore",
        "--aiderignore", str(ignore_path),
        "--message", message,
    ]
    if model_settings_path:
        cmd.extend(["--model-settings-file", str(model_settings_path)])
    elif thinking == "off":
        cmd.extend(["--thinking-tokens", "0"])
    if not (cwd / ".git").exists():
        cmd.append("--no-git")

    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    except Exception as e:
        print(
            f"[ERROR] Aider scope invocation failed: {_sanitize_secret(str(e))}",
            file=sys.stderr,
        )
        print("[STATUS] AIDER_INVOCATION_FAILED", file=sys.stderr)
        sys.exit(13)
    finally:
        remove_ephemeral_aiderignore(ignore_path)
        remove_ephemeral_file(model_settings_path)

    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if verbose:
        print(filter_turn_output(combined, verbose=True))
    if proc.returncode != 0:
        print(
            f"[ERROR] Aider scope command exited with code {proc.returncode}",
            file=sys.stderr,
        )
        sys.exit(proc.returncode)

    parsed = parse_context_output(combined, cwd)
    editable_files = parsed.get("editable_files") or []
    relevant_symbols = parsed.get("relevant_symbols") or []
    read_files, matching_test_files = discover_read_files(
        editable_files, relevant_symbols, cwd
    )
    tokens_sent, cache_write, cache_hit, tokens_rcvd, cost_msg, cost_session = \
        parse_cost_and_tokens(combined)
    suggested_test_cmd = (
        f"uv run pytest {' '.join(matching_test_files)} -q"
        if matching_test_files else None
    )
    suggested_tier = classify_task(
        message, files=editable_files, read_files=read_files, test_cmd=suggested_test_cmd
    )[0]
    confidence = "high" if editable_files else "low"

    result = {
        "editable_files": editable_files,
        "relevant_symbols": relevant_symbols,
        "read_files": read_files,
        "suggested_test_cmd": suggested_test_cmd,
        "suggested_tier": suggested_tier,
        "confidence": confidence,
        "tokens_sent": tokens_sent,
        "cache_write": cache_write,
        "cache_hit": cache_hit,
        "tokens_rcvd": tokens_rcvd,
        "cost_usd": cost_msg,
        "session_cost_usd": cost_session,
    }

    if as_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("[SCOPE] Repo inspection result:")
        print(f"  Confidence      : {confidence}")
        print(f"  Suggested tier  : {suggested_tier}")
        print(f"  Editable files  : {', '.join(editable_files) or '(none)'}")
        print(f"  Relevant symbols: {', '.join(relevant_symbols) or '(none)'}")
        print(f"  Read files      : {', '.join(read_files) or '(none)'}")
        print(f"  Test command    : {suggested_test_cmd or '(none)'}")
        print(f"  Tokens sent     : {tokens_sent}")
        print(f"  Cost (USD)      : ${cost_msg} (session ${cost_session})")
    return result


@contextmanager
def with_git_lock(cwd: Path, timeout: float = 60.0, retry_interval: float = 0.5):
    """
    Acquire an atomic directory lock (<git-common-dir>/aider_merge.lock.d) to serialize
    git worktree merges across concurrent sub-agents without index.lock collisions.
    """
    git_dir = None
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=cwd, capture_output=True, text=True,
        )
        if res.returncode == 0 and res.stdout.strip():
            common_path = Path(res.stdout.strip())
            if not common_path.is_absolute():
                common_path = (cwd / common_path).resolve()
            else:
                common_path = common_path.resolve()
            if common_path.exists():
                git_dir = common_path
    except Exception:
        git_dir = None

    if git_dir is None:
        candidate = cwd / ".git"
        if candidate.is_dir():
            git_dir = candidate
        elif candidate.is_file():
            try:
                content = candidate.read_text().strip()
                if content.startswith("gitdir:"):
                    gd = Path(content.split("gitdir:", 1)[1].strip())
                    if not gd.is_absolute():
                        gd = (cwd / gd).resolve()
                    else:
                        gd = gd.resolve()
                    git_dir = gd.parent.parent if gd.parent.name == "worktrees" else gd
            except Exception:
                git_dir = None

    lock_dir = git_dir / "aider_merge.lock.d" if (git_dir and git_dir.exists()) else cwd / ".aider_merge.lock.d"
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
    branch onto the updated base branch and restore checkout to base branch.
    """
    head_check = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=cwd, capture_output=True, text=True
    )
    current_head = head_check.stdout.strip() if head_check.returncode == 0 else base_branch
    target_base = current_head if current_head and current_head != "HEAD" else base_branch

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
            # Restore cwd to target_base because `git rebase <base> <branch>` checks out <branch>
            if target_base and target_base != "HEAD":
                subprocess.run(["git", "checkout", target_base], cwd=cwd, capture_output=True, text=True)


def create_task_worktree(task_id: str, cwd: Path, base_branch: str = None) -> tuple[Path, str]:
    """Create (or reuse) the isolated worktree for a task on branch 'task/<task_id>'."""
    validate_task_id(task_id)
    worktree_path = cwd / ".worktrees" / task_id
    branch_name = f"task/{task_id}"
    if worktree_path.exists():
        print(f"[WORKTREE] Reusing existing worktree: {worktree_path}")
        return worktree_path, branch_name
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
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


def _is_git_tracked(repo_path: Path, rel_path: str) -> bool:
    """Return True if rel_path is tracked in repo_path."""
    res = subprocess.run(
        ["git", "-C", str(repo_path), "ls-files", "--error-unmatch", rel_path],
        capture_output=True, text=True,
    )
    return res.returncode == 0


def _commit_worktree_changes_if_dirty(task_id: str, worktree_path: Path):
    """Stage and commit any uncommitted edits in worktree_path before removing it."""
    if not worktree_path.exists():
        return
    # Remove bridge-copied .env* or ephemeral .aider* files/dirs if untracked
    for pattern in (".env*", ".aider*"):
        for ep_path in worktree_path.glob(pattern):
            if _is_git_tracked(worktree_path, ep_path.name):
                continue
            try:
                if ep_path.is_dir():
                    shutil.rmtree(ep_path, ignore_errors=True)
                else:
                    ep_path.unlink()
            except OSError:
                pass

    status = subprocess.run(
        ["git", "-C", str(worktree_path), "status", "--porcelain"],
        capture_output=True, text=True,
    )
    if status.returncode == 0 and status.stdout.strip():
        add_cmd = [
            "git", "-C", str(worktree_path), "add", "-A", "--", ".",
            ":(exclude).env", ":(exclude).env.*",
            ":(exclude).aider*", ":(exclude).aider-sessions",
            ":(exclude,glob)**/.env", ":(exclude,glob)**/.env.*",
            ":(exclude,glob)**/.aider*",
        ]
        add_res = subprocess.run(add_cmd, capture_output=True, text=True)
        if add_res.returncode != 0:
            print(
                f"[WORKTREE] Failed to stage changes in {worktree_path}: {add_res.stderr.strip()}",
                file=sys.stderr,
            )
            return
        staged_check = subprocess.run(
            ["git", "-C", str(worktree_path), "diff", "--cached", "--quiet"],
            capture_output=True, text=True,
        )
        if staged_check.returncode == 0:
            return
        commit_res = subprocess.run(
            ["git", "-C", str(worktree_path), "commit", "-m", f"chore(aider): apply task {task_id} changes"],
            capture_output=True, text=True,
        )
        if commit_res.returncode != 0:
            print(
                f"[WORKTREE] Failed to commit changes in {worktree_path}: {commit_res.stderr.strip()}",
                file=sys.stderr,
            )
        else:
            print(f"[WORKTREE] Committed uncommitted changes in {worktree_path}")


def _merge_branch_under_lock(branch_name: str, cwd: Path):
    """Rebase, merge, abort on conflict, and delete branch under the git merge mutex."""
    with with_git_lock(cwd):
        auto_rebase_branch_if_needed(branch_name, cwd)
        merge = subprocess.run(
            ["git", "merge", "--no-edit", branch_name],
            cwd=cwd, capture_output=True, text=True
        )
        if merge.returncode != 0:
            subprocess.run(["git", "merge", "--abort"], cwd=cwd, capture_output=True, text=True)
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
        else:
            print(f"[WORKTREE] Deleted branch {branch_name}")


def _handle_worktree_turn_exit(task_id: str, worktree_path: Path, branch_name: str,
                               cwd: Path, auto_merge: bool, code: int):
    """Finalize worktree lifecycle after an isolated turn exits."""
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


def merge_and_remove_worktree(task_id: str, worktree_path: Path, branch_name: str,
                              cwd: Path, auto_merge: bool = True):
    """Commit dirty edits, remove the worktree, and if auto_merge, rebase/merge/delete branch."""
    _commit_worktree_changes_if_dirty(task_id, worktree_path)
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
    _merge_branch_under_lock(branch_name, cwd)


def merge_worktree_task(task_id: str, cwd: Path):
    """Explicitly merge and cleanup an isolated task worktree under mutex protection."""
    try:
        validate_task_id(task_id)
    except ValueError as e:
        print(f"[ERROR] Security violation: {e}", file=sys.stderr)
        sys.exit(12)

    # Strict safety merge guard (exit 5)
    state_path = get_session_dir(task_id, cwd) / "state.json"
    if state_path.exists():
        try:
            guard_state = json.loads(state_path.read_text() or "{}")
            if not isinstance(guard_state, dict):
                raise ValueError("state.json must be a JSON object")
        except Exception as e:
            print(
                f"[ERROR] Cannot merge worktree for task '{task_id}': corrupt state.json ({e}).",
                file=sys.stderr,
            )
            sys.exit(5)
        status = (guard_state.get("status") or "").upper()
        if status.startswith("AWAITING_"):
            print(
                f"[ERROR] Cannot merge worktree for task '{task_id}': status={status}. "
                f"Resolve the pending question via 'reply --task-id {task_id} --answer ...' first.",
                file=sys.stderr,
            )
            sys.exit(5)
        if status in ("RUNNING", "FAILED"):
            print(
                f"[ERROR] Cannot merge worktree for task '{task_id}': status={status}. "
                f"Only completed sessions can be merged.",
                file=sys.stderr,
            )
            sys.exit(5)

    worktree_path = cwd / ".worktrees" / task_id
    branch_name = f"task/{task_id}"
    if worktree_path.exists():
        _commit_worktree_changes_if_dirty(task_id, worktree_path)
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree_path)],
            cwd=cwd, capture_output=True, text=True
        )
    _merge_branch_under_lock(branch_name, cwd)


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
             verbose: bool = False, thinking: str = "off",
             reasoning_effort: str = None, map_tokens: int = 0,
             plan_align: bool = False, session_root: Path = None,
             tier: str = None, max_context_kb: int = 128,
             allow_large_context: bool = False,
             _is_suppress_retry: bool = False):
    try:
        validate_task_id(task_id)
    except ValueError as e:
        print(f"[ERROR] Security violation: {e}", file=sys.stderr)
        sys.exit(12)

    session_root = session_root or cwd

    # Path containment validation first (security guard)
    cwd_resolved = cwd.resolve()
    for f in (files or []) + (read_files or []):
        resolved = (cwd / f).resolve()
        if resolved == cwd_resolved or not resolved.is_relative_to(cwd_resolved):
            print(f"[ERROR] Security violation: path '{f}' is outside project root!", file=sys.stderr)
            sys.exit(12)

    load_project_env(cwd)
    if session_root != cwd:
        load_project_env(session_root)

    resolved_editor_for_key = _resolve_editor_model(
        model, architect=architect, editor_model=editor_model
    )
    uses_deepseek = bool(
        (model and "deepseek" in model.lower())
        or (resolved_editor_for_key and "deepseek" in resolved_editor_for_key.lower())
    )
    if uses_deepseek and not os.environ.get("DEEPSEEK_API_KEY"):
        print("[ERROR] DEEPSEEK_API_KEY not found in project .env file!", file=sys.stderr)
        sys.exit(1)

    if worktree:
        worktree_path, branch_name = create_task_worktree(task_id, cwd)
        env_src = cwd / ".env"
        if env_src.exists() and not _is_git_tracked(worktree_path, ".env"):
            shutil.copy(env_src, worktree_path / ".env")
        try:
            run_turn(task_id, message, files, worktree_path,
                     read_files=read_files, restore=restore, model=model,
                     lint_cmd=lint_cmd, test_cmd=test_cmd, dry_run=dry_run,
                     architect=architect, editor_model=editor_model,
                     worktree=False, auto_merge=auto_merge, verbose=verbose,
                     thinking=thinking, reasoning_effort=reasoning_effort,
                     map_tokens=map_tokens, plan_align=plan_align,
                     session_root=session_root, tier=tier,
                     max_context_kb=max_context_kb,
                     allow_large_context=allow_large_context)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
            _handle_worktree_turn_exit(task_id, worktree_path, branch_name, cwd, auto_merge, code)
            raise
    run_preflight_checks(read_files, cwd)

    # --- Hard Context Budget Circuit Breaker (COST-GUARD) ---
    requested_paths = []
    seen_requested = set()
    for _f in (files or []) + (read_files or []):
        if not _f:
            continue
        try:
            _resolved_req = (cwd / _f).resolve()
        except (OSError, RuntimeError):
            continue
        if _resolved_req in seen_requested:
            continue
        seen_requested.add(_resolved_req)
        requested_paths.append(_resolved_req)

    total_bytes = 0
    for _rp in requested_paths:
        try:
            if _rp.is_file():
                total_bytes += _rp.stat().st_size
        except OSError:
            continue
    if total_bytes > max_context_kb * 1024 and not allow_large_context:
        print(
            f"[COST-GUARD] Total input context ({total_bytes / 1024:.1f} KB) exceeds "
            f"--max-context-kb ({max_context_kb} KB). "
            f"Pass --allow-large-context to override.",
            file=sys.stderr,
        )
        print("[STATUS] PREFLIGHT_FAILED")
        sys.exit(14)

    # --- Structured session directory management ---
    session_dir = get_session_dir(task_id, session_root)
    session_dir.mkdir(parents=True, exist_ok=True)
    state_file = session_dir / "state.json"
    outbox_file = session_dir / "outbox.jsonl"
    inbox_file = session_dir / "inbox.jsonl"
    for aux in (outbox_file, inbox_file):
        if not aux.exists():
            aux.touch()

    state = {}
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text() or "{}")
        except Exception:
            state = {}

    turn_count = int(state.get("turn_count", 0)) + 1
    state["task_id"] = task_id
    state["turn_count"] = turn_count
    state["files"] = list(files or [])
    state["read_files"] = list(read_files or [])
    state["model"] = model
    state["editor_model"] = editor_model
    state["architect"] = bool(architect)
    state["test_cmd"] = test_cmd
    state["lint_cmd"] = lint_cmd
    state["thinking"] = thinking
    state["reasoning_effort"] = reasoning_effort
    state["map_tokens"] = int(map_tokens)
    state["auto_merge"] = bool(auto_merge)
    if tier:
        state["tier"] = tier
    else:
        state.setdefault("tier", None)
    if ".worktrees" in cwd.parts:
        state["worktree"] = True
        state["worktree_path"] = str(cwd)
        state["branch"] = f"task/{task_id}"
    else:
        state.setdefault("worktree", False)
    if plan_align and not restore:
        state["stage"] = "PLANNING"
    if not state.get("stage"):
        state["stage"] = "IMPLEMENTATION"
    state["status"] = "RUNNING"
    state["last_cache_ping_ts"] = int(time.time())
    state["suppressed_count"] = int(state.get("suppressed_count", 0))
    state.setdefault("pending_qid", None)
    _write_state_atomic(state_file, state)

    session_file = session_dir / "chat_history.md"

    # Create parent directories for any new target files (also creates mocked worktree dirs)
    for f in (files or []):
        target_path = cwd / f
        target_path.parent.mkdir(parents=True, exist_ok=True)

    system_rules = (
        "IMPORTANT RULES: Be extremely concise. Output only surgical unified diffs. "
        "Do NOT write conversational filler or explanations. "
        "If any specification, signature, type, or requirement is missing or ambiguous, "
        "DO NOT GUESS and DO NOT modify code. Immediately output 'QUESTION: <your concise question>' and stop."
    )
    planning_rules = (
        "STAGE 1: PLANNING ONLY. Do NOT edit or modify any files in this turn. "
        "Analyze the scope of the task, propose architectural options, and surface trade-offs. "
        "When you need an architectural decision, output "
        "'QUESTION: [STAGE: PLANNING] [CATEGORY: ARCHITECTURE] [IMPACT: HIGH] <inquiry>' "
        "followed by [OPTIONS: A) ... B) ...] and [RECOMMENDATION: ...], then stop."
    )

    # Prefix-cache preservation: never re-prepend system_rules on restore
    if restore:
        full_message = message
    elif plan_align:
        full_message = f"{planning_rules}\n\nTask: {message}"
    else:
        full_message = f"{system_rules}\n\nTask: {message}"
    
    cmd = [
        "uvx", "--from", "aider-chat", "aider",
        "--model", model,
    ]
    if plan_align:
        cmd.extend(["--chat-mode", "ask"])
    else:
        cmd.extend(["--edit-format", "diff"])
    cmd.extend([
        "--cache-prompts",
        "--cache-keepalive-pings", "6",
        "--map-tokens", str(map_tokens),
        "--no-analytics",
        "--no-check-update",
        "--no-show-release-notes",
        "--no-detect-urls",
        "--no-gitignore",
        "--no-suggest-shell-commands",
        "--chat-history-file", str(session_file),
        "--yes-always",
        "--no-auto-commits",
        "--message", full_message
    ])

    # Dynamic .aiderignore shield for secrets, data payloads, and large files
    ignore_path = create_ephemeral_aiderignore(
        cwd, explicit_files=(files or []) + (read_files or [])
    )
    cmd.extend(["--aiderignore", str(ignore_path)])

    # During Stage 1 planning (--chat-mode ask), suppress architect/editor & auto-lint/test
    effective_architect = architect and not plan_align
    resolved_editor_model = _resolve_editor_model(
        model, architect=effective_architect, editor_model=editor_model
    )
    effective_editor_model = None if plan_align else resolved_editor_model

    # Ephemeral model settings for explicit prompt caching and diff format
    model_settings_path = create_ephemeral_model_settings(
        cwd, model, editor_model=effective_editor_model,
        architect=effective_architect, thinking=thinking, reasoning_effort=reasoning_effort
    )
    if model_settings_path:
        cmd.extend(["--model-settings-file", str(model_settings_path)])

    # Auto-lint / syntax check and closed-loop TDD (disabled during read-only Stage 1 planning)
    if not plan_align:
        resolved_lint_cmd = detect_lint_cmd(files, lint_cmd)
        if resolved_lint_cmd:
            cmd.extend(["--auto-lint", "--lint-cmd", resolved_lint_cmd])

        if test_cmd:
            cmd.extend(["--auto-test", "--test-cmd", test_cmd])

    # DeepSeek dynamic thinking / reasoning controls are injected via ephemeral YAML
    # extra_params; only pass CLI flags when no model-settings YAML was generated.
    if not model_settings_path:
        if thinking == "off":
            cmd.extend(["--thinking-tokens", "0"])
        if reasoning_effort:
            cmd.extend(["--reasoning-effort", reasoning_effort])

    # Dry-run: propose diffs without writing to disk
    if dry_run:
        cmd.append("--dry-run")

    # Architect mode (two-model: architect + editor)
    if effective_architect:
        cmd.append("--architect")
    if effective_editor_model:
        cmd.extend(["--editor-model", effective_editor_model])
    
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
        
    print(f"--- Running Aider Turn [Task: {task_id}] (restore={restore}, plan_align={plan_align}) ---")
    run_kwargs = {"cwd": cwd, "capture_output": True, "text": True}
    if (session_root / ".venv").is_dir() and "UV_PROJECT_ENVIRONMENT" not in os.environ:
        env = os.environ.copy()
        env["UV_PROJECT_ENVIRONMENT"] = str((session_root / ".venv").resolve())
        run_kwargs["env"] = env
    try:
        proc = subprocess.run(cmd, **run_kwargs)
    except Exception as e:
        rollback_target_files(files, cwd)
        state["status"] = "FAILED"
        _write_state_atomic(state_file, state)
        print(f"[ERROR] Aider invocation failed: {_sanitize_secret(str(e))}", file=sys.stderr)
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
    
    # --- Anti-noise threshold gate for intercepted questions ---
    questions = extract_questions(combined_output)
    if questions:
        events = []
        for i, q in enumerate(questions, 1):
            ev = parse_structured_question(q, seq=turn_count * 100 + i)
            ev["verdict"] = evaluate_escalation_threshold(ev)
            events.append(ev)

        suppress_all = all(ev["verdict"] == "AUTO_SUPPRESS" for ev in events)
        escalate_any = any(ev["verdict"] == "ESCALATE_PRINCIPAL" for ev in events)

        if suppress_all and not escalate_any:
            with outbox_file.open("a") as fh:
                for ev in events:
                    rec = dict(ev)
                    rec["suppressed"] = True
                    fh.write(json.dumps(rec) + "\n")
            state["suppressed_count"] = int(state.get("suppressed_count", 0)) + len(events)
            if not _is_suppress_retry:
                _write_state_atomic(state_file, state)
                print(
                    f"[THRESHOLD] Suppressed {len(events)} trivial question(s); auto-replying locally.",
                    file=sys.stderr,
                )
                if plan_align:
                    retry_msg = (
                        "STAGE 1: PLANNING ONLY. Do NOT edit or modify any files. "
                        "Summarize your proposed architectural plan or ask a concrete [STAGE: PLANNING] "
                        "question with [OPTIONS: ...] without conversational sign-off."
                    )
                else:
                    retry_msg = "Follow repo conventions and finalize code without conversational sign-off."
                run_turn(
                    task_id,
                    retry_msg,
                    files, cwd, read_files=read_files, restore=True, model=model,
                    lint_cmd=lint_cmd, test_cmd=test_cmd, dry_run=dry_run,
                    architect=architect, editor_model=editor_model,
                    worktree=False, auto_merge=auto_merge, verbose=verbose,
                    thinking=thinking, reasoning_effort=reasoning_effort,
                    map_tokens=map_tokens, plan_align=plan_align,
                    session_root=session_root, tier=tier,
                    max_context_kb=max_context_kb,
                    allow_large_context=allow_large_context,
                    _is_suppress_retry=True,
                )
                return
            _write_state_atomic(state_file, state)

        if escalate_any:
            pending = next(ev for ev in events if ev["verdict"] == "ESCALATE_PRINCIPAL")
            with outbox_file.open("a") as fh:
                for ev in events:
                    rec = dict(ev)
                    if ev["verdict"] == "AUTO_SUPPRESS":
                        rec["suppressed"] = True
                    fh.write(json.dumps(rec) + "\n")
            state["pending_qid"] = pending["qid"]
            new_stage = "PLANNING" if (plan_align or pending.get("stage") == "PLANNING") else "IMPLEMENTATION"
            state["stage"] = new_stage
            state["status"] = "AWAITING_PLAN_APPROVAL" if new_stage == "PLANNING" else "AWAITING_DECISION"
            _write_state_atomic(state_file, state)
            print(f"\n[INTERCEPT] Sub-agent paused with {len(questions)} question(s):")
            for idx, q in enumerate(questions, 1):
                print(f" {idx}. {q}")
            print(f"[STATUS] INTERACTIVE_QUESTION")
            sys.exit(10)
        
    # Check if diff failed to apply
    if not plan_align and not check_diff_applied_successfully(combined_output):
        rollback_target_files(files, cwd)
        state["status"] = "FAILED"
        _write_state_atomic(state_file, state)
        print("[ERROR] Sub-agent search/replace edit failed to apply cleanly to disk!", file=sys.stderr)
        print("[STATUS] DIFF_APPLY_FAILED")
        sys.exit(11)
        
    if proc.returncode != 0:
        rollback_target_files(files, cwd)
        state["status"] = "FAILED"
        _write_state_atomic(state_file, state)
        print(f"[ERROR] Aider exited with code {proc.returncode}", file=sys.stderr)
        sys.exit(proc.returncode)
        
    state["status"] = "COMPLETED"
    state["pending_qid"] = None
    _write_state_atomic(state_file, state)
    print("[STATUS] COMPLETED")
    sys.exit(0)


def cleanup_session(task_id: str, cwd: Path):
    try:
        validate_task_id(task_id)
    except ValueError as e:
        print(f"[ERROR] Security violation: {e}", file=sys.stderr)
        sys.exit(12)
    session_dir = get_session_dir(task_id, cwd)
    if session_dir.exists():
        shutil.rmtree(session_dir, ignore_errors=True)
        print(f"[CLEANUP] Deleted session dir: {session_dir}")
    else:
        print(f"[CLEANUP] No session dir found for task {task_id}.")
    # Legacy compatibility: prior single-file session artifacts
    legacy = cwd / f".aider.session_{task_id}.md"
    if legacy.exists():
        try:
            legacy.unlink()
            print(f"[CLEANUP] Deleted legacy session file: {legacy.name}")
        except OSError:
            pass


def inspect_session(task_id: str, cwd: Path, as_json: bool = False):
    """Read session state + latest outbox event; exit 1 if state.json missing."""
    try:
        validate_task_id(task_id)
    except ValueError as e:
        print(f"[ERROR] Security violation: {e}", file=sys.stderr)
        sys.exit(12)
    session_dir = get_session_dir(task_id, cwd)
    state_file = session_dir / "state.json"
    if not state_file.exists():
        print(f"[ERROR] No state.json for task '{task_id}' at {session_dir}", file=sys.stderr)
        sys.exit(1)
    try:
        state = json.loads(state_file.read_text() or "{}")
    except Exception as e:
        print(f"[ERROR] Failed to parse state.json: {e}", file=sys.stderr)
        sys.exit(1)

    latest_outbox = None
    outbox_file = session_dir / "outbox.jsonl"
    if outbox_file.exists():
        lines = [ln for ln in outbox_file.read_text().splitlines() if ln.strip()]
        if lines:
            try:
                latest_outbox = json.loads(lines[-1])
            except Exception:
                latest_outbox = None

    if as_json:
        print(json.dumps(
            {"task_id": task_id, "state": state, "latest_outbox": latest_outbox},
            indent=2, ensure_ascii=False,
        ))
    else:
        print(f"[INSPECT] Task: {task_id}")
        print(f"  Status        : {state.get('status')}")
        print(f"  Stage         : {state.get('stage')}")
        print(f"  Pending QID   : {state.get('pending_qid')}")
        print(f"  Turn Count    : {state.get('turn_count')}")
        if latest_outbox:
            print(f"  Latest QID    : {latest_outbox.get('qid')}")
            print(f"  Latest Question: {latest_outbox.get('question')}")
    return {"task_id": task_id, "state": state, "latest_outbox": latest_outbox}


def reply_session(task_id: str, answer: str, cwd: Path,
                  source: str = "architect", approved_option: str = None):
    """Append architect decision to inbox, then resume the session with --restore."""
    try:
        validate_task_id(task_id)
    except ValueError as e:
        print(f"[ERROR] Security violation: {e}", file=sys.stderr)
        sys.exit(12)

    session_dir = get_session_dir(task_id, cwd)
    state_file = session_dir / "state.json"
    if not state_file.exists():
        print(f"[ERROR] No state.json for task '{task_id}' at {session_dir}", file=sys.stderr)
        sys.exit(1)
    try:
        state = json.loads(state_file.read_text() or "{}")
        if not isinstance(state, dict):
            raise ValueError("state.json must be a JSON object")
    except Exception as e:
        print(f"[ERROR] Failed to parse state.json: {e}", file=sys.stderr)
        sys.exit(1)

    status = (state.get("status") or "").upper()
    if not status.startswith("AWAITING_"):
        print(
            f"[ERROR] Cannot reply to task '{task_id}': status={status or 'UNKNOWN'} "
            f"(expected AWAITING_PLAN_APPROVAL or AWAITING_DECISION).",
            file=sys.stderr,
        )
        sys.exit(1)

    is_worktree = bool(state.get("worktree"))
    effective_cwd = cwd
    branch_name = state.get("branch") or f"task/{task_id}"
    if is_worktree:
        wp_raw = state.get("worktree_path")
        wp = Path(wp_raw) if wp_raw else None
        if not wp or not wp.exists():
            print(
                f"[ERROR] Worktree for task '{task_id}' not found at {wp_raw!r}. "
                f"Cannot resume worktree session in main repo root.",
                file=sys.stderr,
            )
            sys.exit(1)
        effective_cwd = wp

    pending_qid = state.get("pending_qid")
    inbox_file = session_dir / "inbox.jsonl"
    inbox_file.touch(exist_ok=True)
    entry = {
        "reply_to": pending_qid,
        "source": source,
        "approved_option": approved_option,
        "answer": answer,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with inbox_file.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")

    prior_stage = (state.get("stage") or "").upper()
    state["status"] = "RUNNING"
    state["pending_qid"] = None
    if prior_stage == "PLANNING":
        state["stage"] = "IMPLEMENTATION"
    _write_state_atomic(state_file, state)

    if approved_option:
        decision_msg = f"Architect Decision (Option {approved_option}): {answer}. Proceed with implementation."
    else:
        decision_msg = f"Architect Decision: {answer}. Proceed with implementation."

    auto_merge = bool(state.get("auto_merge", True))
    try:
        run_turn(
            task_id,
            decision_msg,
            state.get("files") or [],
            effective_cwd,
            read_files=state.get("read_files") or [],
            restore=True,
            model=state.get("model") or "deepseek/deepseek-flash",
            lint_cmd=state.get("lint_cmd"),
            test_cmd=state.get("test_cmd"),
            dry_run=False,
            architect=bool(state.get("architect", False)),
            editor_model=state.get("editor_model"),
            worktree=False,
            auto_merge=auto_merge,
            verbose=False,
            thinking=state.get("thinking") or "off",
            reasoning_effort=state.get("reasoning_effort"),
            map_tokens=int(state.get("map_tokens", 0)),
            plan_align=False,
            session_root=cwd,
            tier=state.get("tier"),
        )
    except SystemExit as e:
        if is_worktree:
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
            _handle_worktree_turn_exit(task_id, effective_cwd, branch_name, cwd, auto_merge, code)
        raise


def _resolve_cli_message(args, cwd_path: Path, parser: argparse.ArgumentParser) -> str:
    """Resolve the task message from --message-file (preferred) or --message."""
    message_file = getattr(args, "message_file", None)
    if message_file:
        candidate = Path(message_file)
        if not candidate.is_absolute() and not candidate.exists():
            candidate = cwd_path / message_file
        if not candidate.is_file():
            print(f"[ERROR] Message file not found: {message_file}", file=sys.stderr)
            sys.exit(2)
        return candidate.read_text(encoding="utf-8")
    if getattr(args, "message", None) is None:
        parser.error("one of --message or --message-file is required")
    return args.message


def main():
    parser = argparse.ArgumentParser(description="Token-Sparse Aider Sub-Agent Orchestrator")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    turn_parser = subparsers.add_parser("turn", help="Run a turn in the task session")
    turn_parser.add_argument("--task-id", required=True, help="Unique identifier for the task session")
    turn_parser.add_argument("--message", default=None, help="Prompt/specification for the sub-agent")
    turn_parser.add_argument("--message-file", default=None, help="Path to a file containing the task message (alternative to --message)")
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
    turn_parser.add_argument("--thinking", default="off", choices=["auto", "on", "off"], help="DeepSeek thinking mode (default: off)")
    turn_parser.add_argument("--reasoning-effort", default=None, choices=["low", "high", "max"], help="DeepSeek reasoning effort level")
    turn_parser.add_argument("--map-tokens", type=int, default=0, help="Aider repo-map token budget (default: 0)")
    turn_parser.add_argument("--plan-align", action="store_true",
                             help="Stage 1 planning: propose architecture, do not edit files")
    turn_parser.add_argument("--max-context-kb", type=int, default=128,
                             help="Maximum total requested context size in KB before the cost guard trips (default: 128)")
    turn_parser.add_argument("--allow-large-context", action="store_true",
                             help="Allow requests whose total context exceeds --max-context-kb")
    turn_parser.add_argument("--cwd", default=".", help="Project working directory")

    dispatch_parser = subparsers.add_parser("dispatch", help="Auto-classify and dispatch a task to the optimal tier")
    dispatch_parser.add_argument("--task-id", required=True, help="Unique identifier for the task session")
    dispatch_parser.add_argument("--message", default=None, help="Task specification / research query")
    dispatch_parser.add_argument("--message-file", default=None, help="Path to a file containing the task message (alternative to --message)")
    dispatch_parser.add_argument("--files", nargs="*", default=[], help="Editable target files")
    dispatch_parser.add_argument("--read-files", nargs="*", default=[], help="Read-only reference files")
    dispatch_parser.add_argument("--model", default=None, help="Override resolved tier model")
    dispatch_parser.add_argument("--editor-model", default=None, help="Override architect editor model")
    dispatch_parser.add_argument("--architect", action="store_true", help="Force two-model architect mode (--architect)")
    dispatch_parser.add_argument("--lint-cmd", default=None, help="Custom lint command passed to Aider (enables --auto-lint)")
    dispatch_parser.add_argument("--test-cmd", default=None, help="Test command enabling closed-loop TDD")
    dispatch_parser.add_argument("--map-tokens", type=int, default=None, help="Override Aider repo-map token budget")
    dispatch_parser.add_argument("--tier", default="auto", choices=["auto", "patch", "surgical", "architect", "research"], help="Force a tier (default: auto)")
    dispatch_parser.add_argument("--thinking", default="auto", choices=["auto", "on", "off"], help="Override thinking mode")
    dispatch_parser.add_argument("--reasoning-effort", default=None, choices=["low", "high", "max"], help="Override reasoning effort")
    dispatch_parser.add_argument("--dry-run", action="store_true", help="Print classification + resolved params without executing")
    dispatch_parser.add_argument("--worktree", action="store_true", help="Run inside an isolated git worktree")
    dispatch_parser.add_argument("--no-auto-merge", action="store_true", help="Do not auto-merge the worktree on completion")
    dispatch_parser.add_argument("-v", "--verbose", action="store_true", help="Include full thinking traces in output")
    dispatch_parser.add_argument("--plan-align", action="store_true",
                                 help="Stage 1 planning: propose architecture, do not edit files")
    dispatch_parser.add_argument("--restore", action="store_true",
                                 help="Restore prior session history when dispatch resumes a session")
    dispatch_parser.add_argument("--auto-scope", action="store_true",
                                 help="Run the Smart Pre-Task Repo Inspection Gateway when --files is omitted")
    dispatch_parser.add_argument("--count", type=int, default=5, help="Search result count when routed to Tier 4 research (1-50, default 5)")
    dispatch_parser.add_argument("--need-content", action="store_true", help="Include full page content when routed to Tier 4 research")
    dispatch_parser.add_argument("--json", action="store_true", help="Emit raw JSON response when routed to Tier 4 research")
    dispatch_parser.add_argument("--max-context-kb", type=int, default=128,
                                 help="Maximum total requested context size in KB before the cost guard trips (default: 128)")
    dispatch_parser.add_argument("--allow-large-context", action="store_true",
                                 help="Allow requests whose total context exceeds --max-context-kb")
    dispatch_parser.add_argument("--cwd", default=".", help="Project working directory")
    
    clean_parser = subparsers.add_parser("cleanup", help="Clean up session history file")
    clean_parser.add_argument("--task-id", required=True, help="Task identifier to clean up")
    clean_parser.add_argument("--cwd", default=".", help="Project working directory")

    merge_parser = subparsers.add_parser("merge-worktree", help="Merge isolated task worktree branch and clean up worktree")
    merge_parser.add_argument("--task-id", required=True, help="Task identifier whose worktree should be merged")
    merge_parser.add_argument("--cwd", default=".", help="Project working directory")

    inspect_parser = subparsers.add_parser("inspect", help="Show session state + latest outbox question")
    inspect_parser.add_argument("--task-id", required=True)
    inspect_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    inspect_parser.add_argument("--cwd", default=".", help="Project working directory")

    reply_parser = subparsers.add_parser("reply", help="Answer a pending question and resume the session")
    reply_parser.add_argument("--task-id", required=True)
    reply_parser.add_argument("--answer", required=True, help="Architect decision text")
    reply_parser.add_argument("--source", default="architect",
                              choices=["architect", "user", "auto"])
    reply_parser.add_argument("--approved-option", default=None, help="Chosen option key (A/B/…)")
    reply_parser.add_argument("--cwd", default=".", help="Project working directory")

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

    scope_parser = subparsers.add_parser("scope", help="Inspect repo scope via Aider context mode")
    scope_parser.add_argument("--message", required=True, help="Task/spec to scope")
    scope_parser.add_argument("--cwd", default=".", help="Project working directory")
    scope_parser.add_argument("--map-tokens", type=int, default=1536, help="Repo-map token budget (default: 1536)")
    scope_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    scope_parser.add_argument("--model", default="deepseek/deepseek-flash", help="Model name (default: deepseek/deepseek-flash)")
    scope_parser.add_argument("--thinking", default="off", choices=["auto", "on", "off"], help="DeepSeek thinking mode")
    scope_parser.add_argument("-v", "--verbose", action="store_true", help="Include full thinking traces in output")

    args = parser.parse_args()
    cwd_path = Path(getattr(args, "cwd", ".")).resolve()
    
    if args.command == "turn":
        message = _resolve_cli_message(args, cwd_path, turn_parser)
        run_turn(args.task_id, message, args.files, cwd_path, 
                 read_files=args.read_files, restore=args.restore, model=args.model,
                 lint_cmd=args.lint_cmd, test_cmd=args.test_cmd,
                 dry_run=args.dry_run, architect=args.architect,
                 editor_model=args.editor_model, worktree=args.worktree,
                 auto_merge=not args.no_auto_merge, verbose=args.verbose,
                 thinking=args.thinking, reasoning_effort=args.reasoning_effort,
                 map_tokens=args.map_tokens, plan_align=args.plan_align,
                 max_context_kb=args.max_context_kb,
                 allow_large_context=args.allow_large_context)
    elif args.command == "dispatch":
        message = _resolve_cli_message(args, cwd_path, dispatch_parser)
        run_dispatch(args.task_id, message, args.files, cwd_path,
                     read_files=args.read_files, test_cmd=args.test_cmd,
                     lint_cmd=args.lint_cmd, model=args.model,
                     editor_model=args.editor_model, map_tokens=args.map_tokens,
                     architect=args.architect,
                     tier=args.tier, thinking=args.thinking,
                     reasoning_effort=args.reasoning_effort,
                     dry_run=args.dry_run, worktree=args.worktree,
                     auto_merge=not args.no_auto_merge, verbose=args.verbose,
                     plan_align=args.plan_align, restore=args.restore,
                     count=args.count, need_content=args.need_content,
                     as_json=args.json, auto_scope=args.auto_scope,
                     max_context_kb=args.max_context_kb,
                     allow_large_context=args.allow_large_context)
    elif args.command == "cleanup":
        cleanup_session(args.task_id, cwd_path)
    elif args.command == "inspect":
        inspect_session(args.task_id, cwd_path, as_json=args.json)
    elif args.command == "reply":
        reply_session(args.task_id, args.answer, cwd_path,
                      source=args.source, approved_option=args.approved_option)
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
    elif args.command == "scope":
        run_scope(args.message, cwd_path, map_tokens=args.map_tokens,
                  as_json=args.json, model=args.model, thinking=args.thinking,
                  verbose=args.verbose)


if __name__ == "__main__":
    main()
