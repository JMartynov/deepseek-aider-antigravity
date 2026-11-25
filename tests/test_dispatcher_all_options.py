"""
Comprehensive Deep Test Suite: All DeepSeek & Aider Options, Smart Dispatcher,
Model Settings YAML, CLI Command Construction, and Subcommand Wiring.
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import aider_bridge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_aider_subprocess(monkeypatch, stdout="", stderr="", returncode=0, raise_exc=None):
    """Mock subprocess.run, capturing all uvx/aider invocations."""
    calls = []

    def mock_run(cmd, cwd=None, capture_output=True, text=True):
        if cmd and cmd[0] == "uvx":
            if raise_exc is not None:
                raise raise_exc
            calls.append(list(cmd))
            if "--chat-history-file" in cmd:
                ch_path = Path(cmd[cmd.index("--chat-history-file") + 1])
                ch_path.parent.mkdir(parents=True, exist_ok=True)
                ch_path.touch()
            if "--model-settings-file" in cmd:
                ms_path = Path(cmd[cmd.index("--model-settings-file") + 1])
                if ms_path.exists():
                    calls.append({"_model_settings_yaml": ms_path.read_text()})
        class Ret:
            pass
        ret = Ret()
        ret.returncode = returncode
        ret.stdout = stdout
        ret.stderr = stderr
        return ret

    monkeypatch.setattr(aider_bridge.subprocess, "run", mock_run)
    monkeypatch.setattr(aider_bridge, "fetch_account_balance", lambda *a, **k: None)
    return calls


def _extract_cmd_and_yaml(calls):
    """Split captured uvx command list and optional model settings YAML snapshot."""
    cmd = next(c for c in calls if isinstance(c, list))
    yaml_text = next((c["_model_settings_yaml"] for c in calls if isinstance(c, dict)), None)
    return cmd, yaml_text


# ---------------------------------------------------------------------------
# 1. Every Tier Classification & Exact Resolved Parameters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ext", [
    ".md", ".txt", ".rst", ".toml", ".yaml", ".yml", ".json", ".ini", ".cfg",
])
def test_tier1_patch_all_doc_config_extensions(ext):
    tier, reason, params = aider_bridge.classify_task(
        "Update file", files=[f"docs/sample{ext}"]
    )
    assert tier == "patch"
    assert params == {
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
        "test_cmd": None,
    }


@pytest.mark.parametrize("kw", [
    "typo", "lint", "linter", "bump", "version",
    "docstring", "comment", "readme", "format", "rename",
])
def test_tier1_patch_all_keywords(kw):
    tier, _, params = aider_bridge.classify_task(
        f"Apply {kw} cleanup", files=["module.py"]
    )
    assert tier == "patch"
    assert params["tier_num"] == 1
    assert params["thinking"] == "off"
    assert params["map_tokens"] == 0


def test_tier1_upgrades_to_surgical_when_test_cmd_present():
    tier, reason, params = aider_bridge.classify_task(
        "Fix typo in config", files=["config.toml"], test_cmd="pytest -q"
    )
    assert tier == "surgical"
    assert params["tier_num"] == 2
    assert params["test_cmd"] == "pytest -q"


def test_tier2_surgical_exact_resolved_params():
    tier, reason, params = aider_bridge.classify_task(
        "Add validation helper",
        files=["a.py", "b.py", "c.py"],
        test_cmd="uv run pytest tests/ -q",
    )
    assert tier == "surgical"
    assert params == {
        "tier_num": 2,
        "tier_label": "SURGICAL",
        "model": "deepseek/deepseek-flash",
        "thinking": "auto",
        "reasoning_effort": "low",
        "edit_format": "diff",
        "map_tokens": 512,
        "architect": False,
        "editor_model": None,
        "plan_align": False,
        "test_cmd": "uv run pytest tests/ -q",
    }


@pytest.mark.parametrize("kw", [
    "architect", "refactor", "ast", "visitor", "sqlglot",
    "security guard", "security_guard", "concurrency", "mutex",
    "lock", "cross-module", "cross-repo", "redesign", "rewrite", "migrate",
])
def test_tier3_architect_all_keywords_and_exact_params(kw):
    tier, _, params = aider_bridge.classify_task(
        f"Perform {kw} work", files=["core.py"], test_cmd="pytest -q"
    )
    assert tier == "architect"
    assert params == {
        "tier_num": 3,
        "tier_label": "ARCHITECT",
        "model": "deepseek/deepseek-reasoner",
        "thinking": "on",
        "reasoning_effort": "high",
        "edit_format": "diff",
        "map_tokens": 512,
        "architect": True,
        "editor_model": "deepseek/deepseek-flash",
        "plan_align": True,
        "test_cmd": "pytest -q",
    }


def test_tier3_architect_file_count_threshold():
    # 3 files -> surgical; 4 files -> architect
    t3_below, _, _ = aider_bridge.classify_task("Update modules", files=["a.py", "b.py", "c.py"])
    t3_above, _, _ = aider_bridge.classify_task("Update modules", files=["a.py", "b.py", "c.py", "d.py"])
    assert t3_below == "surgical"
    assert t3_above == "architect"


@pytest.mark.parametrize("kw", [
    "research", "search", "lookup", "look up", "pep", "cve",
    "advisory", "spec", "what is", "how does", "documentation",
])
def test_tier4_research_all_keywords_and_exact_params(kw):
    tier, _, params = aider_bridge.classify_task(f"  {kw} asyncio TaskGroup  ")
    assert tier == "research"
    assert params == {
        "tier_num": 4,
        "tier_label": "RESEARCH",
        "command": "search",
        "query": f"{kw} asyncio TaskGroup",
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


def test_tier4_research_no_files_with_test_cmd_and_non_research_msg_falls_to_surgical():
    tier, _, params = aider_bridge.classify_task(
        "Run suite check", files=[], test_cmd="pytest -q"
    )
    assert tier == "surgical"
    assert params["test_cmd"] == "pytest -q"


# ---------------------------------------------------------------------------
# 2. Every Explicit Override on `dispatch`
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tier_val,expected_label,expected_num", [
    ("patch", "PATCH", 1),
    ("surgical", "SURGICAL", 2),
    ("architect", "ARCHITECT", 3),
    ("research", "RESEARCH", 4),
])
def test_dispatch_dry_run_all_tiers(monkeypatch, capsys, tier_val, expected_label, expected_num):
    def boom(*a, **k):
        raise AssertionError("run_turn/run_search must not be called in --dry-run")

    monkeypatch.setattr(aider_bridge, "run_turn", boom)
    monkeypatch.setattr(aider_bridge, "run_search", boom)
    monkeypatch.setattr(sys, "argv", [
        "aider_bridge.py", "dispatch",
        "--task-id", f"dry_{tier_val}",
        "--message", "Do work",
        "--files", "app.py",
        "--tier", tier_val,
        "--dry-run",
        "--cwd", ".",
    ])
    with pytest.raises(SystemExit) as exc:
        aider_bridge.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert f"[DISPATCHER] Classified as TIER {expected_num}: {expected_label}" in out
    assert f"Tier key: {tier_val}" in out
    assert "[STATUS] DRY_RUN" in out


@pytest.mark.parametrize("thinking_val", ["auto", "on", "off"])
@pytest.mark.parametrize("effort_val", ["low", "high", "max"])
def test_dispatch_overrides_thinking_and_reasoning_effort(monkeypatch, thinking_val, effort_val):
    captured = {}

    def fake_run_turn(task_id, message, files, cwd, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(aider_bridge, "run_turn", fake_run_turn)
    monkeypatch.setattr(sys, "argv", [
        "aider_bridge.py", "dispatch",
        "--task-id", "ovr_think",
        "--message", "Implement parser",
        "--files", "parser.py",
        "--tier", "surgical",
        "--thinking", thinking_val,
        "--reasoning-effort", effort_val,
        "--cwd", ".",
    ])
    aider_bridge.main()
    assert captured["thinking"] == thinking_val
    assert captured["reasoning_effort"] == effort_val


def test_dispatch_wires_all_turn_override_flags(monkeypatch):
    captured = {}

    def fake_run_turn(task_id, message, files, cwd, **kwargs):
        captured["task_id"] = task_id
        captured["message"] = message
        captured["files"] = list(files)
        captured["kwargs"] = kwargs

    monkeypatch.setattr(aider_bridge, "run_turn", fake_run_turn)
    monkeypatch.setattr(sys, "argv", [
        "aider_bridge.py", "dispatch",
        "--task-id", "ovr_all",
        "--message", "Implement module",
        "--files", "mod.py",
        "--read-files", "contracts.py",
        "--tier", "surgical",
        "--model", "deepseek/deepseek-v4-pro",
        "--architect",
        "--editor-model", "deepseek/deepseek-flash",
        "--map-tokens", "1024",
        "--lint-cmd", "ruff check",
        "--test-cmd", "pytest tests/ -q",
        "--plan-align",
        "--restore",
        "--worktree",
        "--no-auto-merge",
        "-v",
        "--cwd", ".",
    ])
    aider_bridge.main()

    assert captured["task_id"] == "ovr_all"
    assert captured["files"] == ["mod.py"]
    kw = captured["kwargs"]
    assert kw["read_files"] == ["contracts.py"]
    assert kw["model"] == "deepseek/deepseek-v4-pro"
    assert kw["architect"] is True
    assert kw["editor_model"] == "deepseek/deepseek-flash"
    assert kw["map_tokens"] == 1024
    assert kw["lint_cmd"] == "ruff check"
    assert kw["test_cmd"] == "pytest tests/ -q"
    assert kw["plan_align"] is True
    assert kw["restore"] is True
    assert kw["worktree"] is True
    assert kw["auto_merge"] is False
    assert kw["verbose"] is True
    assert kw["tier"] == "surgical"


def test_dispatch_architect_v4_pro_auto_resolves_editor_model(monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("should not execute on dry-run")

    monkeypatch.setattr(aider_bridge, "run_turn", boom)
    monkeypatch.setattr(sys, "argv", [
        "aider_bridge.py", "dispatch",
        "--task-id", "v4_auto_ed",
        "--message", "Update module",
        "--files", "mod.py",
        "--tier", "surgical",
        "--model", "deepseek/deepseek-v4-pro",
        "--architect",
        "--dry-run",
        "--cwd", ".",
    ])
    with pytest.raises(SystemExit) as exc:
        aider_bridge.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "model: deepseek/deepseek-v4-pro" in out
    assert "architect: True" in out
    assert "editor_model: deepseek/deepseek-flash" in out


def test_dispatch_tier3_restore_disables_default_plan_align(monkeypatch):
    captured = {}

    def fake_run_turn(task_id, message, files, cwd, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(aider_bridge, "run_turn", fake_run_turn)
    monkeypatch.setattr(sys, "argv", [
        "aider_bridge.py", "dispatch",
        "--task-id", "t3_restore",
        "--message", "Proceed with implementation",
        "--files", "a.py", "b.py", "c.py", "d.py",
        "--restore",
        "--cwd", ".",
    ])
    aider_bridge.main()
    assert captured["tier"] == "architect"
    assert captured["restore"] is True
    assert captured["plan_align"] is False
    assert captured["architect"] is True


def test_dispatch_research_forwards_search_flags(monkeypatch):
    captured = {}

    def fake_run_search(query, cwd, **kwargs):
        captured["query"] = query
        captured["kwargs"] = kwargs

    monkeypatch.setattr(aider_bridge, "run_search", fake_run_search)
    monkeypatch.setattr(sys, "argv", [
        "aider_bridge.py", "dispatch",
        "--task-id", "res_flags",
        "--message", "What is PEP 695?",
        "--count", "12",
        "--need-content",
        "--json",
        "--cwd", ".",
    ])
    aider_bridge.main()
    assert captured["query"] == "What is PEP 695?"
    assert captured["kwargs"] == {"count": 12, "need_content": True, "as_json": True}


# ---------------------------------------------------------------------------
# 3. Model Settings YAML Generation (create_ephemeral_model_settings)
# ---------------------------------------------------------------------------

def test_model_settings_flash_thinking_off():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = aider_bridge.create_ephemeral_model_settings(
            Path(tmpdir), "deepseek/deepseek-flash", thinking="off"
        )
        assert p is not None
        text = p.read_text()
        assert "- name: deepseek/deepseek-flash" in text
        assert "edit_format: diff" in text
        assert "caches_by_default: true" in text
        assert "examples_as_sys_msg: true" in text
        assert "extra_params:\n    thinking:\n      type: disabled" in text
        assert "reasoning_effort" not in text


def test_model_settings_flash_thinking_on_effort_low():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = aider_bridge.create_ephemeral_model_settings(
            Path(tmpdir), "deepseek/deepseek-flash", thinking="on", reasoning_effort="low"
        )
        assert p is not None
        text = p.read_text()
        assert "- name: deepseek/deepseek-flash" in text
        assert "edit_format: diff" in text
        assert "caches_by_default: true" in text
        assert "examples_as_sys_msg: true" in text
        assert "thinking:\n      type: enabled" in text
        assert "reasoning_effort: low" in text


def test_model_settings_flash_thinking_auto_no_effort():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = aider_bridge.create_ephemeral_model_settings(
            Path(tmpdir), "deepseek/deepseek-flash", thinking="auto", reasoning_effort=None
        )
        assert p is not None
        text = p.read_text()
        assert "caches_by_default: true" in text
        assert "extra_params" not in text


def test_model_settings_reasoner_architect_primary_and_editor():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = aider_bridge.create_ephemeral_model_settings(
            Path(tmpdir),
            "deepseek/deepseek-reasoner",
            editor_model="deepseek/deepseek-flash",
            architect=True,
            thinking="on",
            reasoning_effort="high",
        )
        assert p is not None
        text = p.read_text()
        assert "- name: deepseek/deepseek-reasoner" in text
        assert "edit_format: diff" in text
        assert "type: enabled" in text
        assert "reasoning_effort: high" in text
        assert "- name: deepseek/deepseek-flash" in text
        assert "edit_format: editor-diff" in text
        assert "type: disabled" in text


@pytest.mark.parametrize("primary_model", [
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-reasoner",
])
def test_model_settings_auto_editor_routing_when_architect_enabled(primary_model):
    with tempfile.TemporaryDirectory() as tmpdir:
        p = aider_bridge.create_ephemeral_model_settings(
            Path(tmpdir),
            primary_model,
            editor_model=None,
            architect=True,
            thinking="on",
            reasoning_effort="max",
        )
        assert p is not None
        text = p.read_text()
        assert f"- name: {primary_model}" in text
        assert "reasoning_effort: max" in text
        assert "- name: deepseek/deepseek-flash" in text
        assert "edit_format: editor-diff" in text
        assert "type: disabled" in text


def test_model_settings_same_model_and_editor_does_not_duplicate():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = aider_bridge.create_ephemeral_model_settings(
            Path(tmpdir),
            "deepseek/deepseek-flash",
            editor_model="deepseek/deepseek-flash",
            architect=True,
            thinking="off",
        )
        assert p is not None
        text = p.read_text()
        assert text.count("- name: deepseek/deepseek-flash") == 1


def test_model_settings_non_deepseek_returns_none():
    with tempfile.TemporaryDirectory() as tmpdir:
        assert aider_bridge.create_ephemeral_model_settings(Path(tmpdir), "openai/gpt-4o") is None


# ---------------------------------------------------------------------------
# 4. Aider CLI Command Construction (subprocess.run mocked)
# ---------------------------------------------------------------------------

def test_cli_construction_tier1_patch(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        calls = _mock_aider_subprocess(monkeypatch, stdout="Applied edit to README.md\n")

        monkeypatch.setattr(sys, "argv", [
            "aider_bridge.py", "dispatch",
            "--task-id", "cli_t1",
            "--message", "Fix typo in README",
            "--files", "README.md",
            "--cwd", str(tmppath),
        ])
        with pytest.raises(SystemExit) as exc:
            aider_bridge.main()
        assert exc.value.code == 0

        cmd, yaml_text = _extract_cmd_and_yaml(calls)
        assert cmd[cmd.index("--model") + 1] == "deepseek/deepseek-flash"
        assert cmd[cmd.index("--edit-format") + 1] == "diff"
        assert cmd[cmd.index("--map-tokens") + 1] == "0"
        assert "--thinking-tokens" not in cmd
        assert "--cache-prompts" in cmd
        assert cmd[cmd.index("--cache-keepalive-pings") + 1] == "6"
        assert "--no-detect-urls" in cmd
        assert "--no-analytics" in cmd
        assert "--no-check-update" in cmd
        assert "--no-show-release-notes" in cmd
        assert "--no-gitignore" in cmd
        assert "--yes-always" in cmd
        assert "--no-auto-commits" in cmd
        assert "--architect" not in cmd
        assert "--editor-model" not in cmd
        assert "--chat-mode" not in cmd
        assert "--auto-test" not in cmd
        assert "--auto-lint" not in cmd
        assert "--reasoning-effort" not in cmd
        assert "README.md" in cmd
        assert "type: disabled" in yaml_text
        # Ephemeral files cleaned up in finally
        assert not (tmppath / ".aiderignore.ephemeral").exists()
        assert not list(tmppath.glob(".aider.model.settings.*.yml"))


def test_cli_construction_tier2_surgical_with_test_and_lint(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / ".git").mkdir()
        (tmppath / "spec.md").write_text("# contract\n")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        calls = _mock_aider_subprocess(monkeypatch, stdout="Applied edit to auth.py\n")

        monkeypatch.setattr(sys, "argv", [
            "aider_bridge.py", "dispatch",
            "--task-id", "cli_t2",
            "--message", "Implement verify_token",
            "--files", "auth.py",
            "--read-files", "spec.md",
            "--test-cmd", "pytest tests/test_auth.py -q",
            "--lint-cmd", "ruff check",
            "--cwd", str(tmppath),
        ])
        with pytest.raises(SystemExit) as exc:
            aider_bridge.main()
        assert exc.value.code == 0

        cmd, yaml_text = _extract_cmd_and_yaml(calls)
        assert cmd[cmd.index("--model") + 1] == "deepseek/deepseek-flash"
        assert cmd[cmd.index("--edit-format") + 1] == "diff"
        assert cmd[cmd.index("--map-tokens") + 1] == "512"
        assert "--reasoning-effort" not in cmd
        assert "reasoning_effort: low" in yaml_text
        assert "--auto-test" in cmd
        assert cmd[cmd.index("--test-cmd") + 1] == "pytest tests/test_auth.py -q"
        assert "--auto-lint" in cmd
        assert cmd[cmd.index("--lint-cmd") + 1] == "ruff check"
        assert cmd[cmd.index("--read") + 1] == "spec.md"
        assert "--no-git" not in cmd
        assert "--thinking-tokens" not in cmd
        assert "--architect" not in cmd
        assert "--chat-mode" not in cmd


def test_cli_construction_tier3_stage1_plan_align(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        calls = _mock_aider_subprocess(
            monkeypatch,
            stdout=(
                "QUESTION: [STAGE: PLANNING] [CATEGORY: ARCHITECTURE] [IMPACT: HIGH]\n"
                "Select visitor design?\n"
                "[OPTIONS:\n  A) Pure AST\n  B) CST\n]\n"
                "[RECOMMENDATION: Option A]\n"
            ),
        )

        monkeypatch.setattr(sys, "argv", [
            "aider_bridge.py", "dispatch",
            "--task-id", "cli_t3_s1",
            "--message", "Refactor AST visitors",
            "--files", "a.py", "b.py", "c.py", "d.py",
            "--test-cmd", "pytest -q",
            "--cwd", str(tmppath),
        ])
        with pytest.raises(SystemExit) as exc:
            aider_bridge.main()
        assert exc.value.code == 10

        cmd, yaml_text = _extract_cmd_and_yaml(calls)
        assert cmd[cmd.index("--model") + 1] == "deepseek/deepseek-reasoner"
        assert cmd[cmd.index("--chat-mode") + 1] == "ask"
        assert cmd[cmd.index("--map-tokens") + 1] == "512"
        assert "--reasoning-effort" not in cmd
        assert "reasoning_effort: high" in yaml_text
        assert "--edit-format" not in cmd
        assert "--architect" not in cmd
        assert "--editor-model" not in cmd
        assert "--auto-test" not in cmd
        assert "--auto-lint" not in cmd
        msg = cmd[cmd.index("--message") + 1]
        assert "IMPORTANT RULES" in msg
        assert "STAGE 1: PLANNING ONLY" in msg
        assert "- name: deepseek/deepseek-reasoner" in yaml_text
        assert "edit_format: editor-diff" not in yaml_text


def test_cli_construction_tier3_stage2_reply_restore(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        sd = tmppath / ".aider-sessions" / "cli_t3_s2"
        sd.mkdir(parents=True)
        (sd / "state.json").write_text(json.dumps({
            "task_id": "cli_t3_s2",
            "stage": "PLANNING",
            "status": "AWAITING_PLAN_APPROVAL",
            "pending_qid": "q_plan1",
            "files": ["a.py", "b.py"],
            "read_files": [],
            "model": "deepseek/deepseek-reasoner",
            "editor_model": "deepseek/deepseek-flash",
            "architect": True,
            "test_cmd": "pytest tests/ -q",
            "lint_cmd": None,
            "thinking": "on",
            "reasoning_effort": "high",
            "map_tokens": 512,
            "turn_count": 1,
            "auto_merge": True,
            "worktree": False,
        }))
        (sd / "chat_history.md").touch()
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        calls = _mock_aider_subprocess(monkeypatch, stdout="Applied edit to a.py\n")

        monkeypatch.setattr(sys, "argv", [
            "aider_bridge.py", "reply",
            "--task-id", "cli_t3_s2",
            "--approved-option", "A",
            "--answer", "Use Pure AST",
            "--cwd", str(tmppath),
        ])
        with pytest.raises(SystemExit) as exc:
            aider_bridge.main()
        assert exc.value.code == 0

        cmd, yaml_text = _extract_cmd_and_yaml(calls)
        assert cmd[cmd.index("--model") + 1] == "deepseek/deepseek-reasoner"
        assert cmd[cmd.index("--edit-format") + 1] == "diff"
        assert "--architect" in cmd
        assert cmd[cmd.index("--editor-model") + 1] == "deepseek/deepseek-flash"
        assert "--restore-chat-history" in cmd
        assert "--auto-test" in cmd
        assert cmd[cmd.index("--test-cmd") + 1] == "pytest tests/ -q"
        assert "--auto-lint" in cmd
        assert cmd[cmd.index("--lint-cmd") + 1] == "python:python -m py_compile"
        assert "--chat-mode" not in cmd
        msg = cmd[cmd.index("--message") + 1]
        assert "IMPORTANT RULES" not in msg
        assert "Architect Decision (Option A): Use Pure AST." in msg
        assert "edit_format: editor-diff" in yaml_text


def test_cli_construction_tier4_research_never_invokes_aider(monkeypatch, capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

        def fail_subproc(*a, **k):
            raise AssertionError("subprocess.run (aider) must never be called for Tier 4 research")

        class _Resp:
            def read(self):
                return json.dumps({
                    "results": [{"title": "PEP 695", "url": "https://peps.python.org/pep-0695/", "summary": "Type params"}]
                }).encode()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False

        monkeypatch.setattr(aider_bridge.subprocess, "run", fail_subproc)
        monkeypatch.setattr(aider_bridge.urllib.request, "urlopen", lambda req, timeout=15: _Resp())

        monkeypatch.setattr(sys, "argv", [
            "aider_bridge.py", "dispatch",
            "--task-id", "cli_t4",
            "--message", "What is PEP 695?",
            "--cwd", str(tmppath),
        ])
        aider_bridge.main()
        out = capsys.readouterr().out
        assert "[DISPATCHER] Classified as TIER 4: RESEARCH" in out
        assert "PEP 695" in out


def test_cli_construction_turn_v4_pro_architect_auto_editor(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        calls = _mock_aider_subprocess(monkeypatch, stdout="Applied edit to core.py\n")

        monkeypatch.setattr(sys, "argv", [
            "aider_bridge.py", "turn",
            "--task-id", "v4_turn",
            "--message", "Redesign core",
            "--files", "core.py",
            "--model", "deepseek/deepseek-v4-pro",
            "--architect",
            "--thinking", "on",
            "--reasoning-effort", "max",
            "--map-tokens", "256",
            "--dry-run",
            "--cwd", str(tmppath),
        ])
        with pytest.raises(SystemExit) as exc:
            aider_bridge.main()
        assert exc.value.code == 0

        cmd, yaml_text = _extract_cmd_and_yaml(calls)
        assert cmd[cmd.index("--model") + 1] == "deepseek/deepseek-v4-pro"
        assert "--architect" in cmd
        assert cmd[cmd.index("--editor-model") + 1] == "deepseek/deepseek-flash"
        assert "--reasoning-effort" not in cmd
        assert "reasoning_effort: max" in yaml_text
        assert cmd[cmd.index("--map-tokens") + 1] == "256"
        assert "--dry-run" in cmd
        assert "- name: deepseek/deepseek-v4-pro" in yaml_text
        assert "- name: deepseek/deepseek-flash" in yaml_text
        assert "edit_format: editor-diff" in yaml_text


def test_turn_verbose_vs_quiet_output(monkeypatch, capsys):
    raw = (
        "Aider v0.86.2\n"
        "Detected dumb terminal, disabling fancy input.\n"
        "► **THINKING**\nsecret reasoning\n► **ANSWER**\n"
        "Applied edit to app.py\n"
    )
    for verbose_flag, expect_thinking in [(False, False), (True, True)]:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
            _mock_aider_subprocess(monkeypatch, stdout=raw)
            argv = [
                "aider_bridge.py", "turn",
                "--task-id", f"verb_{verbose_flag}",
                "--message", "Edit app", "--files", "app.py",
                "--cwd", str(tmppath),
            ]
            if verbose_flag:
                argv.append("-v")
            monkeypatch.setattr(sys, "argv", argv)
            with pytest.raises(SystemExit) as exc:
                aider_bridge.main()
            assert exc.value.code == 0
            out = capsys.readouterr().out
            assert ("► **THINKING**" in out) is expect_thinking
            assert "Applied edit to app.py" in out


# ---------------------------------------------------------------------------
# 5. All Subcommands CLI Wiring & Error / Exit Codes
# ---------------------------------------------------------------------------

def test_subcommand_cleanup_cli_wiring(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        sd = tmppath / ".aider-sessions" / "c1"
        sd.mkdir(parents=True)
        (sd / "state.json").write_text("{}")
        legacy = tmppath / ".aider.session_c1.md"
        legacy.write_text("old")

        monkeypatch.setattr(sys, "argv", [
            "aider_bridge.py", "cleanup", "--task-id", "c1", "--cwd", str(tmppath)
        ])
        aider_bridge.main()
        assert not sd.exists()
        assert not legacy.exists()


def test_subcommand_inspect_cli_wiring_and_missing(monkeypatch, capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        sd = tmppath / ".aider-sessions" / "i1"
        sd.mkdir(parents=True)
        (sd / "state.json").write_text(json.dumps({
            "task_id": "i1", "status": "AWAITING_DECISION",
            "stage": "IMPLEMENTATION", "pending_qid": "q_1", "turn_count": 2,
        }))
        (sd / "outbox.jsonl").write_text(json.dumps({"qid": "q_1", "question": "Use SQLite?"}) + "\n")

        monkeypatch.setattr(sys, "argv", [
            "aider_bridge.py", "inspect", "--task-id", "i1", "--json", "--cwd", str(tmppath)
        ])
        aider_bridge.main()
        data = json.loads(capsys.readouterr().out)
        assert data["task_id"] == "i1"
        assert data["latest_outbox"]["question"] == "Use SQLite?"

        # Missing session exits 1
        monkeypatch.setattr(sys, "argv", [
            "aider_bridge.py", "inspect", "--task-id", "missing", "--cwd", str(tmppath)
        ])
        with pytest.raises(SystemExit) as exc:
            aider_bridge.main()
        assert exc.value.code == 1


def test_subcommand_compact_log_cli_wiring_and_missing(monkeypatch, capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        log_file = tmppath / "run.log"
        log_file.write_text("\n".join(f"\x1b[31mline {i}\x1b[0m" for i in range(10)))

        monkeypatch.setattr(sys, "argv", [
            "aider_bridge.py", "compact-log", "--log-file", str(log_file), "--max-lines", "3"
        ])
        aider_bridge.main()
        out = capsys.readouterr().out.strip().splitlines()
        assert out == ["line 7", "line 8", "line 9"]

        monkeypatch.setattr(sys, "argv", [
            "aider_bridge.py", "compact-log", "--log-file", str(tmppath / "nope.log")
        ])
        with pytest.raises(SystemExit) as exc:
            aider_bridge.main()
        assert exc.value.code == 1


def test_subcommand_merge_worktree_cli_wiring(monkeypatch):
    called = {}

    def fake_merge(task_id, cwd):
        called["task_id"] = task_id
        called["cwd"] = cwd

    monkeypatch.setattr(aider_bridge, "merge_worktree_task", fake_merge)
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr(sys, "argv", [
            "aider_bridge.py", "merge-worktree", "--task-id", "mw1", "--cwd", tmpdir
        ])
        aider_bridge.main()
        assert called["task_id"] == "mw1"
        assert called["cwd"] == Path(tmpdir).resolve()


def test_run_turn_exit_codes_diff_fail_invocation_fail_and_missing_key(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Exit 1: missing API key
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(SystemExit) as exc:
            aider_bridge.run_turn("k1", "msg", ["a.py"], tmppath)
        assert exc.value.code == 1

        # Exit 12: path traversal via read_files
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        with pytest.raises(SystemExit) as exc:
            aider_bridge.run_turn("k2", "msg", [], tmppath, read_files=["../secret.txt"])
        assert exc.value.code == 12

        # Exit 11: diff apply failed
        _mock_aider_subprocess(monkeypatch, stdout="Failed to apply edit to a.py\n")
        with pytest.raises(SystemExit) as exc:
            aider_bridge.run_turn("k3", "msg", ["a.py"], tmppath)
        assert exc.value.code == 11
        assert json.loads((tmppath / ".aider-sessions" / "k3" / "state.json").read_text())["status"] == "FAILED"

        # Exit 13: subprocess invocation raises OSError
        _mock_aider_subprocess(monkeypatch, raise_exc=OSError("uvx missing"))
        with pytest.raises(SystemExit) as exc:
            aider_bridge.run_turn("k4", "msg", ["a.py"], tmppath)
        assert exc.value.code == 13
        assert json.loads((tmppath / ".aider-sessions" / "k4" / "state.json").read_text())["status"] == "FAILED"


def test_cli_construction_non_deepseek_passes_thinking_and_reasoning_flags(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        calls = _mock_aider_subprocess(monkeypatch, stdout="Applied edit to app.py\n")

        monkeypatch.setattr(sys, "argv", [
            "aider_bridge.py", "turn",
            "--task-id", "non_ds",
            "--message", "Edit app",
            "--files", "app.py",
            "--model", "openai/o3-mini",
            "--thinking", "off",
            "--reasoning-effort", "low",
            "--cwd", str(tmppath),
        ])
        with pytest.raises(SystemExit) as exc:
            aider_bridge.main()
        assert exc.value.code == 0

        cmd, yaml_text = _extract_cmd_and_yaml(calls)
        assert yaml_text is None
        assert cmd[cmd.index("--thinking-tokens") + 1] == "0"
        assert cmd[cmd.index("--reasoning-effort") + 1] == "low"
