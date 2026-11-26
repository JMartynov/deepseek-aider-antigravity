"""
Unit tests for the Smart Pre-Task Repo Inspection Gateway
(parse_context_output, discover_read_files, run_scope, --auto-scope wiring).
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

def _mock_scope_subprocess(monkeypatch, stdout="", stderr="", returncode=0, raise_exc=None):
    calls = []

    def mock_run(cmd, cwd=None, capture_output=True, text=True):
        if raise_exc is not None:
            raise raise_exc
        calls.append(list(cmd))

        class Ret:
            pass

        ret = Ret()
        ret.returncode = returncode
        ret.stdout = stdout
        ret.stderr = stderr
        return ret

    monkeypatch.setattr(aider_bridge.subprocess, "run", mock_run)
    return calls


# ---------------------------------------------------------------------------
# 1. parse_context_output
# ---------------------------------------------------------------------------

def test_parse_context_output_backtick_and_bullet_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "app.py").write_text("x")
        (tmppath / "lib").mkdir()
        (tmppath / "lib" / "util.py").write_text("y")
        output = "ALL files\n- app.py\n  `lib/util.py`\n"
        res = aider_bridge.parse_context_output(output, tmppath)
        assert res["editable_files"] == ["app.py", "lib/util.py"]


def test_parse_context_output_bare_path_token():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "svc").mkdir()
        (tmppath / "svc" / "core.py").write_text("x")
        res = aider_bridge.parse_context_output("svc/core.py\n", tmppath)
        assert res["editable_files"] == ["svc/core.py"]


def test_parse_context_output_nonexistent_path_excluded():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        res = aider_bridge.parse_context_output("- ghost.py\n", tmppath)
        assert res["editable_files"] == []


def test_parse_context_output_traversal_excluded():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath.parent / "scope_secret.py").write_text("s")
        try:
            res = aider_bridge.parse_context_output("- ../scope_secret.py\n", tmppath)
            assert res["editable_files"] == []
        finally:
            (tmppath.parent / "scope_secret.py").unlink(missing_ok=True)


def test_parse_context_output_absolute_path_excluded():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "app.py").write_text("x")
        abs_path = str((tmppath / "app.py").resolve())
        res = aider_bridge.parse_context_output(f"- {abs_path}\n", tmppath)
        assert res["editable_files"] == []


def test_parse_context_output_dedup_preserves_order():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "app.py").write_text("x")
        (tmppath / "lib").mkdir()
        (tmppath / "lib" / "util.py").write_text("y")
        output = "- app.py\n- lib/util.py\n- app.py\n"
        res = aider_bridge.parse_context_output(output, tmppath)
        assert res["editable_files"] == ["app.py", "lib/util.py"]


def test_parse_context_output_other_section_excludes_editable_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "app.py").write_text("x")
        (tmppath / "other.py").write_text("y")
        output = (
            "ALL files\n"
            "- app.py\n"
            "Relevant symbols from OTHER files\n"
            "- other.py\n"
        )
        res = aider_bridge.parse_context_output(output, tmppath)
        assert res["editable_files"] == ["app.py"]


def test_parse_context_output_other_section_still_collects_symbols():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        output = (
            "ALL files\n"
            "- app.py\n"
            "Relevant symbols from OTHER files\n"
            "- `ExternalWidget`\n"
        )
        res = aider_bridge.parse_context_output(output, tmppath)
        assert "ExternalWidget" in res["relevant_symbols"]


def test_parse_context_output_backtick_symbols_dotted_and_call():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        output = "ALL files\n- `Buzzer.buzz_buzz()` method\n- `verify_token(...)`\n"
        res = aider_bridge.parse_context_output(output, tmppath)
        assert "Buzzer.buzz_buzz" in res["relevant_symbols"]
        assert "verify_token" in res["relevant_symbols"]


def test_parse_context_output_unbackticked_bullet_symbols():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        output = "ALL files\n- AlarmManager class\n- verify_token(...)\n"
        res = aider_bridge.parse_context_output(output, tmppath)
        syms = res["relevant_symbols"]
        assert "AlarmManager" in syms
        assert "verify_token" in syms
        assert "class" not in syms


def test_parse_context_output_symbol_dedup():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        output = "ALL files\n- `AlarmManager`\n- `AlarmManager`\n"
        res = aider_bridge.parse_context_output(output, tmppath)
        assert res["relevant_symbols"].count("AlarmManager") == 1


def test_parse_context_output_symbol_excludes_file_extensions():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        output = "ALL files\n- `app.py`\n"
        res = aider_bridge.parse_context_output(output, tmppath)
        assert "app.py" not in res["relevant_symbols"]


def test_parse_context_output_returns_expected_keys():
    with tempfile.TemporaryDirectory() as tmpdir:
        res = aider_bridge.parse_context_output("", Path(tmpdir))
        assert set(res.keys()) == {"editable_files", "relevant_symbols"}


# ---------------------------------------------------------------------------
# 2. discover_read_files
# ---------------------------------------------------------------------------

def test_discover_read_files_editable_test_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "test_mod.py").write_text("x")
        read_files, tests = aider_bridge.discover_read_files(["test_mod.py"], [], tmppath)
        assert "test_mod.py" in tests


def test_discover_read_files_finds_tests_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "app.py").write_text("x")
        (tmppath / "tests").mkdir()
        (tmppath / "tests" / "test_app.py").write_text("t")
        read_files, tests = aider_bridge.discover_read_files(["app.py"], [], tmppath)
        assert "tests/test_app.py" in tests
        assert "tests/test_app.py" in read_files


def test_discover_read_files_finds_tests_unit_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "app.py").write_text("x")
        (tmppath / "tests" / "unit").mkdir(parents=True)
        (tmppath / "tests" / "unit" / "test_app.py").write_text("t")
        read_files, tests = aider_bridge.discover_read_files(["app.py"], [], tmppath)
        assert "tests/unit/test_app.py" in tests


def test_discover_read_files_symbol_scan():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "app.py").write_text("x")
        (tmppath / "core.py").write_text("class AlarmManager:\n    pass\n")
        read_files, _ = aider_bridge.discover_read_files(["app.py"], ["AlarmManager"], tmppath)
        assert "core.py" in read_files
        assert "app.py" not in read_files


def test_discover_read_files_skips_ignored_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        for d in (".git", ".venv", ".worktrees"):
            (tmppath / d).mkdir()
            (tmppath / d / "skip.py").write_text("AlarmManager")
        (tmppath / "good.py").write_text("AlarmManager")
        read_files, _ = aider_bridge.discover_read_files([], ["AlarmManager"], tmppath)
        assert "good.py" in read_files
        assert not any(
            r.startswith(".git/") or r.startswith(".venv/") or r.startswith(".worktrees/")
            for r in read_files
        )


def test_discover_read_files_max_symbol_files_cap():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        for i in range(5):
            (tmppath / f"f{i}.py").write_text("AlarmManager")
        read_files, _ = aider_bridge.discover_read_files(
            [], ["AlarmManager"], tmppath, max_symbol_files=2
        )
        assert len(read_files) == 2


def test_discover_read_files_excludes_editable_from_read():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "app.py").write_text("AlarmManager")
        read_files, _ = aider_bridge.discover_read_files(["app.py"], ["AlarmManager"], tmppath)
        assert "app.py" not in read_files


def test_discover_read_files_dedup_read_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "app.py").write_text("x")
        (tmppath / "tests").mkdir()
        (tmppath / "tests" / "test_app.py").write_text("class AlarmManager: pass\n")
        read_files, _ = aider_bridge.discover_read_files(
            ["app.py"], ["AlarmManager"], tmppath
        )
        assert read_files.count("tests/test_app.py") == 1


# ---------------------------------------------------------------------------
# 3. create_ephemeral_aiderignore
# ---------------------------------------------------------------------------

def test_create_ephemeral_aiderignore_base_patterns():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        p = aider_bridge.create_ephemeral_aiderignore(tmppath)
        text = p.read_text()
        for pat in (".worktrees/", "artifacts/", "passports/", "advisories/", ".aider-sessions/"):
            assert pat in text


def test_create_ephemeral_aiderignore_scope_mode_adds_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        p = aider_bridge.create_ephemeral_aiderignore(tmppath, scope_mode=True)
        assert "*.json" in p.read_text()


def test_create_ephemeral_aiderignore_default_no_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        p = aider_bridge.create_ephemeral_aiderignore(tmppath)
        assert "*.json" not in p.read_text()


# ---------------------------------------------------------------------------
# 4. create_ephemeral_model_settings edit_format
# ---------------------------------------------------------------------------

def test_create_ephemeral_model_settings_edit_format_none_omits():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        p = aider_bridge.create_ephemeral_model_settings(
            tmppath, "deepseek/deepseek-flash", edit_format=None
        )
        assert p is not None
        text = p.read_text()
        assert "- name: deepseek/deepseek-flash" in text
        assert "edit_format: diff" not in text
        assert "caches_by_default: true" in text


def test_create_ephemeral_model_settings_default_edit_format():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        p = aider_bridge.create_ephemeral_model_settings(
            tmppath, "deepseek/deepseek-flash"
        )
        assert p is not None
        assert "edit_format: diff" in p.read_text()


def test_create_ephemeral_model_settings_editor_block_unchanged():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        p = aider_bridge.create_ephemeral_model_settings(
            tmppath,
            "deepseek/deepseek-reasoner",
            editor_model="deepseek/deepseek-flash",
            architect=True,
            edit_format=None,
        )
        assert p is not None
        text = p.read_text()
        assert "edit_format: editor-diff" in text


# ---------------------------------------------------------------------------
# 5. run_scope
# ---------------------------------------------------------------------------

def test_run_scope_cli_flags_and_cleanup(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "app.py").write_text("x")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        calls = _mock_scope_subprocess(monkeypatch, stdout="ALL files\n- app.py\n")
        result = aider_bridge.run_scope("Inspect app", tmppath)

        cmd = calls[0]
        assert cmd[cmd.index("--model") + 1] == "deepseek/deepseek-flash"
        assert cmd[cmd.index("--chat-mode") + 1] == "context"
        assert cmd[cmd.index("--map-tokens") + 1] == "1536"
        assert cmd[cmd.index("--map-multiplier-no-files") + 1] == "1"
        assert "--no-auto-commits" in cmd
        assert "--yes-always" in cmd
        assert cmd[cmd.index("--aiderignore") + 1].endswith(".aiderignore.ephemeral")
        assert cmd[cmd.index("--message") + 1] == "Inspect app"
        assert "--model-settings-file" in cmd
        assert "--thinking-tokens" not in cmd
        assert "--no-git" in cmd
        assert result["editable_files"] == ["app.py"]
        assert result["confidence"] == "high"
        assert not (tmppath / ".aiderignore.ephemeral").exists()
        assert not list(tmppath.glob(".aider.model.settings.*.yml"))


def test_run_scope_json_output(monkeypatch, capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "app.py").write_text("x")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        _mock_scope_subprocess(monkeypatch, stdout="ALL files\n- app.py\n")
        aider_bridge.run_scope("Inspect app", tmppath, as_json=True)
        out = capsys.readouterr().out
        data = json.loads(out)
        for key in (
            "editable_files", "relevant_symbols", "read_files",
            "suggested_test_cmd", "suggested_tier", "confidence",
            "tokens_sent", "cache_write", "cache_hit", "tokens_rcvd",
            "cost_usd", "session_cost_usd",
        ):
            assert key in data


def test_run_scope_missing_api_key_exit_1(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(SystemExit) as exc:
            aider_bridge.run_scope("x", Path(tmpdir))
        assert exc.value.code == 1


def test_run_scope_invocation_failure_exit_13(monkeypatch, capsys):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    with tempfile.TemporaryDirectory() as tmpdir:
        _mock_scope_subprocess(monkeypatch, raise_exc=OSError("uvx missing"))
        with pytest.raises(SystemExit) as exc:
            aider_bridge.run_scope("x", Path(tmpdir))
        assert exc.value.code == 13
        err = capsys.readouterr().err
        assert "AIDER_INVOCATION_FAILED" in err


def test_run_scope_nonzero_returncode(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    with tempfile.TemporaryDirectory() as tmpdir:
        _mock_scope_subprocess(monkeypatch, returncode=2, stderr="boom")
        with pytest.raises(SystemExit) as exc:
            aider_bridge.run_scope("x", Path(tmpdir))
        assert exc.value.code == 2


def test_run_scope_suggested_test_cmd_high_confidence(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "app.py").write_text("x")
        (tmppath / "tests").mkdir()
        (tmppath / "tests" / "test_app.py").write_text("t")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        _mock_scope_subprocess(monkeypatch, stdout="ALL files\n- app.py\n")
        result = aider_bridge.run_scope("Implement app", tmppath)
        assert result["confidence"] == "high"
        assert result["editable_files"] == ["app.py"]
        assert result["suggested_test_cmd"] == "uv run pytest tests/test_app.py -q"
        assert result["suggested_tier"] in ("patch", "surgical", "architect", "research")


def test_run_scope_low_confidence_when_no_files(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        _mock_scope_subprocess(monkeypatch, stdout="No relevant files found.\n")
        result = aider_bridge.run_scope("Research asyncio", tmppath, as_json=True)
        assert result["confidence"] == "low"
        assert result["editable_files"] == []
        assert result["suggested_test_cmd"] is None


def test_run_scope_non_deepseek_uses_thinking_tokens(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        calls = _mock_scope_subprocess(monkeypatch, stdout="")
        aider_bridge.run_scope("x", tmppath, model="openai/gpt-4o", thinking="off")
        cmd = calls[0]
        assert "--model-settings-file" not in cmd
        assert cmd[cmd.index("--thinking-tokens") + 1] == "0"


def test_run_scope_omits_no_git_when_git_present(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / ".git").mkdir()
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        calls = _mock_scope_subprocess(monkeypatch, stdout="")
        aider_bridge.run_scope("x", tmppath)
        assert "--no-git" not in calls[0]


# ---------------------------------------------------------------------------
# 6. dispatch --auto-scope
# ---------------------------------------------------------------------------

def test_dispatch_auto_scope_populates_files(monkeypatch):
    monkeypatch.setattr(aider_bridge, "run_scope", lambda **k: {
        "editable_files": ["app.py"],
        "read_files": ["spec.md"],
        "suggested_test_cmd": "uv run pytest tests/test_app.py -q",
        "confidence": "high",
    })
    captured = {}

    def fake_run_turn(task_id, message, files, cwd, **kwargs):
        captured["files"] = list(files)
        captured["kwargs"] = kwargs

    monkeypatch.setattr(aider_bridge, "run_turn", fake_run_turn)
    monkeypatch.setattr(sys, "argv", [
        "aider_bridge.py", "dispatch",
        "--task-id", "as_pop",
        "--message", "Implement app",
        "--auto-scope",
        "--cwd", ".",
    ])
    aider_bridge.main()
    assert captured["files"] == ["app.py"]
    assert captured["kwargs"]["read_files"] == ["spec.md"]
    assert captured["kwargs"]["test_cmd"] == "uv run pytest tests/test_app.py -q"


def test_dispatch_auto_scope_merges_read_files_dedup(monkeypatch):
    monkeypatch.setattr(aider_bridge, "run_scope", lambda **k: {
        "editable_files": ["app.py"],
        "read_files": ["spec.md", "app.py", "extra.md"],
        "suggested_test_cmd": None,
        "confidence": "high",
    })
    captured = {}

    def fake_run_turn(task_id, message, files, cwd, **kwargs):
        captured["kwargs"] = kwargs

    monkeypatch.setattr(aider_bridge, "run_turn", fake_run_turn)
    monkeypatch.setattr(sys, "argv", [
        "aider_bridge.py", "dispatch",
        "--task-id", "as_merge",
        "--message", "Implement app",
        "--read-files", "spec.md",
        "--auto-scope",
        "--cwd", ".",
    ])
    aider_bridge.main()
    assert captured["kwargs"]["read_files"] == ["spec.md", "extra.md"]


def test_dispatch_auto_scope_dry_run(monkeypatch, capsys):
    monkeypatch.setattr(aider_bridge, "run_scope", lambda **k: {
        "editable_files": ["app.py"],
        "read_files": ["spec.md"],
        "suggested_test_cmd": "uv run pytest tests/test_app.py -q",
        "confidence": "high",
    })

    def boom(*a, **k):
        raise AssertionError("run_turn must not be called in --dry-run")

    monkeypatch.setattr(aider_bridge, "run_turn", boom)
    monkeypatch.setattr(sys, "argv", [
        "aider_bridge.py", "dispatch",
        "--task-id", "as_dry",
        "--message", "Implement app",
        "--auto-scope", "--dry-run", "--cwd", ".",
    ])
    with pytest.raises(SystemExit) as exc:
        aider_bridge.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "test_app.py" in out
    assert "[STATUS] DRY_RUN" in out


def test_dispatch_auto_scope_low_confidence_exit_14(monkeypatch, capsys):
    monkeypatch.setattr(aider_bridge, "run_scope", lambda **k: {
        "editable_files": [],
        "read_files": [],
        "suggested_test_cmd": None,
        "confidence": "low",
    })

    def boom(*a, **k):
        raise AssertionError("run_turn must not be called")

    monkeypatch.setattr(aider_bridge, "run_turn", boom)
    monkeypatch.setattr(sys, "argv", [
        "aider_bridge.py", "dispatch",
        "--task-id", "as_low",
        "--message", "Find something",
        "--auto-scope", "--cwd", ".",
    ])
    with pytest.raises(SystemExit) as exc:
        aider_bridge.main()
    assert exc.value.code == 14
    out = capsys.readouterr().out
    assert "low confidence" in out


def test_dispatch_explicit_files_skips_scope(monkeypatch):
    def scope_boom(**k):
        raise AssertionError("run_scope must not run when --files is provided")

    monkeypatch.setattr(aider_bridge, "run_scope", scope_boom)
    captured = {}

    def fake_run_turn(task_id, message, files, cwd, **kwargs):
        captured["files"] = list(files)

    monkeypatch.setattr(aider_bridge, "run_turn", fake_run_turn)
    monkeypatch.setattr(sys, "argv", [
        "aider_bridge.py", "dispatch",
        "--task-id", "ef1",
        "--message", "Edit module",
        "--files", "mod.py",
        "--auto-scope",
        "--cwd", ".",
    ])
    aider_bridge.main()
    assert captured["files"] == ["mod.py"]


# ---------------------------------------------------------------------------
# 7. main() CLI wiring
# ---------------------------------------------------------------------------

def test_main_scope_cli_wiring(monkeypatch):
    called = {}

    def fake_scope(message, cwd, map_tokens=1536, as_json=False,
                   model="deepseek/deepseek-flash", thinking="off", verbose=False):
        called.update({
            "message": message, "cwd": cwd, "map_tokens": map_tokens,
            "as_json": as_json, "model": model, "thinking": thinking,
            "verbose": verbose,
        })
        return {}

    monkeypatch.setattr(aider_bridge, "run_scope", fake_scope)
    monkeypatch.setattr(sys, "argv", [
        "aider_bridge.py", "scope",
        "--message", "Find the module",
        "--map-tokens", "2048",
        "--json",
        "--model", "deepseek/deepseek-reasoner",
        "--thinking", "on",
        "-v",
        "--cwd", ".",
    ])
    aider_bridge.main()
    assert called["message"] == "Find the module"
    assert called["map_tokens"] == 2048
    assert called["as_json"] is True
    assert called["model"] == "deepseek/deepseek-reasoner"
    assert called["thinking"] == "on"
    assert called["verbose"] is True
    assert called["cwd"] == Path(".").resolve()


def test_main_dispatch_auto_scope_wiring(monkeypatch):
    called = {}

    def fake_dispatch(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs

    monkeypatch.setattr(aider_bridge, "run_dispatch", fake_dispatch)
    monkeypatch.setattr(sys, "argv", [
        "aider_bridge.py", "dispatch",
        "--task-id", "d1",
        "--message", "Do work",
        "--auto-scope",
        "--cwd", ".",
    ])
    aider_bridge.main()
    assert called["kwargs"]["auto_scope"] is True
