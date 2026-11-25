"""
Suite 6: Enterprise Jules Best Practices.
Tests:
1. Log compactor: ANSI stripping and tail extraction (Assertion Sieve)
2. Pre-flight sanity linter: Uncommitted changes warning and missing read-only ref failure (exit 14)
3. Worktree isolation lifecycle:
   - create_task_worktree creates isolated branch and directory
   - Idempotent worktree reuse on resumption
   - merge_and_remove_worktree cleanly merges and cleans worktree
"""

import sys
import json
import tempfile
import subprocess
import urllib.error
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import aider_bridge


class _MockHTTPResponse:
    def __init__(self, body_bytes):
        self._body = body_bytes
    def read(self):
        return self._body
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False


def test_compact_failure_trace_ansi_and_tail():
    raw_log = (
        "\x1b[31mE   AssertionError: assert 5 == 10\x1b[0m\n"
        "\x1b[32mtests/test_foo.py:12: AssertionError\x1b[0m\n"
        "\x1b[1m=== FAILURES ===\x1b[0m\n"
    ) * 20  # 60 lines with ANSI
    
    compacted = aider_bridge.compact_failure_trace(raw_log, max_lines=15)
    assert "\x1b[" not in compacted
    assert len(compacted.splitlines()) <= 15
    assert "AssertionError" in compacted


def test_preflight_checks_fails_on_missing_read_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        import pytest
        with pytest.raises(SystemExit) as exc:
            aider_bridge.run_preflight_checks(
                read_files=["missing_contract.py"],
                cwd=tmppath
            )
        assert exc.value.code == 14


def test_preflight_checks_injects_pythonpath():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        src_dir = tmppath / "src"
        src_dir.mkdir()
        import os
        old_pp = os.environ.get("PYTHONPATH", "")
        try:
            aider_bridge.run_preflight_checks(read_files=[], cwd=tmppath)
            assert str(src_dir.resolve()) in os.environ.get("PYTHONPATH", "")
        finally:
            os.environ["PYTHONPATH"] = old_pp


def test_preflight_checks_warns_on_dirty_tree(monkeypatch, capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / ".git").mkdir()
        
        def mock_run(cmd, cwd, capture_output=True, text=True):
            class Ret:
                returncode = 0
                stdout = " M dirty_file.py\n"
            return Ret()
            
        monkeypatch.setattr(aider_bridge.subprocess, "run", mock_run)
        aider_bridge.run_preflight_checks(read_files=[], cwd=tmppath)
        
        captured = capsys.readouterr()
        assert "[WARN]" in captured.err


def test_worktree_creation_idempotency_and_cleanup():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        # Initialize real git repo
        subprocess.run(["git", "init"], cwd=tmppath, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "TestUser"], cwd=tmppath, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmppath, check=True)
        (tmppath / "main_file.py").write_text("# main branch initial")
        subprocess.run(["git", "add", "."], cwd=tmppath, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=tmppath, check=True)

        # 1. Create worktree
        wt_path, branch = aider_bridge.create_task_worktree("unit_test_wt", tmppath)
        assert wt_path.exists()
        assert branch == "task/unit_test_wt"
        
        # 2. Idempotency test (call again)
        wt_path2, branch2 = aider_bridge.create_task_worktree("unit_test_wt", tmppath)
        assert wt_path2 == wt_path
        assert branch2 == branch

        # 3. Modify inside worktree
        (wt_path / "worktree_file.py").write_text("# created inside worktree")
        subprocess.run(["git", "add", "."], cwd=wt_path, check=True)
        subprocess.run(["git", "commit", "-m", "Worktree commit"], cwd=wt_path, check=True)

        # 4. Merge and remove worktree
        aider_bridge.merge_and_remove_worktree("unit_test_wt", wt_path, branch, tmppath, auto_merge=True)
        assert not wt_path.exists()
        assert (tmppath / "worktree_file.py").exists()


def test_ephemeral_model_settings_creation_and_cleanup():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        settings_path = aider_bridge.create_ephemeral_model_settings(tmppath, "deepseek/deepseek-flash")
        assert settings_path is not None
        assert settings_path.exists()
        content = settings_path.read_text()
        assert "name: deepseek/deepseek-flash" in content
        assert "caches_by_default: true" in content
        assert "edit_format: diff" in content
        
        # Test non-deepseek model returns None
        none_settings = aider_bridge.create_ephemeral_model_settings(tmppath, "gpt-4o")
        assert none_settings is None

        # Test cleanup
        aider_bridge.remove_ephemeral_file(settings_path)
        assert not settings_path.exists()


def test_worktree_staged_multi_pass_and_explicit_merge():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        # Initialize repo
        subprocess.run(["git", "init"], cwd=tmppath, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "TestUser"], cwd=tmppath, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmppath, check=True)
        (tmppath / "base.txt").write_text("initial base content\n")
        subprocess.run(["git", "add", "."], cwd=tmppath, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=tmppath, check=True)

        task_id = "staged_task_42"
        wt_path, branch = aider_bridge.create_task_worktree(task_id, tmppath)
        assert wt_path.exists()

        # Pass 1 inside worktree
        (wt_path / "feature.py").write_text("def feature(): return 1\n")
        subprocess.run(["git", "add", "feature.py"], cwd=wt_path, check=True)
        subprocess.run(["git", "commit", "-m", "Pass 1 implementation"], cwd=wt_path, check=True)

        # Worktree kept without auto_merge
        aider_bridge.merge_and_remove_worktree(task_id, wt_path, branch, tmppath, auto_merge=False)
        assert not (tmppath / "feature.py").exists() # Not on main yet!

        # Re-create/reuse for Pass 2 (review & polish)
        wt_path2, branch2 = aider_bridge.create_task_worktree(task_id, tmppath)
        assert wt_path2.exists()
        (wt_path2 / "feature.py").write_text("def feature(): return 42\n")
        subprocess.run(["git", "add", "feature.py"], cwd=wt_path2, check=True)
        subprocess.run(["git", "commit", "-m", "Pass 2 review refinement"], cwd=wt_path2, check=True)

        # Orchestrator runs explicit merge_worktree_task
        aider_bridge.merge_worktree_task(task_id, tmppath)

        assert not wt_path2.exists()
        assert (tmppath / "feature.py").exists()
        assert "return 42" in (tmppath / "feature.py").read_text()


def test_with_git_lock_mutex_serialization():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / ".git").mkdir()

        # Acquire lock
        with aider_bridge.with_git_lock(tmppath, timeout=1.0) as lock1:
            assert lock1.exists()

            # Second concurrent lock should timeout
            import pytest
            with pytest.raises(TimeoutError):
                with aider_bridge.with_git_lock(tmppath, timeout=0.2, retry_interval=0.05):
                    pass

        # After exiting, lock must be released
        assert not lock1.exists()


def test_auto_rebase_on_advancing_base_branch():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        # Initialize repo with main branch
        subprocess.run(["git", "init", "-b", "main"], cwd=tmppath, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "TestUser"], cwd=tmppath, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmppath, check=True)
        (tmppath / "base.txt").write_text("v1\n")
        subprocess.run(["git", "add", "."], cwd=tmppath, check=True)
        subprocess.run(["git", "commit", "-m", "Base v1"], cwd=tmppath, check=True)

        # Worker A: create task/a branch and worktree
        wt_a, branch_a = aider_bridge.create_task_worktree("task_a", tmppath)
        (wt_a / "feature_a.py").write_text("def a(): pass\n")
        subprocess.run(["git", "add", "."], cwd=wt_a, check=True)
        subprocess.run(["git", "commit", "-m", "Commit in Task A"], cwd=wt_a, check=True)

        # Main advances (e.g. Worker B landed first)
        (tmppath / "base.txt").write_text("v2 advanced\n")
        subprocess.run(["git", "add", "."], cwd=tmppath, check=True)
        subprocess.run(["git", "commit", "-m", "Base advanced to v2"], cwd=tmppath, check=True)

        # Now Worker A merges using merge_worktree_task (which auto-rebases under mutex)
        aider_bridge.merge_worktree_task("task_a", tmppath)

        assert not wt_a.exists()
        assert (tmppath / "feature_a.py").exists()
        assert (tmppath / "base.txt").read_text() == "v2 advanced\n"

        # Check linear log history
        log_res = subprocess.run(["git", "log", "--oneline"], cwd=tmppath, capture_output=True, text=True)
        assert "Commit in Task A" in log_res.stdout
        assert "Base advanced to v2" in log_res.stdout


def test_filter_turn_output_strips_thinking_and_banners():
    sample_raw = (
        "Detected dumb terminal, disabling fancy input and pretty output.\n"
        "Aider v0.86.2\n"
        "Cost estimates may be inaccurate when using streaming and caching.\n"
        "Scraping https://github.com/example/repo...\n"
        "Added src/foo.py to the chat.\n"
        "► **THINKING**\n"
        "This is internal chain of thought with thousands of tokens of reasoning\n"
        "that shouldn't waste orchestrator context.\n"
        "► **ANSWER**\n"
        "Applied surgical unified diff:\n"
        "```diff\n"
        "+ def foo(): return True\n"
        "```\n"
    )

    filtered = aider_bridge.filter_turn_output(sample_raw, verbose=False)
    assert "Detected dumb terminal" not in filtered
    assert "Aider v0.86.2" not in filtered
    assert "Scraping https" not in filtered
    assert "► **THINKING**" not in filtered
    assert "internal chain of thought" not in filtered
    assert "Added src/foo.py to the chat." in filtered
    assert "+ def foo(): return True" in filtered

    # Verbose keeps thinking
    verbose_out = aider_bridge.filter_turn_output(sample_raw, verbose=True)
    assert "► **THINKING**" in verbose_out


def _invoke_turn_and_capture(monkeypatch, tmppath, argv_tail):
    """Helper: run main() with a mocked subprocess.run and return the aider (uvx) cmd."""
    import pytest
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    captured = {}

    def mock_run(cmd, cwd=None, capture_output=True, text=True):
        # Only capture the aider invocation; ignore git/worktree housekeeping cmds.
        if cmd and cmd[0] == "uvx":
            captured["cmd"] = list(cmd)
        class Ret:
            returncode = 0
            stdout = ""
            stderr = ""
        return Ret()

    monkeypatch.setattr(aider_bridge.subprocess, "run", mock_run)
    monkeypatch.setattr(aider_bridge, "fetch_account_balance", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["aider_bridge.py", "turn"] + argv_tail + ["--cwd", str(tmppath)])

    with pytest.raises(SystemExit) as exc:
        aider_bridge.main()
    assert exc.value.code == 0
    assert "cmd" in captured, "Aider (uvx) subprocess was not invoked"
    return captured["cmd"]


def test_test_cmd_argument_wiring(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        cmd = _invoke_turn_and_capture(
            monkeypatch, tmppath,
            ["--task-id", "t1", "--message", "m", "--files", "foo.py", "--test-cmd", "pytest tests/"],
        )
        assert "--auto-test" in cmd
        idx = cmd.index("--test-cmd")
        assert cmd[idx + 1] == "pytest tests/"


def test_dry_run_argument_wiring(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        cmd = _invoke_turn_and_capture(
            monkeypatch, tmppath,
            ["--task-id", "t2", "--message", "m", "--files", "foo.py", "--dry-run"],
        )
        assert "--dry-run" in cmd


def test_architect_argument_wiring(monkeypatch):
    # Architect with default flash model: --architect only, no editor override
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = _invoke_turn_and_capture(
            monkeypatch, Path(tmpdir),
            ["--task-id", "t3", "--message", "m", "--files", "foo.py", "--architect"],
        )
        assert "--architect" in cmd
        assert "--editor-model" not in cmd

    # Architect with reasoner model: editor defaults to deepseek/deepseek-flash
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = _invoke_turn_and_capture(
            monkeypatch, Path(tmpdir),
            ["--task-id", "t4", "--message", "m", "--files", "foo.py",
             "--model", "deepseek/deepseek-reasoner", "--architect"],
        )
        assert "--architect" in cmd
        idx = cmd.index("--editor-model")
        assert cmd[idx + 1] == "deepseek/deepseek-flash"

    # Explicit --editor-model override wins over the reasoner default
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = _invoke_turn_and_capture(
            monkeypatch, Path(tmpdir),
            ["--task-id", "t5", "--message", "m", "--files", "foo.py",
             "--model", "deepseek/deepseek-reasoner", "--architect",
             "--editor-model", "custom/editor"],
        )
        assert "--architect" in cmd
        idx = cmd.index("--editor-model")
        assert cmd[idx + 1] == "custom/editor"


# ---------------------------------------------------------------------------
# Smart DeepSeek Classifier & Dispatcher
# ---------------------------------------------------------------------------

def test_classify_task_tier1_patch():
    tier, reason, params = aider_bridge.classify_task(
        "Fix typo in the README", files=["README.md"])
    assert tier == "patch"
    assert params["tier_num"] == 1
    assert params["tier_label"] == "PATCH"
    assert params["model"] == "deepseek/deepseek-flash"
    assert params["thinking"] == "off"
    assert params["reasoning_effort"] is None
    assert params["edit_format"] == "diff"
    assert params["map_tokens"] == 0
    assert params["architect"] is False
    assert params["editor_model"] is None


def test_classify_task_tier1_patch_by_extension():
    tier, reason, params = aider_bridge.classify_task(
        "Update the config", files=["pyproject.toml"])
    assert tier == "patch"
    assert params["tier_num"] == 1


def test_classify_task_tier2_surgical():
    tier, reason, params = aider_bridge.classify_task(
        "Implement verify_token", files=["auth.py"],
        test_cmd="pytest tests/test_auth.py -q")
    assert tier == "surgical"
    assert params["tier_num"] == 2
    assert params["tier_label"] == "SURGICAL"
    assert params["model"] == "deepseek/deepseek-flash"
    assert params["thinking"] == "auto"
    assert params["reasoning_effort"] == "low"
    assert params["map_tokens"] == 512
    assert params["architect"] is False
    assert params["test_cmd"] == "pytest tests/test_auth.py -q"


def test_classify_task_tier2_surgical_no_test_cmd():
    tier, reason, params = aider_bridge.classify_task(
        "Fix bug in the parser", files=["parser.py"])
    assert tier == "surgical"
    assert params["tier_num"] == 2


def test_classify_task_tier3_architect_by_file_count():
    tier, reason, params = aider_bridge.classify_task(
        "Implement feature", files=["a.py", "b.py", "c.py", "d.py"])
    assert tier == "architect"
    assert params["tier_num"] == 3
    assert params["tier_label"] == "ARCHITECT"
    assert params["model"] == "deepseek/deepseek-reasoner"
    assert params["thinking"] == "on"
    assert params["reasoning_effort"] == "high"
    assert params["map_tokens"] == 512
    assert params["architect"] is True
    assert params["editor_model"] == "deepseek/deepseek-flash"


def test_classify_task_tier3_architect_by_keyword():
    tier, reason, params = aider_bridge.classify_task(
        "Refactor the sqlglot visitor", files=["visitors.py"])
    assert tier == "architect"
    assert params["tier_num"] == 3


def test_classify_task_tier4_research_no_files():
    tier, reason, params = aider_bridge.classify_task("What is PEP 695?")
    assert tier == "research"
    assert params["tier_num"] == 4
    assert params["tier_label"] == "RESEARCH"
    assert params["command"] == "search"
    assert params["query"] == "What is PEP 695?"
    assert params["model"] is None
    assert params["test_cmd"] is None


def test_classify_task_tier4_research_query_truncated():
    long_msg = "research " + ("x" * 300)
    tier, reason, params = aider_bridge.classify_task(long_msg)
    assert tier == "research"
    assert len(params["query"]) == 128


def test_classify_task_manual_override():
    tier, reason, params = aider_bridge.classify_task(
        "Refactor everything", files=["a.py", "b.py", "c.py", "d.py"],
        forced_tier="patch")
    assert tier == "patch"
    assert "manual override" in reason
    assert params["tier_num"] == 1

    tier, reason, params = aider_bridge.classify_task(
        "typo", files=["README.md"], forced_tier="architect")
    assert tier == "architect"
    assert params["tier_num"] == 3


def test_classify_task_invalid_forced_tier():
    import pytest
    with pytest.raises(ValueError):
        aider_bridge.classify_task("msg", forced_tier="bogus")


def test_ephemeral_settings_architect_primary_and_editor():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        path = aider_bridge.create_ephemeral_model_settings(
            tmppath, "deepseek/deepseek-reasoner",
            editor_model="deepseek/deepseek-flash", architect=True,
            thinking="on", reasoning_effort="high")
        assert path is not None
        content = path.read_text()
        assert "- name: deepseek/deepseek-reasoner" in content
        assert "edit_format: diff" in content
        assert "caches_by_default: true" in content
        assert "examples_as_sys_msg: true" in content
        assert "type: enabled" in content
        assert "reasoning_effort: high" in content
        assert "- name: deepseek/deepseek-flash" in content
        assert "edit_format: editor-diff" in content
        assert "type: disabled" in content


def test_ephemeral_settings_thinking_off():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        path = aider_bridge.create_ephemeral_model_settings(
            tmppath, "deepseek/deepseek-flash", thinking="off")
        assert path is not None
        content = path.read_text()
        assert "type: disabled" in content
        assert "type: enabled" not in content


def test_dispatch_dry_run_patch(monkeypatch, capsys):
    import pytest
    called = {"run": False}

    def boom(*a, **k):
        called["run"] = True
        raise AssertionError("run_turn/run_search must not run under --dry-run")

    monkeypatch.setattr(aider_bridge, "run_turn", boom)
    monkeypatch.setattr(aider_bridge, "run_search", boom)
    monkeypatch.setattr(sys, "argv", [
        "aider_bridge.py", "dispatch", "--task-id", "d1",
        "--message", "Fix typo in docs", "--files", "README.md",
        "--tier", "auto", "--dry-run", "--cwd", ".",
    ])
    with pytest.raises(SystemExit) as exc:
        aider_bridge.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "[DISPATCHER] Classified as TIER 1: PATCH" in out
    assert "model: deepseek/deepseek-flash" in out
    assert "thinking: off" in out
    assert "map_tokens: 0" in out
    assert called["run"] is False


def test_dispatch_routes_to_run_turn(monkeypatch):
    captured = {}

    def fake_run_turn(task_id, message, files, cwd, **kwargs):
        captured["task_id"] = task_id
        captured["files"] = list(files)
        captured["kwargs"] = kwargs

    monkeypatch.setattr(aider_bridge, "run_turn", fake_run_turn)
    monkeypatch.setattr(sys, "argv", [
        "aider_bridge.py", "dispatch", "--task-id", "d2",
        "--message", "Implement verify_token", "--files", "auth.py",
        "--test-cmd", "pytest tests/test_auth.py -q", "--cwd", ".",
    ])
    aider_bridge.main()

    assert captured["task_id"] == "d2"
    assert captured["files"] == ["auth.py"]
    kw = captured["kwargs"]
    assert kw["test_cmd"] == "pytest tests/test_auth.py -q"
    assert kw["model"] == "deepseek/deepseek-flash"
    assert kw["thinking"] == "auto"
    assert kw["map_tokens"] == 512
    assert kw["architect"] is False


def test_dispatch_applies_thinking_override(monkeypatch):
    captured = {}

    def fake_run_turn(task_id, message, files, cwd, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(aider_bridge, "run_turn", fake_run_turn)
    monkeypatch.setattr(sys, "argv", [
        "aider_bridge.py", "dispatch", "--task-id", "d4",
        "--message", "Fix typo", "--files", "README.md",
        "--thinking", "on", "--reasoning-effort", "high", "--cwd", ".",
    ])
    aider_bridge.main()

    assert captured["thinking"] == "on"
    assert captured["reasoning_effort"] == "high"


def test_dispatch_routes_research_to_search(monkeypatch):
    captured = {}

    def fake_run_search(query, cwd, **kwargs):
        captured["query"] = query

    monkeypatch.setattr(aider_bridge, "run_search", fake_run_search)
    monkeypatch.setattr(sys, "argv", [
        "aider_bridge.py", "dispatch", "--task-id", "d3",
        "--message", "What is PEP 695?", "--cwd", ".",
    ])
    aider_bridge.main()

    assert captured["query"] == "What is PEP 695?"


def test_classify_task_word_boundary_prevents_ast_false_positive():
    tier, _, _ = aider_bridge.classify_task(
        "Fix bug in fast parser", files=["parser.py"])
    assert tier == "surgical"


def test_search_default_post_request(monkeypatch):
    captured = {}

    def mock_urlopen(req, timeout=None):
        captured["req"] = req
        return _MockHTTPResponse(json.dumps({"results": []}).encode())

    monkeypatch.setattr(aider_bridge.urllib.request, "urlopen", mock_urlopen)

    aider_bridge.search_web_deepseek("deepseek", "sk-test")

    req = captured["req"]
    assert req.get_full_url() == "https://api.deepseek.com/v1/tools/search"
    assert req.get_method() == "POST"
    assert req.get_header("Authorization") == "Bearer sk-test"
    assert req.get_header("Content-type") == "application/json"
    assert req.get_header("Accept") == "application/json"
    assert json.loads(req.data) == {"query": "deepseek", "count": 5, "need_content": False}


def test_search_custom_flags(monkeypatch):
    captured = {}

    def mock_urlopen(req, timeout=None):
        captured["req"] = req
        return _MockHTTPResponse(json.dumps({"results": []}).encode())

    monkeypatch.setattr(aider_bridge.urllib.request, "urlopen", mock_urlopen)

    aider_bridge.search_web_deepseek("query", "sk-test", count=10, need_content=True)

    assert json.loads(captured["req"].data) == {"query": "query", "count": 10, "need_content": True}


def test_format_search_results_basic():
    data = {"results": [{"title": "T1", "url": "https://ex.com", "summary": "S1"}]}
    out = aider_bridge.format_search_results(data, need_content=False)
    assert "T1" in out
    assert "https://ex.com" in out
    assert "S1" in out
    assert "Content:" not in out


def test_format_search_results_with_content():
    data = {"results": [{"title": "T1", "url": "u1", "summary": "s1", "content": "full body"}]}
    out = aider_bridge.format_search_results(data, need_content=True)
    assert "Content: full body" in out


def test_format_search_results_empty():
    assert aider_bridge.format_search_results({"results": []}) == "No search results found."


def test_run_search_json_cli(monkeypatch, capsys):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    resp = {"results": [{"title": "T", "url": "U", "summary": "S"}]}

    def mock_urlopen(req, timeout=None):
        return _MockHTTPResponse(json.dumps(resp).encode())

    monkeypatch.setattr(aider_bridge.urllib.request, "urlopen", mock_urlopen)
    with tempfile.TemporaryDirectory() as tmpdir:
        aider_bridge.run_search("query", Path(tmpdir), as_json=True)

    out = capsys.readouterr().out
    assert json.loads(out) == resp


def test_run_search_loads_env_from_cwd(monkeypatch):
    captured = {}

    def mock_urlopen(req, timeout=None):
        captured["req"] = req
        return _MockHTTPResponse(json.dumps({"results": []}).encode())

    monkeypatch.setattr(aider_bridge.urllib.request, "urlopen", mock_urlopen)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / ".env").write_text('DEEPSEEK_API_KEY="sk-from-env"\n')
        aider_bridge.run_search("query", tmppath)

    assert captured["req"].get_header("Authorization") == "Bearer sk-from-env"


def test_run_search_missing_key(monkeypatch):
    import pytest
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(SystemExit) as exc:
            aider_bridge.run_search("query", Path(tmpdir))
        assert exc.value.code == 1


def test_run_search_invalid_query_empty(monkeypatch):
    import pytest
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(SystemExit) as exc:
            aider_bridge.run_search("   ", Path(tmpdir))
        assert exc.value.code == 2


def test_run_search_invalid_query_overlong(monkeypatch):
    import pytest
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(SystemExit) as exc:
            aider_bridge.run_search("x" * 129, Path(tmpdir))
        assert exc.value.code == 2


def test_run_search_invalid_count(monkeypatch):
    import pytest
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    with tempfile.TemporaryDirectory() as tmpdir:
        for bad in (0, 51):
            with pytest.raises(SystemExit) as exc:
                aider_bridge.run_search("query", Path(tmpdir), count=bad)
            assert exc.value.code == 2


def test_run_search_rate_limit_429(monkeypatch):
    import pytest
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    def mock_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.get_full_url(), 429, "Too Many", {}, None)

    monkeypatch.setattr(aider_bridge.urllib.request, "urlopen", mock_urlopen)
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(SystemExit) as exc:
            aider_bridge.run_search("query", Path(tmpdir))
        assert exc.value.code == 16


def test_run_search_api_error_500(monkeypatch):
    import pytest
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    def mock_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.get_full_url(), 500, "Server Error", {}, None)

    monkeypatch.setattr(aider_bridge.urllib.request, "urlopen", mock_urlopen)
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(SystemExit) as exc:
            aider_bridge.run_search("query", Path(tmpdir))
        assert exc.value.code == 17


def test_run_search_404_fallback(monkeypatch, capsys):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    seen_urls = []

    def mock_urlopen(req, timeout=None):
        seen_urls.append(req.get_full_url())
        if len(seen_urls) == 1:
            raise urllib.error.HTTPError(req.get_full_url(), 404, "Not Found", {}, None)
        body = {
            "choices": [{
                "message": {
                    "content": json.dumps({"results": [{"title": "T", "url": "U", "summary": "S"}]})
                }
            }]
        }
        return _MockHTTPResponse(json.dumps(body).encode())

    monkeypatch.setattr(aider_bridge.urllib.request, "urlopen", mock_urlopen)
    with tempfile.TemporaryDirectory() as tmpdir:
        aider_bridge.run_search("query", Path(tmpdir))

    out = capsys.readouterr().out
    assert "T" in out
    assert "U" in out
    assert "S" in out
    assert len(seen_urls) == 2
    assert seen_urls[1] == "https://api.deepseek.com/v1/chat/completions"


def test_run_search_urlerror_network(monkeypatch):
    import pytest
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    def mock_urlopen(req, timeout=None):
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(aider_bridge.urllib.request, "urlopen", mock_urlopen)
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(SystemExit) as exc:
            aider_bridge.run_search("query", Path(tmpdir))
        assert exc.value.code == 18


def test_run_search_timeout(monkeypatch):
    import pytest
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    def mock_urlopen(req, timeout=None):
        raise TimeoutError()

    monkeypatch.setattr(aider_bridge.urllib.request, "urlopen", mock_urlopen)
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(SystemExit) as exc:
            aider_bridge.run_search("query", Path(tmpdir))
        assert exc.value.code == 18


def test_run_search_invalid_json(monkeypatch):
    import pytest
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    def mock_urlopen(req, timeout=None):
        return _MockHTTPResponse(b"not json")

    monkeypatch.setattr(aider_bridge.urllib.request, "urlopen", mock_urlopen)
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(SystemExit) as exc:
            aider_bridge.run_search("query", Path(tmpdir))
        assert exc.value.code == 19


def test_search_cli_main_wiring(monkeypatch, capsys):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    captured = {}
    payload = {"results": [{"title": "CLI Title", "url": "https://cli.test", "summary": "CLI Sum", "content": "Full"}]}

    def mock_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data)
        return _MockHTTPResponse(json.dumps(payload).encode())

    monkeypatch.setattr(aider_bridge.urllib.request, "urlopen", mock_urlopen)
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr(
            sys,
            "argv",
            ["aider_bridge.py", "search", "my cli query", "--count", "3", "--need-content", "--json", "--cwd", tmpdir],
        )
        aider_bridge.main()

    assert captured["body"] == {"query": "my cli query", "count": 3, "need_content": True}
    out = capsys.readouterr().out
    assert json.loads(out) == payload


def test_worktree_forwards_advanced_flags(monkeypatch):
    """worktree=True recursion must still forward test_cmd/dry_run/architect/editor_model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = _invoke_turn_and_capture(
            monkeypatch, Path(tmpdir),
            ["--task-id", "wt1", "--message", "m", "--files", "foo.py",
             "--test-cmd", "pytest -q", "--dry-run",
             "--model", "deepseek/deepseek-reasoner", "--architect",
             "--worktree", "--no-auto-merge"],
        )
        assert "--auto-test" in cmd
        idx = cmd.index("--test-cmd")
        assert cmd[idx + 1] == "pytest -q"
        assert "--dry-run" in cmd
        assert "--architect" in cmd
        idx = cmd.index("--editor-model")
        assert cmd[idx + 1] == "deepseek/deepseek-flash"


def test_worktree_uncommitted_edits_preserved_on_merge_worktree_task():
    """Uncommitted edits in .worktrees/<task_id> (--no-auto-commits) must be committed and merged."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        subprocess.run(["git", "init", "-b", "main"], cwd=tmppath, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "TestUser"], cwd=tmppath, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmppath, check=True)
        (tmppath / "app.py").write_text("def greet():\n    return 'helo'\n")
        subprocess.run(["git", "add", "."], cwd=tmppath, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=tmppath, check=True)
        # Untracked .env in main repo (copied to worktree during run_turn)
        (tmppath / ".env").write_text("DEEPSEEK_API_KEY=sk-secret\n")

        wt_path, branch = aider_bridge.create_task_worktree("uncommitted_task", tmppath)
        (wt_path / ".env").write_text("DEEPSEEK_API_KEY=sk-secret\n")
        # Simulate Aider --no-auto-commits edits (uncommitted!)
        (wt_path / "app.py").write_text("def greet():\n    return 'hello'\n")
        (wt_path / "new_mod.py").write_text("VALUE = 123\n")

        aider_bridge.merge_worktree_task("uncommitted_task", tmppath)

        assert not wt_path.exists()
        assert "return 'hello'" in (tmppath / "app.py").read_text()
        assert (tmppath / "new_mod.py").read_text() == "VALUE = 123\n"
        # .env must NOT be tracked/committed on main
        ls_env = subprocess.run(["git", "ls-files", ".env"], cwd=tmppath, capture_output=True, text=True)
        assert ls_env.stdout.strip() == ""


def test_worktree_uncommitted_edits_preserved_on_auto_merge():
    """merge_and_remove_worktree must also commit dirty worktree edits before removing worktree."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        subprocess.run(["git", "init", "-b", "main"], cwd=tmppath, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "TestUser"], cwd=tmppath, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmppath, check=True)
        (tmppath / "README.md").write_text("teh docs\n")
        subprocess.run(["git", "add", "."], cwd=tmppath, check=True)
        subprocess.run(["git", "commit", "-m", "Initial"], cwd=tmppath, check=True)

        wt_path, branch = aider_bridge.create_task_worktree("auto_dirty", tmppath)
        (wt_path / "README.md").write_text("the docs\n")  # uncommitted

        aider_bridge.merge_and_remove_worktree("auto_dirty", wt_path, branch, tmppath, auto_merge=True)
        assert not wt_path.exists()
        assert (tmppath / "README.md").read_text() == "the docs\n"


def test_worktree_recursion_records_auto_merge_false(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        _invoke_turn_and_capture(
            monkeypatch, tmppath,
            ["--task-id", "wt_am", "--message", "m", "--files", "foo.py",
             "--worktree", "--no-auto-merge"],
        )
        state = json.loads((tmppath / ".aider-sessions" / "wt_am" / "state.json").read_text())
        assert state["worktree"] is True
        assert state["auto_merge"] is False


def test_parse_cost_and_tokens_architect_multi_call():
    output = (
        "Architect model:\n"
        "Tokens: 4.0k sent, 1.0k cache write, 2.0k cache hit, 400 received. "
        "Cost: $0.00120 message, $0.00120 session.\n"
        "Editor model:\n"
        "Tokens: 2.5k sent, 100 received. "
        "Cost: $0.00035 message, $0.00155 session.\n"
    )
    sent, write, hit, rcvd, cost_m, cost_s = aider_bridge.parse_cost_and_tokens(output)
    assert sent == "6.5k"
    assert write == "1.0k"
    assert hit == "2.0k"
    assert rcvd == "500"
    assert cost_m == "0.00155"
    assert cost_s == "0.00155"


def test_validate_task_id_blocks_traversal_and_unsafe_git_refs():
    import pytest
    bad_ids = ["", "   ", "../evil", "a/b", "a\\b", "-rf", ".hidden", "foo..bar", "task.lock", "bad;cmd"]
    for bad in bad_ids:
        with pytest.raises(ValueError):
            aider_bridge.validate_task_id(bad)
        with pytest.raises(ValueError):
            aider_bridge.get_session_dir(bad, Path("/tmp"))
        with pytest.raises(ValueError):
            aider_bridge.create_task_worktree(bad, Path("/tmp"))
        for fn in (
            lambda b=bad: aider_bridge.run_turn(b, "m", [], Path("/tmp")),
            lambda b=bad: aider_bridge.merge_worktree_task(b, Path("/tmp")),
            lambda b=bad: aider_bridge.cleanup_session(b, Path("/tmp")),
            lambda b=bad: aider_bridge.inspect_session(b, Path("/tmp")),
            lambda b=bad: aider_bridge.reply_session(b, "ans", Path("/tmp")),
        ):
            with pytest.raises(SystemExit) as exc:
                fn()
            assert exc.value.code == 12


def test_path_containment_blocks_sibling_prefix_bypass():
    import pytest
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        proj = base / "proj"
        sibling = base / "proj_evil"
        proj.mkdir()
        sibling.mkdir()
        (sibling / "secret.py").write_text("SECRET = 1\n")

        with pytest.raises(SystemExit) as exc:
            aider_bridge.run_turn("sib_test", "m", ["../proj_evil/secret.py"], proj)
        assert exc.value.code == 12

        with pytest.raises(SystemExit) as exc2:
            aider_bridge.run_turn("dot_test", "m", ["."], proj)
        assert exc2.value.code == 12


def test_with_git_lock_uses_git_common_dir_inside_worktree():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        subprocess.run(["git", "init", "-b", "main"], cwd=tmppath, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "TestUser"], cwd=tmppath, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmppath, check=True)
        (tmppath / "a.txt").write_text("1\n")
        subprocess.run(["git", "add", "."], cwd=tmppath, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmppath, check=True)

        wt_path, _ = aider_bridge.create_task_worktree("lock_wt", tmppath)
        expected_lock = (tmppath / ".git").resolve() / "aider_merge.lock.d"
        with aider_bridge.with_git_lock(wt_path, timeout=1.0) as lock_dir:
            assert lock_dir.resolve() == expected_lock
            assert lock_dir.exists()
        assert not expected_lock.exists()


def test_auto_rebase_restores_base_branch_checkout_and_updates_main():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        subprocess.run(["git", "init", "-b", "main"], cwd=tmppath, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "TestUser"], cwd=tmppath, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmppath, check=True)
        (tmppath / "base.txt").write_text("v1\n")
        subprocess.run(["git", "add", "."], cwd=tmppath, check=True)
        subprocess.run(["git", "commit", "-m", "v1"], cwd=tmppath, check=True)

        wt_a, branch_a = aider_bridge.create_task_worktree("rebase_fix", tmppath)
        (wt_a / "feat.py").write_text("x = 1\n")
        subprocess.run(["git", "add", "."], cwd=wt_a, check=True)
        subprocess.run(["git", "commit", "-m", "feat"], cwd=wt_a, check=True)

        # Advance main
        (tmppath / "base.txt").write_text("v2\n")
        subprocess.run(["git", "add", "."], cwd=tmppath, check=True)
        subprocess.run(["git", "commit", "-m", "v2"], cwd=tmppath, check=True)

        aider_bridge.merge_worktree_task("rebase_fix", tmppath)

        head = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=tmppath, capture_output=True, text=True, check=True,
        )
        assert head.stdout.strip() == "main"
        branches = subprocess.run(
            ["git", "branch", "--list", branch_a],
            cwd=tmppath, capture_output=True, text=True, check=True,
        )
        assert branches.stdout.strip() == ""
        assert (tmppath / "feat.py").read_text() == "x = 1\n"


def test_merge_conflict_aborts_cleanly_on_exit_15():
    import pytest
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        subprocess.run(["git", "init", "-b", "main"], cwd=tmppath, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "TestUser"], cwd=tmppath, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmppath, check=True)
        (tmppath / "shared.txt").write_text("line1\n")
        subprocess.run(["git", "add", "."], cwd=tmppath, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmppath, check=True)

        wt_c, _ = aider_bridge.create_task_worktree("conflict_task", tmppath)
        (wt_c / "shared.txt").write_text("from_task\n")
        subprocess.run(["git", "add", "."], cwd=wt_c, check=True)
        subprocess.run(["git", "commit", "-m", "task edit"], cwd=wt_c, check=True)

        (tmppath / "shared.txt").write_text("from_main\n")
        subprocess.run(["git", "add", "."], cwd=tmppath, check=True)
        subprocess.run(["git", "commit", "-m", "main edit"], cwd=tmppath, check=True)

        with pytest.raises(SystemExit) as exc:
            aider_bridge.merge_worktree_task("conflict_task", tmppath)
        assert exc.value.code == 15
        # MERGE_HEAD must be aborted so repo is not left wedged
        assert not (tmppath / ".git" / "MERGE_HEAD").exists()
        head = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=tmppath, capture_output=True, text=True, check=True,
        )
        assert head.stdout.strip() == "main"


def test_merge_worktree_task_guards_running_failed_and_corrupt_state():
    import pytest
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        for status in ("RUNNING", "FAILED"):
            tid = f"guard_{status}"
            sd = tmppath / ".aider-sessions" / tid
            sd.mkdir(parents=True)
            (sd / "state.json").write_text(json.dumps({"status": status}))
            with pytest.raises(SystemExit) as exc:
                aider_bridge.merge_worktree_task(tid, tmppath)
            assert exc.value.code == 5

        sd_bad = tmppath / ".aider-sessions" / "guard_corrupt"
        sd_bad.mkdir(parents=True)
        (sd_bad / "state.json").write_text("{corrupt")
        with pytest.raises(SystemExit) as exc:
            aider_bridge.merge_worktree_task("guard_corrupt", tmppath)
        assert exc.value.code == 5


def test_commit_worktree_never_stages_tracked_or_untracked_env_and_aider_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        subprocess.run(["git", "init", "-b", "main"], cwd=tmppath, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "TestUser"], cwd=tmppath, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmppath, check=True)
        (tmppath / ".env").write_text("DEEPSEEK_API_KEY=placeholder\n")
        (tmppath / "app.py").write_text("x = 1\n")
        subprocess.run(["git", "add", "."], cwd=tmppath, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmppath, check=True)

        wt_path, _ = aider_bridge.create_task_worktree("env_guard", tmppath)
        # Modify tracked .env, plus create untracked .env.local, .aider*, .aider-sessions/
        (wt_path / ".env").write_text("DEEPSEEK_API_KEY=sk-REAL-SECRET-123456\n")
        (wt_path / ".env.local").write_text("SECRET=xyz\n")
        (wt_path / ".aider.model.settings.999.yml").write_text("secret\n")
        (wt_path / ".aider-sessions" / "env_guard").mkdir(parents=True)
        (wt_path / ".aider-sessions" / "env_guard" / "state.json").write_text("{}")
        (wt_path / "app.py").write_text("x = 2\n")

        aider_bridge.merge_worktree_task("env_guard", tmppath)
        assert (tmppath / "app.py").read_text() == "x = 2\n"
        assert (tmppath / ".env").read_text() == "DEEPSEEK_API_KEY=placeholder\n"
        assert not (tmppath / ".env.local").exists()
        assert not (tmppath / ".aider-sessions").exists()


def test_load_project_env_export_quotes_and_error_secret_sanitization(monkeypatch, capsys):
    import os
    import pytest
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / ".env").write_text(
            'export DEEPSEEK_API_KEY="sk-live-secret-99999" # inline comment\n'
            "SINGLE_QUOTED='abc''def'\n"
            "MIXED_QUOTES=\"val'inner'\" # comment\n"
            "UNQUOTED_COMMENT=rawval # trailing\n"
        )
        for k in ("DEEPSEEK_API_KEY", "SINGLE_QUOTED", "MIXED_QUOTES", "UNQUOTED_COMMENT"):
            monkeypatch.delenv(k, raising=False)

        aider_bridge.load_project_env(tmppath)
        assert os.environ["DEEPSEEK_API_KEY"] == "sk-live-secret-99999"
        assert os.environ["MIXED_QUOTES"] == "val'inner'"
        assert os.environ["UNQUOTED_COMMENT"] == "rawval"

        # Verify error sanitization redacts secret in run_search & fetch_account_balance
        def boom_urlopen(req, timeout=None):
            raise urllib.error.URLError("Bearer sk-live-secret-99999 failed")

        monkeypatch.setattr(aider_bridge.urllib.request, "urlopen", boom_urlopen)
        bal = aider_bridge.fetch_account_balance("sk-live-secret-99999")
        assert "sk-live-secret-99999" not in bal["error"]
        assert "[REDACTED]" in bal["error"]

        with pytest.raises(SystemExit) as exc:
            aider_bridge.run_search("query", tmppath)
        assert exc.value.code == 18
        err_out = capsys.readouterr().err
        assert "sk-live-secret-99999" not in err_out
        assert "[REDACTED]" in err_out


# ---------------------------------------------------------------------------
# Regression fixes: API-key preflight, binary archive ignores, reflection stalls
# ---------------------------------------------------------------------------

def test_run_turn_validates_api_key_before_creating_worktree(monkeypatch, tmp_path):
    import pytest
    subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "TestUser"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    (tmp_path / "foo.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True)

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(SystemExit) as exc:
        aider_bridge.run_turn(
            "no_key_wt", "Implement feature", ["foo.py"], tmp_path,
            model="deepseek/deepseek-flash", worktree=True,
        )
    assert exc.value.code == 1
    assert not (tmp_path / ".worktrees" / "no_key_wt").exists()
    branches = subprocess.run(
        ["git", "branch", "--list", "task/no_key_wt"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert branches.stdout.strip() == ""


def test_run_turn_loads_dotenv_before_creating_worktree(monkeypatch, tmp_path):
    import os
    import pytest
    subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "TestUser"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    (tmp_path / "foo.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True)
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=sk-from-project-env\n")

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    seen = {}

    class _Stop(Exception):
        pass

    def spy(task_id, cwd, base_branch=None):
        seen["key"] = os.environ.get("DEEPSEEK_API_KEY")
        raise _Stop()

    monkeypatch.setattr(aider_bridge, "create_task_worktree", spy)

    with pytest.raises(_Stop):
        aider_bridge.run_turn(
            "dotenv_wt", "Implement feature", ["foo.py"], tmp_path,
            model="deepseek/deepseek-flash", worktree=True,
        )
    assert seen["key"] == "sk-from-project-env"


def test_run_turn_non_deepseek_proceeds_without_api_key(monkeypatch, tmp_path):
    import pytest
    from types import SimpleNamespace
    subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "TestUser"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    (tmp_path / "foo.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True)

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    captured = {}

    def mock_run(cmd, cwd=None, capture_output=True, text=True):
        if cmd and cmd[0] == "uvx":
            captured["cmd"] = list(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(aider_bridge.subprocess, "run", mock_run)
    monkeypatch.setattr(aider_bridge, "fetch_account_balance", lambda *a, **k: None)

    with pytest.raises(SystemExit) as exc:
        aider_bridge.run_turn("non_ds", "Implement feature", ["foo.py"], tmp_path,
                              model="openai/gpt-4o")
    assert exc.value.code == 0
    assert "cmd" in captured


def test_aiderignore_excludes_binary_archives(tmp_path):
    ignore_path = aider_bridge.create_ephemeral_aiderignore(tmp_path)
    content = ignore_path.read_text()
    for pattern in ("*.gz", "*.zip", "*.tar", "*.tgz", "*.pickle", "*.pkl", "*.bin"):
        assert pattern in content
    aider_bridge.remove_ephemeral_aiderignore(ignore_path)


def test_check_diff_detects_reflection_and_decode_stalls():
    assert aider_bridge.check_diff_applied_successfully(
        "Only 3 reflections allowed, stopping.") is False
    assert aider_bridge.check_diff_applied_successfully(
        "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0") is False
    assert aider_bridge.check_diff_applied_successfully("Applied edit to foo.py") is True


def test_run_turn_reflection_and_decode_stalls_exit_11_and_mark_failed(monkeypatch, tmp_path):
    import pytest
    from types import SimpleNamespace
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(aider_bridge, "fetch_account_balance", lambda *a, **k: None)

    stalls = {
        "refl_stall": "Only 3 reflections allowed, stopping.\n",
        "decode_stall": "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff\n",
    }
    for task_id, stall in stalls.items():
        monkeypatch.setattr(
            aider_bridge.subprocess, "run",
            lambda *a, _s=stall, **k: SimpleNamespace(returncode=0, stdout=_s, stderr=""),
        )
        with pytest.raises(SystemExit) as exc:
            aider_bridge.run_turn(task_id, "Implement feature", ["foo.py"], tmp_path)
        assert exc.value.code == 11
        state = json.loads(
            (tmp_path / ".aider-sessions" / task_id / "state.json").read_text()
        )
        assert state["status"] == "FAILED"
