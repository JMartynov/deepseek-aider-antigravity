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
import tempfile
import subprocess
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import aider_bridge


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
