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
