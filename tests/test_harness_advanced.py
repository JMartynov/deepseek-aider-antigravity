"""
Suite 5: Advanced Features & Harness Safety.
Tests:
1. Ephemeral .aiderignore creation, contents, and cleanup
2. Atomic rollback on failure: tracked file restored
3. Atomic rollback on failure: newly created untracked file deleted
4. Syntax / lint command detection
"""

import sys
import tempfile
import subprocess
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import aider_bridge


def test_ephemeral_aiderignore_creation_and_removal():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        ignore_file = aider_bridge.create_ephemeral_aiderignore(tmppath)
        assert ignore_file.exists()
        
        contents = ignore_file.read_text()
        assert ".env*" in contents
        assert "*.pem" in contents
        assert "*.key" in contents
        assert "__pycache__/" in contents
        assert ".pytest_cache/" in contents
        
        aider_bridge.remove_ephemeral_aiderignore(ignore_file)
        assert not ignore_file.exists()


def test_detect_lint_cmd():
    # User-provided override takes precedence
    assert aider_bridge.detect_lint_cmd(["foo.py"], user_lint_cmd="ruff check") == "ruff check"
    
    # Python files trigger py_compile default
    res = aider_bridge.detect_lint_cmd(["bar.py"])
    assert res == "python:python -m py_compile"
    
    # Non-python files return None
    assert aider_bridge.detect_lint_cmd(["data.json", "style.css"]) is None


def test_rollback_target_files_tracked(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        git_dir = tmppath / ".git"
        git_dir.mkdir()
        
        tracked_file = tmppath / "tracked.py"
        tracked_file.write_text("corrupted content")
        
        calls = []
        def mock_run(cmd, cwd, capture_output=True, text=True):
            calls.append(cmd)
            class Ret:
                returncode = 0
            return Ret()
            
        monkeypatch.setattr(aider_bridge.subprocess, "run", mock_run)
        aider_bridge.rollback_target_files(["tracked.py"], tmppath)
        
        assert any("checkout" in c for c in calls)


def test_rollback_target_files_untracked():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        git_dir = tmppath / ".git"
        git_dir.mkdir()
        
        untracked_file = tmppath / "untracked_new.py"
        untracked_file.write_text("temporary bad file")
        assert untracked_file.exists()
        
        # In this temp dir, git ls-files --error-unmatch will fail (returncode 1)
        aider_bridge.rollback_target_files(["untracked_new.py"], tmppath)
        
        # Newly created untracked file must be deleted on rollback
        assert not untracked_file.exists()
