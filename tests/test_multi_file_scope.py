"""
Suite 4: Multi-File Scope & Path Safety.
Tests:
- Path containment checks against directory traversal (e.g. ../../etc)
- Argument parsing for editable files vs read-only files
"""

import sys
import tempfile
import pytest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import aider_bridge


def test_path_containment_violation_exits():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        # Attempt to pass an escaping file path
        with pytest.raises(SystemExit) as exc:
            aider_bridge.run_turn(
                task_id="sec_test",
                message="test",
                files=["../../outside.py"],
                cwd=tmppath
            )
        assert exc.value.code == 12


def test_parent_directory_creation_for_new_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        env_file = tmppath / ".env"
        env_file.write_text("DEEPSEEK_API_KEY=sk-test")
        
        nested_file = "src/components/parser.py"
        full_path = tmppath / nested_file
        
        # Ensure parent directory does not exist yet
        assert not full_path.parent.exists()
        
        # Verify run_turn creates directory structure before delegating
        try:
            aider_bridge.run_turn(
                task_id="dir_test",
                message="test",
                files=[nested_file],
                cwd=tmppath
            )
        except SystemExit:
            pass
            
        assert full_path.parent.exists()
