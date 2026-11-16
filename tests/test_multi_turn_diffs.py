"""
Suite 3: Multi-turn & Diff Collision Recovery.
Tests:
- Detection of search/replace mismatch ('Failed to apply edit to...')
- Normal success verification
"""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import aider_bridge


def test_diff_applied_successfully():
    output = """
    Applied edit to service.py
    Commit 1234567 added.
    """
    assert aider_bridge.check_diff_applied_successfully(output) is True


def test_diff_collision_detected_failed_to_apply():
    output = """
    Failed to apply edit to service.py
    The search block did not match the file content.
    """
    assert aider_bridge.check_diff_applied_successfully(output) is False


def test_diff_collision_detected_did_not_find():
    output = """
    Aider could not apply changes: did not find the search block in utils.py.
    """
    assert aider_bridge.check_diff_applied_successfully(output) is False
