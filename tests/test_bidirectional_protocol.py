"""
Suite: Bidirectional Asynchronous Messaging & Decision Escalation Protocol.
"""

import sys
import json
import tempfile
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import aider_bridge


def test_question_inside_thinking_block_is_ignored():
    output = (
        "► **THINKING**\n"
        "I could output QUESTION: should I proceed?\n"
        "► **ANSWER**\n"
        "Applied edit to foo.py\n"
    )
    assert aider_bridge.extract_questions(output) == []


def test_question_after_thinking_block_is_extracted():
    output = (
        "► **THINKING**\n"
        "internal reasoning QUESTION: should I proceed?\n"
        "► **ANSWER**\n"
        "QUESTION: Which DB driver should we use?"
    )
    qs = aider_bridge.extract_questions(output)
    assert len(qs) == 1
    assert "DB driver" in qs[0]


def test_parse_structured_question_tags():
    raw = (
        "[STAGE: PLANNING] [CATEGORY: ARCHITECTURE] [IMPACT: HIGH]\n"
        "Which token strategy?\n"
        "[OPTIONS:\n  A) JWT stateless\n  B) Redis JTI\n]\n"
        "[RECOMMENDATION: Option A]"
    )
    ev = aider_bridge.parse_structured_question(raw, seq=7)
    assert ev["stage"] == "PLANNING"
    assert ev["category"] == "ARCHITECTURE"
    assert ev["impact"] == "HIGH"
    assert ev["seq"] == 7
    assert ev["type"] == "question"
    assert ev["blocking"] is True
    assert ev["suppressed"] is False
    assert ev["qid"].startswith("q_")
    assert len(ev["options"]) == 2
    assert ev["options"][0]["key"] == "A"
    assert "JWT stateless" in ev["options"][0]["summary"]
    assert ev["options"][1]["key"] == "B"
    assert "Redis JTI" in ev["options"][1]["summary"]
    assert ev["recommendation"] == "Option A"
    assert ev["ts"].endswith("Z")


def test_evaluate_escalation_threshold():
    assert aider_bridge.evaluate_escalation_threshold("Should I proceed?") == "AUTO_SUPPRESS"
    assert aider_bridge.evaluate_escalation_threshold("Would you like me to add tests?") == "AUTO_SUPPRESS"
    assert aider_bridge.evaluate_escalation_threshold({"question": "Q", "impact": "LOW"}) == "AUTO_SUPPRESS"
    assert aider_bridge.evaluate_escalation_threshold({"question": "Q", "stage": "PLANNING"}) == "ESCALATE_PRINCIPAL"
    assert aider_bridge.evaluate_escalation_threshold({"question": "Q", "category": "ARCHITECTURE"}) == "ESCALATE_PRINCIPAL"
    assert aider_bridge.evaluate_escalation_threshold({"question": "Q", "category": "SECURITY"}) == "ESCALATE_PRINCIPAL"
    assert aider_bridge.evaluate_escalation_threshold({"question": "Q", "category": "UNRESOLVABLE_FAILURE"}) == "ESCALATE_PRINCIPAL"
    assert aider_bridge.evaluate_escalation_threshold({"question": "Q", "category": "AMBIGUITY"}) == "ESCALATE_PRINCIPAL"
    assert aider_bridge.evaluate_escalation_threshold({"question": "Q", "impact": "HIGH"}) == "ESCALATE_PRINCIPAL"
    assert aider_bridge.evaluate_escalation_threshold("Which DB driver should we use?") == "ESCALATE_PRINCIPAL"


def _mock_subprocess(monkeypatch, aider_stdout_for_uvx=""):
    captured = {}

    def mock_run(cmd, cwd=None, capture_output=True, text=True):
        class Ret:
            returncode = 0
            stderr = ""
        if cmd and cmd[0] == "uvx":
            captured["cmd"] = list(cmd)
            Ret.stdout = aider_stdout_for_uvx
        else:
            Ret.stdout = ""
        return Ret()

    monkeypatch.setattr(aider_bridge.subprocess, "run", mock_run)
    monkeypatch.setattr(aider_bridge, "fetch_account_balance", lambda *a, **k: None)
    return captured


def test_session_dir_and_aiderignore(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        _mock_subprocess(monkeypatch)
        monkeypatch.setattr(sys, "argv", [
            "aider_bridge.py", "turn", "--task-id", "sess1",
            "--message", "m", "--files", "foo.py", "--cwd", str(tmppath),
        ])
        with pytest.raises(SystemExit) as exc:
            aider_bridge.main()
        assert exc.value.code == 0

        sd = tmppath / ".aider-sessions" / "sess1"
        assert sd.is_dir()
        assert (sd / "state.json").exists()
        assert (sd / "outbox.jsonl").exists()
        assert (sd / "inbox.jsonl").exists()

        ignore_path = aider_bridge.create_ephemeral_aiderignore(tmppath)
        try:
            assert ".aider-sessions/" in ignore_path.read_text()
        finally:
            aider_bridge.remove_ephemeral_aiderignore(ignore_path)


def test_inspect_session_json(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        sd = tmppath / ".aider-sessions" / "insp1"
        sd.mkdir(parents=True)
        (sd / "state.json").write_text(json.dumps({"task_id": "insp1", "status": "AWAITING_DECISION", "pending_qid": "q_x"}))
        (sd / "outbox.jsonl").write_text(json.dumps({"qid": "q_x", "question": "test q"}) + "\n")

        aider_bridge.inspect_session("insp1", tmppath, as_json=True)
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["task_id"] == "insp1"
        assert parsed["state"]["status"] == "AWAITING_DECISION"
        assert parsed["latest_outbox"]["qid"] == "q_x"


def test_reply_writes_inbox_and_preserves_prefix(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        sd = tmppath / ".aider-sessions" / "reply1"
        sd.mkdir(parents=True)
        (sd / "state.json").write_text(json.dumps({
            "task_id": "reply1", "stage": "PLANNING", "status": "AWAITING_PLAN_APPROVAL",
            "pending_qid": "q_abc12345", "files": [], "read_files": [],
            "model": "deepseek/deepseek-flash", "editor_model": None,
            "architect": False, "test_cmd": None, "lint_cmd": None,
            "thinking": "auto", "reasoning_effort": None, "map_tokens": 0,
            "turn_count": 1, "auto_merge": True,
        }))
        (sd / "outbox.jsonl").touch()
        (sd / "inbox.jsonl").touch()
        (sd / "chat_history.md").touch()

        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        captured = _mock_subprocess(monkeypatch)
        monkeypatch.setattr(sys, "argv", [
            "aider_bridge.py", "reply", "--task-id", "reply1",
            "--answer", "Use JWT stateless", "--cwd", str(tmppath),
        ])
        with pytest.raises(SystemExit) as exc:
            aider_bridge.main()
        assert exc.value.code == 0

        inbox_lines = [ln for ln in (sd / "inbox.jsonl").read_text().splitlines() if ln.strip()]
        assert len(inbox_lines) == 1
        entry = json.loads(inbox_lines[0])
        assert entry["reply_to"] == "q_abc12345"
        assert entry["answer"] == "Use JWT stateless"
        assert entry["source"] == "architect"

        cmd = captured["cmd"]
        assert "--restore-chat-history" in cmd
        idx = cmd.index("--message")
        assert "IMPORTANT RULES" not in cmd[idx + 1]
        assert "Architect Decision" in cmd[idx + 1]


def test_plan_align_sets_awaiting_plan_approval(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        _mock_subprocess(
            monkeypatch,
            aider_stdout_for_uvx=(
                "QUESTION: [STAGE: PLANNING] [CATEGORY: ARCHITECTURE] [IMPACT: HIGH]\n"
                "Token revocation strategy?\n"
                "[OPTIONS:\n  A) JWT\n  B) Redis\n]\n"
                "[RECOMMENDATION: A]\n"
            ),
        )
        monkeypatch.setattr(sys, "argv", [
            "aider_bridge.py", "turn", "--task-id", "plan1",
            "--message", "Plan the auth refactor", "--files", "auth.py",
            "--plan-align", "--cwd", str(tmppath),
        ])
        with pytest.raises(SystemExit) as exc:
            aider_bridge.main()
        assert exc.value.code == 10

        state = json.loads((tmppath / ".aider-sessions" / "plan1" / "state.json").read_text())
        assert state["status"] == "AWAITING_PLAN_APPROVAL"
        assert state["stage"] == "PLANNING"
        assert state["pending_qid"]


def test_merge_worktree_guard_exit_5():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        for status in ("AWAITING_PLAN_APPROVAL", "AWAITING_DECISION"):
            sd = tmppath / ".aider-sessions" / f"guard_{status}"
            sd.mkdir(parents=True)
            (sd / "state.json").write_text(json.dumps({"status": status}))
            with pytest.raises(SystemExit) as exc:
                aider_bridge.merge_worktree_task(f"guard_{status}", tmppath)
            assert exc.value.code == 5


def test_dispatch_restore_wiring(monkeypatch):
    captured = {}

    def fake_run_turn(task_id, message, files, cwd, **kwargs):
        captured["kwargs"] = kwargs

    monkeypatch.setattr(aider_bridge, "run_turn", fake_run_turn)
    monkeypatch.setattr(sys, "argv", [
        "aider_bridge.py", "dispatch", "--task-id", "d_restore",
        "--message", "Implement verify_token", "--files", "auth.py",
        "--restore", "--cwd", ".",
    ])
    aider_bridge.main()
    assert captured["kwargs"].get("restore") is True


def test_question_inside_fenced_code_block_is_ignored():
    output = (
        "Here is the documented contract:\n"
        "```text\n"
        "QUESTION: [STAGE: PLANNING] [CATEGORY: ARCHITECTURE]\n"
        "Example inquiry inside markdown fence\n"
        "```\n"
        "Applied edit to README.md\n"
    )
    assert aider_bridge.extract_questions(output) == []


def test_auto_suppress_local_reply_in_run_turn(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        calls = []

        def mock_run(cmd, cwd=None, capture_output=True, text=True):
            class Ret:
                returncode = 0
                stderr = ""
            if cmd and cmd[0] == "uvx":
                calls.append(list(cmd))
                if "--chat-history-file" in cmd:
                    ch_path = Path(cmd[cmd.index("--chat-history-file") + 1])
                    ch_path.parent.mkdir(parents=True, exist_ok=True)
                    ch_path.touch()
                if len(calls) == 1:
                    Ret.stdout = "QUESTION: Should I proceed?\n"
                else:
                    Ret.stdout = "Applied edit to foo.py\n"
            else:
                Ret.stdout = ""
            return Ret()

        monkeypatch.setattr(aider_bridge.subprocess, "run", mock_run)
        monkeypatch.setattr(aider_bridge, "fetch_account_balance", lambda *a, **k: None)
        monkeypatch.setattr(sys, "argv", [
            "aider_bridge.py", "turn", "--task-id", "sup1",
            "--message", "Update foo", "--files", "foo.py", "--cwd", str(tmppath),
        ])
        with pytest.raises(SystemExit) as exc:
            aider_bridge.main()
        assert exc.value.code == 0
        assert len(calls) == 2
        assert "--restore-chat-history" in calls[1]
        msg2 = calls[1][calls[1].index("--message") + 1]
        assert "IMPORTANT RULES" not in msg2
        assert "Follow repo conventions" in msg2
        state = json.loads((tmppath / ".aider-sessions" / "sup1" / "state.json").read_text())
        assert state["suppressed_count"] == 1
        assert state["status"] == "COMPLETED"


def test_dispatch_plan_align_wiring(monkeypatch):
    captured = {}

    def fake_run_turn(task_id, message, files, cwd, **kwargs):
        captured["kwargs"] = kwargs

    monkeypatch.setattr(aider_bridge, "run_turn", fake_run_turn)
    monkeypatch.setattr(sys, "argv", [
        "aider_bridge.py", "dispatch", "--task-id", "d_plan",
        "--message", "Design auth", "--files", "auth.py",
        "--plan-align", "--cwd", ".",
    ])
    aider_bridge.main()
    assert captured["kwargs"].get("plan_align") is True
