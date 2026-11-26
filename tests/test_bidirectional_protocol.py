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


def test_extract_questions_deduplicates_identical_blocks():
    output = (
        "QUESTION: Which DB driver?\n"
        "Applied edit to foo.py\n"
        "QUESTION: Which   DB   driver?\n"
        "QUESTION: Which cache backend?\n"
    )
    qs = aider_bridge.extract_questions(output)
    assert len(qs) == 2
    assert qs[0] == "Which DB driver?"
    assert "cache backend" in qs[1]


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

    # Single-line [OPTIONS: A) ... B) ...] must also split cleanly
    inline = (
        "[STAGE: PLANNING] [CATEGORY: ARCHITECTURE] [IMPACT: HIGH]\n"
        "Store choice?\n"
        "[OPTIONS: A) In-memory RLock + dict B) sqlite3 WAL store]\n"
        "[RECOMMENDATION: Option A]"
    )
    ev_inline = aider_bridge.parse_structured_question(inline, seq=8)
    assert [(o["key"], o["summary"]) for o in ev_inline["options"]] == [
        ("A", "In-memory RLock + dict"),
        ("B", "sqlite3 WAL store"),
    ]


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


def test_extract_questions_preserves_multiline_structured_block():
    output = (
        "Aider v0.86.2\n"
        "QUESTION: [STAGE: PLANNING] [CATEGORY: ARCHITECTURE] [IMPACT: HIGH]\n"
        "Which persistence backend should we use for session tokens?\n"
        "\n"
        "- Local file store keeps deployment simple\n"
        "* Redis enables multi-worker horizontal scaling\n"
        "\n"
        "[OPTIONS:\n"
        "  A) Local JSON store\n"
        "  B) Redis cluster\n"
        "]\n"
        "\n"
        "[RECOMMENDATION: Option A for single-node CLI]\n"
        "\n"
        "Tokens: 1.5k sent, 120 received. Cost: $0.00050 message, $0.00050 session.\n"
    )
    qs = aider_bridge.extract_questions(output)
    assert len(qs) == 1
    q = qs[0]
    assert "- Local file store" in q
    assert "* Redis enables" in q
    assert "[OPTIONS:" in q
    assert "A) Local JSON store" in q
    assert "B) Redis cluster" in q
    assert "[RECOMMENDATION: Option A for single-node CLI]" in q
    assert "Tokens:" not in q

    ev = aider_bridge.parse_structured_question(q, seq=1)
    assert ev["stage"] == "PLANNING"
    assert ev["category"] == "ARCHITECTURE"
    assert ev["impact"] == "HIGH"
    assert [o["key"] for o in ev["options"]] == ["A", "B"]
    assert ev["recommendation"] == "Option A for single-node CLI"


def test_evaluate_escalation_threshold_precedence_over_polite_phrases():
    assert aider_bridge.evaluate_escalation_threshold({
        "question": "Would you like me to proceed with JWT or Redis?",
        "stage": "PLANNING",
    }) == "ESCALATE_PRINCIPAL"
    assert aider_bridge.evaluate_escalation_threshold({
        "question": "Should I proceed with Option A?",
        "category": "ARCHITECTURE",
    }) == "ESCALATE_PRINCIPAL"
    assert aider_bridge.evaluate_escalation_threshold({
        "question": "Would you like me to rewrite the lock?",
        "impact": "HIGH",
    }) == "ESCALATE_PRINCIPAL"
    # Polite implementation-stage low/unspecified question is still suppressed
    assert aider_bridge.evaluate_escalation_threshold({
        "question": "Would you like me to format docstrings?",
        "stage": "IMPLEMENTATION",
    }) == "AUTO_SUPPRESS"


def test_plan_align_uses_read_only_ask_mode_without_architect_or_tests(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        captured = _mock_subprocess(
            monkeypatch,
            aider_stdout_for_uvx=(
                "QUESTION: [STAGE: PLANNING] [CATEGORY: ARCHITECTURE] [IMPACT: HIGH]\n"
                "Choose layout?\n"
                "[OPTIONS:\n  A) Split modules\n  B) Single file\n]\n"
                "[RECOMMENDATION: Option A]\n"
            ),
        )
        monkeypatch.setattr(sys, "argv", [
            "aider_bridge.py", "turn", "--task-id", "plan_ro",
            "--message", "Plan refactor", "--files", "app.py",
            "--model", "deepseek/deepseek-reasoner", "--architect",
            "--test-cmd", "pytest -q", "--plan-align", "--cwd", str(tmppath),
        ])
        with pytest.raises(SystemExit) as exc:
            aider_bridge.main()
        assert exc.value.code == 10

        cmd = captured["cmd"]
        assert "--chat-mode" in cmd
        assert cmd[cmd.index("--chat-mode") + 1] == "ask"
        assert "--edit-format" not in cmd
        assert "--architect" not in cmd
        assert "--editor-model" not in cmd
        assert "--auto-test" not in cmd

        state = json.loads((tmppath / ".aider-sessions" / "plan_ro" / "state.json").read_text())
        assert state["architect"] is True
        assert state["stage"] == "PLANNING"


def test_dispatch_auto_plan_align_only_for_tier3(monkeypatch):
    captured = {}

    def fake_run_turn(task_id, message, files, cwd, **kwargs):
        captured[task_id] = kwargs

    monkeypatch.setattr(aider_bridge, "run_turn", fake_run_turn)

    # Tier 3 (architect) auto-enables plan_align
    monkeypatch.setattr(sys, "argv", [
        "aider_bridge.py", "dispatch", "--task-id", "t3_auto",
        "--message", "Refactor the concurrency lock architecture", "--files", "lock.py",
        "--cwd", ".",
    ])
    aider_bridge.main()
    assert captured["t3_auto"]["plan_align"] is True

    # Tier 1 (patch) keeps plan_align=False
    monkeypatch.setattr(sys, "argv", [
        "aider_bridge.py", "dispatch", "--task-id", "t1_auto",
        "--message", "Fix typo in README", "--files", "README.md",
        "--cwd", ".",
    ])
    aider_bridge.main()
    assert captured["t1_auto"]["plan_align"] is False

    # Tier 2 (surgical) keeps plan_align=False
    monkeypatch.setattr(sys, "argv", [
        "aider_bridge.py", "dispatch", "--task-id", "t2_auto",
        "--message", "Implement add function", "--files", "calc.py",
        "--test-cmd", "pytest -q", "--cwd", ".",
    ])
    aider_bridge.main()
    assert captured["t2_auto"]["plan_align"] is False


def test_reply_includes_approved_option_and_fails_on_missing_worktree(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        sd = tmppath / ".aider-sessions" / "rep_opt"
        sd.mkdir(parents=True)
        (sd / "state.json").write_text(json.dumps({
            "task_id": "rep_opt", "stage": "PLANNING", "status": "AWAITING_PLAN_APPROVAL",
            "pending_qid": "q_1234", "files": ["app.py"], "read_files": [],
            "model": "deepseek/deepseek-reasoner", "editor_model": "deepseek/deepseek-flash",
            "architect": True, "test_cmd": None, "lint_cmd": None,
            "thinking": "on", "reasoning_effort": "high", "map_tokens": 512,
            "turn_count": 1, "auto_merge": True, "worktree": False,
        }))
        (sd / "chat_history.md").touch()
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        captured = _mock_subprocess(monkeypatch)
        monkeypatch.setattr(sys, "argv", [
            "aider_bridge.py", "reply", "--task-id", "rep_opt",
            "--approved-option", "A", "--answer", "Use local store",
            "--cwd", str(tmppath),
        ])
        with pytest.raises(SystemExit) as exc:
            aider_bridge.main()
        assert exc.value.code == 0
        cmd = captured["cmd"]
        msg = cmd[cmd.index("--message") + 1]
        assert "Architect Decision (Option A): Use local store. Proceed with implementation." == msg

    # Missing worktree must fail with exit 1 instead of falling back to main repo root
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        sd = tmppath / ".aider-sessions" / "rep_missing_wt"
        sd.mkdir(parents=True)
        missing_wt = tmppath / ".worktrees" / "rep_missing_wt"
        (sd / "state.json").write_text(json.dumps({
            "task_id": "rep_missing_wt", "stage": "PLANNING", "status": "AWAITING_PLAN_APPROVAL",
            "pending_qid": "q_9999", "worktree": True, "worktree_path": str(missing_wt),
        }))
        with pytest.raises(SystemExit) as exc:
            aider_bridge.reply_session("rep_missing_wt", "Proceed", tmppath, approved_option="A")
        assert exc.value.code == 1
        # State must remain intact when failing pre-flight
        state_after = json.loads((sd / "state.json").read_text())
        assert state_after["status"] == "AWAITING_PLAN_APPROVAL"


def test_parse_structured_question_with_nested_brackets():
    raw = (
        "[STAGE: PLANNING] [CATEGORY: API_CONTRACT] [IMPACT: HIGH]\n"
        "Which return type should we expose?\n"
        "[OPTIONS:\n"
        "  A) Return dict[str, int]\n"
        "  B) Return list[tuple[str, int]]\n"
        "]\n"
        "[RECOMMENDATION: Option A (dict[str, int])]"
    )
    ev = aider_bridge.parse_structured_question(raw, seq=11)
    assert [(o["key"], o["summary"]) for o in ev["options"]] == [
        ("A", "Return dict[str, int]"),
        ("B", "Return list[tuple[str, int]]"),
    ]
    assert ev["recommendation"] == "Option A (dict[str, int])"
    assert ev["question"] == "Which return type should we expose?"

    # Inline single-line options containing nested brackets
    inline = (
        "Choose type?\n"
        "[OPTIONS: A) dict[str, int] B) list[str]]\n"
        "[RECOMMENDATION: dict[str, int]]"
    )
    ev2 = aider_bridge.parse_structured_question(inline, seq=12)
    assert [(o["key"], o["summary"]) for o in ev2["options"]] == [
        ("A", "dict[str, int]"),
        ("B", "list[str]"),
    ]
    assert ev2["recommendation"] == "dict[str, int]"

    # Blank line inside [OPTIONS: ...] after dict[str, int] must not split the question
    extracted = aider_bridge.extract_questions(
        "QUESTION: Choose type?\n"
        "[OPTIONS:\n"
        "  A) dict[str, int]\n"
        "\n"
        "  B) list[str]\n"
        "]\n"
    )
    assert len(extracted) == 1
    assert "B) list[str]" in extracted[0]


def test_plan_align_auto_suppress_retry_stays_in_planning_mode(monkeypatch):
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
                    ch = Path(cmd[cmd.index("--chat-history-file") + 1])
                    ch.parent.mkdir(parents=True, exist_ok=True)
                    ch.touch()
                if len(calls) == 1:
                    Ret.stdout = "QUESTION: Should I proceed?\n"
                else:
                    Ret.stdout = (
                        "QUESTION: [STAGE: PLANNING] [CATEGORY: ARCHITECTURE] [IMPACT: HIGH]\n"
                        "Pick storage?\n"
                        "[OPTIONS:\n  A) SQLite\n  B) JSON\n]\n"
                    )
            else:
                Ret.stdout = ""
            return Ret()

        monkeypatch.setattr(aider_bridge.subprocess, "run", mock_run)
        monkeypatch.setattr(aider_bridge, "fetch_account_balance", lambda *a, **k: None)
        with pytest.raises(SystemExit) as exc:
            aider_bridge.run_turn(
                "plan_sup", "Plan backend", ["app.py"], tmppath,
                architect=True, plan_align=True,
            )
        assert exc.value.code == 10
        assert len(calls) == 2
        retry_cmd = calls[1]
        assert "--chat-mode" in retry_cmd
        assert retry_cmd[retry_cmd.index("--chat-mode") + 1] == "ask"
        assert "--architect" not in retry_cmd
        retry_msg = retry_cmd[retry_cmd.index("--message") + 1]
        assert "PLANNING ONLY" in retry_msg
        assert "finalize code" not in retry_msg


def test_reply_session_rejects_non_awaiting_and_corrupt_state():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        for bad_status in ("COMPLETED", "RUNNING", "FAILED"):
            tid = f"bad_{bad_status}"
            sd = tmppath / ".aider-sessions" / tid
            sd.mkdir(parents=True)
            (sd / "state.json").write_text(json.dumps({"task_id": tid, "status": bad_status}))
            with pytest.raises(SystemExit) as exc:
                aider_bridge.reply_session(tid, "Proceed", tmppath)
            assert exc.value.code == 1

        sd_corrupt = tmppath / ".aider-sessions" / "corrupt_state"
        sd_corrupt.mkdir(parents=True)
        (sd_corrupt / "state.json").write_text("{not-json")
        with pytest.raises(SystemExit) as exc:
            aider_bridge.reply_session("corrupt_state", "Proceed", tmppath)
        assert exc.value.code == 1


def test_reply_session_worktree_auto_merge_on_completion(monkeypatch):
    import subprocess
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        subprocess.run(["git", "init", "-b", "main"], cwd=tmppath, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "TestUser"], cwd=tmppath, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmppath, check=True)
        (tmppath / "app.py").write_text("VAL = 0\n")
        subprocess.run(["git", "add", "."], cwd=tmppath, check=True)
        subprocess.run(["git", "commit", "-m", "Init"], cwd=tmppath, check=True)

        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        monkeypatch.setattr(aider_bridge, "fetch_account_balance", lambda *a, **k: None)

        real_run = subprocess.run
        turn_phase = {"n": 0}

        def mock_run(cmd, cwd=None, capture_output=True, text=True):
            if cmd and cmd[0] == "uvx":
                turn_phase["n"] += 1
                if "--chat-history-file" in cmd:
                    ch = Path(cmd[cmd.index("--chat-history-file") + 1])
                    ch.parent.mkdir(parents=True, exist_ok=True)
                    ch.touch()
                class Ret:
                    returncode = 0
                    stderr = ""
                if turn_phase["n"] == 1:
                    Ret.stdout = (
                        "QUESTION: [STAGE: PLANNING] [CATEGORY: ARCHITECTURE] [IMPACT: HIGH]\n"
                        "Pick value?\n"
                    )
                else:
                    # Simulate Aider editing app.py inside the worktree
                    (Path(cwd) / "app.py").write_text("VAL = 99\n")
                    Ret.stdout = "Applied edit to app.py\n"
                return Ret()
            return real_run(cmd, cwd=cwd, capture_output=capture_output, text=text)

        monkeypatch.setattr(aider_bridge.subprocess, "run", mock_run)

        # Turn 1: pauses in worktree with exit 10
        with pytest.raises(SystemExit) as exc1:
            aider_bridge.run_turn(
                "wt_reply", "Update VAL", ["app.py"], tmppath,
                worktree=True, auto_merge=True, plan_align=True,
            )
        assert exc1.value.code == 10
        wt_path = tmppath / ".worktrees" / "wt_reply"
        assert wt_path.exists()

        # Turn 2: reply completes with exit 0 -> auto_merge=True merges and removes worktree
        with pytest.raises(SystemExit) as exc2:
            aider_bridge.reply_session("wt_reply", "Use 99", tmppath, approved_option="A")
        assert exc2.value.code == 0
        assert not wt_path.exists()
        assert (tmppath / "app.py").read_text() == "VAL = 99\n"


def test_atomic_state_json_write_no_tmp_leftover():
    with tempfile.TemporaryDirectory() as tmpdir:
        sf = Path(tmpdir) / "sess" / "state.json"
        aider_bridge._write_state_atomic(sf, {"task_id": "t", "status": "RUNNING"})
        assert json.loads(sf.read_text()) == {"task_id": "t", "status": "RUNNING"}
        assert not list(sf.parent.glob(".state.*.tmp"))
