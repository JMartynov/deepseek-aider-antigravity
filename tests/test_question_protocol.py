"""
Suite 2: Question Protocol & Sentinel Disambiguation.
Tests:
- Multi-line question blocks
- Questions with embedded code snippets
- Multiple distinct questions
- Immunity against false positives in code diffs (lines with + or - containing QUESTION:)
"""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import aider_bridge


def test_multiline_question_extraction():
    output = """
    Aider v0.86.2
    QUESTION: Should the rate be passed as decimal (0.2) or percentage (20)?
    Option A: Decimal 0.0 to 1.0
    Option B: Integer 0 to 100

    Applied edit to calc.py
    """
    questions = aider_bridge.extract_questions(output)
    assert len(questions) == 1
    assert "Option A" in questions[0]
    assert "Option B" in questions[0]


def test_question_with_code_block():
    output = """
    QUESTION: Should the signature be:
    ```python
    def compute(x: int, y: int) -> int:
    ```
    or do you want keyword-only arguments?

    Tokens: 2k sent, 50 received.
    """
    questions = aider_bridge.extract_questions(output)
    assert len(questions) == 1
    assert "def compute" in questions[0]
    assert "keyword-only" in questions[0]


def test_multiple_questions_extracted():
    output = """
    QUESTION: What database driver should we target (asyncpg or psycopg3)?

    QUESTION: Should timestamps be UTC or local time?
    """
    questions = aider_bridge.extract_questions(output)
    assert len(questions) == 2
    assert "asyncpg" in questions[0]
    assert "UTC" in questions[1]


def test_false_positive_immunity_on_diff_lines():
    """Ensure that diff additions/deletions containing the word QUESTION: are ignored."""
    output = """
    Aider v0.86.2
    Applied edit to test_runner.py
    <<<<<<< SEARCH
    =======
    # QUESTION: Check thread safety before calling this
    QUESTION_PATTERN = "QUESTION: (.*)"
    >>>>>>> REPLACE
    + # QUESTION: Add documentation later
    + def handle_question(): pass
    Tokens: 2.5k sent, 150 received. Cost: $0.00076 message, $0.00076 session.
    """
    questions = aider_bridge.extract_questions(output)
    assert len(questions) == 0, f"Expected 0 questions, but got: {questions}"
