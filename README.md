# DeepSeek + Aider Antigravity Interactive Sub-Agent

Repository: [https://github.com/JMartynov/deepseek-aider-antigravity](https://github.com/JMartynov/deepseek-aider-antigravity)

This repository demonstrates the end-to-end interactive workflow between **Google Antigravity** (acting as Lead Architect) and **DeepSeek V3** (invoked as an ultra-cheap coding sub-agent via Aider).

## Architecture & Principles
- **Global Skill**: Configured at `~/.gemini/config/skills/aider-deepseek/SKILL.md`.
- **Token Sparsity**:
  - Model: `deepseek/deepseek-chat` (cheapest non-reasoning model).
  - Search/replace unified diffs (`edit-format: diff`).
  - No bloated repository map (`map-tokens: 0`).
  - Exact prefix-caching across multi-turn sessions (`cache-prompts: true`).
- **Secrets Management**:
  - `DEEPSEEK_API_KEY` is loaded exclusively from the project `.env` file (never hardcoded, ignored by `.gitignore`).
- **Interactive Turn Protocol**:
  - Sub-agent halts on missing specification and raises `QUESTION: <question>` (exiting with code 10).
  - Antigravity intercepts the question and provides the architectural answer.
  - The session resumes via `--restore`, leveraging DeepSeek's server-side prefix cache.
  - Ephemeral session files are cleaned up after tests pass.

---

## Interactive Demo Run Log (Bank Account)

### Turn 1: Delegation with Ambiguity
- **Prompt**: Implement `BankAccount` with `deposit` and `withdraw`. Stop and ask if negative balances / overdrafts are permitted.
- **Result**: Sub-agent halted without altering code.
  - Intercepted: `QUESTION: Are negative balances/overdrafts permitted, and if not, what exception should be raised on insufficient funds (e.g., ValueError, a custom InsufficientFundsError)? Also, what should the initial balance be and should deposit/withdraw reject non-positive amounts?`
  - Cost: **$0.00071**

### Turn 2: Architect Clarification & Implementation
- **Architect Response**: Overdrafts not permitted. Raise `ValueError('Insufficient funds')`. Initial balance defaults to 0. Reject non-positive amounts.
- **Result**: DeepSeek restored session history and created `bank_account.py`.
  - Cost: **$0.00082**

### Turn 3: Adding PyTest Suite
- **Prompt**: Add comprehensive pytest suite in `test_bank_account.py`.
- **Result**: DeepSeek generated 11 unit test cases.
  - Cost: **$0.00100**

### Verification & Cleanup
- Tests ran: `uv run --with pytest pytest test_bank_account.py -v` ➔ **11 passed in 0.01s**.
- Ephemeral session file `.aider.session_bank_task_01.md` was deleted automatically.
- **Total Workflow Cost**: **`$0.00253`** (~0.25 of a cent).
