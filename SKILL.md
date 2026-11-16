---
name: aider-deepseek
description: Global token-sparse coding sub-agent using Aider and DeepSeek. Features interactive turn-based message emulation, automated clarification intercept (QUESTION stop-word), prefix-cache preservation, and automatic session cleanup.
---

# Aider-DeepSeek Token-Sparse Sub-Agent

Use this skill whenever delegating code writing, refactoring, or test generation to a cost-effective, high-intelligence coding model (`deepseek/deepseek-chat`).

## Core Principles
1. **Ultra Token-Sparsity & Flash Architecture**:
   - Model: `deepseek/deepseek-flash` (DeepSeek V4.1-Flash with 1M context, MoE architecture, and prompt caching).
   - `edit-format: diff`: Only unified search/replace diffs are returned to minimize output tokens.
   - `map-tokens: 0`: No repo-map scans to prevent context window bloat and external build issues.
   - Prefix Caching (`cache-prompts: true` + `cache-keepalive-pings: 6`): DeepSeek hardware caches static system instructions and target files, reducing subsequent turn token costs by ~90–98%.
2. **Strict Scope Bounding & Reference Protection**:
   - Target editable files are explicitly passed to `--files`.
   - Read-only contract/schema files are explicitly passed to `--read-files`, preventing unwanted edits.
   - Never dump entire repositories into the prompt.
3. **Secrets Isolation**:
   - `DEEPSEEK_API_KEY` is loaded strictly from the target project's `.env` file (never hardcoded, ignored by git).
4. **Interactive Message-Passing & Question Intercept**:
   - If requirements or signatures are ambiguous, the sub-agent halts and outputs `QUESTION: <question>` without modifying code.
   - The runner intercepts the question, displays it clearly, and exits with status `10` (`INTERACTIVE_QUESTION`).
   - Antigravity catches exit code 10, formulates an architectural decision, and re-invokes with `--restore` to continue the exact same session under cache discount.
5. **Live Token, Cost & Balance Telemetry**:
   - Queries `https://api.deepseek.com/user/balance` after every turn to display real-time account balances alongside turn and session costs.
6. **Session Cleanup**:
   - Once tests pass, the ephemeral `.aider.session_<id>.md` is cleaned up immediately via `cleanup`.

---

## CLI Orchestrator Usage

The skill provides the bridge runner at `/Users/ivan/.gemini/config/skills/aider-deepseek/scripts/aider_bridge.py`.

### 1. Launching an Initial Turn
```bash
python3 /Users/ivan/.gemini/config/skills/aider-deepseek/scripts/aider_bridge.py turn \
  --task-id "<unique_task_id>" \
  --message "<concise_task_contract>" \
  --files <file1> <file2> \
  --cwd "<project_root>"
```

### 2. Resuming an Interactive Session (After Question Intercept)
If the runner exits with code `10` (`INTERACTIVE_QUESTION`):
```bash
python3 /Users/ivan/.gemini/config/skills/aider-deepseek/scripts/aider_bridge.py turn \
  --task-id "<unique_task_id>" \
  --restore \
  --message "Architect Answer: <clarification>. Proceed with implementation." \
  --files <file1> <file2> \
  --cwd "<project_root>"
```

### 3. Cleaning Up Ephemeral Session
After tests pass:
```bash
python3 /Users/ivan/.gemini/config/skills/aider-deepseek/scripts/aider_bridge.py cleanup \
  --task-id "<unique_task_id>" \
  --cwd "<project_root>"
```
