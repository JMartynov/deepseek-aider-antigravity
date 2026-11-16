---
name: aider-deepseek
description: Global token-sparse coding sub-agent using Aider and DeepSeek. Features interactive turn-based message emulation, automated clarification intercept (QUESTION stop-word), prefix-cache preservation, and automatic session cleanup.
---

# Aider-DeepSeek Token-Sparse Sub-Agent

Use this skill whenever delegating code writing, refactoring, or test generation to a cost-effective, high-intelligence coding model (`deepseek/deepseek-chat`).

## Core Principles
1. **Ultra Token-Sparsity**:
   - Model: `deepseek/deepseek-chat` (cheapest non-reasoning model).
   - `edit-format: diff`: Only unified search/replace diffs are returned.
   - `map-tokens: 0`: No repo-map scans to avoid large context bloat.
   - Prefix Caching: Prompts and session files are structured to hit DeepSeek's server-side prefix cache (~90% cost reduction on subsequent turns).
2. **Strict Scope Bounding**:
   - Always restrict the sub-agent to an explicit set of target editable/reference files. Never pass entire repository trees.
3. **Secrets Isolation**:
   - `DEEPSEEK_API_KEY` is loaded exclusively from the target project's `.env` file.
4. **Interactive Message-Passing & Question Intercept**:
   - If requirements or signatures are ambiguous, the sub-agent halts and outputs `QUESTION: <question>`.
   - The runner exits with status `10` (`INTERACTIVE_QUESTION`).
   - Antigravity intercepts the question, formulates the architectural answer from project context, and re-invokes with `--restore` to continue the session under cache discount.
5. **Session Cleanup**:
   - Once the sub-agent finishes and verification tests pass, remove the ephemeral session file using `cleanup`.

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
