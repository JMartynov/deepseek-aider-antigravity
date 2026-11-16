---
name: aider-deepseek
description: Global token-sparse coding sub-agent using Aider and DeepSeek. Features interactive turn-based message emulation, automated clarification intercept (QUESTION stop-word), prefix-cache preservation, dynamic secret shields, atomic git rollbacks, and automatic session cleanup.
---

# Aider-DeepSeek Token-Sparse Sub-Agent

A high-efficiency coding sub-agent skill designed for the **Antigravity Lead Architect**. It delegates heavy code editing, implementation, and test suite generation to **DeepSeek V4.1-Flash** via Aider CLI while guaranteeing minimal token burn, security isolation, and turn-based interactive dialogue.

---

## 🎯 Architectural Concept: Architect & Coder Division of Labor

```
┌─────────────────────────────────────────────────────────────┐
│                 Antigravity (Lead Architect)                │
│  - Strategic Planning & System Design                       │
│  - Contract Formulation & File Scoping                      │
│  - Code Review & Question Intercept Resolution              │
└──────────────────────────────┬──────────────────────────────┘
                               │
                Invokes Bridge via Subprocess
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 aider_bridge.py Orchestrator                │
│  - Injects Ephemeral .aiderignore (Secret Shield)           │
│  - Loads DEEPSEEK_API_KEY from Project .env Only            │
│  - Manages Session State (.aider.session_<id>.md)           │
│  - Runs Auto-Lint / Syntax Verification (py_compile)        │
│  - Enforces Atomic Rollback on Edit Failures                │
│  - Fetches Live DeepSeek Account Balance                    │
└──────────────────────────────┬──────────────────────────────┘
                               │
               Dispatches Headless Turn (diff mode)
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│               DeepSeek V4.1-Flash Sub-Agent                 │
│  - 1 Million Context Window                                 │
│  - Server-Side Prefix Caching (~90-98% Cost Reduction)     │
│  - Halts on Ambiguity: outputs 'QUESTION: <q>'              │
│  - Returns Surgical Search/Replace Unified Diffs            │
└─────────────────────────────────────────────────────────────┘
```

---

## 💡 Best Practices for Orchestrators & Developers

### 1. Ultra Token-Sparsity & Model Routing
- **Default Engine**: `deepseek/deepseek-flash` (DeepSeek V4.1-Flash). It offers state-of-the-art coding abilities with 1M context at ~$0.15/1M input and ~$0.60/1M output, and down to ~$0.003–$0.014/1M on cached tokens.
- **Diff Edit Format (`edit-format: diff`)**: Forces the model to emit only search/replace blocks. Never let models reprint full 500-line files.
- **Map Tokens Disabled (`map-tokens: 0`)**: Disabling Aider's AST repo-map avoids token bloat, prevents large cross-file scans, and avoids platform-dependent native compiler prompts.
- **Prefix Caching (`cache-prompts: true` + `cache-keepalive-pings: 6`)**: DeepSeek hardware caches identical prompt prefixes. The bridge automatically supplies an ephemeral model settings file for `deepseek/deepseek-flash` ensuring `caches_by_default: true` and `edit_format: diff`. When resuming sessions with `--restore`, unchanged system instructions and file context are served from cache at a ~90% discount.

### 2. Strict Scope Bounding & Reference Safety
- **Pass explicit target files only**: Never pass broad glob patterns (e.g. `src/**/*.py`).
- **Editable vs Read-Only Segregation**:
  - Pass files to be modified via `--files <file1> <file2>`.
  - Pass reference interfaces, schemas, or docs via `--read-files <ref1> <ref2>`. Aider opens these with `--read`, protecting them from unintended edits.
- **Git Worktree Isolation (`--worktree`)**: Run turns on dedicated temporary worktrees (`.worktrees/<task-id>`) and branches (`task/<task-id>`). Use `--no-auto-merge` for multi-pass workflows (Pass 1 Impl -> Pass 2 Review/Restore -> Tests -> Merge), followed by explicit `merge-worktree --task-id <task-id>` when verification succeeds.
  - Pass reference interfaces, schemas, or docs via `--read-files <ref1> <ref2>`. Aider opens these with `--read`, protecting them from unintended edits.

### 3. Secrets Isolation
- `DEEPSEEK_API_KEY` is loaded strictly from the target project's `.env` file.
- The bridge dynamically generates `.aiderignore.ephemeral` during execution, shielding `.env*`, `*.pem`, `*.key`, `node_modules/`, and test cache files from being ingested into LLM prompts.

### 4. Interactive Turn Protocol (`QUESTION:` Stop-Word)
- When specifications, parameter types, or boundary conditions are ambiguous, DeepSeek is instructed **not to guess** and output `QUESTION: <question>`.
- The bridge detects the question, prints it, and exits with code `10` (`INTERACTIVE_QUESTION`).
- **Orchestrator Role**:
  1. Inspect the question from the sub-agent output.
  2. Answer the question using high-level architectural knowledge or user preferences.
  3. Re-invoke the bridge with `--restore` and the answer.
  4. DeepSeek restores session history and proceeds with implementation under prefix cache.

### 5. Automated Syntax Verification & Atomic Rollback
- **Auto-Linting**: Enabled by default using `python -m py_compile` (or user `--lint-cmd`). If DeepSeek introduces a syntax error, Aider feeds the error back to DeepSeek to self-heal in the same turn.
- **Atomic Rollback**: If a diff collision occurs or the edit fails, all modified target files are restored (`git checkout -- <file>`) and incomplete untracked files are deleted.

---

## 🛠️ CLI Orchestrator Command Reference

The runner script is located at:
`~/.gemini/config/skills/aider-deepseek/scripts/aider_bridge.py`

### Step 1: Initial Turn (Task Dispatch)
```bash
python3 ~/.gemini/config/skills/aider-deepseek/scripts/aider_bridge.py turn \
  --task-id "auth_service_01" \
  --message "Implement verify_token in auth.py: return user dict if token is valid, raise ValueError on expiry. If token format (JWT vs Opaque) is not specified, ask QUESTION: ..." \
  --files auth.py \
  --read-files interfaces/user.py \
  --cwd "/path/to/project"
```

### Step 2: Handling Interactive Questions (Exit Code 10)
If the sub-agent returns exit code `10`:
```bash
python3 ~/.gemini/config/skills/aider-deepseek/scripts/aider_bridge.py turn \
  --task-id "auth_service_01" \
  --restore \
  --message "Architect Answer: Use standard JWT with RS256 algorithm. Proceed with implementation." \
  --files auth.py \
  --read-files interfaces/user.py \
  --cwd "/path/to/project"
```

### Step 3: Verifying & Cleaning Up
After test verification succeeds:
```bash
python3 ~/.gemini/config/skills/aider-deepseek/scripts/aider_bridge.py cleanup \
  --task-id "auth_service_01" \
  --cwd "/path/to/project"
```

---

## 📊 Exit Codes & Diagnostics

| Exit Code | Status Identifier | Description / Action Required |
| :--- | :--- | :--- |
| `0` | `COMPLETED` | Edit applied and verified. Proceed to test runner. |
| `1` | `MISSING_API_KEY` | `DEEPSEEK_API_KEY` missing from project `.env`. |
| `10` | `INTERACTIVE_QUESTION` | Sub-agent paused with a clarifying question. Resolve and re-invoke with `--restore`. |
| `11` | `DIFF_APPLY_FAILED` | Aider search block did not match. Target files automatically rolled back. |
| `12` | `PATH_TRAVERSAL_VIOLATION` | Target file path resolves outside project root. |
| `13` | `AIDER_INVOCATION_FAILED` | Process error during launch. Target files automatically rolled back. |
| `14` | `PREFLIGHT_FAILED` | Missing read-only reference file before starting Aider turn. |
| `15` | `MERGE_FAILED` | Worktree branch failed to cleanly merge into base branch. |
