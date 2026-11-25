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
- **Orchestrator Token Sieve**:
  - Automatically strips DeepSeek internal `► **THINKING** ... ► **ANSWER**` blocks before reporting turn output, preventing thousands of reasoning tokens from bloating orchestrator conversation context.
  - Automatically suppresses startup noise and disables web URL scraping via `--no-detect-urls`.
  - Pass `-v` / `--verbose` only when low-level reasoning inspection is required.

### 2. Strict Scope Bounding & Reference Safety
- **Pass explicit target files only**: Never pass broad glob patterns (e.g. `src/**/*.py`).
- **Editable vs Read-Only Segregation**:
  - Pass files to be modified via `--files <file1> <file2>`.
  - Pass reference interfaces, schemas, or docs via `--read-files <ref1> <ref2>`. Aider opens these with `--read`, protecting them from unintended edits.
- **Git Worktree Isolation (`--worktree`)**: Run turns on dedicated temporary worktrees (`.worktrees/<task-id>`) and branches (`task/<task-id>`). Use `--no-auto-merge` for multi-pass workflows (Pass 1 Impl -> Pass 2 Review/Restore -> Tests -> Merge), followed by explicit `merge-worktree --task-id <task-id>` when verification succeeds.
- **Atomic Mutex & Auto-Rebase (`with_git_lock`)**: Merging worktrees automatically locks `.git/aider_merge.lock.d`, checks if the base branch advanced during the turn, and auto-rebases before merging to serialize concurrent sub-agents without index collisions or conflicts.
- **Worktree PYTHONPATH Isolation**: Test suites executed within an isolated worktree should always prepend `PYTHONPATH="${WORKTREE}/src"`; `run_preflight_checks` automatically exports local `src/` to `PYTHONPATH` if present.

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

### 6. Dependency, Environment & Capability Division of Labor

DeepSeek runs as a pure remote Large Language Model accessed via API. It **does NOT install packages or execute arbitrary shell commands directly**. The execution environment and dependency lifecycle are strictly divided between the Orchestrator and DeepSeek:

| Capability | Orchestrator (Antigravity) | DeepSeek Sub-Agent (via Aider) |
| :--- | :--- | :--- |
| **Dependency Resolution** | Owns `uv add`, `uv sync`, `pip install`, or `.venv` creation. | Edits `pyproject.toml` or `requirements.txt` via unified diffs. |
| **Python Environment** | Manages local virtualenv (`uv` / `.venv`) and parent tasks. | Runs inside the parent project / worktree Python environment. |
| **Test Execution** | Runs integration suites (`uv run pytest tests/`, `npm test`, etc.). | Auto-lints via `--lint-cmd`; fixes tests via `--auto-test --test-cmd`. |
| **Web Research** | Performs live web searches (`search_web`) or dynamic browser exploration (`browser` sub-agent). | Disabled inside Aider (`--no-detect-urls`). Orchestrator can invoke `aider_bridge.py search` for curated web research via DeepSeek's search tool. |
| **OS CLI Execution** | Full shell access via `run_command` (git, build systems, background tasks). | Restricted to unified diff file modifications inside isolated worktrees. |

### 7. Orchestrator Pre-Check Gate (Task Readiness Protocol)

Before delegating any implementation or refactoring task to DeepSeek, the lead orchestrator **MUST pre-check that all prerequisite conditions are ready**:

1. **Environment & Dependency Pre-Check**:
   - Ensure the virtual environment is synced (`uv sync --extra dev` or equivalent). If DeepSeek is asked to write code using a third-party library, verify the library is already installed or explicitly instruct DeepSeek to update `pyproject.toml`.
2. **File & Dependency Scoping Pre-Check**:
   - Verify that all target editable files (`--files`) and reference contracts (`--read-files`) exist. Missing read-only references trigger exit code `14` (`PREFLIGHT_FAILED`).
   - Pre-check that internal dependencies (fixtures, helper modules, subpackages) are located beforehand so the worker sub-agent doesn't waste tokens running exploratory `grep` / `find` loops.
3. **External Research Pre-Synthesis**:
   - If a task involves external APIs or third-party library behaviors, the orchestrator must perform the web research upfront and supply the contract in the `--message` prompt or as a `--read-files` specification.
4. **Isolated Worktree Hygiene**:
   - Ensure working tree is clean or use `--worktree` to branch off cleanly without dirty working tree collisions.

### 8. Closed-Loop TDD Delegation & Advanced Turn Flags

- **Closed-Loop TDD Delegation Pattern**: Pass `--test-cmd <cmd>` (e.g. `pytest tests/ -q`) so Aider runs `--auto-test --test-cmd <cmd>` after applying edits and feeds failing test output back to DeepSeek to self-heal within the same turn. Combine with `--worktree` and `--no-auto-merge` for isolated multi-pass TDD: implementation → auto-test self-heal → review → explicit `merge-worktree`.
- **`--test-cmd <cmd>`**: Enables `--auto-test --test-cmd <cmd>` so the sub-agent verifies and repairs failing tests before returning (exit `0` only when tests pass).
- **`--dry-run`**: Runs Aider in dry-run mode — diffs are proposed but not written to disk, ideal for low-cost plan/spec review.
- **`--architect` / `--editor-model`**: Enables a two-model architect workflow (`--architect`). When using `--model deepseek/deepseek-reasoner --architect` and no `--editor-model` is supplied, the editor model defaults to the cheaper/faster `deepseek/deepseek-flash`; pass `--editor-model <name>` to override. `--editor-model` is also honoured without `--architect`.
- **`--plan-align`**: Stage 1 planning turn — propose architecture, do NOT edit files, escalate via structured `QUESTION:`; sets `AWAITING_PLAN_APPROVAL`.

**Mandatory Delegation Rules for Orchestrator Sub-Agents:**
1. **NEVER write tests manually in a separate orchestrator turn.** Always pass `--test-cmd "<cmd>"` so Aider runs `--auto-test` itself and self-heals failing assertions inside the same turn (closed-loop TDD). This preserves prefix cache and minimizes token cost.
2. **NEVER manually re-prompt after a test failure.** Provide the failing test command up front via `--test-cmd`; the sub-agent will iterate internally.
3. **For complex/multi-file refactors, always use `--architect`** (with `--model deepseek/deepseek-reasoner`) so the plan is authored by the stronger reasoner while surgical diffs are emitted by the cheaper `deepseek/deepseek-flash` editor.
4. Combine with `--worktree --no-auto-merge` for isolation: review the worktree branch, then `merge-worktree --task-id <id>`.

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

### Step 2a: Inspect Session State & Reply (Bidirectional Protocol)

```bash
# Dump session state + latest outbox question
python3 ~/.gemini/config/skills/aider-deepseek/scripts/aider_bridge.py inspect \
  --task-id "auth_service_01" --cwd "/path/to/project"

python3 ~/.gemini/config/skills/aider-deepseek/scripts/aider_bridge.py inspect \
  --task-id "auth_service_01" --json --cwd "/path/to/project"

# Resolve pending question and resume (system_rules NOT re-prepended → 95%+ cache hit)
python3 ~/.gemini/config/skills/aider-deepseek/scripts/aider_bridge.py reply \
  --task-id "auth_service_01" \
  --answer "Use Redis JTI revocation set, inject RedisClient Protocol" \
  --approved-option B \
  --cwd "/path/to/project"
```

Structured session artifacts live under `.aider-sessions/<task_id>/` (git-ignored):

```
.aider-sessions/<task_id>/
├── state.json         # RUNNING | AWAITING_PLAN_APPROVAL | AWAITING_DECISION | COMPLETED | FAILED
├── chat_history.md    # Aider --chat-history-file
├── outbox.jsonl       # Structured questions emitted by DeepSeek
└── inbox.jsonl        # Architect decisions written by `reply`
```

### Structured QUESTION Contract

```text
QUESTION: [STAGE: PLANNING|IMPLEMENTATION] [CATEGORY: ARCHITECTURE|SCHEMA|...] [IMPACT: HIGH]
<concise technical inquiry explaining what decision is needed>
[OPTIONS:
  A) <Option A>
  B) <Option B>
]
[RECOMMENDATION: Option A (Reason)]
```

### Anti-Noise Threshold Gate

- **`AUTO_SUPPRESS`**: boilerplate sign-offs (`Should I proceed?`, `Would you like me to…`) or `IMPACT: LOW`. Auto-replied locally at most once — **0 principal tokens**.
- **`ESCALATE_PRINCIPAL`**: `[STAGE: PLANNING]`, `ARCHITECTURE`, `SCHEMA`, `API_CONTRACT`, `CONCURRENCY`, `TRADEOFF`, `BREAKING_CHANGE`, `SECURITY`, `UNRESOLVABLE_FAILURE`, `AMBIGUITY`, `IMPACT: HIGH`, or substantive unstructured questions. Writes to `outbox.jsonl`, sets `state.json` to `AWAITING_*`, exits `10`.

### Stage 1 Planning Alignment (`--plan-align`)

```bash
python3 .../aider_bridge.py turn \
  --task-id "auth_service_01" \
  --message "Plan the auth refactor" \
  --files auth.py \
  --plan-align \
  --cwd "/path/to/project"
```

Injects Stage 1 planning rules; DeepSeek MUST NOT edit files — it analyzes scope, surfaces trade-offs and emits a structured `QUESTION: [STAGE: PLANNING] ...`. Session state moves to `AWAITING_PLAN_APPROVAL` until `reply` resolves it.

### Step 3: Verifying & Cleaning Up
After test verification succeeds:
```bash
python3 ~/.gemini/config/skills/aider-deepseek/scripts/aider_bridge.py cleanup \
  --task-id "auth_service_01" \
  --cwd "/path/to/project"
```

### Step 4: Advanced Turn Flags (TDD, Dry-Run, Architect)

```bash
# Closed-loop TDD inside an isolated worktree (auto-test self-heal, then explicit merge)
python3 ~/.gemini/config/skills/aider-deepseek/scripts/aider_bridge.py turn \
  --task-id "auth_service_02" \
  --message "Implement verify_token and make the tests pass." \
  --files auth.py \
  --test-cmd "uv run pytest tests/test_auth.py -q" \
  --worktree --no-auto-merge \
  --cwd "/path/to/project"

# Dry-run plan review (proposes diffs without writing to disk)
python3 ~/.gemini/config/skills/aider-deepseek/scripts/aider_bridge.py turn \
  --task-id "auth_plan" --message "Plan the refactor." --files auth.py --dry-run \
  --cwd "/path/to/project"

# Architect mode with reasoner model (editor defaults to deepseek/deepseek-flash)
python3 ~/.gemini/config/skills/aider-deepseek/scripts/aider_bridge.py turn \
  --task-id "auth_arch" --message "Design and implement verify_token." --files auth.py \
  --model deepseek/deepseek-reasoner --architect \
  --cwd "/path/to/project"
```

### Step 5: Web Search (via DeepSeek Search API)

```bash
# Basic search (default --count 5)
python3 ~/.gemini/config/skills/aider-deepseek/scripts/aider_bridge.py search \
  --cwd "/path/to/project" "Aider 0.86 release notes"

# Custom count + full page content
python3 ~/.gemini/config/skills/aider-deepseek/scripts/aider_bridge.py search \
  --count 10 --need-content --cwd "/path/to/project" "pytest async fixtures"

# Machine-readable JSON output for downstream tooling
python3 ~/.gemini/config/skills/aider-deepseek/scripts/aider_bridge.py search \
  --json --cwd "/path/to/project" "Python 3.12 PEP 695 syntax"
```

Options: `--count N` (1-50, default 5), `--need-content` (include full page content), `--json` (raw JSON), `--cwd PATH`.

### Step 6: Smart Dispatcher (Auto Tier Routing)

`dispatch` classifies the task and auto-selects the optimal model, thinking mode, edit format, and test wiring:

```bash
# Dry-run: print the verdict + resolved params without running anything
python3 ~/.gemini/config/skills/aider-deepseek/scripts/aider_bridge.py dispatch \
  --task-id "patch_01" --message "Fix typo in README" --files README.md \
  --dry-run --cwd "/path/to/project"

# Execute the matched tier (Tier 2 surgical + closed-loop TDD here)
python3 ~/.gemini/config/skills/aider-deepseek/scripts/aider_bridge.py dispatch \
  --task-id "auth_01" --message "Implement verify_token" --files auth.py \
  --test-cmd "uv run pytest tests/test_auth.py -q" --worktree --cwd "/path/to/project"

# Force a tier / override thinking effort
python3 ~/.gemini/config/skills/aider-deepseek/scripts/aider_bridge.py dispatch \
  --task-id "arch_01" --message "Refactor AST visitors" --files a.py b.py c.py d.py \
  --tier architect --thinking on --reasoning-effort high --cwd "/path/to/project"
```

| Tier | Label | Trigger | Model / Thinking | Test-Cmd |
| :---: | :--- | :--- | :--- | :--- |
| 1 | `PATCH` | 1 doc/config file (`.md/.txt/.rst/.toml/.yaml/.yml/.json/.ini/.cfg`) or patch keywords (typo, lint, bump, version, docstring, comment, readme, format, rename); no `--test-cmd` | `deepseek-flash` / `off` | off |
| 2 | `SURGICAL` | 1–3 domain files, localized feature/bugfix, or active `--test-cmd` | `deepseek-flash` / `auto` (low) | active |
| 3 | `ARCHITECT` | >3 files or complex/AST/concurrency keywords (architect, refactor, ast, visitor, sqlglot, security guard, concurrency, mutex, lock, cross-module, cross-repo, redesign, rewrite, migrate) | `deepseek-reasoner` + `flash` editor / `on` (high) | active |
| 4 | `RESEARCH` | No editable files with research intent (research, search, lookup, pep, cve, advisory, spec, what is, how does, documentation) or no `--test-cmd` | DeepSeek `search` API | n/a |

Flags: `--tier {auto,patch,surgical,architect,research}`, `--thinking {auto,on,off}`, `--reasoning-effort {low,high,max}`, `--dry-run`, `--worktree`, `--no-auto-merge`, `-v/--verbose`, `--restore` (resumes prior session on dispatch).

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
| `5`  | `MERGE_BLOCKED` | `merge-worktree` aborted because `state.json` status is `AWAITING_*`. Resolve via `reply` first. |
| `2`  | `INVALID_ARGUMENT`        | Query empty/overlong (1-128 chars) or `--count` out of range (1-50). |
| `16` | `SEARCH_RATE_LIMITED`     | DeepSeek search API returned HTTP 429. Retry after backoff. |
| `17` | `SEARCH_API_ERROR`        | Non-429 HTTP error from DeepSeek search API. |
| `18` | `SEARCH_NETWORK_ERROR`    | `URLError`/`TimeoutError`/`OSError` reaching DeepSeek search. |
| `19` | `SEARCH_INVALID_RESPONSE` | Malformed JSON from DeepSeek search API. |
