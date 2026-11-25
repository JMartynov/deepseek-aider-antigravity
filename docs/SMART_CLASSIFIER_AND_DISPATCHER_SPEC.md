# Architecture & Specification: Smart DeepSeek Classifier & Task Dispatcher

## 1. Executive Summary & Objective

In enterprise multi-agent workflows, one of the largest sources of cost, latency, and failure is **misconfigured model settings**:
1. **Overkill / Wasteful Thinking**: DeepSeek models (`deepseek-flash` and `deepseek-v4-pro`) have Chain-of-Thought (CoT) "Thinking Mode" enabled by default at `high` effort. On mechanical edits, single-line bug fixes, documentation updates, or linter fixes, the model spends 1,000–5,000+ reasoning tokens analyzing the problem before generating a diff. This inflates turn latency from ~3s to 25s+ and burns budget on tokens that are stripped from output.
2. **Under-Reasoning on Complex AST / Architecture**: Conversely, attempting to execute cross-module refactors or complex AST visitors (`SnowflakeSecurityGuard`, SQLGlot AST traversals) with a fast non-thinking model leads to logic hallucinations, broken edge cases, and test failures.
3. **Editor Degradation in Architect Mode**: When using `--architect`, if the editor model runs with thinking enabled or whole-file formats, it re-thinks the architecture that was already solved by the reasoner, duplicating cost and latency.

The **Smart DeepSeek Classifier & Dispatcher** (`aider_bridge.py dispatch`) eliminates manual guesswork by analyzing the task prompt, target files, and test contracts to automatically classify each task into a matched precision tier with optimal model, thinking, edit-format, and test-runner settings.

---

## 2. Ranked DeepSeek API & Aider Capabilities

| Rank | Feature / API | Primary Benefit | Token & Speed Impact | Product Value |
| :---: | :--- | :--- | :--- | :--- |
| **#1** | **`--test-cmd` Closed-Loop Self-Healing** | Code Quality: In-turn test verification before exiting turn. | Zero orchestrator roundtrips, no log re-reading. | **Critical (Must Have)** |
| **#2** | **Dynamic Thinking Control (`--thinking {off,low,high}`)** | Token Sparseness: Disables CoT on mechanical tasks. | 50%–80% cost reduction; latency drops from 25s to 2–4s. | **Critical (Must Have)** |
| **#3** | **Split Architect/Editor Settings** | Architecture & Speed: Reasoner plans, Flash editor diffs with thinking `off`. | Eliminates full-file reprints and editor CoT duplication. | **Critical (Must Have)** |
| **#4** | **`search` & `fetch` API Tools** | Autonomy: Direct API doc/CVE extraction into specs. | Avoids spinning up heavy browser subagents. | **High Value** |
| **#5** | **FIM (Fill-In-The-Middle) Code Infill** | Ultra-Fast Repair: Sub-second surgical infill. | Zero reasoning tokens, direct single POST request. | **Medium Value** |
| **#6** | **RepoMap Token Tuning (`--map-tokens 512`)** | Prefix Cache Freezing: Stable AST symbol graph across `--restore`. | Preserves 64-token KV cache alignment across turns. | **Medium Value** |

---

## 3. Four-Tier Classification Matrix

The Dispatcher evaluates:
- **File count & types** (e.g., single `.md`/`pyproject.toml` vs multiple `.py` domain files).
- **Keyword heuristics** (e.g., "typo", "bump", "lint", "docstring" vs "refactor", "AST", "visitor", "concurrency", "mutex", "security guard").
- **Presence of test commands** (`--test-cmd`).
- **Explicit override flags** (e.g., `--tier patch`, `--tier surgical`, `--tier architect`, `--tier research`).

```
                                  ┌─────────────────────────────┐
                                  │      Task Input Contract    │
                                  │  (Prompt + Files + Tests)   │
                                  └──────────────┬──────────────┘
                                                 │
                                                 ▼
                                 ┌───────────────────────────────┐
                                 │   Smart DeepSeek Classifier   │
                                 │  • Analyzes file count & diff │
                                 │  • Evaluates AST/logic risk   │
                                 │  • Checks for test contracts  │
                                 └───────────────┬───────────────┘
                                                 │
         ┌───────────────────┬───────────────────┴───────────────────┬───────────────────┐
         ▼                   ▼                                       ▼                   ▼
   [Tier 1: PATCH]    [Tier 2: SURGICAL]                      [Tier 3: ARCHITECT]  [Tier 4: RESEARCH]
   • 1 file           • 1-3 files                             • Multi-file AST     • No local code
   • Typos, lints,    • Unit tests, logic                     • Security Guards    • Spec lookup,
     version bumps,     implementations                       • Cross-repo rules     PEPs, CVEs
     doc edits
   ─────────────────  ─────────────────                       ───────────────────  ──────────────────
   • Model: Flash     • Model: Flash                          • Model: Reasoner    • Tool: DeepSeek
   • Thinking: OFF    • Thinking: LOW/AUTO                      (Architect) +        Search/Fetch
   • Time: 2-4s       • Test-Cmd: ACTIVE                        Flash (Editor OFF) • Time: 1-2s
   • Cost: ~$0.001    • Cost: ~$0.01                          • Test-Cmd: ACTIVE   • Cost: ~$0.0007
                                                              • Cost: ~$0.04
```

### Tier Specifications

#### Tier 1: Quick Patch (`PATCH`)
- **Criteria**: Single file, documentation, version bumps, linter fixes, comments, docstrings, or simple string edits without complex logic.
- **Settings**:
  - `model`: `deepseek/deepseek-flash`
  - `thinking`: `off`
  - `edit_format`: `diff`
  - `use_repo_map`: `false`
  - `test_cmd`: none (or optional fast lint)
- **Outcome**: Turn completes in 2–4 seconds for ~$0.001.

#### Tier 2: Surgical TDD (`SURGICAL`)
- **Criteria**: 1 to 3 domain files, localized feature addition, bug fix, or locked ATDD test suite implementation.
- **Settings**:
  - `model`: `deepseek/deepseek-flash`
  - `thinking`: `low` / `auto`
  - `edit_format`: `diff`
  - `test_cmd`: wired to target pytest suite (e.g. `uv run pytest <test_file>`)
  - `use_repo_map`: `true` (512 tokens)
- **Outcome**: High developer velocity, automated self-healing test pass, modest token usage (~$0.01).

#### Tier 3: Complex / AST / Architecture (`ARCHITECT`)
- **Criteria**: Cross-module changes (e.g. Free vs Pro boundary), AST syntax tree rewrites (`sqlglot` visitors, `SnowflakeSecurityGuard`), concurrency (`with_git_lock`), or high-cardinality refactors.
- **Settings**:
  - `model`: `deepseek/deepseek-reasoner` (Architect)
  - `editor_model`: `deepseek/deepseek-flash` (Editor with thinking `off`)
  - `mode`: `architect`
  - `editor_edit_format`: `editor-diff`
  - `test_cmd`: wired to full regression gate
- **Outcome**: Thorough mathematical/logical reasoning on the architecture, combined with fast surgical diff application by the editor.

#### Tier 4: Technical Research (`RESEARCH`)
- **Criteria**: Inquiry, library spec check, PEP verification, or CVE advisory lookup without modifying source files.
- **Settings**:
  - `command`: `aider_bridge.py search` or `fetch`
  - Zero Aider / LLM edit overhead.
- **Outcome**: Direct structured markdown or JSON returned in 1–2 seconds.

---

## 4. CLI Interface & Dispatch Protocol

The new subcommand is exposed via:
```bash
python3 scripts/aider_bridge.py dispatch \
  --task-id "<task_id>" \
  --files <file1> <file2> ... \
  [--read-files <contract1> <contract2> ...] \
  [--test-cmd "<test_command>"] \
  [--tier {auto,patch,surgical,architect,research}] \
  [--dry-run] \
  --message "<Task specification>"
```

- When `--tier auto` (default), the built-in rule engine parses the inputs, emits a classification verdict banner (e.g., `[DISPATCHER] Classified as TIER 2: SURGICAL (reason: 2 domain files with active --test-cmd)`), configures the exact model parameters, and executes the turn.
- With `--dry-run`, it prints the classification verdict and resolved Aider arguments without executing the turn.
