# AI Engineer Agent

A lightweight, local AI engineering agent that uses OpenRouter API to assist with software engineering tasks through structured tool calling.

## Evolution

- **V1.0**: Core agent loop, tool system, CLI, reviewer, security hardening, 270 tests
- **V1.5**: State machine (`TaskPhase`), retry/recovery logic, anti-fabrication, 301 tests
- **V2.0**: Context engine, structured planning with validation, diagnostics, edit tool, budget enforcement, 569 tests
- **V2.1**: Model routing, persistent project memory, trust-level evidence, security check phase, validation blocking, 608 tests

## Architecture

```
User → CLI → Orchestrator → AgentCore → OpenRouter → Model
                                          ↓
                                    Native tool_calls (OpenAI format)
                                    OR ````tool``` fallback
                                          ↓
                                    Tool validation → Security → Execution
                                          ↓
                                    ToolExecution → History → Next LLM call
                                          ↓
                                    Final response → TaskReport
```

### V2.1 Orchestrator Flow

```
UNDERSTAND → PLAN → INSPECT → IMPLEMENT → TEST → SECURITY_CHECK
                                                      ↓
                                    TEST pass → SECURITY_CHECK → REVIEW → VALIDATE → REPORT → DONE
                                    TEST fail → DIAGNOSE → FIX → RETEST (loop)
                                    REVIEW reject → FIX → RETEST → REVIEW
                                    VALIDATION fail → FAILED
```

### Key Components

| Module | Purpose |
|---|---|
| `app/main.py` | CLI interface with /budget, /history commands |
| `app/config.py` | Configuration via env vars |
| `app/agent/core.py` | Agent loop, tool dispatch, message management |
| `app/agent/orchestrator.py` | Phase state machine, tool registration, mode management, report building |
| `app/agent/context.py` | TaskContext, ContextBuilder, TrustLevel (5 levels), bounded collections |
| `app/agent/planner.py` | TaskPlan, Subtask, PlanValidator (Pydantic-based deterministic validation) |
| `app/agent/diagnostics.py` | FailureAnalyzer, 13 error categories, pattern-based classification, suggested fixes |
| `app/agent/validator.py` | Validator, ValidationReport (blocks on failure, excludes expected security rejections) |
| `app/agent/prompts.py` | System and review prompts (with trust boundary instructions) |
| `app/agent/reviewer.py` | Independent code review (NEEDS_MORE_EVIDENCE verdict, trust-level evidence) |
| `app/agent/router.py` | ModelRouter, task classification (DEBUGGING, CODING, REVIEW, etc.) |
| `app/agent/memory.py` | ProjectMemory with JSON persistence across sessions |
| `app/agent/quota.py` | BudgetTracker, BudgetLimits (LLM calls, tool calls, retry cycles, duration) |
| `app/agent/phases.py` | TaskPhase enum with SECURITY_CHECK phase |
| `app/llm/openrouter.py` | OpenRouter client with ChatResponse |
| `app/models/schemas.py` | Pydantic models (ToolCall, ToolExecution, ExecutionEvidence, TaskReport, ReviewVerdict) |
| `app/tools/base.py` | BaseTool, ToolSchema, argument validation |
| `app/tools/filesystem.py` | ListFilesTool, ReadFileTool, WriteFileTool |
| `app/tools/terminal.py` | RunCommandTool with allowlist |
| `app/tools/testing.py` | RunTestsTool (pytest) |
| `app/tools/edit.py` | EditFileTool (exact-match text replacement) |
| `app/tools/git.py` | GitStatusTool, GitDiffTool (read-only) |
| `app/tools/env.py` | Environment discovery and project-local executable resolution |
| `app/tools/security.py` | Centralized security: `safe_path()`, secret protection, shell metacharacter blocking |

## Context Engine

`TaskContext` persists across all phases, accumulating evidence:

- **inspected_files**: Files read during inspection (bounded to 30)
- **observations**: Typed entries with trust levels (bounded to 50)
- **recent_executions**: Tool execution history (bounded to 20)
- **failures**: Test failure outputs (bounded to 10)
- **diagnoses**: Automated and model diagnoses (bounded to 15)
- **fixes**: Applied fix descriptions (bounded to 15)
- **decisions**: Decision records (bounded to 20)
- **plan**: Validated structured plan (when LLM produces valid JSON)

`ContextBuilder` generates phase-specific prompts from accumulated context, including trust-level annotations on observations.

### Trust Levels

| Level | Source | Example |
|---|---|---|
| `USER_ASSERTED` | User input | Task description |
| `TOOL_VERIFIED` | Actual tool execution | `read_file` result, `run_tests` pass |
| `SYSTEM_DERIVED` | Deterministic computation | Plan validation success, validation report |
| `MODEL_PROPOSED` | LLM claim (not verified) | "I fixed the bug" |
| `MODEL_INFERRED` | LLM inference | Diagnosis hypothesis, classification |

Trust levels are surfaced in phase prompts and review context. Model claims never silently become tool-verified evidence.

## Structured Planning

The LLM proposes a plan. `PlanValidator` (Pydantic-based, deterministic) validates:

- Required `objective` field
- Subtasks must have `id` and `description`
- No duplicate subtask IDs
- Dependencies reference existing subtasks
- At least one subtask required

If the LLM produces valid structured JSON, the validated plan is stored in `TaskContext`. If validation fails, the task fails immediately. Free-text (non-JSON) plans are accepted as informational guidance.

## Diagnostics

`FailureAnalyzer` classifies test failures into 13 categories:

SYNTAX_ERROR, TYPE_ERROR, IMPORT_ERROR, TEST_FAILURE, LOGIC_ERROR, CONFIGURATION_ERROR, DEPENDENCY_ERROR, ENVIRONMENT_ERROR, PERMISSION_ERROR, TIMEOUT, TOOL_ERROR, LLM_ERROR, UNKNOWN

Pattern-based classification generates hypotheses and suggested fixes labeled as `MODEL_INFERRED`.

## Validation

`Validator` performs programmatic checks and **blocks on failure**:

- Test exit code (0 = pass, handles nested result format)
- Expected files present in modified list
- Execution success (excludes expected security rejections like blocked commands)
- Security rejections (unknown executable, blocked git subcommand) are excluded from failure count

The reviewer LLM remains the primary quality gate.

## Security Check Phase

After tests pass, a `SECURITY_CHECK` phase runs before review:

- Scans modified files for suspicious patterns (network fetches, dynamic code execution, potential secrets)
- Checks executed commands for suspicious operations
- Trust-level labeled observations fed to reviewer

## Reviewer

The `Reviewer` component performs independent code review:

- Receives enriched context: task, plan, changes, diff, test results, diagnoses, fixes, trust-level evidence
- Three verdicts: `APPROVE`, `REJECT`, `NEEDS_MORE_EVIDENCE`
- `NEEDS_MORE_EVIDENCE` triggers additional inspection before re-review
- Separate LLM call with dedicated review prompt (curly-brace safe)
- Rejection triggers FIX → RETEST → REVIEW recovery loop
- Reviewer does not have access to tools

## Model Routing

`ModelRouter` classifies tasks by complexity and selects appropriate models:

| Category | Keywords |
|---|---|
| DEBUGGING | debug, fix, error, bug, fail, broken, crash |
| CODING | implement, write, create, add, build, develop, code |
| REVIEW | review, audit, check, inspect, analyze, assess |
| PLANNING | plan, design, architect, structure, organize |
| REASONING | explain, why, how, reason, compare, evaluate |
| SIMPLE | (default) |

## Project Memory

`ProjectMemory` persists structured knowledge across sessions:

- Important files, architecture notes, known commands
- Previous failures, successful fixes, project conventions
- Stored in `.opencode/memory.json`
- Loaded on orchestrator init, saved after each task

## Edit Tool

`EditFileTool` enables precise text replacement:

- Exact-match required (`content.count(old_text) == 1`)
- Zero matches: error
- Multiple matches: error (ambiguous)
- Path traversal protection
- Secret path protection
- Mode-gated (ALLOW_EDITS / FULL_AUTONOMOUS only)

## Autonomous Recovery Loop

```
TEST fail → DIAGNOSE → FIX → RETEST → TEST pass → SECURITY_CHECK → REVIEW
                                          RETEST fail → DIAGNOSE (loop, bounded by MAX_RETRY_CYCLES)
                                   REVIEW reject → FIX → RETEST → REVIEW
                                   REVIEW needs_more_evidence → INSPECT → REVIEW
                                   VALIDATION fail → FAILED
```

- Bounded by `MAX_RETRY_CYCLES` (default: 3)
- Bounded by `MAX_ITERATIONS` (default: 50)
- Bounded by budget limits

## Anti-Fabrication / Execution Evidence

`TaskReport` is built from actual `ToolExecution` records:

- `files_modified` only includes paths from successful `write_file`/`edit_file` executions
- `files_inspected` only includes paths from successful `read_file` executions
- Model claims ("I modified main.py") without tool execution produce no entries
- `ExecutionEvidence` provides structured provenance with trust levels

## Budget Enforcement

`BudgetTracker` enforces local limits (not remote quota):

| Limit | Default | Env Var |
|---|---|---|
| LLM calls | 50 | `MAX_LLM_CALLS` |
| Tool calls | 100 | `MAX_TOOL_CALLS` |
| Retry cycles | 3 | `MAX_RETRY_CYCLES` |
| Task duration | 600s | `MAX_TASK_DURATION` |

Actual LLM API requests are counted at the `AgentCore._call_llm` boundary (one callback per HTTP request). Budget checked at each orchestrator loop iteration.

## Security Model

- **Project root restriction**: All file operations confined to project directory
- **Secret protection**: `.env`, `*.pem`, `*.key`, `id_rsa`, `.ssh/`, `.gnupg/`, `.aws/` blocked
- **Path traversal protection**: `../` and absolute paths outside root rejected
- **Terminal security**: `shell=False`, 16-command allowlist, metacharacter blocking (`;`, `&&`, `||`, `|`, `$()`, backticks, `>`, `>>`, `<`)
- **Git write blocking**: push, commit, reset, checkout, branch, merge, rebase blocked
- **Argument validation**: Tool arguments validated against JSON Schema
- **API key protection**: Never exposed in errors, logs, or reports
- **Edit tool protection**: Exact-match, path traversal, secret blocking, mode gating
- **Prompt injection hardening**: System prompt explicitly marks repository contents as untrusted data
- **Review prompt safety**: String concatenation instead of `.format()` to prevent KeyError from curly braces in interpolated values

## Agent Modes

| Mode | Read Files | Write Files | Edit Files | Terminal | Git |
|---|---|---|---|---|---|
| `READ_ONLY` | Yes | No | No | Safe commands | Read-only |
| `ALLOW_EDITS` | Yes | Yes | Yes | Safe commands | Read-only |
| `FULL_AUTONOMOUS` | Yes | Yes | Yes | Safe commands | Read-only |

All modes subject to security validation, project-root restrictions, and secret protection.

## CLI Usage

```bash
python app/main.py
```

Commands:
- `/help` — Show available commands
- `/status` — Show current configuration
- `/mode` — Change execution mode
- `/plan` — Show current task plan
- `/context` — Show current task context
- `/budget` — Show budget usage and limits
- `/history` — Show project memory (previous tasks, fixes, conventions)
- `/clear` — Clear conversation history
- `/exit` — Exit the agent

## Configuration

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_key_here
PRIMARY_MODEL=openrouter/free
REVIEWER_MODEL=openrouter/free
FALLBACK_MODEL=openrouter/free
PROJECT_ROOT=.
AGENT_MODE=READ_ONLY
MAX_AGENT_STEPS=15
COMMAND_TIMEOUT=30
MAX_LLM_CALLS=50
MAX_TOOL_CALLS=100
MAX_TASK_DURATION=600
```

## Testing

```bash
.venv/bin/python -m pytest tests/ -v
```

**608 tests** covering:
- V1.5 regression (301+ tests)
- V2.0 context engine, planning, diagnostics, validation, edit tool (185+ tests)
- V2.1 security check, memory persistence, model routing, reviewer verdicts, trust evidence (39 tests)
- Security: terminal, filesystem, edit tool, prompt injection, anti-fabrication (109 tests)
- E2E integration (30+ tests)

## Known Limitations

- Local engineering assistant, NOT a fully isolated OS sandbox
- Terminal commands restricted to safe allowlist
- Git operations read-only
- Secret files and directories protected
- Agent operates within configured project root only
- Reviewer integration requires additional LLM call
- Models without native tool calling use text fallback
- Plan validation is deterministic but plan generation relies on LLM
- Budget is local enforcement only (does not query OpenRouter quota)
- Model routing uses keyword-based classification (not LLM-based)
- Prompt injection protection is defense-in-depth, not guaranteed immunity
- No OS-level sandboxing beyond the tool allowlist
