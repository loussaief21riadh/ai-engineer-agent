# AI Engineer Agent V5 — Security Model

## Overview

This agent operates with **zero trust**. All file contents, user inputs, and external data are treated as untrusted by default.

## Defense Layers

### 1. File System Security
- **Project-scoped writes**: All file writes validated via `safe_path()` — resolves symlinks and verifies path stays within project root
- **Secret file protection**: `.env`, `.git/config`, SSH keys, cloud credentials are read-only or blocked
- **No system directory writes**: `/etc`, `/usr`, `/var`, `/tmp` are blocked

### 2. Command Injection Prevention
- **Command allowlist**: Only whitelisted commands (git, python, pytest, npm, etc.) permitted
- **Shell=False**: All subprocess calls use `shell=False`
- **Metacharacter blocking**: `;`, `&&`, `||`, backticks, `$()`, `|` in arguments are rejected
- **Git safety**: `git push`, `git reset`, `git clean`, `git rebase`, `git checkout .` blocked

### 3. AST Code Analysis
- **Dangerous patterns detected**: `eval()`, `exec()`, `__import__()`, `subprocess.call()`, `os.system()`
- **Network imports blocked**: `requests`, `urllib`, `httpx`, `socket` in generated code blocked
- **Critical findings**: Block execution immediately

### 4. LLM Security
- **System prompt hardening**: Anti-injection rules, trust level definitions
- **Injection detection**: Regex patterns scan file contents for known injection attempts
- **Tool argument validation**: All LLM-proposed tool arguments validated against schemas
- **Mode gating**: Tools restricted by agent mode (readOnly/planning/full)

### 5. Budget Enforcement
- **Per-task limits**: 50 LLM calls, 100 tool calls, 5 retries
- **Pre-call checks**: Budget verified before every LLM and tool call
- **Violation tracking**: Budget violations logged and reported

### 6. State Machine Hardening
- **Exhaustive transition validation**: All state transitions must be in VALID_TRANSITIONS map
- **Invalid transitions rejected**: Raises ValueError with clear message
- **Execution history**: Full state transition log with timestamps

### 7. Checkpoint Integrity
- **SHA-256 checksums**: All checkpoint files verified on load
- **Tamper detection**: Invalid checksums cause checkpoint to be discarded
- **Fallback**: Corrupted checkpoint triggers fresh start

### 8. Evidence-Based Decisions
- **All decisions traced**: Every state transition, tool call, and LLM decision logged as evidence
- **Evidence integrity**: Timestamps, source attribution, provenance chain

## Threat Model

| Threat ID | Category | Severity | Defense |
|-----------|----------|----------|---------|
| T001 | Prompt Injection | HIGH | System prompt hardening + trust levels |
| T002 | Command Injection | CRITICAL | Allowlist + shell=False + metachar blocking |
| T003 | Path Traversal | HIGH | safe_path() with resolve() |
| T004 | Credential Leakage | CRITICAL | Secret path protection + redaction |
| T005 | Dangerous Code | HIGH | AST analysis + pattern matching |
| T006 | Tool Abuse | MEDIUM | Argument validation + mode gating |
| T007 | Memory Poisoning | MEDIUM | Sanitization + trust levels |
| T008 | Checkpoint Tampering | HIGH | SHA-256 checksum verification |
| T009 | Scope Escalation | HIGH | File-level scope enforcement |

## Human Override

Users can interrupt autonomous execution at any time:
- `/override STOP` — Halt execution immediately
- `/override CANCEL` — Cancel current task
- `/override PAUSE` — Pause before next step
- `/override RESUME` — Resume paused execution

## Limitations

- **No runtime sandboxing**: Commands execute in user's terminal (Phase 12 deferred)
- **No network isolation**: Agent can make HTTP requests if tool allows
- **Python AST only**: Code analysis limited to Python; JS/TS patterns not analyzed
- **No perfect injection detection**: Sophisticated multi-turn attacks may bypass defenses
