# AI Engineer Agent

A lightweight, local AI engineering agent that uses OpenRouter API to assist with software engineering tasks through structured tool calling.

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

### Key Components

| Module | Purpose |
|---|---|
| `app/main.py` | CLI interface |
| `app/config.py` | Configuration via env vars |
| `app/agent/core.py` | Agent loop, tool dispatch, message management |
| `app/agent/orchestrator.py` | Tool registration, mode management, report building |
| `app/agent/prompts.py` | System and review prompts |
| `app/agent/reviewer.py` | Independent code review component |
| `app/llm/openrouter.py` | OpenRouter client with ChatResponse |
| `app/models/schemas.py` | Pydantic models (ToolCall, ToolExecution, TaskReport, etc.) |
| `app/tools/base.py` | BaseTool, ToolSchema, argument validation |
| `app/tools/filesystem.py` | ListFilesTool, ReadFileTool, WriteFileTool |
| `app/tools/terminal.py` | RunCommandTool with allowlist |
| `app/tools/testing.py` | RunTestsTool (pytest) |
| `app/tools/git.py` | GitStatusTool, GitDiffTool (read-only) |
| `app/tools/env.py` | Environment discovery and project-local executable resolution |
| `app/tools/security.py` | Secret protection, shell metacharacter blocking |

## Available Tools

| Tool | Description | Mode |
|---|---|---|
| `list_files` | List files in a directory | All modes |
| `read_file` | Read file contents | All modes |
| `run_command` | Execute safe shell commands | All modes |
| `run_tests` | Run pytest | All modes |
| `git_status` | Show git status | All modes |
| `git_diff` | Show git diff | All modes |
| `write_file` | Create/modify files | ALLOW_EDITS, FULL_AUTONOMOUS |

## Security Model

- **Project root restriction**: All file operations are confined to the project directory
- **Secret protection**: `.env`, `*.pem`, `*.key`, `id_rsa`, `.ssh/`, `.gnupg/`, `.aws/` are blocked
- **Path traversal protection**: `../` and absolute paths outside project root are rejected
- **Terminal security**: `shell=False`, 16-command allowlist, shell metacharacter blocking (`;`, `&&`, `||`, `|`, `$()`, backticks, `>`, `>>`, `<`)
- **Git write blocking**: push, commit, reset, checkout, branch, merge, rebase, etc. are blocked
- **Argument validation**: Tool arguments are validated against JSON Schema definitions
- **API key protection**: Never exposed in errors, logs, execution history, or reports

## Agent Modes

| Mode | Read Files | Write Files | Terminal | Git |
|---|---|---|---|---|
| `READ_ONLY` | Yes | No | Safe commands | Read-only |
| `ALLOW_EDITS` | Yes | Yes | Safe commands | Read-only |
| `FULL_AUTONOMOUS` | Yes | Yes | Safe commands | Read-only |

All modes are subject to security validation, project-root restrictions, and secret protection.

## Native Tool Calling

The agent uses OpenAI-compatible native tool calling when the model supports it. Tool definitions are built from the project's `ToolSchema` objects:

```python
{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the contents of a text file...",
        "parameters": {"type": "object", "properties": {...}, "required": [...]}
    }
}
```

Multiple tool calls per LLM response are supported. Each tool call preserves its `tool_call_id` for proper OpenAI message format.

## Legacy Fallback

If the model does not return native tool calls, the agent falls back to parsing ````tool` blocks:

````
```tool
{"name": "read_file", "arguments": {"path": "app/main.py"}}
```
````

Native tool calls always take precedence. The fallback is retained for compatibility with models that don't support native tool calling.

## Reviewer

The `Reviewer` component can independently review code changes using a separate LLM call. It inspects actual execution results rather than trusting model claims. The reviewer does not have access to tools and operates as a read-only review component.

## Audit / Reporting

Every tool execution generates a `ToolExecution` record with:
- `step`, `tool_name`, `arguments`, `success`, `result`, `error`, `duration_ms`, `tool_call_id`

`TaskReport` is built from actual runtime executions, NOT from model claims. This prevents fabrication — if the model claims it read a file but didn't, the report shows no files inspected.

## Installation

```bash
cd ~/Documents/ai-engineer-agent
python3 -m venv .venv
source .venv/bin/activate
pip install httpx pydantic python-dotenv pytest
```

## Configuration

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_key_here
PRIMARY_MODEL=openrouter/free
REVIEWER_MODEL=openrouter/free
PROJECT_ROOT=.
AGENT_MODE=READ_ONLY
MAX_AGENT_STEPS=15
COMMAND_TIMEOUT=30
```

## Running the CLI

```bash
python app/main.py
```

Commands:
- `/help` — Show available commands
- `/status` — Show current configuration
- `/mode` — Change execution mode
- `/clear` — Clear conversation history
- `/exit` — Exit the agent

## Running Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

## Current Limitations

- This is a local engineering assistant, NOT a fully isolated OS sandbox
- Terminal commands are restricted to a safe allowlist
- Git operations are read-only
- Secret files and directories are protected
- The agent can only operate within the configured project root
- Reviewer integration requires an additional LLM call
- Models that don't support native tool calling will use the text fallback
