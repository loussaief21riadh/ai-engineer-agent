# AI Engineer Agent V5 — Architecture

## System Layers

```
┌─────────────────────────────────────────────┐
│                  CLI Layer                   │
│  app/main.py — run, budget, inspect, trace  │
├─────────────────────────────────────────────┤
│              V5 Autonomy Layer              │
│  Policy Engine │ Task Graph │ Self-Reflection│
│  Eng. Memory   │ Adaptive Planning          │
│  Threat Model  │ Validation Gate            │
│  Human Override│ Adversarial Testing         │
├─────────────────────────────────────────────┤
│              V4 Intelligence Layer          │
│  Project Understanding │ Codebase Graph      │
│  Impact Analysis       │ Test Selection      │
│  Context Budget        │ Change Validator    │
│  Observability 2.0                         │
├─────────────────────────────────────────────┤
│              V3.2 Reliability Layer         │
│  Evidence Engine │ State Machine            │
│  Regression Engine │ Execution Controller    │
│  Diagnostics (19 categories + strategies)   │
├─────────────────────────────────────────────┤
│              V3 Core Runtime                │
│  Orchestrator │ LLM Client │ Terminal Core   │
│  Checkpoint   │ Planner    │ Budget          │
├─────────────────────────────────────────────┤
│              Security Baseline              │
│  safe_path │ allowlist │ AST analysis        │
│  injection detection │ trust levels          │
└─────────────────────────────────────────────┘
```

## Module Inventory

### V3 Core (existing)
| Module | Purpose |
|--------|---------|
| `orchestrator.py` | Main execution loop, phase management |
| `llm_client.py` | OpenRouter API communication |
| `terminal_core.py` | Command execution with safety |
| `planner.py` | Subtask decomposition |
| `checkpoint.py` | Save/resume with integrity |
| `quota.py` | Budget tracking |
| `config.py` | Configuration constants |
| `schemas.py` | Pydantic models |

### V3.2 Reliability (new)
| Module | Purpose |
|--------|---------|
| `evidence.py` | Centralized evidence store |
| `state_machine.py` | Deterministic state transitions |
| `regression.py` | Baseline comparison engine |
| `execution_controller.py` | Bounded subtask lifecycle |
| `diagnostics.py` | Extended to 19 error categories |

### V4 Intelligence (new)
| Module | Purpose |
|--------|---------|
| `project_understanding.py` | Repository analysis |
| `codebase_graph.py` | Module/function/test graph |
| `impact_analysis.py` | Change impact prediction |
| `test_selection.py` | Relevant test selection |
| `context_budget.py` | Priority-based context |
| `change_validator.py` | Change safety checks |
| `observability.py` | Execution trace reconstruction |

### V5 Autonomy (new)
| Module | Purpose |
|--------|---------|
| `policy_engine.py` | Permission decisions |
| `task_graph.py` | DAG-based planning |
| `adaptive_planning.py` | Evidence-driven revision |
| `self_reflection.py` | Structured failure analysis |
| `engineering_memory.py` | Strategy capture |
| `threat_model.py` | Threat documentation |
| `adversarial_testing.py` | Protection verification |
| `human_override.py` | STOP/CANCEL/PAUSE/RESUME |
| `validation_gate.py` | Final validation checks |

## Data Flow

1. User input → Orchestrator
2. Orchestrator → Policy Engine (can_execute_tool?)
3. Orchestrator → Planner (task decomposition)
4. Orchestrator → Execution Controller (lifecycle management)
5. Orchestrator → State Machine (transition validation)
6. LLM calls → Evidence Store (logged)
7. Tool calls → Evidence Store + Observability (traced)
8. Failures → Self-Reflection → Adaptive Planning
9. Test results → Regression Engine (baseline comparison)
10. Finalization → Validation Gate → Evidence Store → Report

## Design Principles

1. **Evidence-driven**: Every decision has structured evidence
2. **Deterministic**: State transitions, budgets, permissions are predictable
3. **Self-correcting**: Failures trigger diagnosis and replanning
4. **Observable**: Full trace reconstruction available
5. **Secure**: Zero trust, defense in depth
6. **Bounded**: All resources have explicit limits
