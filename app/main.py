from __future__ import annotations

from app.agent.orchestrator import Orchestrator
from app.config import AGENT_MODE, MAX_RETRY_CYCLES, PROJECT_ROOT, AgentMode
from app.models.schemas import TaskReport


def print_header(orchestrator: Orchestrator) -> None:
    print("\n=== AI ENGINEER AGENT V5 ===")
    print(f"Project: {PROJECT_ROOT}")
    print(f"Mode: {orchestrator.mode.value}")
    print("Type /help for commands, /exit to quit.\n")


def print_help() -> None:
    print("\nCommands:")
    print("  /help              - Show this help")
    print("  /status            - Show current configuration")
    print("  /mode              - Change execution mode")
    print("  /plan              - Show current plan")
    print("  /context           - Show current context summary")
    print("  /budget            - Show budget usage and limits")
    print("  /history           - Show task history from memory")
    print("  /inspect           - Show evidence store contents")
    print("  /trace             - Show execution trace")
    print("  /threats           - Show security threat model")
    print("  /adversarial       - Show adversarial test suite status")
    print("  /gate              - Show final validation gate status")
    print("  /override [cmd]    - STOP/CANCEL/PAUSE/RESUME execution")
    print("  /clear             - Clear conversation history")
    print("  /exit              - Exit the agent")
    print()


def print_status(orchestrator: Orchestrator) -> None:
    print(f"\nProject: {PROJECT_ROOT}")
    print(f"Mode: {orchestrator.mode.value}")
    print(f"Tools: {', '.join(orchestrator.core.tools.keys())}")
    print(f"Max retry cycles: {MAX_RETRY_CYCLES}")
    budget = orchestrator.budget.status()
    print(f"Budget: {budget['llm_calls']}/{budget['max_llm_calls']} LLM calls, "
          f"{budget['tool_calls']}/{budget['max_tool_calls']} tool calls")
    if orchestrator.context.plan:
        print(f"Plan: {len(orchestrator.context.plan.get('subtasks', []))} subtasks")
    print()


def print_plan(orchestrator: Orchestrator) -> None:
    plan = orchestrator.context.plan
    if not plan:
        print("\nNo plan available.\n")
        return

    print(f"\n--- Plan ---")
    print(f"Objective: {plan.get('objective', 'N/A')}")
    subtasks = plan.get("subtasks", [])
    if subtasks:
        for st in subtasks:
            st_id = st.get("id", "?")
            desc = st.get("description", "")
            status = st.get("status", "PENDING")
            print(f"  [{status}] {st_id}: {desc}")
    print()


def print_context(orchestrator: Orchestrator) -> None:
    ctx = orchestrator.context
    print(f"\n--- Context ---")
    print(f"Phase: {ctx.current_phase}")
    print(f"Phase history: {' -> '.join(ctx.phase_history)}")
    print(f"Iterations: {ctx.iteration_count}, Retries: {ctx.retry_count}")

    if ctx.inspected_files:
        print(f"Inspected files: {len(ctx.inspected_files)}")

    if ctx.diagnoses:
        print(f"Diagnoses: {len(ctx.diagnoses)}")

    if ctx.fixes:
        print(f"Fixes: {len(ctx.fixes)}")

    if ctx.review_feedback:
        print(f"Review feedback: pending")
    print()


def print_report(report: TaskReport) -> None:
    print("\nAgent:")
    print(report.final_response)

    has_actions = (
        report.files_inspected
        or report.files_modified
        or report.commands_executed
        or report.test_results
    )

    if has_actions:
        print("\n--- Actions ---")

        if report.files_inspected:
            print("  Inspected:")
            for f in report.files_inspected:
                print(f"    - {f}")

        if report.files_modified:
            print("  Modified:")
            for f in report.files_modified:
                print(f"    - {f}")

        if report.commands_executed:
            print("  Commands:")
            for c in report.commands_executed:
                print(f"    - {c}")

        if report.test_results is not None:
            exit_code = report.test_results.get("exit_code", "?")
            status = "PASS" if report.test_results.get("success") else "FAIL"
            print(f"  Tests: {status} (exit code {exit_code})")

    if report.review is not None:
        print("\n--- Review ---")
        status = "APPROVED" if report.review.approved else "CHANGES REQUESTED"
        print(f"  Status: {status}")
        print(f"  Summary: {report.review.summary}")
        if report.review.findings:
            print("  Findings:")
            for f in report.review.findings:
                print(f"    [{f.severity.value}] {f.category}: {f.description}")
                if f.file:
                    print(f"      at {f.file}:{f.line}" if f.line else f"      at {f.file}")

    print(f"\n--- Execution ---")
    print(f"  Phase: {report.final_phase}")
    print(f"  Iterations: {report.iteration_count}")
    if report.retry_count > 0:
        print(f"  Retries: {report.retry_count}")
    if report.diagnoses:
        print(f"  Diagnoses: {len(report.diagnoses)}")
    if report.fixes:
        print(f"  Fixes: {len(report.fixes)}")
    if report.stop_reason != "completed":
        print(f"  Stop reason: {report.stop_reason}")

    print()


def main() -> None:
    orchestrator = Orchestrator()

    print_header(orchestrator)

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"/exit", "exit", "quit"}:
            print("\nGoodbye.")
            break

        if user_input == "/help":
            print_help()
            continue

        if user_input == "/status":
            print_status(orchestrator)
            continue

        if user_input == "/plan":
            print_plan(orchestrator)
            continue

        if user_input == "/context":
            print_context(orchestrator)
            continue

        if user_input == "/mode":
            print("\nAvailable modes:")
            for m in AgentMode:
                print(f"  {m.value}")
            print()
            choice = input("Select mode: ").strip()
            try:
                new_mode = AgentMode(choice)
                orchestrator.set_mode(new_mode)
                print(f"\nMode set to: {new_mode.value}\n")
            except ValueError:
                print(f"\nInvalid mode: {choice}\n")
            continue

        if user_input == "/clear":
            orchestrator.core.history.clear()
            orchestrator.core.executions.clear()
            print("\nConversation cleared.\n")
            continue

        if user_input == "/budget":
            budget = orchestrator.budget.status()
            print(f"\n--- Budget ---")
            print(f"LLM calls:   {budget['llm_calls']}/{budget['max_llm_calls']}")
            print(f"Tool calls:  {budget['tool_calls']}/{budget['max_tool_calls']}")
            print(f"Retries:     {budget['retry_cycles']}/{budget['max_retry_cycles']}")
            within = "YES" if budget['within_budget'] else "NO"
            print(f"Within budget: {within}")
            violation = orchestrator.budget.budget_violation()
            if violation:
                print(f"Violation: {violation}")
            print()
            continue

        if user_input == "/inspect":
            try:
                from app.agent.evidence import EvidenceStore
                store = EvidenceStore()
                ev_list = store.list_evidence()
                print(f"\n--- Evidence Store ---")
                print(f"Total evidence: {len(ev_list)}")
                if ev_list:
                    by_type: dict[str, int] = {}
                    for ev in ev_list:
                        by_type[ev.evidence_type.value] = by_type.get(ev.evidence_type.value, 0) + 1
                    for etype, count in sorted(by_type.items()):
                        print(f"  {etype}: {count}")
                    print("\nLatest evidence:")
                    for ev in ev_list[-5:]:
                        print(f"  [{ev.evidence_type.value}] {ev.description[:70]}")
                else:
                    print("  No evidence recorded yet.")
            except Exception as e:
                print(f"\n  Evidence store not available: {e}")
            print()
            continue

        if user_input == "/trace":
            try:
                from app.agent.observability import ExecutionTrace
                trace = ExecutionTrace()
                spans = trace.spans
                print(f"\n--- Execution Trace ---")
                print(f"Total spans: {len(spans)}")
                if spans:
                    print("Latest spans:")
                    for span in spans[-5:]:
                        duration_ms = (span.end_time - span.start_time) * 1000
                        status = "OK" if not span.error else "ERROR"
                        print(f"  [{status}] {span.name} ({duration_ms:.0f}ms)")
                        if span.error:
                            print(f"    Error: {span.error}")
                else:
                    print("  No execution spans yet.")
            except Exception as e:
                print(f"\n  Trace not available: {e}")
            print()
            continue

        if user_input == "/threats":
            try:
                from app.agent.threat_model import SecurityThreatModel
                tm = SecurityThreatModel()
                summary = tm.summary()
                print(f"\n--- Security Threat Model ---")
                print(f"Total threats: {summary['total_threats']}")
                print(f"Tested: {summary['tested']}, Untested: {summary['untested']}")
                for sev, count in summary['by_severity'].items():
                    print(f"  {sev}: {count}")
            except Exception as e:
                print(f"\n  Threat model not available: {e}")
            print()
            continue

        if user_input == "/adversarial":
            try:
                from app.agent.adversarial_testing import AdversarialTestSuite
                suite = AdversarialTestSuite()
                summary = suite.summary()
                print(f"\n--- Adversarial Test Suite ---")
                print(f"Total tests: {summary['total_tests']}")
                print(f"Passed: {summary['passed']}, Failed: {summary['failed']}")
                for cat, count in summary['categories'].items():
                    print(f"  {cat}: {count}")
            except Exception as e:
                print(f"\n  Adversarial suite not available: {e}")
            print()
            continue

        if user_input == "/gate":
            try:
                from app.agent.validation_gate import FinalValidationGate
                gate = FinalValidationGate()
                result = gate.validate(
                    policy_passed=True,
                    tests_passed=True,
                    security_passed=True,
                    no_regressions=True,
                    scope_valid=True,
                    has_evidence=True,
                    review_approved=True,
                )
                print(f"\n--- Final Validation Gate ---")
                for r in result.results:
                    status = "PASS" if r.passed else "FAIL"
                    print(f"  [{status}] {r.check.value}: {r.details}")
                print(f"\nOverall: {'PASS' if result.overall_passed else 'BLOCKED'}")
            except Exception as e:
                print(f"\n  Validation gate not available: {e}")
            print()
            continue

        if user_input.startswith("/override"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[1].upper() if len(parts) > 1 else ""
            try:
                from app.agent.human_override import HumanOverride
                override = HumanOverride()
                if cmd == "STOP":
                    override.stop("User-initiated stop")
                    print("\nExecution stopped.\n")
                elif cmd == "CANCEL":
                    override.cancel("User-initiated cancel")
                    print("\nExecution cancelled.\n")
                elif cmd == "PAUSE":
                    override.pause("User-initiated pause")
                    print("\nExecution paused.\n")
                elif cmd == "RESUME":
                    override.resume()
                    print("\nExecution resumed.\n")
                else:
                    print("\nOverride commands: STOP, CANCEL, PAUSE, RESUME")
                    print("Usage: /override STOP|CANCEL|PAUSE|RESUME\n")
            except Exception as e:
                print(f"\n  Override not available: {e}")
            continue

        if user_input == "/clear":
            orchestrator.core.history.clear()
            orchestrator.core.executions.clear()
            print("\nConversation cleared.\n")
            continue

        if user_input == "/history":
            memory = orchestrator.memory
            print(f"\n--- Project Memory ---")
            if memory.important_files:
                print(f"Important files: {len(memory.important_files)}")
                for path, desc in list(memory.important_files.items())[:5]:
                    print(f"  {path}: {desc[:60]}")
            if memory.architecture_notes:
                print(f"Architecture notes: {len(memory.architecture_notes)}")
                for note in memory.architecture_notes[-3:]:
                    print(f"  - {note[:80]}")
            if memory.known_commands:
                print(f"Known commands: {len(memory.known_commands)}")
            if memory.known_test_commands:
                print(f"Test commands: {memory.known_test_commands}")
            if memory.important_decisions:
                print(f"Decisions: {len(memory.important_decisions)}")
                for dec in memory.important_decisions[-3:]:
                    print(f"  - {dec[:80]}")
            if memory.previous_failures:
                print(f"Previous failures: {len(memory.previous_failures)}")
                for fail in memory.previous_failures[-3:]:
                    print(f"  - {fail[:80]}")
            if memory.successful_fixes:
                print(f"Successful fixes: {len(memory.successful_fixes)}")
                for fix in memory.successful_fixes[-3:]:
                    print(f"  - {fix[:80]}")
            if memory.project_conventions:
                print(f"Conventions: {len(memory.project_conventions)}")
                for conv in memory.project_conventions[-3:]:
                    print(f"  - {conv[:80]}")
            if not any([memory.important_files, memory.architecture_notes,
                       memory.known_commands, memory.known_test_commands,
                       memory.important_decisions, memory.previous_failures,
                       memory.successful_fixes, memory.project_conventions]):
                print("  No entries yet.")
            print()
            continue

        try:
            report = orchestrator.run_task(user_input)
            print_report(report)
        except Exception as exc:
            print(f"\nAgent error: {exc}\n")


if __name__ == "__main__":
    main()
