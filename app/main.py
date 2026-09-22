from __future__ import annotations

from app.agent.orchestrator import Orchestrator
from app.config import AGENT_MODE, MAX_RETRY_CYCLES, PROJECT_ROOT, AgentMode
from app.models.schemas import TaskReport


def print_header(orchestrator: Orchestrator) -> None:
    print("\n=== AI ENGINEER AGENT V2.0 ===")
    print(f"Project: {PROJECT_ROOT}")
    print(f"Mode: {orchestrator.mode.value}")
    print("Type /help for commands, /exit to quit.\n")


def print_help() -> None:
    print("\nCommands:")
    print("  /help    - Show this help")
    print("  /status  - Show current configuration")
    print("  /mode    - Change execution mode")
    print("  /plan    - Show current plan")
    print("  /context - Show current context summary")
    print("  /clear   - Clear conversation history")
    print("  /exit    - Exit the agent")
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

        try:
            report = orchestrator.run_task(user_input)
            print_report(report)
        except Exception as exc:
            print(f"\nAgent error: {exc}\n")


if __name__ == "__main__":
    main()
