from __future__ import annotations

from app.agent.orchestrator import Orchestrator
from app.config import AGENT_MODE, MAX_RETRY_CYCLES, PROJECT_ROOT, AgentMode
from app.models.schemas import TaskReport


def print_header(orchestrator: Orchestrator) -> None:
    print("\n=== AI ENGINEER AGENT ===")
    print(f"Project: {PROJECT_ROOT}")
    print(f"Mode: {orchestrator.mode.value}")
    print("Type /help for commands, /exit to quit.\n")


def print_help() -> None:
    print("\nCommands:")
    print("  /help    - Show this help")
    print("  /status  - Show current configuration")
    print("  /mode    - Change execution mode")
    print("  /clear   - Clear conversation history")
    print("  /exit    - Exit the agent")
    print()


def print_status(orchestrator: Orchestrator) -> None:
    print(f"\nProject: {PROJECT_ROOT}")
    print(f"Mode: {orchestrator.mode.value}")
    print(f"Tools: {', '.join(orchestrator.core.tools.keys())}")
    print(f"Max retry cycles: {MAX_RETRY_CYCLES}")
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
