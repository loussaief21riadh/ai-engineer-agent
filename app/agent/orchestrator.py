from __future__ import annotations

import json
from typing import Any

from app.agent.context import ContextBuilder, TaskContext, TrustLevel
from app.agent.core import AgentCore
from app.agent.diagnostics import FailureAnalyzer
from app.agent.phases import TaskPhase, can_transition
from app.agent.planner import PlanValidator
from app.agent.quota import BudgetTracker
from app.agent.reviewer import Reviewer
from app.agent.validator import Validator
from app.config import AGENT_MODE, MAX_RETRY_CYCLES, AgentMode
from app.llm.openrouter import OpenRouterClient
from app.models.schemas import TaskReport, ToolExecution
from app.tools.base import BaseTool
from app.tools.edit import EditFileTool
from app.tools.filesystem import ListFilesTool, ReadFileTool, WriteFileTool
from app.tools.terminal import RunCommandTool
from app.tools.testing import RunTestsTool
from app.tools.git import GitDiffTool, GitStatusTool


def _build_report_from_executions(
    task: str,
    mode: str,
    final_response: str,
    steps_taken: int,
    executions: list[ToolExecution],
    review: Any = None,
    final_phase: str = "DONE",
    phase_history: list[str] | None = None,
    iteration_count: int = 0,
    retry_count: int = 0,
    diagnoses: list[str] | None = None,
    fixes: list[str] | None = None,
    stop_reason: str = "completed",
    validation_report: Any = None,
) -> TaskReport:
    files_inspected: list[str] = []
    files_modified: list[str] = []
    commands_executed: list[str] = []
    test_results: dict[str, Any] | None = None

    for ex in executions:
        if not ex.success:
            continue

        if ex.tool_name == "read_file":
            path = ex.arguments.get("path", "")
            if path and path not in files_inspected:
                files_inspected.append(path)

        elif ex.tool_name == "list_files":
            path = ex.arguments.get("path", ".")
            label = f"[dir] {path}"
            if label not in files_inspected:
                files_inspected.append(label)

        elif ex.tool_name in ("write_file", "edit_file"):
            path = ex.arguments.get("path", "")
            if path and path not in files_modified:
                files_modified.append(path)

        elif ex.tool_name == "run_command":
            cmd = ex.arguments.get("command", "")
            if cmd and cmd not in commands_executed:
                commands_executed.append(cmd)

        elif ex.tool_name == "run_tests":
            if isinstance(ex.result, dict):
                test_results = ex.result

        elif ex.tool_name in ("git_status", "git_diff"):
            label = f"{ex.tool_name}"
            if label not in commands_executed:
                commands_executed.append(label)

    return TaskReport(
        task=task,
        mode=mode,
        files_inspected=files_inspected,
        files_modified=files_modified,
        commands_executed=commands_executed,
        test_results=test_results,
        review=review,
        final_response=final_response,
        steps_taken=steps_taken,
        executions=executions,
        final_phase=final_phase,
        phase_history=phase_history or [],
        iteration_count=iteration_count,
        retry_count=retry_count,
        diagnoses=diagnoses or [],
        fixes=fixes or [],
        stop_reason=stop_reason,
    )


class Orchestrator:
    def __init__(
        self,
        client: OpenRouterClient | None = None,
        mode: AgentMode | None = None,
    ) -> None:
        self.client = client or OpenRouterClient()
        self.mode = mode or AGENT_MODE

        self.context = TaskContext()
        self.context_builder = ContextBuilder()
        self.failure_analyzer = FailureAnalyzer()
        self.validator = Validator()
        self.budget = BudgetTracker()

        self.core = AgentCore(client=self.client, on_llm_call=self.budget.record_llm_call)
        self.reviewer = Reviewer(client=self.client)
        self._accumulated_executions: list[ToolExecution] = []
        self._last_review_result: Any = None
        self._review_retry_count: int = 0
        self._review_feedback: str = ""
        self._plan_validated = False
        self._plan_rejected = False

        self._register_default_tools()

    def _register_default_tools(self) -> None:
        self.core.register_tool(ListFilesTool())
        self.core.register_tool(ReadFileTool())
        self.core.register_tool(RunCommandTool())
        self.core.register_tool(RunTestsTool())
        self.core.register_tool(GitStatusTool())
        self.core.register_tool(GitDiffTool())

        if self.mode in (AgentMode.ALLOW_EDITS, AgentMode.FULL_AUTONOMOUS):
            self.core.register_tool(WriteFileTool())
            self.core.register_tool(EditFileTool())

    def set_mode(self, mode: AgentMode) -> None:
        self.mode = mode
        self.core.tools.clear()
        self._register_default_tools()

    def run_task(self, user_message: str) -> TaskReport:
        phase_history: list[str] = []
        diagnoses: list[str] = []
        fixes: list[str] = []
        retry_count = 0
        iteration_count = 0

        current_phase = TaskPhase.UNDERSTAND
        phase_history.append(current_phase.value)

        self.core.history = []
        self.core.executions = []
        self._accumulated_executions = []
        self._last_review_result = None
        self._review_retry_count = 0
        self._review_feedback = ""

        self.context = TaskContext(task=user_message)
        self.context.transition_to("UNDERSTAND")
        self.context.add_observation("Task received by orchestrator", trust=TrustLevel.USER_ASSERTED)

        self.budget.start()
        self._plan_validated = False
        self._plan_rejected = False

        max_iterations = 50

        while current_phase not in (TaskPhase.DONE, TaskPhase.FAILED):
            if not self.budget.is_within_budget():
                violation = self.budget.budget_violation() or "Budget exceeded"
                return _build_report_from_executions(
                    task=user_message,
                    mode=self.mode.value,
                    final_response=f"Budget limit reached: {violation}",
                    steps_taken=len(self.core.history),
                    executions=list(self._accumulated_executions),
                    final_phase=current_phase.value,
                    phase_history=phase_history,
                    iteration_count=iteration_count,
                    retry_count=retry_count,
                    diagnoses=diagnoses,
                    fixes=fixes,
                    stop_reason="budget_exceeded",
                )

            iteration_count += 1
            self.context.iteration_count = iteration_count
            self.context.retry_count = retry_count

            if iteration_count > max_iterations:
                return _build_report_from_executions(
                    task=user_message,
                    mode=self.mode.value,
                    final_response="Reached maximum iterations without completing the task.",
                    steps_taken=len(self.core.history),
                    executions=list(self._accumulated_executions),
                    final_phase=current_phase.value,
                    phase_history=phase_history,
                    iteration_count=iteration_count,
                    retry_count=retry_count,
                    diagnoses=diagnoses,
                    fixes=fixes,
                    stop_reason="iteration_limit_reached",
                )

            if current_phase == TaskPhase.UNDERSTAND:
                current_phase = self._phase_understand(user_message, phase_history)

            elif current_phase == TaskPhase.PLAN:
                current_phase = self._phase_plan(user_message, phase_history)

            elif current_phase == TaskPhase.INSPECT:
                current_phase = self._phase_inspect(user_message, phase_history)

            elif current_phase == TaskPhase.IMPLEMENT:
                current_phase = self._phase_implement(
                    user_message, phase_history
                )

            elif current_phase == TaskPhase.TEST:
                test_passed, current_phase = self._phase_test(
                    user_message, phase_history
                )

            elif current_phase == TaskPhase.DIAGNOSE:
                current_phase = self._phase_diagnose(
                    user_message, phase_history, diagnoses
                )

            elif current_phase == TaskPhase.FIX:
                if self.mode == AgentMode.READ_ONLY:
                    current_phase = TaskPhase.FAILED
                    phase_history.append(current_phase.value)
                else:
                    current_phase = self._phase_fix(
                        user_message, phase_history, fixes
                    )

            elif current_phase == TaskPhase.RETEST:
                current_phase = self._phase_retest(
                    user_message, phase_history, retry_count, MAX_RETRY_CYCLES
                )
                if current_phase == TaskPhase.DIAGNOSE:
                    retry_count += 1

            elif current_phase == TaskPhase.REVIEW:
                current_phase = self._phase_review(
                    user_message, phase_history
                )

            elif current_phase == TaskPhase.VALIDATE:
                current_phase = self._phase_validate(
                    user_message, phase_history
                )

            elif current_phase == TaskPhase.REPORT:
                current_phase = self._phase_report(
                    user_message, phase_history
                )

            else:
                current_phase = TaskPhase.FAILED
                phase_history.append(current_phase.value)

        self.budget.stop()
        final_response = self._get_final_response()

        return _build_report_from_executions(
            task=user_message,
            mode=self.mode.value,
            final_response=final_response,
            steps_taken=len(self.core.history),
            executions=list(self._accumulated_executions),
            review=self._last_review_result,
            final_phase=current_phase.value,
            phase_history=phase_history,
            iteration_count=iteration_count,
            retry_count=retry_count,
            diagnoses=diagnoses,
            fixes=fixes,
            stop_reason="completed"
            if current_phase == TaskPhase.DONE
            else "failed",
        )

    def _run_agent_step(self, phase_prompt: str) -> str:
        self.core.history.clear()
        self.core.executions.clear()
        result = self.core.run(phase_prompt, max_steps=10)
        self._accumulated_executions.extend(self.core.executions)

        for ex in self.core.executions:
            self.budget.record_tool_call()
            self.context.record_execution({
                "tool_name": ex.tool_name,
                "success": ex.success,
                "arguments": ex.arguments,
            })
            if ex.tool_name == "read_file" and ex.success:
                path = ex.arguments.get("path", "")
                content = str(ex.result)[:500] if ex.result else ""
                self.context.record_inspected_file(path, content)
                self.context.add_observation(
                    f"Read file {path}",
                    trust=TrustLevel.TOOL_VERIFIED,
                )

        return result

    def _record_test_execution(self, result: dict) -> None:
        self._accumulated_executions.append(ToolExecution(
            step=len(self._accumulated_executions) + 1,
            tool_name="run_tests",
            arguments={"args": "-v"},
            success=result.get("success", False),
            result=result.get("result"),
            error=result.get("error"),
        ))
        self.context.test_results = result
        if not result.get("success", False):
            stdout = ""
            if isinstance(result.get("result"), dict):
                stdout = result["result"].get("stdout", "")
            elif result.get("error"):
                stdout = result["error"]
            self.context.record_failure(stdout[:500])

    def _phase_understand(
        self, task: str, phase_history: list[str]
    ) -> TaskPhase:
        self.context.transition_to("UNDERSTAND")
        prompt = self.context_builder.build_phase_prompt(
            self.context, "UNDERSTAND", task
        )
        prompt += "\n\nAnalyze the task. What is being asked? What files are likely involved? Do not make changes yet."
        self._run_agent_step(prompt)
        return self._advance(TaskPhase.UNDERSTAND, TaskPhase.PLAN, phase_history)

    def _phase_plan(
        self, task: str, phase_history: list[str]
    ) -> TaskPhase:
        self.context.transition_to("PLAN")
        prompt = self.context_builder.build_phase_prompt(
            self.context, "PLAN", task
        )
        prompt += (
            "\n\nCreate a step-by-step plan. List the files to inspect and changes to make. "
            "Do not implement yet."
        )
        response = self._run_agent_step(prompt)

        try:
            import json as _json
            data = _json.loads(response)
            is_valid, errors, plan = PlanValidator.validate_plan(data)
            if is_valid and plan is not None:
                self.context.set_plan(plan.to_dict())
                self._plan_validated = True
                self._plan_rejected = False
                self.context.add_observation(
                    "Plan validated successfully",
                    trust=TrustLevel.SYSTEM_DERIVED,
                )
            else:
                self._plan_validated = False
                self._plan_rejected = True
                self.context.add_observation(
                    f"Plan validation failed: {'; '.join(errors)}",
                    trust=TrustLevel.MODEL_INFERRED,
                )
                phase_history.append(TaskPhase.FAILED.value)
                return TaskPhase.FAILED
        except (_json.JSONDecodeError, ValueError):
            self._plan_validated = False
            self._plan_rejected = False
            self.context.add_observation(
                "Plan response was not structured JSON; proceeding with free-text plan",
                trust=TrustLevel.MODEL_INFERRED,
            )

        return self._advance(TaskPhase.PLAN, TaskPhase.INSPECT, phase_history)

    def _phase_inspect(
        self, task: str, phase_history: list[str]
    ) -> TaskPhase:
        self.context.transition_to("INSPECT")
        prompt = self.context_builder.build_phase_prompt(
            self.context, "INSPECT", task
        )
        prompt += (
            "\n\nRead the relevant files. Understand the current code before making changes."
        )
        self._run_agent_step(prompt)
        return self._advance(TaskPhase.INSPECT, TaskPhase.IMPLEMENT, phase_history)

    def _phase_implement(
        self, task: str, phase_history: list[str]
    ) -> TaskPhase:
        self.context.transition_to("IMPLEMENT")
        if self.mode == AgentMode.READ_ONLY:
            prompt = self.context_builder.build_phase_prompt(
                self.context, "IMPLEMENT (READ-ONLY)", task
            )
            prompt += (
                "\n\nYou are in read-only mode. You cannot write files. "
                "Analyze what changes would be needed and report them."
            )
        else:
            prompt = self.context_builder.build_phase_prompt(
                self.context, "IMPLEMENT", task
            )
            prompt += (
                "\n\nImplement the planned changes. Make minimal, targeted modifications."
            )
        self._run_agent_step(prompt)
        return self._advance(TaskPhase.IMPLEMENT, TaskPhase.TEST, phase_history)

    def _phase_test(
        self, task: str, phase_history: list[str]
    ) -> tuple[bool, TaskPhase]:
        self.context.transition_to("TEST")
        test_tool = self.core.tools.get("run_tests")
        if test_tool is None:
            return True, self._advance(
                TaskPhase.TEST, TaskPhase.REVIEW, phase_history
            )

        result = test_tool.execute(args="-v")
        self.budget.record_tool_call()
        self._record_test_execution(result)
        success = result.get("success", False)

        if success:
            self.context.add_observation(
                "Tests passed", trust=TrustLevel.TOOL_VERIFIED
            )
            return True, self._advance(
                TaskPhase.TEST, TaskPhase.REVIEW, phase_history
            )
        else:
            self.context.add_observation(
                "Tests failed", trust=TrustLevel.TOOL_VERIFIED
            )
            return False, self._advance(
                TaskPhase.TEST, TaskPhase.DIAGNOSE, phase_history
            )

    def _phase_diagnose(
        self,
        task: str,
        phase_history: list[str],
        diagnoses: list[str],
    ) -> TaskPhase:
        self.context.transition_to("DIAGNOSE")
        test_failures = ""
        for ex in reversed(self._accumulated_executions):
            if ex.tool_name == "run_tests" and not ex.success:
                if isinstance(ex.result, dict):
                    test_failures = ex.result.get("stdout", "")
                elif ex.error:
                    test_failures = ex.error
                break

        diag = self.failure_analyzer.analyze(test_failures)
        self.context.add_observation(
            f"Diagnosis: {diag.category.value} - {diag.hypothesis}",
            trust=TrustLevel.MODEL_INFERRED,
        )

        prompt = self.context_builder.build_phase_prompt(
            self.context, "DIAGNOSE", task
        )
        prompt += (
            f"\n\nTests failed. Actual failure output:\n{test_failures[:3000]}\n\n"
            f"Automated classification: {diag.category.value}\n"
            f"Hypothesis: {diag.hypothesis}\n\n"
            f"Analyze the failure. Identify the root cause, failing tests, "
            f"and relevant files. Do not make changes yet."
        )
        response = self._run_agent_step(prompt)
        diagnoses.append(response[:500])
        self.context.record_diagnosis(response[:500])
        return self._advance(TaskPhase.DIAGNOSE, TaskPhase.FIX, phase_history)

    def _phase_fix(
        self,
        task: str,
        phase_history: list[str],
        fixes: list[str],
    ) -> TaskPhase:
        self.context.transition_to("FIX")
        if self._review_feedback:
            self.context.review_feedback = self._review_feedback
            prompt = self.context_builder.build_phase_prompt(
                self.context, "FIX (reviewer rejection)", task
            )
            prompt += (
                "\n\nThe reviewer rejected the implementation with the feedback above. "
                "Fix the issues identified by the reviewer. "
                "Do not refactor unrelated code."
            )
        else:
            prompt = self.context_builder.build_phase_prompt(
                self.context, "FIX", task
            )
            prompt += (
                "\n\nApply the minimal fix based on your diagnosis. "
                "Do not refactor unrelated code."
            )
        response = self._run_agent_step(prompt)
        fixes.append(response[:500])
        self.context.record_fix(response[:500])
        self._review_feedback = ""
        self.context.review_feedback = ""
        return self._advance(TaskPhase.FIX, TaskPhase.RETEST, phase_history)

    def _phase_retest(
        self,
        task: str,
        phase_history: list[str],
        retry_count: int,
        max_retries: int,
    ) -> TaskPhase:
        self.context.transition_to("RETEST")
        if retry_count >= max_retries:
            phase_history.append(TaskPhase.FAILED.value)
            return TaskPhase.FAILED

        test_tool = self.core.tools.get("run_tests")
        if test_tool is None:
            return self._advance(
                TaskPhase.RETEST, TaskPhase.TEST, phase_history
            )

        result = test_tool.execute(args="-v")
        self.budget.record_tool_call()
        self._record_test_execution(result)
        success = result.get("success", False)

        if success:
            self.context.add_observation(
                "Retest passed", trust=TrustLevel.TOOL_VERIFIED
            )
            return self._advance(
                TaskPhase.RETEST, TaskPhase.REVIEW, phase_history
            )
        else:
            self.context.add_observation(
                "Retest failed", trust=TrustLevel.TOOL_VERIFIED
            )
            return self._advance(
                TaskPhase.RETEST, TaskPhase.DIAGNOSE, phase_history
            )

    def _phase_review(
        self, task: str, phase_history: list[str]
    ) -> TaskPhase:
        self.context.transition_to("REVIEW")
        if self._review_retry_count >= MAX_RETRY_CYCLES:
            phase_history.append(TaskPhase.FAILED.value)
            return TaskPhase.FAILED

        try:
            review_result = self._run_review(
                task, list(self._accumulated_executions), self._get_final_response()
            )
        except Exception:
            from app.models.schemas import ReviewResult
            review_result = ReviewResult(
                approved=False,
                summary="Review failed due to an internal error.",
            )

        self._last_review_result = review_result

        if review_result.approved:
            self.context.add_observation(
                "Review approved", trust=TrustLevel.TOOL_VERIFIED
            )
            return self._advance(TaskPhase.REVIEW, TaskPhase.VALIDATE, phase_history)
        else:
            findings_text = "; ".join(
                f"[{f.severity.value}] {f.description}" for f in review_result.findings
            ) if review_result.findings else review_result.summary
            self._review_feedback = (
                f"Reviewer rejected the implementation.\n"
                f"Summary: {review_result.summary}\n"
                f"Findings: {findings_text}"
            )
            self.context.review_feedback = self._review_feedback
            self._review_retry_count += 1
            return self._advance(TaskPhase.REVIEW, TaskPhase.FIX, phase_history)

    def _phase_validate(
        self, task: str, phase_history: list[str]
    ) -> TaskPhase:
        self.context.transition_to("VALIDATE")
        test_results = None
        for ex in reversed(self._accumulated_executions):
            if ex.tool_name == "run_tests" and isinstance(ex.result, dict):
                test_results = ex.result
                break

        files_modified = []
        for ex in self._accumulated_executions:
            if ex.success and ex.tool_name in ("write_file", "edit_file"):
                path = ex.arguments.get("path", "")
                if path and path not in files_modified:
                    files_modified.append(path)

        validation = self.validator.validate_all(
            test_results=test_results,
            files_modified=files_modified,
        )

        self.context.add_observation(
            f"Validation: {'passed' if validation.overall_passed else 'failed'}",
            trust=TrustLevel.SYSTEM_DERIVED,
        )

        return self._advance(TaskPhase.VALIDATE, TaskPhase.REPORT, phase_history)

    def _phase_report(
        self, task: str, phase_history: list[str]
    ) -> TaskPhase:
        self.context.transition_to("REPORT")
        return self._advance(TaskPhase.REPORT, TaskPhase.DONE, phase_history)

    def _advance(
        self,
        current: TaskPhase,
        target: TaskPhase,
        phase_history: list[str],
    ) -> TaskPhase:
        if can_transition(current, target):
            phase_history.append(target.value)
            return target
        return TaskPhase.FAILED

    def _get_final_response(self) -> str:
        if self.core.history:
            for msg in reversed(self.core.history):
                if msg.role == "assistant" and msg.content:
                    return msg.content
        return "Task completed."

    def _run_review(
        self,
        task: str,
        executions: list[ToolExecution],
        final_response: str,
    ) -> Any:
        changes_lines: list[str] = []
        diff_lines: list[str] = []
        test_results_str = ""

        for ex in executions:
            if ex.success and ex.tool_name == "read_file":
                path = ex.arguments.get("path", "")
                content_preview = str(ex.result)[:500] if ex.result else ""
                changes_lines.append(f"Read {path}:\n{content_preview}")

            elif ex.success and ex.tool_name in ("write_file", "edit_file"):
                path = ex.arguments.get("path", "")
                action = "Wrote" if ex.tool_name == "write_file" else "Edited"
                changes_lines.append(f"{action} {path}")

            elif ex.success and ex.tool_name == "run_command":
                cmd = ex.arguments.get("command", "")
                result_data = ex.result if isinstance(ex.result, dict) else {}
                stdout = result_data.get("stdout", "")
                diff_lines.append(f"$ {cmd}\n{stdout[:1000]}")

            elif ex.success and ex.tool_name in ("git_status", "git_diff"):
                result_str = str(ex.result) if ex.result else ""
                diff_lines.append(f"[{ex.tool_name}]\n{result_str[:1000]}")

            elif ex.tool_name == "run_tests" and isinstance(ex.result, dict):
                test_results_str = json.dumps(ex.result, indent=2, default=str)[:2000]

        changes_text = "\n".join(changes_lines) if changes_lines else "No file changes recorded."
        diff_text = "\n".join(diff_lines) if diff_lines else "No diff available."

        review_context = self.context_builder.build_review_context(
            self.context, task, changes_text, diff_text, test_results_str,
        )

        return self.reviewer.review(
            task=task,
            changes=changes_text,
            diff=diff_text,
            test_results=test_results_str,
            context=review_context,
        )

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [tool.to_dict() for tool in self.core.tools.values()]
