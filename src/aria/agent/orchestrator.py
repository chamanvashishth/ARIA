"""Approval-gated orchestration for explicitly registered local tools."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Literal
from uuid import uuid4

from aria.agent.tools import ToolDefinition, ToolRegistry, ToolResult
from aria.verification import VerificationReport, verify_tool_result


@dataclass(frozen=True)
class AgentStep:
    """One declarative tool call in an agent plan."""

    step_id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentPlan:
    """A proposed sequence/DAG of local tool calls for a single goal."""

    goal: str
    steps: tuple[AgentStep, ...]
    plan_id: str = field(default_factory=lambda: uuid4().hex)


@dataclass(frozen=True)
class StepExecution:
    """Auditable outcome of one planned step."""

    step_id: str
    tool_name: str
    status: Literal["completed", "rejected", "failed"]
    result: ToolResult | None = None
    verification: VerificationReport | None = None
    message: str = ""


@dataclass(frozen=True)
class ExecutionReport:
    """Immutable summary of an approval-gated plan execution."""

    plan_id: str
    goal: str
    status: Literal["completed", "rejected", "failed"]
    steps: tuple[StepExecution, ...]
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "completed"


PlanProvider = Callable[[str, list[ToolDefinition]], AgentPlan]
ApprovalCallback = Callable[[AgentStep], bool]


class PlanValidationError(ValueError):
    """Raised when a proposed plan is malformed or references unavailable tools."""


class AgentOrchestrator:
    """Validate plans and execute allowlisted tools only after explicit approval.

    The orchestrator does not generate plans, execute shell commands, import
    tools dynamically, or infer permission from model output. A caller supplies
    a planner and an approval callback; every tool invocation passes through
    the existing ToolRegistry validation boundary.
    """

    def __init__(self, registry: ToolRegistry, *, max_steps: int = 32) -> None:
        if not isinstance(registry, ToolRegistry):
            raise TypeError("registry must be a ToolRegistry instance")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        self.registry = registry
        self.max_steps = max_steps

    def _ordered_steps(self, plan: AgentPlan) -> tuple[AgentStep, ...]:
        if not isinstance(plan, AgentPlan):
            raise PlanValidationError("planner must return an AgentPlan")
        if not isinstance(plan.plan_id, str) or not plan.plan_id.strip():
            raise PlanValidationError("plan_id must be a non-empty string")
        if not isinstance(plan.goal, str) or not plan.goal.strip():
            raise PlanValidationError("plan goal must be a non-empty string")
        if not isinstance(plan.steps, tuple) or not plan.steps:
            raise PlanValidationError("plan must contain at least one step")
        if len(plan.steps) > self.max_steps:
            raise PlanValidationError(f"plan exceeds the {self.max_steps}-step limit")

        by_id: dict[str, AgentStep] = {}
        for step in plan.steps:
            if not isinstance(step, AgentStep):
                raise PlanValidationError("every plan entry must be an AgentStep")
            if not isinstance(step.step_id, str) or not step.step_id.strip():
                raise PlanValidationError("step IDs must be non-empty strings")
            if step.step_id in by_id:
                raise PlanValidationError(f"duplicate step ID: {step.step_id}")
            if not isinstance(step.tool_name, str) or not step.tool_name.strip():
                raise PlanValidationError(f"step {step.step_id} has an empty tool name")
            if not isinstance(step.arguments, dict):
                raise PlanValidationError(f"arguments for step {step.step_id} must be an object")
            if not isinstance(step.depends_on, tuple):
                raise PlanValidationError(f"dependencies for step {step.step_id} must be a tuple")
            if any(not isinstance(dep, str) or not dep.strip() for dep in step.depends_on):
                raise PlanValidationError(f"dependencies for step {step.step_id} must be non-empty IDs")
            if len(set(step.depends_on)) != len(step.depends_on):
                raise PlanValidationError(f"step {step.step_id} contains duplicate dependencies")
            if step.step_id in step.depends_on:
                raise PlanValidationError(f"step {step.step_id} cannot depend on itself")
            by_id[step.step_id] = step

        for step in plan.steps:
            missing = sorted(set(step.depends_on) - set(by_id))
            if missing:
                raise PlanValidationError(
                    f"step {step.step_id} references unknown dependencies: {missing}"
                )
            try:
                self.registry.get(step.tool_name)
            except KeyError as exc:
                raise PlanValidationError(str(exc)) from exc

        pending = list(plan.steps)
        ordered: list[AgentStep] = []
        completed_ids: set[str] = set()
        while pending:
            ready = [
                step for step in pending
                if set(step.depends_on).issubset(completed_ids)
            ]
            if not ready:
                raise PlanValidationError("plan dependencies contain a cycle")
            for step in ready:
                ordered.append(step)
                completed_ids.add(step.step_id)
                pending.remove(step)
        return tuple(ordered)

    def execute(
        self,
        plan: AgentPlan,
        *,
        approve: ApprovalCallback,
    ) -> ExecutionReport:
        """Execute a validated plan, requiring approval before every tool call."""
        if not callable(approve):
            raise TypeError("approve must be a callable approval callback")
        ordered_steps = self._ordered_steps(plan)
        outcomes: list[StepExecution] = []

        for step in ordered_steps:
            try:
                approved = approve(step)
            except Exception as exc:
                message = f"approval callback failed: {type(exc).__name__}: {exc}"
                outcomes.append(StepExecution(
                    step.step_id, step.tool_name, "failed", message=message
                ))
                return ExecutionReport(
                    plan.plan_id, plan.goal, "failed", tuple(outcomes), error=message
                )

            if approved is not True:
                message = "tool invocation was not approved"
                outcomes.append(StepExecution(
                    step.step_id, step.tool_name, "rejected", message=message
                ))
                return ExecutionReport(
                    plan.plan_id, plan.goal, "rejected", tuple(outcomes), error=message
                )

            result = self.registry.invoke(
                step.tool_name, deepcopy(step.arguments)
            )
            verification = verify_tool_result(result, require_success=True)
            if not result.success or not verification.passed:
                message = result.error or verification.summary
                outcomes.append(StepExecution(
                    step.step_id,
                    step.tool_name,
                    "failed",
                    result=result,
                    verification=verification,
                    message=message,
                ))
                return ExecutionReport(
                    plan.plan_id, plan.goal, "failed", tuple(outcomes), error=message
                )

            outcomes.append(StepExecution(
                step.step_id,
                step.tool_name,
                "completed",
                result=result,
                verification=verification,
                message=verification.summary,
            ))

        return ExecutionReport(
            plan.plan_id, plan.goal, "completed", tuple(outcomes)
        )

    def run(
        self,
        goal: str,
        *,
        planner: PlanProvider,
        approve: ApprovalCallback,
    ) -> ExecutionReport:
        """Ask an injected planner for a plan, then execute it with approvals.

        The planner receives only the goal and the explicitly registered tool
        descriptions. It can be backed by ARIA's local model; no remote provider
        or network access is introduced by this orchestration layer.
        """
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must be a non-empty string")
        if not callable(planner):
            raise TypeError("planner must be a callable plan provider")
        plan = planner(goal, self.registry.list_tools())
        if not isinstance(plan, AgentPlan):
            raise PlanValidationError("planner must return an AgentPlan")
        return self.execute(plan, approve=approve)
