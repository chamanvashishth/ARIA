from __future__ import annotations

import pytest

from aria.agent import (
    AgentOrchestrator,
    AgentPlan,
    AgentStep,
    PlanValidationError,
    ToolRegistry,
)


def _registry(events: list[str] | None = None) -> ToolRegistry:
    registry = ToolRegistry()

    def record(label: str) -> str:
        if events is not None:
            events.append(label)
        return label

    def fail(label: str) -> str:
        raise RuntimeError(label)

    registry.register(
        "record",
        "Record a label",
        record,
        {
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
            "additionalProperties": False,
        },
    )
    registry.register(
        "fail",
        "Raise a controlled failure",
        fail,
        {
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
            "additionalProperties": False,
        },
    )
    return registry


def test_orchestrator_orders_dependencies_and_verifies_tool_results() -> None:
    events: list[str] = []
    orchestrator = AgentOrchestrator(_registry(events))
    plan = AgentPlan(
        goal="record two ordered actions",
        steps=(
            AgentStep("second", "record", {"label": "second"}, ("first",)),
            AgentStep("first", "record", {"label": "first"}),
        ),
        plan_id="plan-1",
    )
    approvals: list[str] = []

    report = orchestrator.execute(
        plan,
        approve=lambda step: approvals.append(step.step_id) or True,
    )

    assert report.succeeded
    assert report.plan_id == "plan-1"
    assert events == ["first", "second"]
    assert approvals == ["first", "second"]
    assert [step.status for step in report.steps] == ["completed", "completed"]
    assert all(step.verification is not None and step.verification.passed for step in report.steps)


def test_orchestrator_rejection_prevents_tool_execution() -> None:
    events: list[str] = []
    orchestrator = AgentOrchestrator(_registry(events))
    plan = AgentPlan(
        goal="do not run without approval",
        steps=(AgentStep("one", "record", {"label": "should-not-run"}),),
    )

    report = orchestrator.execute(plan, approve=lambda _step: False)

    assert report.status == "rejected"
    assert report.steps[0].status == "rejected"
    assert events == []


def test_orchestrator_stops_after_tool_failure() -> None:
    events: list[str] = []
    orchestrator = AgentOrchestrator(_registry(events))
    plan = AgentPlan(
        goal="stop when a tool fails",
        steps=(
            AgentStep("bad", "fail", {"label": "expected failure"}),
            AgentStep("later", "record", {"label": "must-not-run"}),
        ),
    )

    report = orchestrator.execute(plan, approve=lambda _step: True)

    assert report.status == "failed"
    assert len(report.steps) == 1
    assert report.steps[0].status == "failed"
    assert "expected failure" in (report.error or "")
    assert events == []


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        (
            (AgentStep("same", "record"), AgentStep("same", "record")),
            "duplicate step ID",
        ),
        (
            (AgentStep("one", "record", depends_on=("missing",)),),
            "unknown dependencies",
        ),
        (
            (
                AgentStep("one", "record", depends_on=("two",)),
                AgentStep("two", "record", depends_on=("one",)),
            ),
            "cycle",
        ),
        (
            (AgentStep("one", "not-registered"),),
            "unknown tool",
        ),
    ],
)
def test_orchestrator_rejects_invalid_plans_before_execution(
    steps: tuple[AgentStep, ...], message: str
) -> None:
    orchestrator = AgentOrchestrator(_registry())
    with pytest.raises(PlanValidationError, match=message):
        orchestrator.execute(
            AgentPlan(goal="invalid plan", steps=steps),
            approve=lambda _step: True,
        )


def test_orchestrator_requires_approval_callback_and_non_empty_plan() -> None:
    orchestrator = AgentOrchestrator(_registry())
    with pytest.raises(TypeError, match="approve must be a callable"):
        orchestrator.execute(
            AgentPlan(goal="missing approval", steps=(AgentStep("one", "record"),)),
            approve=None,  # type: ignore[arg-type]
        )
    with pytest.raises(PlanValidationError, match="at least one step"):
        orchestrator.execute(
            AgentPlan(goal="empty", steps=()),
            approve=lambda _step: True,
        )


def test_run_uses_injected_planner_and_only_registered_tools() -> None:
    orchestrator = AgentOrchestrator(_registry())
    seen: list[str] = []

    def planner(goal, available_tools):
        seen.extend(tool.name for tool in available_tools)
        return AgentPlan(
            goal=goal,
            steps=(AgentStep("one", "record", {"label": "planned"}),),
            plan_id="local-planner",
        )

    report = orchestrator.run(
        "complete a local task",
        planner=planner,
        approve=lambda _step: True,
    )

    assert report.succeeded
    assert report.plan_id == "local-planner"
    assert seen == ["fail", "record"]
