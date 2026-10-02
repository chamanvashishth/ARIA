"""Deterministic structural checks for ARIA outputs.

These checks validate contracts and shape, not factual correctness or safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one verification check."""

    name: str
    passed: bool
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationReport:
    """Collection of check outcomes with aggregate status."""

    checks: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failed_checks(self) -> tuple[CheckResult, ...]:
        return tuple(check for check in self.checks if not check.passed)

    @property
    def summary(self) -> str:
        total = len(self.checks)
        failed = len(self.failed_checks)
        return f"{total - failed}/{total} checks passed; {failed} failed"


def check_type(value: Any, expected_type: type | tuple[type, ...], *, name: str = "type") -> CheckResult:
    """Check whether a value matches a Python type or tuple of types."""
    passed = isinstance(value, expected_type)
    expected_name = (
        ", ".join(item.__name__ for item in expected_type)
        if isinstance(expected_type, tuple)
        else expected_type.__name__
    )
    return CheckResult(
        name=name,
        passed=passed,
        message=f"value matches {expected_name}" if passed else f"expected {expected_name}, got {type(value).__name__}",
        details={"expected_type": expected_name, "actual_type": type(value).__name__},
    )


def check_required_keys(value: Any, required_keys: tuple[str, ...] | list[str], *, name: str = "required_keys") -> CheckResult:
    """Check that a mapping contains every required key."""
    if not isinstance(value, Mapping):
        return CheckResult(name, False, "value is not a mapping", {"missing": list(required_keys)})
    missing = [key for key in required_keys if key not in value]
    return CheckResult(
        name=name,
        passed=not missing,
        message="all required keys are present" if not missing else f"missing required keys: {', '.join(missing)}",
        details={"missing": missing},
    )


def check_non_empty_string(value: Any, *, name: str = "non_empty_string") -> CheckResult:
    """Check that a value is a string containing non-whitespace characters."""
    passed = isinstance(value, str) and bool(value.strip())
    return CheckResult(
        name=name,
        passed=passed,
        message="string is non-empty" if passed else "expected a non-empty string",
        details={"actual_type": type(value).__name__},
    )


def verify_tool_result(
    result: Any,
    *,
    expected_output_type: type | tuple[type, ...] | None = None,
    require_success: bool = True,
) -> VerificationReport:
    """Validate the structure and optional output type of a ToolResult-like object.

    A passing report does not establish that the tool output is true, complete,
    safe, or appropriate for a particular task.
    """
    checks: list[CheckResult] = []
    required = ("tool_name", "success", "output", "error")
    if any(not hasattr(result, key) for key in required):
        checks.append(CheckResult(
            name="tool_result_shape",
            passed=False,
            message="result is missing one or more required attributes",
            details={"required_attributes": list(required)},
        ))
        return VerificationReport(tuple(checks))

    checks.append(check_non_empty_string(result.tool_name, name="tool_name"))
    checks.append(check_type(result.success, bool, name="success_flag"))

    if result.success:
        checks.append(CheckResult(
            name="success_error_consistency",
            passed=result.error is None,
            message="successful result has no error" if result.error is None else "successful result unexpectedly contains an error",
        ))
    else:
        checks.append(CheckResult(
            name="failure_error_consistency",
            passed=isinstance(result.error, str) and bool(result.error.strip()),
            message="failed result includes an error message" if isinstance(result.error, str) and bool(result.error.strip()) else "failed result must include a non-empty error message",
        ))

    if require_success:
        checks.append(CheckResult(
            name="required_success",
            passed=result.success is True,
            message="tool succeeded" if result.success is True else "tool did not succeed",
        ))

    if expected_output_type is not None and result.success:
        checks.append(check_type(result.output, expected_output_type, name="output_type"))

    return VerificationReport(tuple(checks))


class VerificationSuite:
    """Run named, deterministic checks against a single value."""

    def __init__(self) -> None:
        self._checks: list[tuple[str, Callable[[Any], bool | CheckResult]]] = []

    def register(self, name: str, check: Callable[[Any], bool | CheckResult]) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("check name must be a non-empty string")
        if not callable(check):
            raise TypeError("check must be callable")
        if any(existing_name == name for existing_name, _ in self._checks):
            raise ValueError(f"check already registered: {name}")
        self._checks.append((name, check))

    def run(self, value: Any) -> VerificationReport:
        results: list[CheckResult] = []
        for name, check in self._checks:
            try:
                outcome = check(value)
                if isinstance(outcome, CheckResult):
                    if outcome.name != name:
                        outcome = CheckResult(name, outcome.passed, outcome.message, outcome.details)
                    results.append(outcome)
                else:
                    results.append(CheckResult(
                        name=name,
                        passed=bool(outcome),
                        message="check passed" if bool(outcome) else "check failed",
                    ))
            except Exception as exc:
                results.append(CheckResult(
                    name=name,
                    passed=False,
                    message=f"check raised {type(exc).__name__}: {exc}",
                ))
        return VerificationReport(tuple(results))
