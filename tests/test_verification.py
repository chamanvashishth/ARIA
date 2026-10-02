import pytest

from aria.agent import ToolResult
from aria.verification import (
    CheckResult,
    VerificationSuite,
    check_non_empty_string,
    check_required_keys,
    check_type,
    verify_tool_result,
)


def test_basic_checks_pass_and_fail() -> None:
    assert check_type(4, int).passed
    assert not check_type("4", int).passed
    assert check_required_keys({"a": 1, "b": 2}, ["a", "b"]).passed
    missing = check_required_keys({"a": 1}, ["a", "b"])
    assert not missing.passed
    assert missing.details["missing"] == ["b"]
    assert check_non_empty_string("hello").passed
    assert not check_non_empty_string("  ").passed


def test_verify_successful_tool_result_and_output_type() -> None:
    result = ToolResult(tool_name="sum", success=True, output=5)
    report = verify_tool_result(result, expected_output_type=int)
    assert report.passed
    assert report.failed_checks == ()
    assert report.summary == "5/5 checks passed; 0 failed"


def test_verify_output_type_mismatch() -> None:
    result = ToolResult(tool_name="sum", success=True, output="five")
    report = verify_tool_result(result, expected_output_type=int)
    assert not report.passed
    assert any(check.name == "output_type" for check in report.failed_checks)


def test_failed_tool_result_requires_error_and_can_require_success() -> None:
    result = ToolResult(tool_name="sum", success=False, error="bad input")
    report = verify_tool_result(result)
    assert not report.passed
    assert any(check.name == "required_success" for check in report.failed_checks)
    assert not any(check.name == "failure_error_consistency" for check in report.failed_checks)


def test_malformed_result_is_reported() -> None:
    report = verify_tool_result({"success": True})
    assert not report.passed
    assert report.checks[0].name == "tool_result_shape"


def test_suite_runs_checks_and_catches_exceptions() -> None:
    suite = VerificationSuite()
    suite.register("positive", lambda value: value > 0)
    suite.register("named", lambda value: CheckResult("named", value == 2, "value equals two"))
    suite.register("broken", lambda value: (_ for _ in ()).throw(RuntimeError("boom")))

    report = suite.run(2)
    assert not report.passed
    assert [check.passed for check in report.checks] == [True, True, False]
    assert "RuntimeError" in report.checks[2].message


def test_suite_rejects_duplicate_names_and_non_callables() -> None:
    suite = VerificationSuite()
    suite.register("positive", lambda value: value > 0)
    with pytest.raises(ValueError, match="already registered"):
        suite.register("positive", lambda value: True)
    with pytest.raises(TypeError, match="callable"):
        suite.register("not-callable", 42)  # type: ignore[arg-type]
