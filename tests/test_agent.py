import pytest

from aria.agent import ToolRegistry, ToolValidationError


def test_register_list_and_invoke_local_tool() -> None:
    registry = ToolRegistry()
    registry.register(
        "add",
        "Add two integers",
        lambda left, right: left + right,
        {
            "type": "object",
            "properties": {
                "left": {"type": "integer"},
                "right": {"type": "integer"},
            },
            "required": ["left", "right"],
            "additionalProperties": False,
        },
    )

    assert [tool.name for tool in registry.list_tools()] == ["add"]
    result = registry.invoke("add", {"left": 2, "right": 3})
    assert result.success
    assert result.output == 5
    assert result.error is None


def test_rejects_missing_extra_and_wrong_type_arguments() -> None:
    registry = ToolRegistry()
    registry.register(
        "greet",
        "Return a greeting",
        lambda name: f"Hello, {name}",
        {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
    )

    assert "missing required" in registry.invoke("greet", {}).error
    assert "unexpected argument" in registry.invoke("greet", {"name": "ARIA", "extra": 1}).error
    assert "must have type" in registry.invoke("greet", {"name": 123}).error


def test_unknown_tools_and_handler_errors_are_structured() -> None:
    registry = ToolRegistry()
    assert registry.invoke("missing").success is False

    def fail() -> None:
        raise RuntimeError("expected failure")

    registry.register("fail", "A test failure", fail)
    result = registry.invoke("fail")
    assert not result.success
    assert "RuntimeError" in result.error
    assert "expected failure" in result.error


def test_rejects_duplicate_names_and_invalid_schemas() -> None:
    registry = ToolRegistry()
    registry.register("noop", "Do nothing", lambda: None)
    with pytest.raises(ToolValidationError, match="already registered"):
        registry.register("noop", "Duplicate", lambda: None)
    with pytest.raises(ToolValidationError, match="schema"):
        registry.register("bad", "Bad schema", lambda: None, {"type": "array"})
