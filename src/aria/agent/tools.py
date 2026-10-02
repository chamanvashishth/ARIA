"""Explicit registry for validated, locally executed ARIA tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
ToolHandler = Callable[..., Any]


@dataclass(frozen=True)
class ToolDefinition:
    """A trusted local callable plus a small JSON-Schema-like input contract."""

    name: str
    description: str
    handler: ToolHandler
    input_schema: Mapping[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """Structured outcome of a tool invocation."""

    tool_name: str
    success: bool
    output: Any = None
    error: str | None = None


class ToolValidationError(ValueError):
    """Raised when a tool definition or its input arguments are invalid."""


def _check_type(value: Any, expected: str) -> bool:
    checks = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "null": lambda v: v is None,
    }
    if expected not in checks:
        raise ToolValidationError(f"unsupported schema type: {expected}")
    return checks[expected](value)


class ToolRegistry:
    """Register and invoke explicitly trusted local functions.

    This registry never imports tools by name, evaluates code, or executes
    shell commands on its own. Callers must register each callable explicitly.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        handler: ToolHandler,
        input_schema: Mapping[str, Any] | None = None,
    ) -> ToolDefinition:
        if not isinstance(name, str) or not name.strip():
            raise ToolValidationError("tool name must be a non-empty string")
        if name in self._tools:
            raise ToolValidationError(f"tool already registered: {name}")
        if not callable(handler):
            raise ToolValidationError("tool handler must be callable")
        schema = dict(input_schema or {"type": "object", "properties": {}})
        if schema.get("type") != "object":
            raise ToolValidationError("tool input schema must have type 'object'")
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            raise ToolValidationError("schema properties must be an object and required must be a list")
        for key in required:
            if not isinstance(key, str) or key not in properties:
                raise ToolValidationError(f"required field has no property definition: {key!r}")
        for key, rule in properties.items():
            if not isinstance(key, str) or not isinstance(rule, dict) or "type" not in rule:
                raise ToolValidationError("each property must have a name and a type")
            if rule["type"] not in {"object", "array", "string", "integer", "number", "boolean", "null"}:
                raise ToolValidationError(f"unsupported schema type: {rule['type']}")
        definition = ToolDefinition(name, description, handler, schema)
        self._tools[name] = definition
        return definition

    def list_tools(self) -> list[ToolDefinition]:
        return [self._tools[name] for name in sorted(self._tools)]

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def validate_arguments(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        definition = self.get(name)
        if not isinstance(arguments, Mapping):
            raise ToolValidationError("tool arguments must be an object")
        schema = definition.input_schema
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        args = dict(arguments)
        missing = [key for key in required if key not in args]
        if missing:
            raise ToolValidationError(f"missing required argument(s): {', '.join(missing)}")
        if schema.get("additionalProperties", False) is False:
            extras = sorted(set(args) - set(properties))
            if extras:
                raise ToolValidationError(f"unexpected argument(s): {', '.join(extras)}")
        for key, value in args.items():
            if key not in properties:
                continue
            rule = properties[key]
            if not _check_type(value, rule["type"]):
                raise ToolValidationError(
                    f"argument {key!r} must have type {rule['type']!r}"
                )
            if "enum" in rule and value not in rule["enum"]:
                raise ToolValidationError(f"argument {key!r} must be one of {rule['enum']!r}")
        return args

    def invoke(self, name: str, arguments: Mapping[str, Any] | None = None) -> ToolResult:
        if name not in self._tools:
            return ToolResult(tool_name=name, success=False, error=f"unknown tool: {name}")
        try:
            args = self.validate_arguments(name, arguments or {})
        except (ToolValidationError, TypeError) as exc:
            return ToolResult(tool_name=name, success=False, error=str(exc))
        try:
            output = self._tools[name].handler(**args)
        except Exception as exc:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"tool execution failed: {type(exc).__name__}: {exc}",
            )
        return ToolResult(tool_name=name, success=True, output=output)
