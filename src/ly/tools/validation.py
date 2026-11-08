"""Minimal JSON-Schema validation for tool arguments.

Only the subset of JSON Schema that ly's tool definitions actually use is
implemented: object schemas with ``properties`` and ``required``, primitive
types, and arrays with ``items``. There is no dependency on ``jsonschema``
and no support for ``$ref``, ``oneOf``, ``pattern``, ``enum``, and friends.
Add constructs here as tool schemas grow.

One deliberate leniency: unknown properties are allowed. Providers sometimes
attach extra keys, and every built-in tool already tolerates them through
``**kwargs``; rejecting them would turn a harmless extra field into a failed
tool call. Wrong types and missing required fields are the cases worth failing
on, because they produce silent nonsense downstream (e.g. iterating the
characters of a string that was supposed to be a list).
"""

from typing import Any

_ARTICLES = {"array": "an", "integer": "an", "object": "an"}

_PRIMITIVES = {
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _check(schema: dict, value: Any, path: str, errors: list[str]) -> None:
    expected = schema.get("type")
    if isinstance(expected, str):
        check = _PRIMITIVES.get(expected)
        if check is None:
            return  # unknown type keyword: nothing sensible to check
        if not check(value):
            article = _ARTICLES.get(expected, "a")
            errors.append(
                f"{path} must be {article} {expected} (got {_type_name(value)})"
            )
            return  # type is wrong; nested checks would only add noise

    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key} is required")
        for key, subschema in (schema.get("properties") or {}).items():
            if key in value and isinstance(subschema, dict):
                _check(subschema, value[key], f"{path}.{key}", errors)

    elif isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                _check(items, item, f"{path}[{index}]", errors)


def validate_arguments(schema: dict | None, arguments: Any) -> list[str]:
    """Return a list of human-readable violations; empty means valid."""
    errors: list[str] = []
    _check(schema or {}, arguments, "arguments", errors)
    return errors
