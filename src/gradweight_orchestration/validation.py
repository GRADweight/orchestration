"""Strikte Validierung von JSON-Werten gegen das von :mod:`.schema` erzeugte Schema.

Bewusst kein vollständiger JSON-Schema-Validator, sondern genau die
Schlüsselwörter, die :func:`~gradweight_orchestration.schema.json_schema`
erzeugt (``type``, ``enum``, ``const``, ``properties``, ``required``,
``additionalProperties``, ``items``, ``anyOf``, ``minimum``, ``maximum``,
``minLength``, ``maxLength``, ``pattern``, ``minItems``, ``maxItems``,
``format`` für ``date``/``date-time``). Keine stillen Typumwandlungen: ``"12.5"``
ist kein ``number``. Die Befunde gehen als strukturierte Beobachtung an das
Modell zurück, damit es den Aufruf korrigieren kann.

:func:`from_jsonable` wandelt validierte JSON-Werte in die annotierten
Python-Typen um (Dataclass-Instanzen, ``Enum``, ``Decimal``, ``date`` ...).
"""

from __future__ import annotations

import collections.abc
import dataclasses
import datetime as _dt
import enum
import functools
import math
import re
import types
import typing
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from ._json import JSONValue
from .errors import ValidationIssue
from .schema import json_schema

__all__ = ["from_jsonable", "validate"]

_NoneType = type(None)
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$")


def _is_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _is_integer(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value) and value.is_integer()


def _never(_value: object) -> bool:
    return False


_TYPE_CHECKS: dict[str, typing.Callable[[object], bool]] = {
    "null": lambda v: v is None,
    "boolean": lambda v: isinstance(v, bool),
    "integer": _is_integer,
    "number": _is_number,
    "string": lambda v: isinstance(v, str),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


def _json_type(value: object) -> str:
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


def _preview(value: object, limit: int = 40) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _same(a: object, b: object) -> bool:
    """Gleichheit ohne Pythons ``True == 1``-Falle."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    return a == b


@functools.lru_cache(maxsize=256)
def _pattern(expr: str) -> re.Pattern[str]:
    return re.compile(expr)


def validate(instance: object, schema: Mapping[str, Any], path: str = "$") -> list[ValidationIssue]:
    """Prüft ``instance`` gegen ``schema``; liefert alle Befunde (leer = gültig)."""
    issues: list[ValidationIssue] = []
    _validate(instance, schema, path, issues)
    return issues


def _validate(
    value: object, schema: Mapping[str, Any], path: str, issues: list[ValidationIssue]
) -> None:
    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        if not any(not validate(value, sub, path) for sub in any_of if isinstance(sub, Mapping)):
            kinds = sorted(
                {str(sub.get("type", "?")) for sub in any_of if isinstance(sub, Mapping)}
            )
            issues.append(
                ValidationIssue(
                    path,
                    f"passt zu keiner erlaubten Variante ({', '.join(kinds)}); "
                    f"erhalten {_json_type(value)} {_preview(value)}",
                )
            )
        return

    declared = schema.get("type")
    if declared is not None:
        kinds = [declared] if isinstance(declared, str) else list(declared)
        if not any(_TYPE_CHECKS.get(kind, _never)(value) for kind in kinds):
            issues.append(
                ValidationIssue(
                    path,
                    f"erwartet {' oder '.join(kinds)}, erhalten {_json_type(value)} "
                    f"{_preview(value)}",
                )
            )
            return

    if "const" in schema and not _same(value, schema["const"]):
        issues.append(ValidationIssue(path, f"erwartet genau {_preview(schema['const'])}"))
        return
    allowed = schema.get("enum")
    if isinstance(allowed, list) and not any(_same(value, option) for option in allowed):
        options = ", ".join(_preview(o) for o in allowed)
        issues.append(
            ValidationIssue(path, f"Wert {_preview(value)} nicht erlaubt; erlaubt: {options}")
        )
        return

    if isinstance(value, str):
        _validate_string(value, schema, path, issues)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        _validate_number(value, schema, path, issues)
    elif isinstance(value, list):
        _validate_array(value, schema, path, issues)
    elif isinstance(value, dict):
        _validate_object(value, schema, path, issues)


def _validate_string(
    value: str, schema: Mapping[str, Any], path: str, issues: list[ValidationIssue]
) -> None:
    min_len, max_len = schema.get("minLength"), schema.get("maxLength")
    if isinstance(min_len, int) and len(value) < min_len:
        issues.append(ValidationIssue(path, f"mindestens {min_len} Zeichen verlangt"))
    if isinstance(max_len, int) and len(value) > max_len:
        issues.append(ValidationIssue(path, f"höchstens {max_len} Zeichen erlaubt"))
    pattern = schema.get("pattern")
    if isinstance(pattern, str) and not _pattern(pattern).search(value):
        issues.append(ValidationIssue(path, f"entspricht nicht dem Muster {pattern!r}"))
    fmt = schema.get("format")
    if fmt == "date" and not _valid_date(value):
        issues.append(ValidationIssue(path, "erwartet Datum im Format JJJJ-MM-TT"))
    elif fmt == "date-time" and not _valid_datetime(value):
        issues.append(ValidationIssue(path, "erwartet Zeitstempel nach ISO 8601"))


def _validate_number(
    value: float, schema: Mapping[str, Any], path: str, issues: list[ValidationIssue]
) -> None:
    minimum, maximum = schema.get("minimum"), schema.get("maximum")
    if isinstance(minimum, (int, float)) and value < minimum:
        issues.append(ValidationIssue(path, f"muss ≥ {minimum} sein"))
    if isinstance(maximum, (int, float)) and value > maximum:
        issues.append(ValidationIssue(path, f"muss ≤ {maximum} sein"))


def _validate_array(
    value: list[Any], schema: Mapping[str, Any], path: str, issues: list[ValidationIssue]
) -> None:
    min_items, max_items = schema.get("minItems"), schema.get("maxItems")
    if isinstance(min_items, int) and len(value) < min_items:
        issues.append(ValidationIssue(path, f"mindestens {min_items} Elemente verlangt"))
    if isinstance(max_items, int) and len(value) > max_items:
        issues.append(ValidationIssue(path, f"höchstens {max_items} Elemente erlaubt"))
    items = schema.get("items")
    if isinstance(items, Mapping):
        for index, item in enumerate(value):
            _validate(item, items, f"{path}[{index}]", issues)


def _validate_object(
    value: dict[Any, Any], schema: Mapping[str, Any], path: str, issues: list[ValidationIssue]
) -> None:
    properties = schema.get("properties")
    props: Mapping[str, Any] = properties if isinstance(properties, Mapping) else {}
    required = schema.get("required")
    if isinstance(required, list):
        issues.extend(
            ValidationIssue(f"{path}.{name}", "Pflichtfeld fehlt")
            for name in required
            if name not in value
        )
    additional = schema.get("additionalProperties", True)
    for key, item in value.items():
        child = f"{path}.{key}"
        if not isinstance(key, str):
            issues.append(ValidationIssue(path, f"Schlüssel {_preview(key)} ist kein String"))
        elif key in props and isinstance(props[key], Mapping):
            _validate(item, props[key], child, issues)
        elif additional is False:
            known = ", ".join(sorted(props)) or "keine"
            issues.append(ValidationIssue(child, f"unbekanntes Feld; erlaubt: {known}"))
        elif isinstance(additional, Mapping):
            _validate(item, additional, child, issues)


def _valid_date(value: str) -> bool:
    if not _DATE.match(value):
        return False
    try:
        _dt.date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _parse_datetime(value: str) -> _dt.datetime:
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    return _dt.datetime.fromisoformat(text)


def _valid_datetime(value: str) -> bool:
    if not _DATETIME.match(value):
        return False
    try:
        _parse_datetime(value)
    except ValueError:
        return False
    return True


# --------------------------------------------------------------------------- #
# JSON -> Python                                                              #
# --------------------------------------------------------------------------- #


def from_jsonable(tp: Any, value: JSONValue) -> Any:
    """Wandelt einen (bereits validierten) JSON-Wert in den annotierten Typ um.

    Löst ``ValueError``/``TypeError`` aus, wenn der Wert nicht passt – Aufrufer
    validieren vorher mit :func:`validate`, damit das Modell lesbare Befunde erhält.
    """
    origin = get_origin(tp)
    if origin is Annotated:
        return from_jsonable(get_args(tp)[0], value)
    if tp is Any:
        return value
    if tp is None or tp is _NoneType:
        if value is not None:
            raise TypeError("erwartet null")
        return None
    if tp is bool:
        if not isinstance(value, bool):
            raise TypeError("erwartet boolean")
        return value
    if tp is int:
        if not _is_integer(value):
            raise TypeError("erwartet integer")
        return int(typing.cast(float, value))
    if tp is float:
        if not _is_number(value):
            raise TypeError("erwartet number")
        return float(typing.cast(float, value))
    if tp is Decimal:
        if not _is_number(value):
            raise TypeError("erwartet number")
        try:
            return Decimal(str(value))
        except InvalidOperation as exc:  # pragma: no cover - durch _is_number ausgeschlossen
            raise TypeError(str(exc)) from exc
    if tp is str:
        if not isinstance(value, str):
            raise TypeError("erwartet string")
        return value
    if tp is _dt.datetime:
        return _parse_datetime(_expect_str(value))
    if tp is _dt.date:
        return _dt.date.fromisoformat(_expect_str(value))
    if origin is Literal:
        if not any(_same(value, option) for option in get_args(tp)):
            raise ValueError(f"{value!r} nicht in {get_args(tp)!r}")
        return value
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return tp(value)
    if origin is Union or origin is types.UnionType:
        members = get_args(tp)
        if value is None and _NoneType in members:
            return None
        for member in members:
            if member is _NoneType:
                continue
            if not validate(value, json_schema(member)):
                return from_jsonable(member, value)
        raise TypeError(f"{value!r} passt zu keinem Typ in {tp!r}")
    if tp in (list, tuple) or origin in (list, tuple, collections.abc.Sequence):
        if not isinstance(value, list):
            raise TypeError("erwartet array")
        args = get_args(tp)
        item_type = args[0] if args else Any
        items = [from_jsonable(item_type, item) for item in value]
        return tuple(items) if (tp is tuple or origin is tuple) else items
    if tp is dict or origin in (dict, collections.abc.Mapping):
        if not isinstance(value, dict):
            raise TypeError("erwartet object")
        args = get_args(tp)
        value_type = args[1] if args else Any
        return {key: from_jsonable(value_type, item) for key, item in value.items()}
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise TypeError("erwartet object")
        hints = typing.get_type_hints(tp, include_extras=True)
        kwargs = {
            f.name: from_jsonable(hints[f.name], value[f.name])
            for f in dataclasses.fields(tp)
            if f.init and f.name in value
        }
        return tp(**kwargs)
    if isinstance(tp, type) and typing.is_typeddict(tp):
        if not isinstance(value, dict):
            raise TypeError("erwartet object")
        hints = typing.get_type_hints(tp, include_extras=True)
        return {key: from_jsonable(hints.get(key, Any), item) for key, item in value.items()}
    raise TypeError(f"nicht unterstützter Typ {tp!r}")


def _expect_str(value: JSONValue) -> str:
    if not isinstance(value, str):
        raise TypeError("erwartet string")
    return value
