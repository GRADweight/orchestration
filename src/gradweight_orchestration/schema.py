"""JSON-Schema aus Python-Typannotationen und Docstrings.

Unterstützt werden genau die Typen, die sich verlustfrei als JSON darstellen und
zu einfachen, flachen Schemas führen (keine ``$ref``, keine Rekursion):

``str``, ``int``, ``float``, ``bool``, ``Decimal`` (als ``number``),
``datetime.date``/``datetime.datetime`` (ISO 8601), ``Literal[...]``, ``Enum``
(String- oder Ganzzahlwerte), ``list[T]``/``Sequence[T]``/``tuple[T, ...]``,
``dict[str, T]``, ``Optional[T]``/``T | None``, ``Union[...]``, Dataclasses,
``TypedDict`` und ``Any``. Mit ``Annotated[T, "Beschreibung"]`` oder
``Annotated[T, Param(...)]`` lassen sich Beschreibungen und Grenzen setzen.

Alles andere führt bei der Deklaration zu :class:`ToolDefinitionError` – ein
Werkzeug mit unklarem Schema soll gar nicht erst registriert werden.
"""

from __future__ import annotations

import collections.abc
import contextlib
import dataclasses
import datetime as _dt
import enum
import functools
import inspect
import re
import types
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from ._json import JSONObject, JSONValue, NotJSONableError, to_jsonable
from .errors import ToolDefinitionError

__all__ = [
    "DocInfo",
    "FunctionSchema",
    "Param",
    "function_schema",
    "json_schema",
    "parse_docstring",
]

_NoneType = type(None)


@dataclass(frozen=True)
class Param:
    """Zusatzangaben für einen Parameter, verwendet über ``Annotated[T, Param(...)]``.

    ``ge``/``le`` gelten für Zahlen, ``min_length``/``max_length`` für Strings
    (Zeichen) und Listen (Elemente), ``pattern`` für Strings (``re.search``).
    """

    description: str | None = None
    ge: float | None = None
    le: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None


# --------------------------------------------------------------------------- #
# Docstrings                                                                  #
# --------------------------------------------------------------------------- #

_PARAM_SECTIONS = {
    "args",
    "arguments",
    "parameters",
    "params",
    "argumente",
    "parameter",
    "attributes",
    "attribute",
    "felder",
}
_RETURN_SECTIONS = {"returns", "return", "rückgabe", "liefert"}
_OTHER_SECTIONS = {
    "raises",
    "examples",
    "example",
    "note",
    "notes",
    "yields",
    "see also",
    "warnings",
    "beispiel",
    "beispiele",
    "fehler",
    "hinweis",
    "hinweise",
}
_ALL_SECTIONS = _PARAM_SECTIONS | _RETURN_SECTIONS | _OTHER_SECTIONS
_HEADER = re.compile(r"^\s*([A-Za-zÄÖÜäöü][A-Za-zÄÖÜäöü ]*?)\s*:?\s*$")
_GOOGLE_ENTRY = re.compile(r"^(\*{0,2}[A-Za-z_]\w*)\s*(?:\([^)]*\))?\s*:\s*(.*)$")
_NUMPY_ENTRY = re.compile(r"^(\*{0,2}[A-Za-z_]\w*)\s*(?::\s*.*)?$")
_REST_PARAM = re.compile(r"^\s*:param\s+(?:[^:]*\s)?(\w+)\s*:\s*(.*)$")
_REST_RETURN = re.compile(r"^\s*:returns?\s*:\s*(.*)$")


@dataclass(frozen=True)
class DocInfo:
    """Zerlegter Docstring: Beschreibungstext, Parameterbeschreibungen, Rückgabe."""

    summary: str
    params: Mapping[str, str]
    returns: str | None


def _section_name(lines: list[str], i: int) -> str | None:
    """Erkennt Abschnittsköpfe im Google- (``Args:``) und NumPy-Stil (Unterstrich)."""
    line = lines[i]
    match = _HEADER.match(line)
    if not match:
        return None
    name = match.group(1).strip().lower()
    if name not in _ALL_SECTIONS:
        return None
    if line.rstrip().endswith(":"):
        return name
    nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
    if nxt and set(nxt) == {"-"}:
        return name
    return None


def parse_docstring(doc: str | None) -> DocInfo:
    """Liest Google-, NumPy- und reST-Docstrings (auch mit deutschen Abschnittsnamen)."""
    if not doc:
        return DocInfo(summary="", params={}, returns=None)
    lines = inspect.cleandoc(doc).splitlines()
    params: dict[str, str] = {}
    returns: str | None = None
    body: list[str] = []
    section: str | None = None
    current: str | None = None
    base_indent: int | None = None
    numpy_style = False
    i = 0
    while i < len(lines):
        line = lines[i]
        rest_param = _REST_PARAM.match(line)
        rest_return = _REST_RETURN.match(line)
        if rest_param:
            params[rest_param.group(1)] = rest_param.group(2).strip()
            current, section = rest_param.group(1), "rest"
            i += 1
            continue
        if rest_return:
            returns = rest_return.group(1).strip()
            current, section = None, "rest-return"
            i += 1
            continue
        name = _section_name(lines, i)
        if name is not None:
            section, current, base_indent = name, None, None
            numpy_style = not line.rstrip().endswith(":")
            if numpy_style:
                i += 1  # NumPy-Unterstreichung überspringen
            i += 1
            continue
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if section is None:
            body.append(line)
        elif section in _PARAM_SECTIONS:
            if not stripped:
                current = None
            elif base_indent is None or indent <= base_indent:
                base_indent = indent if base_indent is None else base_indent
                entry = (_NUMPY_ENTRY if numpy_style else _GOOGLE_ENTRY).match(stripped)
                if entry:
                    current = entry.group(1).lstrip("*")
                    params[current] = "" if numpy_style else entry.group(2).strip()
                else:
                    current = None
            elif current is not None:
                params[current] = f"{params[current]} {stripped}".strip()
        elif section in _RETURN_SECTIONS:
            if stripped:
                returns = f"{returns} {stripped}".strip() if returns else stripped
        elif section == "rest" and current is not None and stripped:
            params[current] = f"{params[current]} {stripped}".strip()
        i += 1
    paragraphs = "\n".join(body).strip().split("\n\n")
    summary = "\n".join(" ".join(p.split()) for p in paragraphs if p.strip())
    return DocInfo(summary=summary, params=params, returns=returns)


# --------------------------------------------------------------------------- #
# Typen -> Schema                                                             #
# --------------------------------------------------------------------------- #

_SEQUENCE_ORIGINS = (list, collections.abc.Sequence, tuple)
_MAPPING_ORIGINS = (dict, collections.abc.Mapping)


def _is_union(tp: Any) -> bool:
    origin = get_origin(tp)
    return origin is Union or origin is types.UnionType


def _nullable(schema: JSONObject) -> JSONObject:
    """Macht ein Schema zusätzlich ``null``-fähig (``T | None``)."""
    out = dict(schema)
    kind = out.get("type")
    if isinstance(kind, str) and "anyOf" not in out:
        out["type"] = [kind, "null"]
        enum_values = out.get("enum")
        if isinstance(enum_values, list) and None not in enum_values:
            out["enum"] = [*enum_values, None]
        return out
    description = out.pop("description", None)
    wrapped: JSONObject = {"anyOf": [out, {"type": "null"}]}
    if description is not None:
        wrapped["description"] = description
    return wrapped


def _apply_param(schema: JSONObject, meta: Param, where: str) -> None:
    kind = schema.get("type")
    kinds = set(kind) if isinstance(kind, list) else {kind}
    numeric = bool(kinds & {"number", "integer"})
    sized = bool(kinds & {"string", "array"})
    if (meta.ge is not None or meta.le is not None) and not numeric:
        raise ToolDefinitionError(f"{where}: ge/le nur für Zahlen erlaubt")
    if (meta.min_length is not None or meta.max_length is not None) and not sized:
        raise ToolDefinitionError(f"{where}: min_length/max_length nur für Strings/Listen")
    if meta.pattern is not None and "string" not in kinds:
        raise ToolDefinitionError(f"{where}: pattern nur für Strings erlaubt")
    if meta.description:
        schema["description"] = meta.description
    if meta.ge is not None:
        schema["minimum"] = meta.ge
    if meta.le is not None:
        schema["maximum"] = meta.le
    if meta.pattern is not None:
        try:
            re.compile(meta.pattern)
        except re.error as exc:
            raise ToolDefinitionError(f"{where}: ungültiges pattern: {exc}") from exc
        schema["pattern"] = meta.pattern
    if "array" in kinds:
        if meta.min_length is not None:
            schema["minItems"] = meta.min_length
        if meta.max_length is not None:
            schema["maxItems"] = meta.max_length
    else:
        if meta.min_length is not None:
            schema["minLength"] = meta.min_length
        if meta.max_length is not None:
            schema["maxLength"] = meta.max_length


def _object_schema(
    hints: Mapping[str, Any],
    required: list[str],
    descriptions: Mapping[str, str],
    *,
    where: str,
    seen: frozenset[Any],
    description: str = "",
) -> JSONObject:
    properties: JSONObject = {}
    for name, hint in hints.items():
        prop = _schema(hint, where=f"{where}.{name}", seen=seen)
        if "description" not in prop and descriptions.get(name):
            prop["description"] = descriptions[name]
        properties[name] = prop
    schema: JSONObject = {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }
    if description:
        schema["description"] = description
    return schema


def _class_doc(cls: type) -> DocInfo:
    doc = cls.__doc__ or ""
    # Dataclasses erzeugen ohne eigenen Docstring "Name(feld: typ, ...)" – ignorieren.
    if doc.startswith(f"{cls.__name__}("):
        doc = ""
    return parse_docstring(doc)


def _type_hints(obj: Any, where: str) -> dict[str, Any]:
    try:
        return typing.get_type_hints(obj, include_extras=True)
    except (NameError, TypeError) as exc:
        raise ToolDefinitionError(
            f"{where}: Typannotationen nicht auflösbar ({exc}); Typen müssen auf "
            "Modulebene definiert sein"
        ) from exc


def _schema(tp: Any, *, where: str, seen: frozenset[Any]) -> JSONObject:
    origin = get_origin(tp)
    if origin is Annotated:
        base, *meta = get_args(tp)
        schema = _schema(base, where=where, seen=seen)
        for item in meta:
            if isinstance(item, Param):
                _apply_param(schema, item, where)
            elif isinstance(item, str):
                schema["description"] = item
        return schema
    if tp is Any:
        return {}
    if tp is None or tp is _NoneType:
        return {"type": "null"}
    if tp is bool:
        return {"type": "boolean"}
    if tp is int:
        return {"type": "integer"}
    if tp is float or tp is Decimal:
        return {"type": "number"}
    if tp is str:
        return {"type": "string"}
    if tp is _dt.datetime:
        return {"type": "string", "format": "date-time"}
    if tp is _dt.date:
        return {"type": "string", "format": "date"}
    if origin is Literal:
        values = list(get_args(tp))
        return _enum_schema(values, where)
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return _enum_schema([member.value for member in tp], where)
    if _is_union(tp):
        members = get_args(tp)
        non_none = [m for m in members if m is not _NoneType]
        if len(non_none) == 1:
            return _nullable(_schema(non_none[0], where=where, seen=seen))
        variants: list[JSONValue] = [_schema(m, where=where, seen=seen) for m in members]
        return {"anyOf": variants}
    if tp in (list, tuple) or origin in _SEQUENCE_ORIGINS:
        args = get_args(tp)
        if origin is tuple and not (len(args) == 2 and args[1] is Ellipsis):
            raise ToolDefinitionError(f"{where}: nur tuple[T, ...] wird unterstützt")
        if not args:
            return {"type": "array"}
        return {"type": "array", "items": _schema(args[0], where=f"{where}[]", seen=seen)}
    if tp is dict or origin in _MAPPING_ORIGINS:
        args = get_args(tp)
        if not args:
            return {"type": "object"}
        if args[0] is not str:
            raise ToolDefinitionError(f"{where}: Mapping-Schlüssel müssen str sein")
        return {
            "type": "object",
            "additionalProperties": _schema(args[1], where=f"{where}{{}}", seen=seen),
        }
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return _dataclass_schema(tp, where=where, seen=seen)
    if isinstance(tp, type) and typing.is_typeddict(tp):
        if tp in seen:
            raise ToolDefinitionError(f"{where}: rekursive Typen werden nicht unterstützt")
        hints = _type_hints(tp, where)
        doc = _class_doc(tp)
        required = [k for k in hints if k in getattr(tp, "__required_keys__", frozenset())]
        return _object_schema(
            hints, required, doc.params, where=where, seen=seen | {tp}, description=doc.summary
        )
    raise ToolDefinitionError(f"{where}: nicht unterstützter Typ {tp!r}")


def _dataclass_schema(tp: type, *, where: str, seen: frozenset[Any]) -> JSONObject:
    if tp in seen:
        raise ToolDefinitionError(f"{where}: rekursive Typen werden nicht unterstützt")
    hints = _type_hints(tp, where)
    doc = _class_doc(tp)
    fields = [f for f in dataclasses.fields(tp) if f.init]
    descriptions = dict(doc.params)
    for f in fields:
        meta_description = f.metadata.get("description")
        if isinstance(meta_description, str):
            descriptions[f.name] = meta_description
    required = [
        f.name
        for f in fields
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    ]
    return _object_schema(
        {f.name: hints[f.name] for f in fields},
        required,
        descriptions,
        where=where,
        seen=seen | {tp},
        description=doc.summary,
    )


def _enum_schema(values: list[Any], where: str) -> JSONObject:
    if not values:
        raise ToolDefinitionError(f"{where}: leere Aufzählung")
    kinds: list[str] = []
    for value in values:
        if isinstance(value, bool):
            kind = "boolean"
        elif isinstance(value, int):
            kind = "integer"
        elif isinstance(value, str):
            kind = "string"
        elif value is None:
            kind = "null"
        else:
            raise ToolDefinitionError(
                f"{where}: Aufzählungswerte müssen str, int, bool oder None sein, nicht {value!r}"
            )
        if kind not in kinds:
            kinds.append(kind)
    schema: JSONObject = {"enum": list(values)}
    schema["type"] = kinds[0] if len(kinds) == 1 else list(kinds)
    return schema


@functools.lru_cache(maxsize=1024)
def _cached_schema(tp: Any) -> JSONObject:
    return _schema(tp, where="$", seen=frozenset())


def json_schema(tp: Any) -> JSONObject:
    """JSON-Schema für einen Python-Typ (Kopie, darf verändert werden)."""
    try:
        schema = _cached_schema(tp)
    except TypeError as exc:  # nicht hashbarer Typ – ohne Cache berechnen
        if "unhashable" not in str(exc):
            raise
        schema = _schema(tp, where="$", seen=frozenset())
    return _deep_copy(schema)


def _deep_copy(value: JSONObject) -> JSONObject:
    copied = _copy_json(value)
    assert isinstance(copied, dict)  # noqa: S101 - Invariante: Objekt bleibt Objekt
    return copied


def _copy_json(value: JSONValue) -> JSONValue:
    if isinstance(value, dict):
        return {k: _copy_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy_json(v) for v in value]
    return value


# --------------------------------------------------------------------------- #
# Funktionen                                                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FunctionSchema:
    """Schema eines Werkzeugs: Name, Beschreibung, Parameter-Schema und Typen."""

    name: str
    description: str
    parameters: JSONObject
    param_types: Mapping[str, Any]
    return_type: Any
    return_schema: JSONObject | None


def function_schema(
    fn: Callable[..., Any], *, name: str | None = None, description: str | None = None
) -> FunctionSchema:
    """Leitet das Werkzeug-Schema aus Signatur, Typannotationen und Docstring ab.

    Regeln:

    * Jeder Parameter braucht eine Typannotation; ``*args``/``**kwargs`` sind
      nicht erlaubt (das Modell muss jeden Parameter benennen können).
    * Parameter ohne Default sind Pflichtfelder; ``additionalProperties`` ist
      ``false``, unbekannte Argumente werden also abgewiesen.
    * Die Beschreibung stammt aus ``description=`` oder dem Docstring; fehlt
      beides, ist das ein Deklarationsfehler.
    """
    where = getattr(fn, "__qualname__", repr(fn))
    signature = inspect.signature(fn)
    hints = _type_hints(fn, where)
    doc = parse_docstring(inspect.getdoc(fn))
    text = (description or doc.summary).strip()
    if not text:
        raise ToolDefinitionError(f"{where}: Beschreibung fehlt (Docstring oder description=)")

    properties: JSONObject = {}
    required: list[str] = []
    param_types: dict[str, Any] = {}
    for param in signature.parameters.values():
        if param.kind not in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
            raise ToolDefinitionError(
                f"{where}: Parameter '{param.name}' ({param.kind}) "
                "nicht unterstützt; nur benannte Parameter"
            )
        if param.name not in hints:
            raise ToolDefinitionError(f"{where}: Typannotation fehlt für '{param.name}'")
        hint = hints[param.name]
        prop = _schema(hint, where=f"{where}.{param.name}", seen=frozenset())
        if "description" not in prop and doc.params.get(param.name):
            prop["description"] = doc.params[param.name]
        if param.default is param.empty:
            required.append(param.name)
        else:
            # Nicht darstellbarer Default: das Modell sieht nur "optional".
            with contextlib.suppress(NotJSONableError):
                prop["default"] = to_jsonable(param.default)
        properties[param.name] = prop
        param_types[param.name] = hint

    return_type = hints.get("return", Any)
    return_schema = (
        None
        if return_type is Any
        else _schema(return_type, where=f"{where}->return", seen=frozenset())
    )
    tool_name = str(name or getattr(fn, "__name__", ""))
    return FunctionSchema(
        name=tool_name,
        description=text,
        parameters={
            "type": "object",
            "properties": properties,
            "required": list(required),
            "additionalProperties": False,
        },
        param_types=param_types,
        return_type=return_type,
        return_schema=return_schema,
    )
