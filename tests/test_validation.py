"""Strikte Validierung und Rückwandlung in Python-Typen (inkl. Property-Tests)."""

from __future__ import annotations

import datetime as dt
import enum
import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Annotated, Any, Literal, TypedDict

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gradweight_orchestration import Param, from_jsonable, json_schema, validate
from gradweight_orchestration._json import to_jsonable


class Farbe(enum.Enum):
    ROT = "rot"
    BLAU = "blau"


@dataclass(frozen=True)
class Zeile:
    text: str
    betrag: Decimal
    menge: int = 1


@dataclass(frozen=True)
class Beleg:
    nummer: Annotated[str, Param(min_length=1)]
    datum: dt.date
    zeilen: list[Zeile]
    farbe: Farbe | None = None


class Notiz(TypedDict):
    titel: str
    tags: list[str]


def paths(schema: dict[str, Any], value: Any) -> list[str]:
    return [i.path for i in validate(value, schema)]


def test_strict_types() -> None:
    assert paths({"type": "number"}, "12.5") == ["$"]
    assert paths({"type": "integer"}, True) == ["$"]
    assert paths({"type": "integer"}, 3.0) == []
    assert paths({"type": "integer"}, 3.5) == ["$"]
    assert paths({"type": "number"}, math.nan) == ["$"]
    assert paths({"type": "boolean"}, 1) == ["$"]
    assert paths({"type": ["string", "null"]}, None) == []
    assert paths({"type": "unbekannt"}, 1) == ["$"]


def test_object_rules() -> None:
    schema = json_schema(Zeile)
    issues = validate({"text": 1, "unbekannt": 2}, schema)
    messages = {i.path: i.message for i in issues}
    assert messages["$.betrag"] == "Pflichtfeld fehlt"
    assert messages["$.text"].startswith("erwartet string")
    assert messages["$.unbekannt"].startswith("unbekanntes Feld; erlaubt: betrag, menge, text")
    assert validate({"text": "a", "betrag": 1}, schema) == []
    assert paths({"type": "object", "additionalProperties": {"type": "integer"}}, {"a": "x"}) == [
        "$.a"
    ]


def test_enum_const_and_bool_trap() -> None:
    assert paths({"enum": [1, 2]}, True) == ["$"]
    assert paths({"enum": [True]}, 1) == ["$"]
    assert paths({"enum": ["rot"]}, "rot") == []
    assert paths({"const": 1}, 1) == []
    assert paths({"const": 1}, 2) == ["$"]


def test_string_number_array_constraints() -> None:
    assert paths({"type": "string", "minLength": 2}, "a") == ["$"]
    assert paths({"type": "string", "maxLength": 2}, "abc") == ["$"]
    assert paths({"type": "string", "pattern": r"^\d{4}$"}, "80x0") == ["$"]
    assert paths({"type": "number", "minimum": 0}, -1) == ["$"]
    assert paths({"type": "number", "maximum": 0}, 1) == ["$"]
    assert paths({"type": "array", "minItems": 1}, []) == ["$"]
    assert paths({"type": "array", "maxItems": 1}, [1, 2]) == ["$"]
    assert paths({"type": "array", "items": {"type": "integer"}}, [1, "a"]) == ["$[1]"]


def test_formats() -> None:
    date_schema = {"type": "string", "format": "date"}
    assert paths(date_schema, "2026-10-26") == []
    assert paths(date_schema, "26.10.2026") == ["$"]
    assert paths(date_schema, "2026-02-30") == ["$"]
    ts = {"type": "string", "format": "date-time"}
    assert paths(ts, "2026-10-26T12:00:00Z") == []
    assert paths(ts, "2026-10-26T12:00:00+02:00") == []
    assert paths(ts, "2026-10-26T25:00:00") == ["$"]
    assert paths(ts, "gestern") == ["$"]


def test_any_of() -> None:
    schema = {"anyOf": [{"type": "integer"}, {"type": "string"}]}
    assert paths(schema, 1) == []
    issues = validate(1.5, schema)
    assert issues[0].message.startswith("passt zu keiner erlaubten Variante")


def test_from_jsonable_builds_typed_objects() -> None:
    raw = {
        "nummer": "B-1",
        "datum": "2026-10-26",
        "zeilen": [{"text": "Papier", "betrag": 12.5}],
        "farbe": "blau",
    }
    assert validate(raw, json_schema(Beleg)) == []
    beleg = from_jsonable(Beleg, raw)
    assert beleg == Beleg(
        "B-1", dt.date(2026, 10, 26), [Zeile("Papier", Decimal("12.5"))], Farbe.BLAU
    )
    assert from_jsonable(Notiz, {"titel": "t", "tags": ["a"]}) == {"titel": "t", "tags": ["a"]}
    assert from_jsonable(tuple[int, ...], [1, 2]) == (1, 2)
    assert from_jsonable(dt.datetime, "2026-10-26T08:00:00Z").tzinfo is not None
    assert from_jsonable(int | str, "x") == "x"
    assert from_jsonable(int | None, None) is None
    assert from_jsonable(Literal["a"], "a") == "a"
    assert from_jsonable(Any, {"a": 1}) == {"a": 1}
    assert from_jsonable(type(None), None) is None


@pytest.mark.parametrize(
    ("tp", "value"),
    [
        (int, "1"),
        (int, True),
        (float, "1.0"),
        (Decimal, "1"),
        (bool, 1),
        (str, 1),
        (type(None), 0),
        (Literal["a"], "b"),
        (int | str, 1.5),
        (list[int], {}),
        (dict[str, int], []),
        (Zeile, []),
        (Notiz, []),
        (dt.date, 1),
        (set, []),
    ],
)
def test_from_jsonable_rejects_mismatches(tp: Any, value: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        from_jsonable(tp, value)


# --------------------------------------------------------- Property-Tests ---

json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(),
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=5), children, max_size=4)
    ),
    max_leaves=12,
)

SCHEMAS = [
    json_schema(t)
    for t in (
        Beleg,
        Zeile,
        Notiz,
        list[int],
        dict[str, float],
        int | str | None,
        Farbe,
        Literal["a", "b"],
        dt.date,
    )
]


@given(value=json_values, schema=st.sampled_from(SCHEMAS))
def test_validator_never_raises(value: Any, schema: dict[str, Any]) -> None:
    issues = validate(value, schema)
    assert all(i.path.startswith("$") for i in issues)


money = st.decimals(
    min_value=Decimal("-1e9"),
    max_value=Decimal("1e9"),
    places=2,
    allow_nan=False,
    allow_infinity=False,
)
zeilen = st.builds(Zeile, text=st.text(), betrag=money, menge=st.integers(-1000, 1000))
belege = st.builds(
    Beleg,
    nummer=st.text(min_size=1),
    datum=st.dates(),
    zeilen=st.lists(zeilen, max_size=3),
    farbe=st.none() | st.sampled_from(Farbe),
)


@settings(max_examples=150)
@given(beleg=belege)
def test_round_trip_python_json_python(beleg: Beleg) -> None:
    """Python → JSON → validieren → Python ergibt wieder denselben Wert."""
    raw = to_jsonable(beleg)
    assert validate(raw, json_schema(Beleg)) == []
    assert from_jsonable(Beleg, raw) == beleg
