"""Schema-Ableitung aus Typannotationen und Docstrings."""

from __future__ import annotations

import datetime as dt
import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Annotated, Any, Literal, Optional, TypedDict, Union

import pytest

from gradweight_orchestration import Param, ToolDefinitionError, function_schema, json_schema
from gradweight_orchestration.schema import parse_docstring


class Waehrung(enum.Enum):
    CHF = "CHF"
    EUR = "EUR"


class Stufe(enum.IntEnum):
    TIEF = 1
    HOCH = 2


@dataclass
class Adresse:
    """Postadresse.

    Attributes:
        plz: Postleitzahl, vierstellig.
        ort: Ortschaft.
    """

    plz: Annotated[str, Param(pattern=r"^\d{4}$")]
    ort: str
    land: str = field(default="CH", metadata={"description": "ISO-Ländercode"})


class Position(TypedDict):
    text: str
    betrag: float


class TeilweisePosition(TypedDict, total=False):
    text: str


@dataclass
class Knoten:
    kind: Knoten | None = None


@pytest.mark.parametrize(
    ("tp", "expected"),
    [
        (str, {"type": "string"}),
        (int, {"type": "integer"}),
        (float, {"type": "number"}),
        (Decimal, {"type": "number"}),
        (bool, {"type": "boolean"}),
        (type(None), {"type": "null"}),
        (Any, {}),
        (dt.date, {"type": "string", "format": "date"}),
        (dt.datetime, {"type": "string", "format": "date-time"}),
        (Literal["a", "b"], {"type": "string", "enum": ["a", "b"]}),
        (Literal[1, "x"], {"type": ["integer", "string"], "enum": [1, "x"]}),
        (Waehrung, {"type": "string", "enum": ["CHF", "EUR"]}),
        (Stufe, {"type": "integer", "enum": [1, 2]}),
        (list[int], {"type": "array", "items": {"type": "integer"}}),
        (Sequence[str], {"type": "array", "items": {"type": "string"}}),
        (tuple[float, ...], {"type": "array", "items": {"type": "number"}}),
        (list, {"type": "array"}),
        (dict[str, float], {"type": "object", "additionalProperties": {"type": "number"}}),
        (Mapping[str, int], {"type": "object", "additionalProperties": {"type": "integer"}}),
        (dict, {"type": "object"}),
        (Optional[str], {"type": ["string", "null"]}),  # noqa: UP045
        (int | None, {"type": ["integer", "null"]}),
        (Literal["a"] | None, {"type": ["string", "null"], "enum": ["a", None]}),
        (Union[int, str], {"anyOf": [{"type": "integer"}, {"type": "string"}]}),  # noqa: UP007
        (Annotated[str, "Kurztext"], {"type": "string", "description": "Kurztext"}),
        (
            Annotated[float, Param(description="Betrag", ge=0, le=10)],
            {"type": "number", "description": "Betrag", "minimum": 0, "maximum": 10},
        ),
        (
            Annotated[list[str], Param(min_length=1, max_length=3)],
            {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
        ),
        (
            Annotated[str, Param(min_length=2, max_length=5)],
            {"type": "string", "minLength": 2, "maxLength": 5},
        ),
    ],
)
def test_type_mapping(tp: Any, expected: dict[str, Any]) -> None:
    assert json_schema(tp) == expected


def test_nullable_variants() -> None:
    """Einfacher Typ: Typliste; ohne festen Typ: anyOf mit angehobener Beschreibung."""
    assert json_schema(Optional[Annotated[dict[str, int], "Zähler"]]) == {  # noqa: UP045
        "type": ["object", "null"],
        "additionalProperties": {"type": "integer"},
        "description": "Zähler",
    }
    assert json_schema(Optional[Annotated[Any, "Beliebig"]]) == {  # noqa: UP045
        "anyOf": [{}, {"type": "null"}],
        "description": "Beliebig",
    }
    assert json_schema(Union[int, str, None]) == {  # noqa: UP007
        "anyOf": [{"type": "integer"}, {"type": "string"}, {"type": "null"}]
    }


def test_dataclass_schema() -> None:
    schema = json_schema(Adresse)
    assert schema == {
        "type": "object",
        "description": "Postadresse.",
        "properties": {
            "plz": {
                "type": "string",
                "pattern": r"^\d{4}$",
                "description": "Postleitzahl, vierstellig.",
            },
            "ort": {"type": "string", "description": "Ortschaft."},
            "land": {"type": "string", "description": "ISO-Ländercode"},
        },
        "required": ["plz", "ort"],
        "additionalProperties": False,
    }


def test_typeddict_schema() -> None:
    assert json_schema(Position)["required"] == ["text", "betrag"]
    assert json_schema(TeilweisePosition)["required"] == []


def test_schema_is_a_copy() -> None:
    first = json_schema(list[int])
    first["items"] = {"type": "string"}
    assert json_schema(list[int]) == {"type": "array", "items": {"type": "integer"}}


@pytest.mark.parametrize(
    "tp",
    [
        bytes,
        set[int],
        tuple[int, str],
        dict[int, str],
        Knoten,
        Literal[1.5],
        Annotated[str, Param(ge=1)],
        Annotated[int, Param(min_length=1)],
        Annotated[int, Param(pattern="x")],
        Annotated[str, Param(pattern="(")],
    ],
)
def test_unsupported_types_are_rejected(tp: Any) -> None:
    with pytest.raises(ToolDefinitionError):
        json_schema(tp)


def test_unhashable_annotation_still_works() -> None:
    unhashable = Annotated[int, {"nicht": "hashbar"}]
    assert json_schema(unhashable) == {"type": "integer"}


# ------------------------------------------------------------ Funktionen ---


def rechnung_pruefen(
    nummer: str,
    betrag: Annotated[Decimal, Param(ge=0)],
    waehrung: Waehrung = Waehrung.CHF,
    *,
    faellig: dt.date | None = None,
) -> dict[str, bool]:
    """Prüft eine Rechnung.

    Der Text darf mehrere Absätze haben.

    Args:
        nummer: Rechnungsnummer, z.B. RE-2026-0042.
        betrag: Bruttobetrag
            inklusive MWST.
        waehrung: Währung.

    Returns:
        Prüfergebnis.
    """
    return {"ok": True}


def test_function_schema() -> None:
    schema = function_schema(rechnung_pruefen)
    assert schema.name == "rechnung_pruefen"
    assert schema.description == "Prüft eine Rechnung.\nDer Text darf mehrere Absätze haben."
    params = schema.parameters
    assert params["required"] == ["nummer", "betrag"]
    assert params["additionalProperties"] is False
    props = params["properties"]
    assert isinstance(props, dict)
    assert props["nummer"] == {
        "type": "string",
        "description": "Rechnungsnummer, z.B. RE-2026-0042.",
    }
    assert props["betrag"] == {
        "type": "number",
        "minimum": 0,
        "description": "Bruttobetrag inklusive MWST.",
    }
    assert props["waehrung"] == {
        "type": "string",
        "enum": ["CHF", "EUR"],
        "description": "Währung.",
        "default": "CHF",
    }
    assert props["faellig"] == {"type": ["string", "null"], "format": "date", "default": None}
    assert schema.return_schema == {"type": "object", "additionalProperties": {"type": "boolean"}}


def test_function_schema_overrides_and_untyped_return() -> None:
    def ohne_rueckgabe(x: int):  # type: ignore[no-untyped-def]
        """Text."""

    schema = function_schema(ohne_rueckgabe, name="anders", description="Neu")
    assert schema.name == "anders"
    assert schema.description == "Neu"
    assert schema.return_schema is None


def test_non_jsonable_default_is_omitted() -> None:
    marker = object()

    def mit_default(x: Any = marker) -> None:
        """Text."""

    props = function_schema(mit_default).parameters["properties"]
    assert isinstance(props, dict)
    assert props["x"] == {}


def test_function_definition_errors() -> None:
    def ohne_doc(x: int) -> int:
        return x

    def ohne_annotation(x) -> int:  # type: ignore[no-untyped-def]
        """Text."""
        return 1

    def varargs(*werte: int) -> int:
        """Text."""
        return 1

    def unaufloesbar(x: GibtEsNicht) -> int:  # type: ignore[name-defined]  # noqa: F821
        """Text."""
        return 1

    for fn in (ohne_doc, ohne_annotation, varargs, unaufloesbar):
        with pytest.raises(ToolDefinitionError):
            function_schema(fn)


# ------------------------------------------------------------- Docstrings ---


def test_parse_google_and_german_sections() -> None:
    doc = parse_docstring(
        """Kurz.

        Argumente:
            a (int): Erstes.
            b: Zweites,
                fortgesetzt.

        Rückgabe:
            Summe.

        Beispiel:
            >>> 1 + 1
        """
    )
    assert doc.summary == "Kurz."
    assert doc.params == {"a": "Erstes.", "b": "Zweites, fortgesetzt."}
    assert doc.returns == "Summe."


def test_parse_numpy_style() -> None:
    doc = parse_docstring(
        """Kurz.

        Parameters
        ----------
        betrag : float
            Der Betrag
            in CHF.
        konto : str
            Konto.

        Returns
        -------
        bool
        """
    )
    assert doc.params == {"betrag": "Der Betrag in CHF.", "konto": "Konto."}
    assert doc.returns == "bool"


def test_parse_rest_style() -> None:
    doc = parse_docstring(
        """Kurz.

        :param str betrag: Der Betrag
            in CHF.
        :param konto: Konto.
        :returns: Ergebnis.
        """
    )
    assert doc.params == {"betrag": "Der Betrag in CHF.", "konto": "Konto."}
    assert doc.returns == "Ergebnis."


def test_parse_empty_and_unknown_headers() -> None:
    assert parse_docstring(None).summary == ""
    doc = parse_docstring("Titel:\nText ohne Abschnitt.\n\nArgs:\n    x: X.\n    nicht passend\n")
    assert doc.summary == "Titel: Text ohne Abschnitt."
    assert doc.params == {"x": "X."}
