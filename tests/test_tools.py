"""@tool-Decorator, Deklarationsprüfungen und Registry."""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal

import pytest

from gradweight_orchestration import (
    ArgumentValidationError,
    Isolation,
    Param,
    SandboxLimits,
    SideEffect,
    Tool,
    ToolDefinitionError,
    ToolRegistry,
    tool,
)


@tool(side_effect=SideEffect.READ_ONLY)
def wechselkurs(waehrung: Literal["EUR", "USD"], datum: str | None = None) -> float:
    """Liefert den Referenzkurs in CHF.

    Args:
        waehrung: ISO-Code der Fremdwährung.
        datum: Stichtag (JJJJ-MM-TT); ohne Angabe der aktuelle Kurs.
    """
    return 0.94 if waehrung == "EUR" else 0.88


@tool(side_effect=SideEffect.MONEY, amount_param="betrag_chf", name="zahlung")
def zahlung_ausloesen(empfaenger: str, betrag_chf: Annotated[Decimal, Param(ge=0)]) -> str:
    """Löst eine Zahlung aus."""
    return f"{empfaenger}:{betrag_chf}"


@tool
def ohne_klammern(x: int) -> int:
    """Verdoppelt."""
    return 2 * x


def test_decorated_function_stays_callable_and_typed() -> None:
    assert isinstance(wechselkurs, Tool)
    assert wechselkurs("EUR") == 0.94
    assert ohne_klammern(2) == 4
    assert ohne_klammern.side_effect is SideEffect.IRREVERSIBLE  # fail-closed Default
    assert ohne_klammern.isolation is Isolation.THREAD


def test_openai_spec() -> None:
    spec = wechselkurs.spec()
    assert spec == {
        "type": "function",
        "function": {
            "name": "wechselkurs",
            "description": "Liefert den Referenzkurs in CHF.",
            "parameters": {
                "type": "object",
                "properties": {
                    "waehrung": {
                        "type": "string",
                        "enum": ["EUR", "USD"],
                        "description": "ISO-Code der Fremdwährung.",
                    },
                    "datum": {
                        "type": ["string", "null"],
                        "default": None,
                        "description": "Stichtag (JJJJ-MM-TT); ohne Angabe der aktuelle Kurs.",
                    },
                },
                "required": ["waehrung"],
                "additionalProperties": False,
            },
        },
    }


def test_parse_arguments() -> None:
    assert zahlung_ausloesen.parse_arguments({"empfaenger": "A", "betrag_chf": 12.5}) == {
        "empfaenger": "A",
        "betrag_chf": Decimal("12.5"),
    }
    with pytest.raises(ArgumentValidationError) as info:
        zahlung_ausloesen.parse_arguments({"empfaenger": "A", "betrag_chf": "12.50"})
    assert info.value.issues[0].path == "$.betrag_chf"
    assert "zahlung" in str(info.value)
    with pytest.raises(ArgumentValidationError):
        zahlung_ausloesen.parse_arguments(None)
    with pytest.raises(ArgumentValidationError):
        zahlung_ausloesen.parse_arguments({"empfaenger": "A", "betrag_chf": -1})


def test_amount_extraction() -> None:
    assert zahlung_ausloesen.amount({"betrag_chf": Decimal("12.50")}) == Decimal("12.50")
    assert zahlung_ausloesen.amount({"betrag_chf": 7}) == Decimal(7)
    assert zahlung_ausloesen.amount({}) is None
    assert zahlung_ausloesen.amount({"betrag_chf": True}) is None
    assert zahlung_ausloesen.amount({"betrag_chf": "abc"}) is None
    assert zahlung_ausloesen.amount({"betrag_chf": float("inf")}) is None
    assert wechselkurs.amount({"waehrung": "EUR"}) is None


def _fn(x: int) -> int:
    """Text."""
    return x


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "ungültig name"},
        {"timeout_s": 0},
        {"retries": -1},
        {"retries": 2},  # schreibend und nicht idempotent
        {"sandbox": SandboxLimits()},  # Limits ohne PROCESS
        {"amount_param": "fehlt"},
        {"isolation": Isolation.PROCESS},  # _fn ist zwar Modulebene, aber siehe unten
    ],
)
def test_declaration_errors(kwargs: dict[str, object]) -> None:
    fn = _fn
    if kwargs.get("isolation") is Isolation.PROCESS:
        fn = lambda x: x  # noqa: E731
        fn.__doc__ = "Text."
        fn.__annotations__ = {"x": int, "return": int}
    with pytest.raises(ToolDefinitionError):
        tool(**kwargs)(fn)  # type: ignore[call-overload]


def test_amount_param_must_be_numeric() -> None:
    def zahlen(betrag: str) -> str:
        """Text."""
        return betrag

    with pytest.raises(ToolDefinitionError):
        tool(amount_param="betrag")(zahlen)


def test_process_isolation_needs_module_level_function() -> None:
    def lokal(x: int) -> int:
        """Text."""
        return x

    with pytest.raises(ToolDefinitionError, match="Modulebene"):
        tool(isolation=Isolation.PROCESS)(lokal)
    proc = tool(isolation=Isolation.PROCESS, side_effect=SideEffect.READ_ONLY)(_fn)
    assert proc.target is not None
    assert proc.target.qualname == "_fn"
    assert proc.sandbox == SandboxLimits()


def test_retries_allowed_for_read_only_or_idempotent() -> None:
    assert tool(retries=2, side_effect=SideEffect.READ_ONLY)(_fn).retries == 2
    assert tool(retries=1, idempotent=True)(_fn).retries == 1


def test_registry() -> None:
    registry = ToolRegistry([wechselkurs, zahlung_ausloesen])
    assert len(registry) == 2
    assert "zahlung" in registry
    assert registry["wechselkurs"] is wechselkurs
    assert registry.find("fehlt") is None
    assert registry.names() == ["wechselkurs", "zahlung"]
    assert [t.name for t in registry] == ["wechselkurs", "zahlung"]
    assert len(registry.specs()) == 2
    with pytest.raises(ToolDefinitionError):
        registry.register(wechselkurs)
    with pytest.raises(ToolDefinitionError):
        registry.register(_fn)  # type: ignore[arg-type]
