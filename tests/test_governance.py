"""Policy: Autonomiegrad × Seiteneffekt, Mandatslimit, Verschärfungslisten."""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated

import pytest
from hypothesis import given
from hypothesis import strategies as st

from gradweight_orchestration import Param, SideEffect, tool
from gradweight_orchestration.governance import (
    MATRIX,
    AutonomyGrade,
    Decision,
    Policy,
    format_amount,
)


def make(effect: SideEffect, name: str = "werkzeug", amount: bool = False):  # type: ignore[no-untyped-def]
    if amount:

        def zahlen(betrag: Annotated[Decimal, Param(ge=0)], empfaenger: str = "x") -> str:
            """Zahlt."""
            return "ok"

        return tool(name=name, side_effect=effect, amount_param="betrag")(zahlen)

    def lesen(x: int = 0) -> int:
        """Liest."""
        return x

    return tool(name=name, side_effect=effect)(lesen)


@pytest.mark.parametrize(
    ("grade", "effect", "expected"),
    [
        (AutonomyGrade.ASSISTANT, SideEffect.READ_ONLY, Decision.ALLOW),
        (AutonomyGrade.ASSISTANT, SideEffect.REVERSIBLE, Decision.DENY),
        (AutonomyGrade.COPILOT, SideEffect.REVERSIBLE, Decision.REQUIRE_APPROVAL),
        (AutonomyGrade.WORKFLOW, SideEffect.REVERSIBLE, Decision.ALLOW),
        (AutonomyGrade.WORKFLOW, SideEffect.IRREVERSIBLE, Decision.REQUIRE_APPROVAL),
        (AutonomyGrade.MANDATE, SideEffect.IRREVERSIBLE, Decision.REQUIRE_APPROVAL),
        (AutonomyGrade.EXCEPTION_ONLY, SideEffect.IRREVERSIBLE, Decision.ALLOW),
    ],
)
def test_matrix(grade: AutonomyGrade, effect: SideEffect, expected: Decision) -> None:
    decision = Policy(grade=grade).decide(make(effect), {})
    assert decision.decision is expected
    assert decision.rule == "grad-matrix"
    assert grade.label in decision.reason


def test_default_is_copilot() -> None:
    assert Policy().grade is AutonomyGrade.COPILOT


def test_money_limit() -> None:
    pay = make(SideEffect.MONEY, amount=True)
    policy = Policy(grade=AutonomyGrade.MANDATE, money_limit=Decimal("5000"))
    inside = policy.decide(pay, {"betrag": Decimal("1081.00")})
    assert inside.decision is Decision.ALLOW
    assert inside.rule == "mandat:innerhalb"
    assert "CHF 1'081.00 innerhalb Mandatslimit CHF 5'000.00" in inside.reason
    edge = policy.decide(pay, {"betrag": Decimal("5000")})
    assert edge.decision is Decision.ALLOW
    over = policy.decide(pay, {"betrag": Decimal("12430.50")})
    assert over.decision is Decision.REQUIRE_APPROVAL
    assert over.rule == "mandat:ueber-limit"
    assert over.amount == Decimal("12430.50")
    assert over.to_json()["amount"] == "12430.50"


def test_money_fail_closed() -> None:
    pay = make(SideEffect.MONEY, amount=True)
    unknown = make(SideEffect.MONEY, name="ohne_betrag")
    policy = Policy(grade=AutonomyGrade.EXCEPTION_ONLY, money_limit=Decimal(100))
    assert policy.decide(unknown, {}).rule == "mandat:betrag-unbekannt"
    assert policy.decide(pay, {"betrag": Decimal(-1)}).rule == "mandat:betrag-negativ"
    no_limit = Policy(grade=AutonomyGrade.MANDATE)
    assert no_limit.decide(pay, {"betrag": Decimal(1)}).rule == "mandat:kein-limit"


def test_lists_only_tighten() -> None:
    reader = make(SideEffect.READ_ONLY, name="lesen")
    policy = Policy(grade=AutonomyGrade.EXCEPTION_ONLY, require_approval=frozenset({"lesen"}))
    assert policy.decide(reader, {}).rule == "liste:require_approval"
    blocked = Policy(grade=AutonomyGrade.EXCEPTION_ONLY, deny=frozenset({"lesen"}))
    assert blocked.decide(reader, {}).decision is Decision.DENY
    # require_approval kann ein DENY der Matrix nicht lockern
    strict = Policy(grade=AutonomyGrade.ASSISTANT, require_approval=frozenset({"w"}))
    assert strict.decide(make(SideEffect.MONEY, name="w"), {}).decision is Decision.DENY


def test_policy_validation_and_json() -> None:
    with pytest.raises(ValueError, match="money_limit"):
        Policy(money_limit=Decimal(-1))
    with pytest.raises(ValueError, match="money_limit"):
        Policy(money_limit=Decimal("NaN"))
    policy = Policy(
        grade=AutonomyGrade.MANDATE,
        money_limit=5000,  # type: ignore[arg-type]
        deny={"b", "a"},
    )  # type: ignore[arg-type]
    assert policy.money_limit == Decimal(5000)
    data = policy.to_json()
    assert data["deny"] == ["a", "b"]
    assert data["grade_label"] == "04 Mandat"


def test_format_amount() -> None:
    assert format_amount(Decimal("12430.5")) == "CHF 12'430.50"
    assert format_amount(Decimal("0")) == "CHF 0.00"
    assert format_amount(Decimal("-1234567.891"), "EUR") == "EUR -1'234'567.89"
    assert format_amount(Decimal("999")) == "CHF 999.00"


def test_matrix_markdown_has_all_grades() -> None:
    table = Policy.matrix_markdown()
    assert table.count("\n") == len(AutonomyGrade) + 1
    assert "| 04 Mandat | erlaubt | erlaubt | Freigabe | bis Limit |" in table


# --------------------------------------------------------- Property-Tests ---

effects = st.sampled_from(list(SideEffect))
grades = st.sampled_from(list(AutonomyGrade))
amounts = st.none() | st.decimals(
    min_value=-10, max_value=10**6, places=2, allow_nan=False, allow_infinity=False
)
limits = st.none() | st.decimals(
    min_value=0, max_value=10**6, places=2, allow_nan=False, allow_infinity=False
)


@given(
    grade=grades,
    effect=effects,
    amount=amounts,
    limit=limits,
    approve=st.booleans(),
    deny=st.booleans(),
)
def test_lists_never_loosen(
    grade: AutonomyGrade,
    effect: SideEffect,
    amount: Decimal | None,
    limit: Decimal | None,
    approve: bool,
    deny: bool,
) -> None:
    item = make(effect, name="t", amount=effect is SideEffect.MONEY)
    args = {} if amount is None else {"betrag": amount}
    base = Policy(grade=grade, money_limit=limit).decide(item, args)
    tightened = Policy(
        grade=grade,
        money_limit=limit,
        require_approval=frozenset({"t"} if approve else ()),
        deny=frozenset({"t"} if deny else ()),
    ).decide(item, args)
    assert tightened.decision.strictness >= base.decision.strictness


@given(effect=effects, amount=amounts, limit=limits)
def test_higher_grade_is_never_stricter(
    effect: SideEffect, amount: Decimal | None, limit: Decimal | None
) -> None:
    item = make(effect, name="t", amount=effect is SideEffect.MONEY)
    args = {} if amount is None else {"betrag": amount}
    previous = None
    for grade in AutonomyGrade:
        current = Policy(grade=grade, money_limit=limit).decide(item, args).decision.strictness
        if previous is not None:
            assert current <= previous
        previous = current


def test_matrix_is_complete() -> None:
    for grade in AutonomyGrade:
        assert set(MATRIX[grade]) == set(SideEffect)
