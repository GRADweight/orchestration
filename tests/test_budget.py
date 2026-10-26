"""Budgets."""

from __future__ import annotations

from decimal import Decimal

import pytest

from gradweight_orchestration.budget import Budget, BudgetState, Pricing
from gradweight_orchestration.llm.types import Usage


def test_pricing_and_state_round_trip() -> None:
    pricing = Pricing(input_per_1k=Decimal("0.002"), output_per_1k=Decimal("0.006"))
    state = BudgetState()
    state.add(Usage(1000, 500), pricing)
    state.add(Usage(10, 0, estimated=True), pricing)
    assert state.tokens == 1510
    assert state.cost == Decimal("0.00502")
    assert state.estimated is True
    assert BudgetState.from_json(state.to_json()) == state


@pytest.mark.parametrize(
    ("budget", "state", "code"),
    [
        (Budget(max_steps=2), BudgetState(steps=2), "max_steps"),
        (
            Budget(max_tokens=100),
            BudgetState(prompt_tokens=60, completion_tokens=40),
            "token_budget",
        ),
        (Budget(max_cost=Decimal("0.01")), BudgetState(cost=Decimal("0.02")), "cost_budget"),
        (Budget(max_seconds=1.0), BudgetState(active_seconds=0.6), "time_budget"),
    ],
)
def test_exceeded(budget: Budget, state: BudgetState, code: str) -> None:
    result = state.exceeded(budget, elapsed=0.5)
    assert result is not None
    assert result[0] == code


def test_not_exceeded_and_remaining() -> None:
    budget = Budget(max_seconds=10)
    state = BudgetState(active_seconds=4)
    assert state.exceeded(budget) is None
    assert state.remaining_seconds(budget, elapsed=1) == 5
    assert BudgetState().remaining_seconds(Budget(max_seconds=None)) is None


def test_validation() -> None:
    for kwargs in ({"max_steps": 0}, {"max_tokens": 0}, {"max_seconds": 0}):
        with pytest.raises(ValueError):  # noqa: PT011
            Budget(**kwargs)  # type: ignore[arg-type]
