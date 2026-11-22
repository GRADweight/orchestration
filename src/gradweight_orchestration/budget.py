"""Budgets für Schritte, Token, Kosten und aktive Laufzeit.

Die Laufzeit zählt nur, solange der Agent arbeitet: Wartezeit auf eine
menschliche Freigabe (zwischen Pause und Fortsetzung) zählt nicht.
Kosten werden aus den gemeldeten Token und einem frei wählbaren Preis
berechnet; lokale Modelle haben standardmässig Preis 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from ._json import JSONObject
from .llm.types import Usage

__all__ = ["Budget", "BudgetState", "Pricing"]


@dataclass(frozen=True)
class Pricing:
    input_per_1k: Decimal = Decimal(0)
    output_per_1k: Decimal = Decimal(0)
    currency: str = "CHF"

    def cost(self, usage: Usage) -> Decimal:
        return (
            Decimal(usage.prompt_tokens) * self.input_per_1k
            + Decimal(usage.completion_tokens) * self.output_per_1k
        ) / Decimal(1000)


@dataclass(frozen=True)
class Budget:
    max_steps: int = 12
    """Höchstzahl Modellaufrufe pro Lauf (inkl. Fortsetzungen)."""
    max_tokens: int | None = None
    max_cost: Decimal | None = None
    max_seconds: float | None = 300.0
    """Aktive Laufzeit; Warten auf Freigaben zählt nicht."""
    pricing: Pricing = field(default_factory=Pricing)

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps muss ≥ 1 sein")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens muss ≥ 1 sein")
        if self.max_seconds is not None and self.max_seconds <= 0:
            raise ValueError("max_seconds muss > 0 sein")


@dataclass
class BudgetState:
    """Verbrauch eines Laufs (Teil des speicherbaren Laufzustands)."""

    steps: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated: bool = False
    cost: Decimal = Decimal(0)
    active_seconds: float = 0.0

    def add(self, usage: Usage, pricing: Pricing) -> None:
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.estimated = self.estimated or usage.estimated
        self.cost += pricing.cost(usage)

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def exceeded(self, budget: Budget, *, elapsed: float = 0.0) -> tuple[str, str] | None:
        """``(Code, Text)``, wenn ein Budget erschöpft ist; sonst ``None``."""
        if self.steps >= budget.max_steps:
            return "max_steps", f"Schrittlimit {budget.max_steps} erreicht"
        if budget.max_tokens is not None and self.tokens >= budget.max_tokens:
            return "token_budget", f"Tokenbudget {budget.max_tokens} erreicht ({self.tokens})"
        if budget.max_cost is not None and self.cost >= budget.max_cost:
            return (
                "cost_budget",
                f"Kostenbudget {budget.pricing.currency} {budget.max_cost} erreicht",
            )
        seconds = self.active_seconds + elapsed
        if budget.max_seconds is not None and seconds >= budget.max_seconds:
            return "time_budget", f"Zeitbudget {budget.max_seconds:g} s erreicht"
        return None

    def remaining_seconds(self, budget: Budget, *, elapsed: float = 0.0) -> float | None:
        if budget.max_seconds is None:
            return None
        return max(0.0, budget.max_seconds - self.active_seconds - elapsed)

    def to_json(self) -> JSONObject:
        return {
            "steps": self.steps,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "estimated": self.estimated,
            "cost": str(self.cost),
            "active_seconds": round(self.active_seconds, 6),
        }

    @classmethod
    def from_json(cls, raw: JSONObject) -> BudgetState:
        return cls(
            steps=int(raw.get("steps", 0)),  # type: ignore[arg-type]
            prompt_tokens=int(raw.get("prompt_tokens", 0)),  # type: ignore[arg-type]
            completion_tokens=int(raw.get("completion_tokens", 0)),  # type: ignore[arg-type]
            estimated=bool(raw.get("estimated", False)),
            cost=Decimal(str(raw.get("cost", "0"))),
            active_seconds=float(raw.get("active_seconds", 0.0)),  # type: ignore[arg-type]
        )
