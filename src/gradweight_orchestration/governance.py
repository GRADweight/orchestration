"""Freigabe-Policy: welcher Werkzeugaufruf darf unbeaufsichtigt laufen?

Die Policy ordnet jedem Aufruf eine Entscheidung zu – ``ALLOW`` (ausführen),
``REQUIRE_APPROVAL`` (Mensch gibt frei) oder ``DENY`` (nie ausführen). Grundlage
sind der Autonomiegrad (01–05, Begriffe aus dem GRADweight-Autonomiemodell) und
der deklarierte Seiteneffekt des Werkzeugs. Die Zuordnung ist eine
**Designentscheidung dieser Runtime**, kein normativer Standard:

======================  ==========  =============  =============  =============
Grad                    lesend      umkehrbar      irreversibel   Geld
======================  ==========  =============  =============  =============
01 Assistent            erlaubt     verboten       verboten       verboten
02 Copilot              erlaubt     Freigabe       Freigabe       Freigabe
03 Workflow             erlaubt     erlaubt        Freigabe       Freigabe
04 Mandat               erlaubt     erlaubt        Freigabe       bis Limit
05 Ausnahmebetrieb      erlaubt     erlaubt        erlaubt        bis Limit
======================  ==========  =============  =============  =============

«bis Limit»: erlaubt, wenn der Betrag bekannt ist und das Mandatslimit nicht
überschreitet; sonst Freigabe (fail-closed: ohne Limit oder ohne Betrag immer
Freigabe). Listen in der Policy können nur *verschärfen*, nie lockern.
Mandatslimits gehören zusätzlich ins Zielsystem (ERP, Bank) – dieses Limit ist
eine zweite Linie, nicht die einzige.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from ._json import JSONObject
from .tools import SideEffect, Tool

__all__ = ["AutonomyGrade", "Decision", "Policy", "PolicyDecision", "format_amount"]


class AutonomyGrade(enum.IntEnum):
    ASSISTANT = 1
    """01 Assistent: nur lesen; jede Wirkung führt der Mensch selbst aus."""
    COPILOT = 2
    """02 Copilot: jede schreibende Aktion braucht eine Freigabe."""
    WORKFLOW = 3
    """03 Workflow: umkehrbare Schritte laufen, der Rest wartet auf Freigabe."""
    MANDATE = 4
    """04 Mandat: Geld bis zum Mandatslimit, Irreversibles per Freigabe."""
    EXCEPTION_ONLY = 5
    """05 Ausnahmebetrieb: nur Ausnahmen (z.B. über Limit) gehen an den Menschen."""

    @property
    def label(self) -> str:
        return _LABELS[self]


_LABELS = {
    AutonomyGrade.ASSISTANT: "01 Assistent",
    AutonomyGrade.COPILOT: "02 Copilot",
    AutonomyGrade.WORKFLOW: "03 Workflow",
    AutonomyGrade.MANDATE: "04 Mandat",
    AutonomyGrade.EXCEPTION_ONLY: "05 Ausnahmebetrieb",
}


class Decision(str, enum.Enum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"

    @property
    def strictness(self) -> int:
        return {"allow": 0, "require_approval": 1, "deny": 2}[self.value]


_A, _F, _V, _L = "allow", "approval", "deny", "limit"
MATRIX: Mapping[AutonomyGrade, Mapping[SideEffect, str]] = {
    AutonomyGrade.ASSISTANT: {
        SideEffect.READ_ONLY: _A,
        SideEffect.REVERSIBLE: _V,
        SideEffect.IRREVERSIBLE: _V,
        SideEffect.MONEY: _V,
    },
    AutonomyGrade.COPILOT: {
        SideEffect.READ_ONLY: _A,
        SideEffect.REVERSIBLE: _F,
        SideEffect.IRREVERSIBLE: _F,
        SideEffect.MONEY: _F,
    },
    AutonomyGrade.WORKFLOW: {
        SideEffect.READ_ONLY: _A,
        SideEffect.REVERSIBLE: _A,
        SideEffect.IRREVERSIBLE: _F,
        SideEffect.MONEY: _F,
    },
    AutonomyGrade.MANDATE: {
        SideEffect.READ_ONLY: _A,
        SideEffect.REVERSIBLE: _A,
        SideEffect.IRREVERSIBLE: _F,
        SideEffect.MONEY: _L,
    },
    AutonomyGrade.EXCEPTION_ONLY: {
        SideEffect.READ_ONLY: _A,
        SideEffect.REVERSIBLE: _A,
        SideEffect.IRREVERSIBLE: _A,
        SideEffect.MONEY: _L,
    },
}
"""Grundmatrix (Grad × Seiteneffekt), Quelle für Tabelle in README und Docstring."""

_CELL_TEXT = {_A: "erlaubt", _F: "Freigabe", _V: "verboten", _L: "bis Limit"}


def format_amount(amount: Decimal, currency: str = "CHF") -> str:
    """Schweizer Schreibweise: ``CHF 12'430.50``."""
    quantized = amount.quantize(Decimal("0.01"))
    sign = "-" if quantized < 0 else ""
    whole, _, cents = f"{abs(quantized):.2f}".partition(".")
    groups: list[str] = []
    while whole:
        groups.insert(0, whole[-3:])
        whole = whole[:-3]
    grouped = "'".join(groups)
    return f"{currency} {sign}{grouped}.{cents}"


@dataclass(frozen=True)
class PolicyDecision:
    decision: Decision
    reason: str
    rule: str
    amount: Decimal | None = None

    def to_json(self) -> JSONObject:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "rule": self.rule,
            "amount": None if self.amount is None else str(self.amount),
        }


@dataclass(frozen=True)
class Policy:
    """Entscheidet pro Werkzeugaufruf. Standard: Grad 02 (Copilot)."""

    grade: AutonomyGrade = AutonomyGrade.COPILOT
    money_limit: Decimal | None = None
    currency: str = "CHF"
    require_approval: frozenset[str] = field(default_factory=frozenset)
    deny: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.money_limit is not None:
            limit = Decimal(str(self.money_limit))
            if not limit.is_finite() or limit < 0:
                raise ValueError("money_limit muss eine endliche Zahl ≥ 0 sein")
            object.__setattr__(self, "money_limit", limit)
        object.__setattr__(self, "require_approval", frozenset(self.require_approval))
        object.__setattr__(self, "deny", frozenset(self.deny))

    def decide(self, tool: Tool[..., Any], arguments: Mapping[str, Any]) -> PolicyDecision:
        """Entscheidung für einen Aufruf mit bereits validierten Argumenten."""
        label = self.grade.label
        if tool.name in self.deny:
            return PolicyDecision(
                Decision.DENY, f"'{tool.name}' steht auf der Sperrliste", "liste:deny"
            )
        cell = MATRIX[self.grade][tool.side_effect]
        amount = tool.amount(arguments) if tool.side_effect is SideEffect.MONEY else None
        effect = tool.side_effect.value
        if cell == _V:
            result = PolicyDecision(
                Decision.DENY,
                f"Grad {label}: Werkzeuge mit Wirkung '{effect}' sind nicht erlaubt",
                "grad-matrix",
                amount,
            )
        elif cell == _F:
            result = PolicyDecision(
                Decision.REQUIRE_APPROVAL,
                f"Grad {label}: Wirkung '{effect}' braucht eine Freigabe",
                "grad-matrix",
                amount,
            )
        elif cell == _L:
            result = self._mandate(amount, label)
        else:
            result = PolicyDecision(
                Decision.ALLOW, f"Grad {label}: Wirkung '{effect}' erlaubt", "grad-matrix", amount
            )
        if result.decision is Decision.ALLOW and tool.name in self.require_approval:
            return PolicyDecision(
                Decision.REQUIRE_APPROVAL,
                f"'{tool.name}' verlangt immer eine Freigabe",
                "liste:require_approval",
                amount,
            )
        return result

    def _mandate(self, amount: Decimal | None, label: str) -> PolicyDecision:
        if self.money_limit is None:
            return PolicyDecision(
                Decision.REQUIRE_APPROVAL,
                f"Grad {label}: kein Mandatslimit konfiguriert",
                "mandat:kein-limit",
                amount,
            )
        limit_text = format_amount(self.money_limit, self.currency)
        if amount is None:
            return PolicyDecision(
                Decision.REQUIRE_APPROVAL,
                "Betrag unbekannt – Mandatslimit nicht prüfbar",
                "mandat:betrag-unbekannt",
                amount,
            )
        if amount < 0:
            return PolicyDecision(
                Decision.REQUIRE_APPROVAL, "negativer Betrag", "mandat:betrag-negativ", amount
            )
        if amount > self.money_limit:
            return PolicyDecision(
                Decision.REQUIRE_APPROVAL,
                f"{format_amount(amount, self.currency)} über Mandatslimit {limit_text}",
                "mandat:ueber-limit",
                amount,
            )
        return PolicyDecision(
            Decision.ALLOW,
            f"{format_amount(amount, self.currency)} innerhalb Mandatslimit {limit_text}",
            "mandat:innerhalb",
            amount,
        )

    def to_json(self) -> JSONObject:
        return {
            "grade": int(self.grade),
            "grade_label": self.grade.label,
            "money_limit": None if self.money_limit is None else str(self.money_limit),
            "currency": self.currency,
            "require_approval": [*map(str, sorted(self.require_approval))],
            "deny": [*map(str, sorted(self.deny))],
        }

    @staticmethod
    def matrix_markdown() -> str:
        """Die Grundmatrix als Markdown-Tabelle (README wird dagegen geprüft)."""
        effects = [
            SideEffect.READ_ONLY,
            SideEffect.REVERSIBLE,
            SideEffect.IRREVERSIBLE,
            SideEffect.MONEY,
        ]
        lines = [
            "| Grad | lesend | umkehrbar | irreversibel | Geld |",
            "|---|---|---|---|---|",
        ]
        for grade in AutonomyGrade:
            cells = " | ".join(_CELL_TEXT[MATRIX[grade][e]] for e in effects)
            lines.append(f"| {grade.label} | {cells} |")
        return "\n".join(lines)
