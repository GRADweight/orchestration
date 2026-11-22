"""Fehlerhierarchie der Runtime.

Alle Fehler, die die Bibliothek selbst auslöst, erben von
:class:`OrchestrationError`. :class:`TransientToolError` ist die einzige
Ausnahme, die *Werkzeuge* auslösen, um einen wiederholbaren Fehler zu melden.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


class OrchestrationError(Exception):
    """Basisklasse für alle Fehler dieser Bibliothek."""


class ToolDefinitionError(OrchestrationError):
    """Ein Werkzeug ist fehlerhaft deklariert (z.B. nicht unterstützter Typ)."""


@dataclass(frozen=True)
class ValidationIssue:
    """Ein einzelner Validierungsbefund mit JSON-Pfad (z.B. ``$.betrag``)."""

    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


class ArgumentValidationError(OrchestrationError):
    """Argumente eines Werkzeugaufrufs verletzen das JSON-Schema."""

    def __init__(self, tool: str, issues: Sequence[ValidationIssue]) -> None:
        self.tool = tool
        self.issues = tuple(issues)
        listing = "; ".join(str(i) for i in self.issues)
        super().__init__(f"Ungültige Argumente für '{tool}': {listing}")


class SandboxUnavailableError(OrchestrationError):
    """Eine verlangte Isolation kann auf dieser Plattform nicht erzwungen werden."""


class ApprovalError(OrchestrationError):
    """Unzulässige Operation auf der Freigabe-Warteschlange."""


class TransientToolError(Exception):
    """Von Werkzeugen auszulösen, wenn ein erneuter Versuch sinnvoll ist.

    Wiederholt wird nur, wenn das Werkzeug ``retries > 0`` deklariert *und*
    lesend oder idempotent ist (siehe :func:`gradweight_orchestration.tool`).
    """
