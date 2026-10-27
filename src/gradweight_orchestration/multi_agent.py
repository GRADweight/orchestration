"""Supervisor-Muster: ein Agent, der Spezialisten mit eigenen Werkzeugen beauftragt.

Die Spezialisten sind gewöhnliche :class:`~gradweight_orchestration.Agent`-Instanzen
mit eigenem Werkzeugsatz, eigener Policy und eigenem Budget. Der Supervisor sieht
sie als Werkzeuge ``delegieren_<name>`` und hat selbst keinen Zugriff auf deren
Werkzeuge. Governance greift dort, wo gehandelt wird: beim Spezialisten.

Für einen zusammenhängenden Trace und ein gemeinsames Audit-Log sollten alle
Agenten dieselben ``tracer``-, ``audit``- und ``approvals``-Objekte erhalten.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .agent import Agent
from .llm.types import ChatModel

__all__ = ["SUPERVISOR_INSTRUCTIONS", "Supervisor"]

SUPERVISOR_INSTRUCTIONS = (
    "Du koordinierst Spezialisten. Zerlege die Aufgabe, beauftrage für jeden Teil den "
    "zuständigen Spezialisten mit einem vollständigen Auftrag (alle Nummern und Beträge "
    "ausschreiben) und fasse die Ergebnisse knapp auf Deutsch zusammen. Erfinde keine "
    "Ergebnisse und wiederhole keinen abgelehnten oder gescheiterten Auftrag unverändert."
)


class Supervisor(Agent):
    """Agent, dessen Werkzeuge (auch) Spezialisten sind."""

    def __init__(
        self,
        *,
        name: str,
        llm: ChatModel,
        specialists: Sequence[Agent],
        instructions: str = SUPERVISOR_INSTRUCTIONS,
        **kwargs: Any,
    ) -> None:
        if not specialists:
            raise ValueError("Ein Supervisor braucht mindestens einen Spezialisten")
        super().__init__(
            name=name, llm=llm, delegates=specialists, instructions=instructions, **kwargs
        )

    @property
    def specialists(self) -> list[Agent]:
        return list(self._delegates.values())
