"""Planner — Task → Sub-Task-Liste (M4).

Die LLM-basierte Variante (gw_shared_lib.inference.chat) ist
`LLMPlanner`. Für den Testpfad (kein Netz, kein 7B nötig) ist
`RulePlanner` der Default: zerlegt einen aufzählenden Text
(„A, B und C" / „erst A, dann B") in Sub-Steps. Zyklerkennung:
wenn zwei aufeinanderfolgende Sub-Steps den gleichen Schritt
beschreiben (gleiche Normalisierung), markiert der Planner den
Plan als `cyclic` — der Runner stoppt den Schleife.

CANONICAL offline: gw_shared_lib.inference.StubClient.
"""
from __future__ import annotations

import re
from typing import Optional

from .types import Plan, SubTask

from gw_shared_lib import StubClient  # type: ignore  (vendor/)

_SPLIT = re.compile(
    r"\s*,\s*|\s*;\s*|\s+und\s+|\s+and\s+",
    re.IGNORECASE,
)


def _split_text(text: str) -> list[str]:
    pieces = _SPLIT.split(text)
    return [p.strip() for p in pieces if p.strip()]


def _normalise(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


class RulePlanner:
    """Regelbasierter Planner (Testpfad, kein LLM, kein Netz)."""

    def plan(self, task_text: str) -> Plan:
        raw = _split_text(task_text)
        if not raw:
            return Plan(text=task_text, subtasks=[], cyclic=False)

        norm_seen: list[str] = []
        subtasks: list[SubTask] = []
        cyclic = False
        for piece in raw:
            n = _normalise(piece)
            if n in norm_seen:
                cyclic = True
            else:
                norm_seen.append(n)
            subtasks.append(SubTask(text=piece))

        if not subtasks:
            # Kein Splitter-Treffer → ein einzelner Schritt
            subtasks = [SubTask(text=task_text)]
        return Plan(text=task_text, subtasks=subtasks, cyclic=cyclic)


class LLMPlanner:
    """LLM-Planner (Produktion): verwendet gw_shared_lib.inference.chat.

    `caller` = eine (system, user) → answer Funktion. `caller=None`
    fällt auf RulePlanner zurück (fail-soft, kein Netz).
    """

    SYSTEM = (
        "Du zerlegst eine Aufgabe in klare, hintereinander auszuführende "
        "Schritte. Antworte nur mit einer Aufzählung, ein pro Zeile, "
        "keine Erklärungen."
    )

    def __init__(self, caller=None, stub: Optional[StubClient] = None) -> None:
        self._caller = caller
        self._stub = stub or StubClient()
        self._rule = RulePlanner()

    def plan(self, task_text: str) -> Plan:
        if self._caller is None:
            return self._rule.plan(task_text)
        try:
            reply = self._caller(
                self.SYSTEM,
                f"Zerlege die Aufgabe in Schritte:\n{task_text}",
            )
            steps = [ln.strip() for ln in reply.splitlines() if ln.strip()]
            if not steps:
                return self._rule.plan(task_text)
            norm_seen: list[str] = []
            subtasks: list[SubTask] = []
            cyclic = False
            for s in steps:
                n = _normalise(s)
                if n in norm_seen:
                    cyclic = True
                else:
                    norm_seen.append(n)
                subtasks.append(SubTask(text=s))
            return Plan(text=task_text, subtasks=subtasks, cyclic=cyclic)
        except Exception:
            return self._rule.plan(task_text)
