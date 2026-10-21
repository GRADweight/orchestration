"""Observer — prüft das Ergebnis, gibt "erfolgreich / falsch / unsicher" (M4).

Der Observer übersetzt ein `ToolResult` in einen `Observation`.
Der Default-Observer ist regelbasiert:
- ok=True + Value vorhanden         → OK (erfolgreich)
- ok=False + Timed-Out               → UERWARTET (unerwartet, Zeitmangel)
- ok=False + anderer Fehler         → UERWARTET
- ok=True aber leerer Wert          → UNSICHER (Signal zu schwach)

`Confidence`:
- OK:        1.0 wenn Value nicht-leer, sonst 0.5
- UERWARTET: 0.8 (klar fehlerhaft)
- UNSICHER:  0.3 (schwach)

Für Produktion: `LLMObserver` (gw_shared_lib.inference.chat)
urteilt semantisch. Im Testpfad: `RuleObserver` (kein Netz).
"""
from __future__ import annotations

from .types import Observation, ObservationStatus, Task, ToolResult

from gw_shared_lib import StubClient  # type: ignore

import re


class RuleObserver:
    """Regelbasierter Observer (Testpfad, kein Netz)."""

    _EMPTY = re.compile(r"^\s*[\{\[].*$")

    def observe(self, result: ToolResult, task: Task) -> Observation:
        if result.timed_out:
            return Observation(
                status=ObservationStatus.UERWARTET,
                confidence=0.8,
                reason=f"Zeitlimit ({task.id}): {result.error}",
                task_id=task.id,
            )
        if not result.ok:
            return Observation(
                status=ObservationStatus.UERWARTET,
                confidence=0.8,
                reason=f"Fehler ({task.id}): {result.error}",
                task_id=task.id,
            )
        val = result.value
        is_empty = val is None or val == "" or (isinstance(val, str) and val.strip() == "")
        if is_empty:
            return Observation(
                status=ObservationStatus.UNSICHER,
                confidence=0.3,
                reason=f"Kein Ergebnis ({task.id}): Wert leer oder nicht-gefunden",
                task_id=task.id,
            )
        return Observation(
            status=ObservationStatus.OK,
            confidence=1.0,
            reason=f"Erfolgreich ({task.id}): {result.tool} lieferte ein Ergebnis",
            task_id=task.id,
        )


class LLMObserver:
    """LLM-Observer (Produktion): semantische Urteile via gw_shared_lib.inference.

    `caller` = (system, user) → answer. `caller=None` → RuleObserver
    (fail-soft, kein Netz).
    """

    SYSTEM = (
        "Du beurteilst ein Tool-Ergebnis. Antworte exakt mit einer Zeile: "
        "'OK', 'UERWARTET', oder 'UNSICHER', gefolgt von '; ' + kurzer Begründung."
    )

    def __init__(self, caller=None, stub: StubClient | None = None) -> None:
        self._caller = caller
        self._stub = stub or StubClient()
        self._rule = RuleObserver()

    def observe(self, result: ToolResult, task: Task) -> Observation:
        if self._caller is None:
            return self._rule.observe(result, task)
        try:
            prompt = (
                f"Tool: {result.tool}\nErgebnis: {result.value}\n"
                f"Task: {task.text}\n"
                f"Beurteilung (OK/UERWARTET/UNSICHER; Begründung):"
            )
            reply = self._caller(self.SYSTEM, prompt)
            first = reply.strip().upper().split(";")[0].strip()
            if first == "UERWARTET":
                st = ObservationStatus.UERWARTET
            elif first == "UNSICHER":
                st = ObservationStatus.UNSICHER
            else:
                st = ObservationStatus.OK  # Default: ok
            reason = reply.strip() if reply.strip() else "LLM-Beurteilung"
            return Observation(status=st, confidence=0.7, reason=reason, task_id=task.id)
        except Exception:
            return self._rule.observe(result, task)
