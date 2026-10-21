"""Executor — führt ein Tool aus (mit Timeout, Retry, Audit) (M4).

Erhält eine `RouterDecision` (vom Router) und das SubTask-Text,
führt das gewählte Tool aus. Retry: bis `max_retries`-mal bei
ToolResult.ok == False. Bei TimeLimit → ToolResult.timed_out.
Audit: jede Ausführung (M7-Präparation) landet im gw_core Audit-Trail.

CANONICAL offline: gw_shared_lib.audit.AuditLog.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from .types import ToolResult, TaskStatus
from .tools import TimeLimit

from gw_shared_lib import audit  # type: ignore  (vendor/)


class Executor:
    """Führt ein Tool aus (M4). Retry + Timeout + Audit."""

    def __init__(
        self,
        registry=None,
        *,
        max_retries: int = 2,
        audit_log=None,
    ) -> None:
        self._registry = registry
        self._max_retries = max_retries
        self._audit = audit_log

    def execute(
        self,
        decision,
        task_text: str,
        *,
        timeout_s: float | None = None,
        mandant: str = "demo",
        prototyp: str = "orchestration",
    ) -> ToolResult:
        """Führt decision.tool.call aus (mit Retry + Timeout + Audit)."""
        spec = decision.tool
        fn = spec.call
        budget = TimeLimit(
            timeout_s if timeout_s is not None else spec.timeout_s
        )

        last: ToolResult = ToolResult(
            ok=False, tool=spec.name, error="nicht ausgeführt"
        )
        for attempt in range(self._max_retries + 1):
            # Budget-Prüfung vor dem Aufruf
            if budget.over():
                last = ToolResult(
                    ok=False,
                    tool=spec.name,
                    error="Zeitlimit erreicht",
                    retries=attempt,
                    timed_out=True,
                )
                break

            start = time.monotonic()
            timed_out = False
            try:
                val = fn()
                result = ToolResult(
                    ok=True,
                    value=val,
                    tool=spec.name,
                    retries=attempt,
                )
            except Exception as e:  # noqa: BLE001 (intentional: Tool-Fehler)
                result = ToolResult(
                    ok=False,
                    tool=spec.name,
                    error=repr(e),
                    retries=attempt,
                )

            # After-caller Zeitmessung: ein einzelner, zu langes Tool
            # wird so trotzdem als TimeOut markiert (KMU-Sweet: kein
            # Signal, kein Thread — nur Zeitvergleich nach dem Aufruf).
            if time.monotonic() - start > budget._limit and budget._limit > 0:
                timed_out = True
                last = ToolResult(
                    ok=False,
                    tool=spec.name,
                    error="Zeitlimit überschritten",
                    retries=attempt,
                    timed_out=True,
                )
                break

            last = result
            if result.ok and not timed_out:
                break

        # Audit (M7-Präparation): jede Ausführung protokolliert
        if self._audit is not None:
            self._audit.append_dict(
                {
                    "action": "tool_ergebnis",
                    "actor": "executor",
                    "output": last.value if last.ok else "",
                    "reason": "" if last.ok else last.error,
                    "category": "tool",
                    "data": {
                        "tool": last.tool,
                        "ok": last.ok,
                        "retries": last.retries,
                        "task": task_text,
                        "timed_out": last.timed_out,
                    },
                    "fall_id": f"{mandant}:{prototyp}",
                }
            )
        return last
