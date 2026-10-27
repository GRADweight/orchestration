"""Freigabe-Warteschlange (Human-in-the-Loop).

Verlangt die Policy eine Freigabe, legt der Agent hier einen Antrag an und
pausiert. Ein Mensch entscheidet – freigeben, mit geänderten Argumenten
freigeben oder ablehnen – und der Agent setzt den Lauf danach fort
(:meth:`gradweight_orchestration.Agent.resume`), auch in einem anderen Prozess.

Speicherung in SQLite (Standardbibliothek). Zustandswechsel sind atomar
(``UPDATE ... WHERE status = 'pending'``): ein Antrag wird genau einmal
entschieden, auch wenn zwei Personen gleichzeitig klicken. Agenten
(``by="agent:..."``) können nicht freigeben.
"""

from __future__ import annotations

import datetime as _dt
import enum
import json
import secrets
import sqlite3
import threading
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._json import JSONObject, JSONValue, canonical_json
from .audit import AuditLog
from .errors import ApprovalError

__all__ = ["ApprovalQueue", "ApprovalRequest", "ApprovalStatus"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    agent TEXT NOT NULL,
    call_id TEXT NOT NULL,
    tool TEXT NOT NULL,
    arguments TEXT NOT NULL,
    side_effect TEXT NOT NULL,
    amount TEXT,
    reason TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    decided_at TEXT,
    decided_by TEXT,
    note TEXT NOT NULL DEFAULT '',
    final_arguments TEXT
);
CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals (status);
CREATE INDEX IF NOT EXISTS idx_approvals_run ON approvals (run_id);
"""
_COLUMNS = (
    "id, run_id, agent, call_id, tool, arguments, side_effect, amount, reason, status, "
    "created_at, decided_at, decided_by, note, final_arguments"
)


class ApprovalStatus(enum.StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    MODIFIED = "approved_with_changes"
    REJECTED = "rejected"

    @property
    def approved(self) -> bool:
        return self in (ApprovalStatus.APPROVED, ApprovalStatus.MODIFIED)


@dataclass(frozen=True)
class ApprovalRequest:
    id: str
    run_id: str
    agent: str
    call_id: str
    tool: str
    arguments: JSONObject
    side_effect: str
    amount: str | None
    reason: str
    status: ApprovalStatus
    created_at: str
    decided_at: str | None = None
    decided_by: str | None = None
    note: str = ""
    final_arguments: JSONObject | None = None

    @property
    def effective_arguments(self) -> JSONObject:
        """Die Argumente, die ausgeführt werden (ggf. vom Menschen geändert)."""
        if self.status is ApprovalStatus.MODIFIED and self.final_arguments is not None:
            return self.final_arguments
        return self.arguments

    def to_json(self) -> JSONObject:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "agent": self.agent,
            "call_id": self.call_id,
            "tool": self.tool,
            "arguments": self.arguments,
            "side_effect": self.side_effect,
            "amount": self.amount,
            "reason": self.reason,
            "status": self.status.value,
            "created_at": self.created_at,
            "decided_at": self.decided_at,
            "decided_by": self.decided_by,
            "note": self.note,
            "final_arguments": self.final_arguments,
        }


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


def _row_to_request(row: tuple[Any, ...]) -> ApprovalRequest:
    (
        rid,
        run_id,
        agent,
        call_id,
        tool,
        arguments,
        side_effect,
        amount,
        reason,
        status,
        created_at,
        decided_at,
        decided_by,
        note,
        final_arguments,
    ) = row
    return ApprovalRequest(
        id=rid,
        run_id=run_id,
        agent=agent,
        call_id=call_id,
        tool=tool,
        arguments=json.loads(arguments),
        side_effect=side_effect,
        amount=amount,
        reason=reason,
        status=ApprovalStatus(status),
        created_at=created_at,
        decided_at=decided_at,
        decided_by=decided_by,
        note=note,
        final_arguments=json.loads(final_arguments) if final_arguments else None,
    )


class ApprovalQueue:
    """Warteschlange für Freigabe-Anträge. ``path=None``: nur im Speicher."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        audit: AuditLog | None = None,
        clock: Callable[[], str] = _now,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._conn = sqlite3.connect(
            str(path) if path is not None else ":memory:",
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.executescript(_SCHEMA)
        self._finalizer = weakref.finalize(self, self._conn.close)
        self._lock = threading.Lock()
        self._audit = audit
        self._clock = clock
        self._new_id = id_factory or (lambda: f"apr_{secrets.token_hex(6)}")

    def close(self) -> None:
        with self._lock:
            self._finalizer()

    def __enter__(self) -> ApprovalQueue:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def submit(
        self,
        *,
        run_id: str,
        agent: str,
        call_id: str,
        tool: str,
        arguments: Mapping[str, JSONValue],
        side_effect: str,
        amount: str | None,
        reason: str,
    ) -> ApprovalRequest:
        """Legt einen Antrag an (Status ``pending``)."""
        request_id = self._new_id()
        with self._lock:
            self._conn.execute(
                f"INSERT INTO approvals ({_COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",  # noqa: S608 - feste Spaltenliste
                (
                    request_id,
                    run_id,
                    agent,
                    call_id,
                    tool,
                    canonical_json(dict(arguments)),
                    side_effect,
                    amount,
                    reason,
                    ApprovalStatus.PENDING.value,
                    self._clock(),
                    None,
                    None,
                    "",
                    None,
                ),
            )
        return self.get(request_id)

    def get(self, request_id: str) -> ApprovalRequest:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {_COLUMNS} FROM approvals WHERE id = ?",  # noqa: S608 - feste Spalten
                (request_id,),
            ).fetchone()
        if row is None:
            raise ApprovalError(f"Kein Antrag mit id={request_id!r}")
        return _row_to_request(row)

    def pending(self, *, run_id: str | None = None) -> list[ApprovalRequest]:
        query = f"SELECT {_COLUMNS} FROM approvals WHERE status = 'pending'"  # noqa: S608
        params: tuple[str, ...] = ()
        if run_id is not None:
            query += " AND (run_id = ? OR run_id LIKE ?)"
            params = (run_id, f"{run_id}/%")
        with self._lock:
            rows = self._conn.execute(query + " ORDER BY created_at, id", params).fetchall()
        return [_row_to_request(r) for r in rows]

    def approve(self, request_id: str, *, by: str, note: str = "") -> ApprovalRequest:
        return self._decide(request_id, ApprovalStatus.APPROVED, by=by, note=note)

    def approve_with_changes(
        self, request_id: str, *, by: str, arguments: Mapping[str, JSONValue], note: str = ""
    ) -> ApprovalRequest:
        return self._decide(
            request_id, ApprovalStatus.MODIFIED, by=by, note=note, final_arguments=dict(arguments)
        )

    def reject(self, request_id: str, *, by: str, reason: str) -> ApprovalRequest:
        if not reason.strip():
            raise ApprovalError("Eine Ablehnung braucht eine Begründung")
        return self._decide(request_id, ApprovalStatus.REJECTED, by=by, note=reason)

    def _decide(
        self,
        request_id: str,
        status: ApprovalStatus,
        *,
        by: str,
        note: str,
        final_arguments: JSONObject | None = None,
    ) -> ApprovalRequest:
        actor = by.strip()
        if not actor:
            raise ApprovalError("Entscheidende Person fehlt (by=...)")
        if actor.lower().startswith("agent:"):
            raise ApprovalError("Agenten dürfen Freigaben nicht selbst erteilen")
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE approvals SET status = ?, decided_at = ?, decided_by = ?, note = ?, "
                "final_arguments = ? WHERE id = ? AND status = 'pending'",
                (
                    status.value,
                    self._clock(),
                    actor,
                    note,
                    canonical_json(final_arguments) if final_arguments is not None else None,
                    request_id,
                ),
            )
            changed = cursor.rowcount
        if changed != 1:
            current = self.get(request_id)  # wirft, wenn es den Antrag nicht gibt
            raise ApprovalError(
                f"Antrag {request_id} ist bereits entschieden ({current.status.value})"
            )
        decided = self.get(request_id)
        if self._audit is not None:
            self._audit.append(
                "approval_decided",
                {
                    "approval_id": decided.id,
                    "tool": decided.tool,
                    "status": status.value,
                    "by": actor,
                    "note": note,
                    "final_arguments": final_arguments,
                },
                run_id=decided.run_id,
                agent=decided.agent,
            )
        return decided
