"""Audit-Trail — lückenlos append-only (wer/was/wann).

Baustein 5 aus `docs/praxis-fit-review.md`: "Wer hat was wann gesehen/
verändert? Schweizer KMU brauchen lückenlose Audit-Trails (v. a. regulierte
Branchen: Treuhand, Versicherung, Gesundheit)."

Design:
- Append-only: es gibt NUR `log()`. Keine UPDATE/DELETE-Methode.
- Null Dependencies (nur stdlib sqlite3) — läuft offline, kein Netzwerk.
- Thread-sicher über ein Lock (kein Per-Thread-Conn-Problem bei in-memory-DBs).
- `export()` liefert CSV für Revision/Freibeweis.
"""

from __future__ import annotations

import csv
import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    mandant TEXT NOT NULL,
    prototyp TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    details TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_audit_mandant ON audit (mandant);
CREATE INDEX IF NOT EXISTS idx_audit_prototyp ON audit (prototyp);
"""


def _jetzt() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


class AuditTrail:
    """Lückenloses, append-only Log. Die einzige Schreibleistung ist `log()`."""

    def __init__(self, dbpath: str | Path = ":memory:"):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(dbpath), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- Schreiben (die EINTIGE) ------------------------------------------
    def log(
        self,
        *,
        mandant: str,
        prototyp: str,
        actor: str,
        action: str,
        details: dict | None = None,
    ) -> int:
        """Einen Eintrag anhängen. Gibt die neue Zeile-ID zurück.

        `actor` = wer (Mensch: Vorname/Zugang, System: 'system'/'llm').
        `details` = freie, aber maschinenlesbare JSON-Beilage (z. B. Aktion-Text,
        alter/neuer Status, ID der betroffener Aktion).
        """
        details_json = json.dumps(details or {}, ensure_ascii=False, sort_keys=True)
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO audit (ts, mandant, prototyp, actor, action, details) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (_jetzt(), mandant, prototyp, actor, action, details_json),
            )
            self._conn.commit()
            new_id = cur.lastrowid
            if new_id is None:
                raise RuntimeError("INSERT lieferte keine Zeile-ID")
            return int(new_id)

    # -- Lesen -------------------------------------------------------------
    def query(
        self,
        *,
        mandant: str | None = None,
        prototyp: str | None = None,
        actor: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Letzte Einträge (neueste zuerst) mit optionalen Filtern."""
        where: list[str] = []
        params: list[object] = []
        if mandant is not None:
            where.append("mandant = ?")
            params.append(mandant)
        if prototyp is not None:
            where.append("prototyp = ?")
            params.append(prototyp)
        if actor is not None:
            where.append("actor = ?")
            params.append(actor)
        sql = "SELECT id, ts, mandant, prototyp, actor, action, details FROM audit"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        out = []
        for rid, ts, man, prot, act, action, details in rows:
            out.append(
                {
                    "id": rid,
                    "ts": ts,
                    "mandant": man,
                    "prototyp": prot,
                    "actor": act,
                    "action": action,
                    "details": json.loads(details) if details else {},
                }
            )
        return out

    def zähler(self, *, mandant: str | None = None, prototyp: str | None = None) -> int:
        """Wie viele Einträge (für KPI/Revision-Prüfung 'lückenlos')."""
        where: list[str] = []
        params: list[object] = []
        if mandant is not None:
            where.append("mandant = ?")
            params.append(mandant)
        if prototyp is not None:
            where.append("prototyp = ?")
            params.append(prototyp)
        sql = "SELECT COUNT(*) FROM audit"
        if where:
            sql += " WHERE " + " AND ".join(where)
        with self._lock:
            return int(self._conn.execute(sql, params).fetchone()[0])

    def export(self, ziel: str | Path, *, mandant: str | None = None) -> Path:
        """Alle Einträge (alt->neu) als CSV. Für Revision/Freibeweis."""
        zeile = self.query(mandant=mandant, limit=10**9)
        zeile.reverse()
        path = Path(ziel)
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "ts", "mandant", "prototyp", "actor", "action", "details"])
            for r in zeile:
                w.writerow(
                    [
                        r["id"],
                        r["ts"],
                        r["mandant"],
                        r["prototyp"],
                        r["actor"],
                        r["action"],
                        json.dumps(r["details"], ensure_ascii=False),
                    ]
                )
        return path

    def schliesse(self) -> None:
        with self._lock:
            self._conn.close()
