"""Approval-Queue — Freigabe / Anpassen / Zurückstellen je Aktion.

Baustein aus `docs/praxis-fit-review.md` (Lücke 2/5) + Kern der
Autonomie-Mappe (`docs/26-autonomie-mappe.md`): "Freigabe je Aktion" (Grad 2)
fehlt bei ALLEN 25 Prototypen. Dies ist der gemeinsame Baustein, der jeden
Prototypen von "Grad 1 (Vorschlag)" auf "Grad 2 (Freigabe)" hebt: die KI
schlägt eine schreibende Aktion vor, ein Mensch gibt sie frei, ändert sie
oder weist sie zurück. Jede Entscheidung landet im Audit-Trail (GB-08/GB-11).

Design:
- Schreibende Aktionen NUR mit Freigabe (GB-03: bei Grad 2 bricht der Mensch
  die Kette — diese Queue IST der strukturelle Kettendurchbruch).
- Rechte-Check: Aktion kann auf bestimmte Freigabeperson begrenzt werden
  und/oder es greift eine zentrale Rollen-Policy (gw_core.authz, GB-08
  Least Privilege). Beide Ebenen müssen durch — AND-Logik.
- Null extra Dependencies (nur stdlib sqlite3 + gw_core.audit), Thread-sicher.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from gw_core_subset.audit import AuditTrail
from gw_core_subset.authz import FreigabePolicy

_STATUS_OFFEN = "offen"
_STATUS_FREIGEGEBEN = "freigegeben"
_STATUS_ANGEPASST = "angepasst"
_STATUS_ZURUECK = "zurueckgestellt"
_GULTIGE = {
    _STATUS_OFFEN,
    _STATUS_FREIGEGEBEN,
    _STATUS_ANGEPASST,
    _STATUS_ZURUECK,
}


def _jetzt() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


@dataclass(frozen=True)
class Aktion:
    """Eine schreibende Aktion, die die KI VORSCHLÄGT (Status: offen)."""

    id: int
    mandant: str
    prototyp: str
    titel: str
    payload: dict
    status: str
    vorgeschlagen_von: str
    freigabe_nur_an: str | None
    anmerkungen: str
    angelegt_um: str
    erledigt_um: str | None = None
    erledigt_von: str | None = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS aktionen (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mandant TEXT NOT NULL,
    prototyp TEXT NOT NULL,
    titel TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'offen',
    vorgeschlagen_von TEXT NOT NULL,
    freigabe_nur_an TEXT,
    anmerkungen TEXT NOT NULL DEFAULT '',
    angelegt_um TEXT NOT NULL,
    erledigt_um TEXT,
    erledigt_von TEXT
);
CREATE INDEX IF NOT EXISTS idx_aktionen_mandant ON aktionen (mandant);
CREATE INDEX IF NOT EXISTS idx_aktionen_status ON aktionen (status);
"""


class ApprovalQueue:
    """Freigabe-Workflow im Zentrum einer 'Grad-2'-Installation.

    Typischer Lauf:
      q = ApprovalQueue(audit=audit)
      aid = q.vorschlage(mandant="K1", prototyp="02-rechnung",
                         titel="Rechnung RE-2026-0042 buchen", payload={...},
                         vorgeschlagen_von="llm")
      q.freigabe(aid, actor="ai-lead")            # App-Code führt sie danach aus
      # ODER
      q.zurueckstellen(aid, actor="ai-lead", begruendung="Betrag falsch")
    """

    def __init__(
        self,
        dbpath: str | Path | None = None,
        audit: AuditTrail | None = None,
        policy: FreigabePolicy | None = None,
    ):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(dbpath) if dbpath is not None else ":memory:",
            check_same_thread=False,
        )
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        # Standard-Trail falls keines übergeben wird (Demos/Tests)
        self._audit = audit or AuditTrail()
        # Optionale zentrale Rollen-Policy (Default: alle Rollen erlaubt —
        # die 'freigabe_nur_an'-Ebenenprüfung bleibt die einzige Bremse).
        self._policy = policy or FreigabePolicy()

    # ------------------------------------------------------------------ #
    # Vorschlag                                                          #
    # ------------------------------------------------------------------ #
    def vorschlage(
        self,
        *,
        mandant: str,
        prototyp: str,
        titel: str,
        payload: dict,
        vorgeschlagen_von: str = "llm",
        freigabe_nur_an: str | None = None,
        anmerkungen: str = "",
    ) -> int:
        """Eine schreibende Aktion als Vorschlag einreichen (Status 'offen')."""
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO aktionen (mandant, prototyp, titel, payload, "
                "status, vorgeschlagen_von, freigabe_nur_an, anmerkungen, "
                "angelegt_um) "
                "VALUES (?, ?, ?, ?, 'offen', ?, ?, ?, ?)",
                (
                    mandant,
                    prototyp,
                    titel,
                    payload_json,
                    vorgeschlagen_von,
                    freigabe_nur_an,
                    anmerkungen,
                    _jetzt(),
                ),
            )
            self._conn.commit()
            new_id = cur.lastrowid
            if new_id is None:
                raise RuntimeError("INSERT lieferte keine Zeile-ID")
        self._audit.log(
            mandant=mandant,
            prototyp=prototyp,
            actor=vorgeschlagen_von,
            action="aktion_vorgeschlagen",
            details={"aktion_id": int(new_id), "titel": titel},
        )
        return int(new_id)

    # ------------------------------------------------------------------ #
    # Entscheidungen                                                     #
    # ------------------------------------------------------------------ #
    def _entscheide(
        self,
        action_id: int,
        *,
        status: str,
        actor: str,
        rolle: str | None = None,
        anmerkungen: str = "",
        neue_payload: dict | None = None,
    ) -> Aktion:
        if status not in _GULTIGE:
            raise ValueError(f"Ungültiger Zielstatus: {status!r}")
        with self._lock:
            row = self._conn.execute(
                "SELECT mandant, prototyp, freigabe_nur_an, payload "
                "FROM aktionen WHERE id = ?",
                (action_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Keine Aktion mit id={action_id}")
            mandant, prototyp, nur_an, payload_json = row
            # Rechte-Check: Freigabe nur für erlaubte Person (GB-08: Least Privilege)
            if (
                nur_an
                and nur_an != actor
                and status in (_STATUS_FREIGEGEBEN, _STATUS_ANGEPASST, _STATUS_ZURUECK)
            ):
                raise PermissionError(
                    f"Nur '{nur_an}' darf Aktion {action_id} entscheiden — abgelehnt."
                )
        # Rollen-Policy (GB-08): schreibt nur — kein Rollenwechsel ohne Freigabe.
        # 'zurueckgestellt' ist immer erlaubt (jeder darf ablehnen — Beziehungsschutz),
        # 'freigegeben' und 'angepasst' brauchen eine gestattete Rolle.
        if rolle is not None and status in (_STATUS_FREIGEGEBEN, _STATUS_ANGEPASST):
            entscheid = self._policy.pruefe(rolle=rolle, prototyp=prototyp)
            if not entscheid.erlaubt:
                self._audit.log(
                    mandant=mandant,
                    prototyp=prototyp,
                    actor=actor,
                    action="entscheid_abgelehnt_policy",
                    details={"aktion_id": action_id, "grund": entscheid.begruendung},
                )
                raise PermissionError(entscheid.begruendung)
        neuer_payload = (
            json.dumps(neue_payload, ensure_ascii=False, sort_keys=True)
            if neue_payload is not None
            else payload_json
        )
        with self._lock:
            self._conn.execute(
                "UPDATE aktionen SET status=?, anmerkungen=?, "
                "erledigt_um=?, erledigt_von=?, payload=? WHERE id=?",
                (status, anmerkungen, _jetzt(), actor, neuer_payload, action_id),
            )
            self._conn.commit()
        self._audit.log(
            mandant=mandant,
            prototyp=prototyp,
            actor=actor,
            action=f"aktion_{status.replace('zurueckgestellt', 'zurueckgestellt')}",
            details={"aktion_id": action_id},
        )
        return self.lade(action_id)

    def freigabe(
        self, action_id: int, *, actor: str, rolle: str | None = None, anmerkungen: str = ""
    ) -> Aktion:
        """Aktion freigeben (-> 'freigegeben'). App-Code führt sie danach aus."""
        return self._entscheide(
            action_id, status=_STATUS_FREIGEGEBEN, actor=actor, rolle=rolle,
            anmerkungen=anmerkungen,
        )

    def anpassen(
        self,
        action_id: int,
        *,
        actor: str,
        rolle: str | None = None,
        neue_payload: dict,
        anmerkungen: str = "",
    ) -> Aktion:
        """Aktion ändern und freigeben (-> 'angepasst')."""
        return self._entscheide(
            action_id,
            status=_STATUS_ANGEPASST,
            actor=actor,
            rolle=rolle,
            anmerkungen=anmerkungen,
            neue_payload=neue_payload,
        )

    def zurueckstellen(
        self, action_id: int, *, actor: str, begruendung: str = ""
    ) -> Aktion:
        """Aktion zurückweisen (-> 'zurueckgestellt'). Wird NICHT ausgeführt.

        `begruendung` wird in Anmerkungen geschrieben (Kunden-Kommunikation).
        """
        return self._entscheide(
            action_id, status=_STATUS_ZURUECK, actor=actor, anmerkungen=begruendung
        )

    # ------------------------------------------------------------------ #
    # Lesen                                                              #
    # ------------------------------------------------------------------ #
    def lade(self, action_id: int) -> Aktion:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, mandant, prototyp, titel, payload, status, "
                "vorgeschlagen_von, freigabe_nur_an, anmerkungen, angelegt_um, "
                "erledigt_um, erledigt_von FROM aktionen WHERE id = ?",
                (action_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Keine Aktion mit id={action_id}")
        return self._zu_aktion(row)

    def offene(self, *, mandant: str | None = None) -> list[Aktion]:
        with self._lock:
            if mandant is None:
                rows = self._conn.execute(
                    "SELECT * FROM aktionen WHERE status='offen' ORDER BY id"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM aktionen WHERE status='offen' AND mandant=? "
                    "ORDER BY id",
                    (mandant,),
                ).fetchall()
        return [self._zu_aktion(r) for r in rows]

    @staticmethod
    def _zu_aktion(row) -> Aktion:
        (
            aid,
            mandant,
            prototyp,
            titel,
            payload_json,
            status,
            vorgeschlagen_von,
            freigabe_nur_an,
            anmerkungen,
            angelegt_um,
            erledigt_um,
            erledigt_von,
        ) = row
        return Aktion(
            id=aid,
            mandant=mandant,
            prototyp=prototyp,
            titel=titel,
            payload=json.loads(payload_json),
            status=status,
            vorgeschlagen_von=vorgeschlagen_von,
            freigabe_nur_an=freigabe_nur_an,
            anmerkungen=anmerkungen,
            angelegt_um=angelegt_um,
            erledigt_um=erledigt_um,
            erledigt_von=erledigt_von,
        )

    # ------------------------------------------------------------------ #
    # KPI-Einseiter (Baustein "was hat das System getan?")               #
    # ------------------------------------------------------------------ #
    def einseiter(self, *, mandant: str | None = None) -> str:
        """Einseiter: was hat das System getan. (Verkauft das Abo im Kunden-Konto.)"""
        with self._lock:
            where = []
            params: list[object] = []
            if mandant is not None:
                where.append("mandant=?")
                params.append(mandant)
            sql = "SELECT status, COUNT(*) FROM aktionen"
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += " GROUP BY status"
            counts = {s: int(n) for s, n in self._conn.execute(sql, params).fetchall()}
            total = sum(counts.values())
        z = lambda s: counts.get(s, 0)  # noqa: E731
        mensch_entscheid = z(_STATUS_ANGEPASST) + z(_STATUS_ZURUECK)
        quote = (
            (mensch_entscheid / total * 100) if total else 0.0
        )
        return (
            "# KPI-Einseiter — Approval-Queue\n\n"
            f"Mandant: {mandant or 'alle'}\n\n"
            f"- Offen (noch unentschieden): {z(_STATUS_OFFEN)}\n"
            f"- Freigegeben (unverändert): {z(_STATUS_FREIGEGEBEN)}\n"
            f"- Angepasst vom Menschen: {z(_STATUS_ANGEPASST)}\n"
            f"- Zurückgestellt: {z(_STATUS_ZURUECK)}\n"
            f"- **Total Aktionen: {total}**\n"
            f"- **Menschliche Änderquote (Anpassung+Rückstellung / Total): "
            f"{quote:.0f}%**\n"
        )

    def schliesse(self) -> None:
        with self._lock:
            self._conn.close()
