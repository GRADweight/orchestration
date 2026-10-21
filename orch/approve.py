"""Approval-Gate + Human-in-the-Loop (M7).

Baut auf dem CANONICAL-Governance-Kern auf (gw_core.approval):
  - ApprovalGate.vorschlag()  → eine schreibende Aktion als 'offen'
  - ApprovalGate.freigabe()   → Mensch gibt frei (-> 'freigegeben')
  - ApprovalGate.anpassen()   → Mensch ändert + gibt frei (-> 'angepasst')
  - ApprovalGate.zurueck()    → Mensch weist zurück (-> 'zurueckgestellt')

Jede Entscheidung landet im gw_core.AuditTrail (wer/was/wann/Grund).
Die Freigabe ist der strukturelle Kettendurchbruch (Grad 2): erst
nach Freigabe führt die App die Aktion aus.

CANONICAL: autonomy/packages/gw_core @ 8982df27 (audited subset,
            vendor/gw_core_subset, siehe SNAPSHOT.txt).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from gw_core_subset import ApprovalQueue, AuditTrail, FreigabePolicy  # type: ignore

_STATUS_FREIGEGEBEN = "freigegeben"
_STATUS_ANGEPASST = "angepasst"
_STATUS_ZURUECK = "zurueckgestellt"


@dataclass
class FreigabeEntscheid:
    """Ergebnis einer Freigabe-Entscheidung (für Tests / Beobachter)."""

    status: str
    aktion_id: int
    ausgefuehrt_aktiv: bool  # True = die Aktion darf jetzt ausgeführt werden

    @classmethod
    def von_aktion(cls, aktion) -> "FreigabeEntscheid":
        ausf = aktion.status in (_STATUS_FREIGEGEBEN, _STATUS_ANGEPASST)
        return cls(status=aktion.status, aktion_id=aktion.id, ausgefuehrt_aktiv=ausf)


@dataclass
class ApprovalGate:
    """Human-in-the-Loop Gate (M7). Kapselt ApprovalQueue + AuditTrail.

    Typischer Lauf (Grad 2):
        gate = ApprovalGate(mandant="K1", prototyp="02-rechnung")
        aid = gate.vorschlag(titel="Rechnung buchen", payload={...})
        # ... Mensch schaut, entscheidet ...
        gate.freigabe(aid, actor="cfo")   # -> ausgefuehrt_aktiv = True
    """

    mandant: str
    prototyp: str
    _queue: ApprovalQueue = field(repr=False)
    _audit: AuditTrail = field(repr=False)

    def __init__(
        self,
        *,
        mandant: str,
        prototyp: str,
        policy: FreigabePolicy | None = None,
    ) -> None:
        self.mandant = mandant
        self.prototyp = prototyp
        self._audit = AuditTrail()  # in-memory (Demos/Tests)
        # Für Produktion: AuditTrail(".audit/approval.db") (persistiert)
        self._queue = ApprovalQueue(audit=self._audit, policy=policy)

    def vorschlag(
        self,
        *,
        titel: str,
        payload: dict,
        vorgeschlagen_von: str = "llm",
        freigabe_nur_an: str | None = None,
        anmerkungen: str = "",
    ) -> int:
        """Reicht eine schreibende Aktion als Vorschlag ein (Status 'offen')."""
        return self._queue.vorschlage(
            mandant=self.mandant,
            prototyp=self.prototyp,
            titel=titel,
            payload=payload,
            vorgeschlagen_von=vorgeschlagen_von,
            freigabe_nur_an=freigabe_nur_an,
            anmerkungen=anmerkungen,
        )

    def freigabe(
        self,
        aktion_id: int,
        *,
        actor: str,
        rolle: str | None = None,
        anmerkungen: str = "",
    ) -> FreigabeEntscheid:
        """Mensch gibt eine Aktion frei (-> 'freigegeben')."""
        aktion = self._queue.freigabe(
            aktion_id, actor=actor, rolle=rolle, anmerkungen=anmerkungen
        )
        return FreigabeEntscheid.von_aktion(aktion)

    def anpassen(
        self,
        aktion_id: int,
        *,
        actor: str,
        neue_payload: dict,
        rolle: str | None = None,
        anmerkungen: str = "",
    ) -> FreigabeEntscheid:
        """Mensch ändert + gibt frei (-> 'angepasst')."""
        aktion = self._queue.anpassen(
            aktion_id,
            actor=actor,
            neue_payload=neue_payload,
            rolle=rolle,
            anmerkungen=anmerkungen,
        )
        return FreigabeEntscheid.von_aktion(aktion)

    def zurueck(
        self,
        aktion_id: int,
        *,
        actor: str,
        begruendung: str = "",
    ) -> FreigabeEntscheid:
        """Mensch weist zurück (-> 'zurueckgestellt'). Wird NICHT ausgeführt."""
        aktion = self._queue.zurueckstellen(
            aktion_id, actor=actor, begruendung=begruendung
        )
        # Die Hülle liefert den CANONICAL-Status 'zurueckgestellt' (gw_core).
        return FreigabeEntscheid(
            status=aktion.status,
            aktion_id=aktion.id,
            ausgefuehrt_aktiv=False,
        )

    def einseiter(self) -> str:
        """KPI: was hat das System getan (menschliche Änderquote)."""
        return self._queue.einseiter(mandant=self.mandant)

    @property
    def audit(self) -> AuditTrail:
        return self._audit

    @property
    def queue(self) -> ApprovalQueue:
        return self._queue
