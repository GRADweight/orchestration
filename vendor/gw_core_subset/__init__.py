"""gw_core_subset — minimal GRADweight Governance-Kern fuer orchestration.

Re-exportiert aus GRADweight/autonomy/packages/gw_core/gw_core:
- audit.py: AuditTrail (luckenlos, append-only, wer/was/wann)
- authz.py: FreigabePolicy, FreigabeEntscheid (Rolle x Prototyp, Least Privilege)
- approval.py: ApprovalQueue (Freigabe/Anpassen/Zurueckstellen, sqlite3-backed)

CANONICAL-Ziel: GRADweight/autonomy @ 8982df27.
Dieses Vendor-Snapshot ist ein flaches Subset — nur die 3 Kernmodule,
die orchestration/approve.py wirklich braucht (keine httpx/bexio).
"""

__version__ = "0.1.0"

from .audit import AuditTrail
from .authz import FreigabePolicy, FreigabeEntscheid
from .approval import ApprovalQueue, Aktion

__all__ = [
    "AuditTrail",
    "FreigabePolicy",
    "FreigabeEntscheid",
    "ApprovalQueue",
    "Aktion",
]
