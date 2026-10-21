"""Freigabe-Autorisierung (AuthZ) — welche Rollen dürfen auf welchem
Prototyp freigeben.

Bisherg: `freigabe_nur_an` als freier Text im Payload — ein Name im
JSON-Feld, keine echte Prüfung. Damit gibt jeder, der den Namen kennt,
frei. Genügt für eine lokale Demo, nicht für den Kunden (Revision:
"Wie gewissen, dass 'ai-lead' wirklich 'ai-lead' war?").

Diese Lösung: Rollen-basierte Policy (JSON via Umgebungsvariable oder
Datei), Prüfung an JEDER Freigabe, Entscheidungsgrund in den
Audit-Trail.

Policy-Format (JSON):

    {
      "default": ["admin"],
      "02-rechnungs-extraktion": ["cfo", "buchhaltung"],
      "11-cv-screening": ["cfo", "personalverantwortung"]
    }

`admin` ist immer erlaubt. Fehlt ein Prototyp-Eintrag, gilt `default`.
Der HTTP-Request liefert `rolle` (die geprüfte Identity); `actor`
bleibt Display-Name für Audit-Zwecke.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FreigabeEntscheid:
    erlaubt: bool
    prototyp: str
    rollen_gestattet: tuple[str, ...]
    begruendung: str


class FreigabePolicy:
    """Zentrale Freigabe-Policy (Rolle × Prototyp).

    Quelle (erster Treffer gewinnt):
    1. direkt übergebene `policy` (Dict) — für Tests und Embeddings
    2. `env["GW_FREIGABE_POLICY"]` (JSON-String)
    3. `env["GW_FREIGABE_POLICY_FILE"]` (Pfad)
    4. Default: nur `admin`
    """

    def __init__(
        self,
        policy: dict[str, Any] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        env = dict(os.environ) if env is None else env
        if policy is None:
            roh = env.get("GW_FREIGABE_POLICY")
            pfad = env.get("GW_FREIGABE_POLICY_FILE")
            if roh:
                policy = json.loads(roh)
            elif pfad and Path(pfad).is_file():
                policy = json.loads(Path(pfad).read_text(encoding="utf-8"))
            else:
                policy = {"default": ["admin"]}
        if not isinstance(policy, dict):
            raise ValueError("Freigabe-Policy muss ein JSON-Objekt sein.")
        self._policy = policy

    def rollen_gestattet(self, prototyp: str) -> tuple[str, ...]:
        """Gestattete Rollen für einen Prototyp — inkl. `admin`."""
        if prototyp in self._policy and isinstance(self._policy[prototyp], list):
            prototyp_rollen = tuple(str(r) for r in self._policy[prototyp])  # type: ignore[union-attr]
        else:
            default = self._policy.get("default", ())
            prototyp_rollen = tuple(str(r) for r in default) if isinstance(default, list) else ()
        return ("admin",) + tuple(r for r in prototyp_rollen if r != "admin")

    def pruefe(self, *, rolle: str, prototyp: str) -> FreigabeEntscheid:
        erlaubte = self.rollen_gestattet(prototyp)
        if rolle == "admin":
            return FreigabeEntscheid(
                erlaubt=True, prototyp=prototyp, rollen_gestattet=erlaubte,
                begruendung="admin-Rolle (Superuser).")
        if rolle in erlaubte:
            return FreigabeEntscheid(
                erlaubt=True, prototyp=prototyp, rollen_gestattet=erlaubte,
                begruendung=f"Rolle '{rolle}' ist für '{prototyp}' gestattet.")
        return FreigabeEntscheid(
            erlaubt=False, prototyp=prototyp, rollen_gestattet=erlaubte,
            begruendung=(
                f"Rolle '{rolle}' NICHT gestattet für '{prototyp}'. "
                f"Gestattet: {', '.join(erlaubte)}."))
