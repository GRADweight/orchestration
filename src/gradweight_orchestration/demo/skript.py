"""Deterministisches «Modell» für die Offline-Demo.

Entscheidet jeden Schritt aus den bisherigen Beobachtungen – wie ein Modell,
nur reproduzierbar. Alle Argumente stammen aus Beobachtungen (Text, Beträge,
UID, IBAN, Vorschlags-ID), nichts ist fest verdrahtet ausser der Reihenfolge.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

from ..governance import format_amount
from ..llm.scripted import ScriptContext, ScriptedLLM, ScriptedTurn

__all__ = ["kreditoren_skript"]

_DOKUMENT = re.compile(r"DOK-\d+")


def _ergebnis(beobachtung: Any) -> dict[str, Any]:
    result = beobachtung.get("result") if isinstance(beobachtung, dict) else None
    return result if isinstance(result, dict) else {}


def naechster_schritt(ctx: ScriptContext) -> ScriptedTurn:
    aufgabe = next(str(m["content"]) for m in ctx.messages if m.get("role") == "user")
    treffer = _DOKUMENT.search(aufgabe)
    if treffer is None:
        return ScriptedTurn.final("Kein Dokument genannt.")
    dokument = treffer.group(0)
    beobachtungen = [o for o in ctx.observations() if isinstance(o, dict)]
    if not beobachtungen:
        return ScriptedTurn.call(
            "posteingang_lesen", {"dokument_id": dokument}, thought="Dokument holen"
        )
    letzte = beobachtungen[-1]
    if letzte.get("ok") is not True:
        fehler = letzte.get("error")
        grund = fehler.get("message", "Fehler") if isinstance(fehler, dict) else "Fehler"
        return ScriptedTurn.final(f"{dokument} nicht gebucht: {grund}")
    schritt = len(beobachtungen)
    if schritt == 1:
        return ScriptedTurn.call(
            "rechnung_parsen",
            {"text": _ergebnis(letzte)["text"]},
            thought="Rechnung lesen und prüfen",
        )
    rechnung = _ergebnis(beobachtungen[1])
    if schritt == 2:
        if rechnung["befunde"]:
            return ScriptedTurn.final(
                f"{rechnung['rechnungsnummer']} nicht gebucht: {'; '.join(rechnung['befunde'])}"
            )
        return ScriptedTurn.call(
            "lieferant_abgleichen",
            {"uid": rechnung["lieferant_uid"], "iban": rechnung["iban"]},
            thought="Lieferant und Bankverbindung abgleichen",
        )
    abgleich = _ergebnis(beobachtungen[2])
    if schritt == 3:
        if not (abgleich["bekannt"] and abgleich["iban_stimmt"]):
            return ScriptedTurn.final(
                f"{rechnung['rechnungsnummer']} nicht gebucht: {abgleich['hinweis']}"
            )
        return ScriptedTurn.call(
            "buchungsvorschlag_erstellen",
            {
                "rechnungsnummer": rechnung["rechnungsnummer"],
                "kreditor_nr": abgleich["kreditor_nr"],
                "netto": rechnung["netto"],
                "mwst_satz": rechnung["mwst_satz"],
                "aufwandkonto": abgleich["standard_konto"],
            },
            thought="Buchungsvorschlag erstellen",
        )
    if schritt == 4:
        vorschlag = _ergebnis(letzte)
        return ScriptedTurn.call(
            "kreditor_buchen",
            {
                "vorschlag_id": vorschlag["vorschlag_id"],
                "betrag_chf": vorschlag["brutto"],
                "zahlung_am": rechnung["faellig_am"],
            },
            thought="Buchen und Zahlung zur Fälligkeit freigeben",
        )
    beleg = _ergebnis(letzte)
    betrag = format_amount(Decimal(str(beleg["betrag"])))
    return ScriptedTurn.final(
        f"{rechnung['rechnungsnummer']} ({rechnung['lieferant']}) gebucht: Beleg "
        f"{beleg['beleg_nr']}, {betrag}, Zahlung am {beleg['zahlung_am']}."
    )


def kreditoren_skript(max_zuege: int = 200) -> ScriptedLLM:
    """Skript-Modell für beliebig viele Dokumente (jeder Zug wird neu berechnet)."""
    return ScriptedLLM([naechster_schritt] * max_zuege, model_name="skript-kreditoren")
