"""Werkzeuge der Kreditoren-Demo.

``rechnung_parsen`` verarbeitet fremden Dokumenttext und läuft deshalb im
Kindprozess (``Isolation.PROCESS``). Die übrigen Werkzeuge sprechen mit dem
(vertrauenswürdigen) Mock-ERP und laufen im Thread.

Konten nach KMU-Kontenrahmen: 1170 Vorsteuer MWST Material, Waren,
Dienstleistungen, Energie; 2000 Verbindlichkeiten aus Lieferungen und
Leistungen (Kreditoren); Aufwandkonto aus den Stammdaten des Lieferanten.
MWST-Sätze Schweiz seit 1.1.2024: 8,1 %, 2,6 %, 3,8 %.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Annotated, Any

from ..governance import format_amount
from ..sandbox import NetworkMode, SandboxLimits, detect_capabilities
from ..schema import Param
from ..tools import Isolation, SideEffect, Tool, tool
from .erp import Buchungszeile, ErpFehler, MockERP

__all__ = [
    "Abgleich",
    "Buchungsbeleg",
    "Buchungsvorschlag",
    "Posteingang",
    "Rechnung",
    "iban_gueltig",
    "kreditoren_werkzeuge",
    "rechnung_parsen",
]

MWST_SAETZE = (Decimal("8.1"), Decimal("2.6"), Decimal("3.8"), Decimal("0"))


@dataclass(frozen=True)
class Posteingang:
    """Ein Dokument aus dem Posteingang."""

    dokument_id: str
    absender: str
    eingang: dt.date
    text: str


@dataclass(frozen=True)
class Rechnung:
    """Aus dem Rechnungstext gelesene Angaben.

    Attributes:
        befunde: Rechenfehler oder fehlende Angaben; leer, wenn alles stimmt.
    """

    rechnungsnummer: str
    lieferant: str
    lieferant_uid: str
    iban: str
    netto: Decimal
    mwst_satz: Decimal
    mwst_betrag: Decimal
    total: Decimal
    faellig_am: dt.date
    befunde: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Abgleich:
    """Ergebnis des Stammdatenabgleichs."""

    bekannt: bool
    iban_stimmt: bool
    kreditor_nr: str | None
    name: str | None
    standard_konto: str | None
    hinweis: str


@dataclass(frozen=True)
class Buchungsvorschlag:
    vorschlag_id: str
    zeilen: list[str]
    brutto: Decimal


@dataclass(frozen=True)
class Buchungsbeleg:
    beleg_nr: str
    betrag: Decimal
    zahlung_am: dt.date


def iban_gueltig(iban: str) -> bool:
    """Prüfziffer nach ISO 13616 (Modulo 97)."""
    compact = iban.replace(" ", "").upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", compact):
        return False
    rearranged = compact[4:] + compact[:4]
    return int("".join(str(int(c, 36)) for c in rearranged)) % 97 == 1


def _betrag(text: str) -> Decimal:
    return Decimal(text.replace("'", ""))


def _suche(muster: str, text: str) -> str | None:
    found = re.search(muster, text, re.MULTILINE)
    return found.group(1).strip() if found else None


_LIMITS = SandboxLimits(
    cpu_seconds=5,
    memory_mb=256 if detect_capabilities().rlimit_memory else None,
    network=NetworkMode.DENY_IF_POSSIBLE,
)


@tool(side_effect=SideEffect.READ_ONLY, isolation=Isolation.PROCESS, timeout_s=10, sandbox=_LIMITS)
def rechnung_parsen(text: str) -> Rechnung:
    """Liest Rechnungsnummer, Lieferant, UID, IBAN, Beträge und Fälligkeit aus Rechnungstext
    und prüft die Rechnung rechnerisch (Netto + MWST = Total, MWST-Satz, IBAN-Prüfziffer).

    Args:
        text: Vollständiger Rechnungstext aus dem Posteingang.
    """
    befunde: list[str] = []
    zeilen = [z.strip() for z in text.strip().splitlines() if z.strip()]
    nummer = _suche(r"Rechnung Nr\.\s*(\S+)", text)
    uid = _suche(r"UID:\s*(CHE-\d{3}\.\d{3}\.\d{3})", text)
    iban = _suche(r"IBAN\s+([A-Z]{2}\d{2}(?: ?[A-Z0-9]{1,4})+)", text)
    netto = _suche(r"Total exkl\. MWST\s+([\d']+\.\d{2})", text)
    satz = _suche(r"MWST\s+(\d+(?:\.\d+)?)\s*%", text)
    mwst = _suche(r"MWST\s+\d+(?:\.\d+)?\s*%\s+([\d']+\.\d{2})", text)
    total = _suche(r"Total inkl\. MWST\s+([\d']+\.\d{2})", text)
    faellig = _suche(r"Zahlbar bis:\s*(\d{2}\.\d{2}\.\d{4})", text)
    fehlend = [
        name
        for name, wert in (
            ("Rechnungsnummer", nummer),
            ("UID", uid),
            ("IBAN", iban),
            ("Netto", netto),
            ("MWST", mwst),
            ("Total", total),
            ("Fälligkeit", faellig),
        )
        if wert is None
    ]
    if fehlend:
        raise ValueError(f"Angaben fehlen: {', '.join(fehlend)}")
    assert nummer and uid and iban and netto and mwst and total and faellig and satz  # noqa: S101, PT018
    netto_d, mwst_d, total_d, satz_d = _betrag(netto), _betrag(mwst), _betrag(total), Decimal(satz)
    if satz_d not in MWST_SAETZE:
        befunde.append(f"MWST-Satz {satz_d} % ist kein Schweizer Satz")
    erwartet = (netto_d * satz_d / 100).quantize(Decimal("0.01"))
    if erwartet != mwst_d:
        befunde.append(f"MWST-Betrag {mwst_d} statt {erwartet}")
    if netto_d + mwst_d != total_d:
        befunde.append(f"Netto + MWST = {netto_d + mwst_d}, Rechnung nennt {total_d}")
    if not iban_gueltig(iban):
        befunde.append("IBAN-Prüfziffer ungültig")
    tag, monat, jahr = faellig.split(".")
    return Rechnung(
        rechnungsnummer=nummer,
        lieferant=zeilen[0],
        lieferant_uid=uid,
        iban=" ".join(iban.split()),
        netto=netto_d,
        mwst_satz=satz_d,
        mwst_betrag=mwst_d,
        total=total_d,
        faellig_am=dt.date(int(jahr), int(monat), int(tag)),
        befunde=befunde,
    )


def kreditoren_werkzeuge(erp: MockERP) -> list[Tool[..., Any]]:
    """Die Werkzeuge für einen Kreditoren-Agenten, gebunden an ein ERP."""

    @tool(side_effect=SideEffect.READ_ONLY, timeout_s=5)
    def posteingang_lesen(dokument_id: Annotated[str, Param(pattern=r"^DOK-\d+$")]) -> Posteingang:
        """Liest ein Dokument aus dem Posteingang.

        Args:
            dokument_id: Kennung des Dokuments, z.B. DOK-1001.
        """
        dokument = erp.posteingang.get(dokument_id)
        if dokument is None:
            raise KeyError(f"Dokument {dokument_id} nicht im Posteingang")
        return Posteingang(dokument.dokument_id, dokument.absender, dokument.eingang, dokument.text)

    @tool(side_effect=SideEffect.READ_ONLY, timeout_s=5)
    def lieferant_abgleichen(
        uid: Annotated[str, Param(pattern=r"^CHE-\d{3}\.\d{3}\.\d{3}$")], iban: str
    ) -> Abgleich:
        """Gleicht Lieferant (UID) und Bankverbindung mit den Kreditoren-Stammdaten ab.

        Args:
            uid: UID des Lieferanten (CHE-xxx.xxx.xxx).
            iban: IBAN auf der Rechnung.
        """
        kreditor = erp.kreditor_nach_uid(uid)
        if kreditor is None:
            return Abgleich(
                False,
                False,
                None,
                None,
                None,
                "Lieferant nicht in den Stammdaten – Neuanlage durch Buchhaltung",
            )
        stimmt = kreditor.iban.replace(" ", "") == iban.replace(" ", "").upper()
        hinweis = (
            "Stammdaten stimmen überein"
            if stimmt
            else (
                f"IBAN weicht von den Stammdaten ab (hinterlegt: …{kreditor.iban[-7:]}). "
                f"Nicht zahlen; Rückfrage über die bekannte Nummer {kreditor.telefon}."
            )
        )
        return Abgleich(True, stimmt, kreditor.nr, kreditor.name, kreditor.standard_konto, hinweis)

    @tool(side_effect=SideEffect.REVERSIBLE, timeout_s=5, idempotent=True)
    def buchungsvorschlag_erstellen(
        rechnungsnummer: str,
        kreditor_nr: Annotated[str, Param(pattern=r"^K-\d{4}$")],
        netto: Annotated[Decimal, Param(ge=0)],
        mwst_satz: Annotated[Decimal, Param(ge=0, le=10)],
        aufwandkonto: Annotated[str, Param(pattern=r"^[3-6]\d{3}$")],
    ) -> Buchungsvorschlag:
        """Erstellt einen Buchungsvorschlag (Entwurf im ERP, jederzeit löschbar).

        Bucht Aufwand und Vorsteuer (1170) im Soll gegen Kreditoren (2000) im Haben.

        Args:
            rechnungsnummer: Nummer der Rechnung.
            kreditor_nr: Kreditorennummer aus dem Abgleich, z.B. K-2001.
            netto: Betrag exkl. MWST in CHF.
            mwst_satz: MWST-Satz in Prozent, z.B. 8.1.
            aufwandkonto: Aufwandkonto (Standardkonto des Lieferanten).
        """
        rappen = Decimal("0.01")
        netto = netto.quantize(rappen)
        mwst = (netto * mwst_satz / 100).quantize(rappen)
        brutto = netto + mwst
        text = f"{rechnungsnummer} {kreditor_nr}"
        zeilen = [
            Buchungszeile(aufwandkonto, netto, Decimal(0), text),
            Buchungszeile("1170", mwst, Decimal(0), f"Vorsteuer {mwst_satz} % {text}"),
            Buchungszeile("2000", Decimal(0), brutto, text),
        ]
        try:
            vorschlag = erp.vorschlag_anlegen(rechnungsnummer, kreditor_nr, zeilen)
        except ErpFehler as exc:
            raise ValueError(str(exc)) from exc
        lines = [
            f"{z.konto} {'Soll' if z.soll else 'Haben'} {format_amount(z.soll or z.haben)}"
            for z in vorschlag.zeilen
        ]
        return Buchungsvorschlag(vorschlag.vorschlag_id, lines, vorschlag.brutto)

    @tool(side_effect=SideEffect.MONEY, amount_param="betrag_chf", timeout_s=5)
    def kreditor_buchen(
        vorschlag_id: Annotated[str, Param(pattern=r"^V-\d{4}$")],
        betrag_chf: Annotated[Decimal, Param(ge=0)],
        zahlung_am: dt.date,
    ) -> Buchungsbeleg:
        """Bucht den Vorschlag und gibt die Zahlung zum Fälligkeitsdatum frei.

        Args:
            vorschlag_id: Kennung des Buchungsvorschlags, z.B. V-0001.
            betrag_chf: Bruttobetrag in CHF; muss dem Vorschlag entsprechen.
            zahlung_am: Zahlungsdatum (Fälligkeit der Rechnung), JJJJ-MM-TT.
        """
        try:
            beleg = erp.buchen(vorschlag_id, betrag_chf, zahlung_am)
        except ErpFehler as exc:
            raise ValueError(f"ERP lehnt ab: {exc}") from exc
        return Buchungsbeleg(beleg.beleg_nr, beleg.betrag, beleg.zahlung_am)

    return [
        posteingang_lesen,
        rechnung_parsen,
        lieferant_abgleichen,
        buchungsvorschlag_erstellen,
        kreditor_buchen,
    ]
