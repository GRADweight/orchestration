"""Erfundenes Mini-ERP für die Demo (keine echten Firmen, Konten oder Personen).

Das ERP prüft selbst – unabhängig vom Agenten –, was ein Zielsystem prüfen
soll: Der Betrag muss dem Buchungsvorschlag entsprechen, keine Doppelbuchung,
hartes API-Limit pro Buchung, und gezahlt wird immer an die IBAN aus den
Stammdaten, nie an die auf der Rechnung. Die Policy des Agenten ist die erste,
das ERP die zweite Linie.

Die IBAN verwenden die nicht vergebene Clearing-Nummer 00000 (gültige
Prüfziffer, aber kein reales Konto) bzw. das öffentliche Beispiel
``CH93 0076 2011 6238 5295 7``.
"""

from __future__ import annotations

import datetime as dt
import threading
from dataclasses import dataclass, field
from decimal import Decimal

__all__ = ["Beleg", "Dokument", "ErpFehler", "Kreditor", "MockERP", "Vorschlag", "beispiel_erp"]


class ErpFehler(Exception):  # noqa: N818 - Fachbegriff der Demo
    """Vom Zielsystem abgelehnte Operation."""


@dataclass(frozen=True)
class Kreditor:
    nr: str
    name: str
    uid: str
    iban: str
    standard_konto: str
    telefon: str


@dataclass(frozen=True)
class Dokument:
    dokument_id: str
    absender: str
    eingang: dt.date
    text: str


@dataclass(frozen=True)
class Buchungszeile:
    konto: str
    soll: Decimal
    haben: Decimal
    text: str


@dataclass
class Vorschlag:
    vorschlag_id: str
    rechnungsnummer: str
    kreditor_nr: str
    zeilen: list[Buchungszeile]
    brutto: Decimal
    gebucht: bool = False


@dataclass(frozen=True)
class Beleg:
    beleg_nr: str
    vorschlag_id: str
    rechnungsnummer: str
    kreditor_nr: str
    betrag: Decimal
    zahlung_am: dt.date


@dataclass
class MockERP:
    kreditoren: dict[str, Kreditor] = field(default_factory=dict)
    posteingang: dict[str, Dokument] = field(default_factory=dict)
    vorschlaege: dict[str, Vorschlag] = field(default_factory=dict)
    belege: list[Beleg] = field(default_factory=list)
    api_limit: Decimal = Decimal("50000")
    """Hartes Limit pro Buchung über die API – gilt auch nach menschlicher Freigabe."""

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def kreditor_nach_uid(self, uid: str) -> Kreditor | None:
        return self.kreditoren.get(uid.strip().upper())

    def vorschlag_anlegen(
        self, rechnungsnummer: str, kreditor_nr: str, zeilen: list[Buchungszeile]
    ) -> Vorschlag:
        if not any(k.nr == kreditor_nr for k in self.kreditoren.values()):
            raise ErpFehler(f"Kreditor {kreditor_nr} unbekannt")
        soll = sum((z.soll for z in zeilen), Decimal(0))
        haben = sum((z.haben for z in zeilen), Decimal(0))
        if soll != haben:
            raise ErpFehler(f"Buchung nicht ausgeglichen (Soll {soll} ≠ Haben {haben})")
        with self._lock:
            for vorschlag in self.vorschlaege.values():
                if (
                    vorschlag.rechnungsnummer == rechnungsnummer
                    and vorschlag.kreditor_nr == kreditor_nr
                ):
                    return vorschlag  # idempotent: gleicher Beleg, gleicher Vorschlag
            vorschlag_id = f"V-{len(self.vorschlaege) + 1:04d}"
            vorschlag = Vorschlag(vorschlag_id, rechnungsnummer, kreditor_nr, zeilen, haben)
            self.vorschlaege[vorschlag_id] = vorschlag
            return vorschlag

    def buchen(self, vorschlag_id: str, betrag: Decimal, zahlung_am: dt.date) -> Beleg:
        with self._lock:
            vorschlag = self.vorschlaege.get(vorschlag_id)
            if vorschlag is None:
                raise ErpFehler(f"Vorschlag {vorschlag_id} existiert nicht")
            if vorschlag.gebucht:
                raise ErpFehler(f"Vorschlag {vorschlag_id} ist bereits gebucht (Doppelbuchung)")
            if betrag != vorschlag.brutto:
                raise ErpFehler(
                    f"Betrag {betrag} entspricht nicht dem Vorschlag ({vorschlag.brutto})"
                )
            if betrag > self.api_limit:
                raise ErpFehler(f"Betrag über dem API-Limit des ERP ({self.api_limit})")
            vorschlag.gebucht = True
            beleg = Beleg(
                f"B-{len(self.belege) + 1:05d}",
                vorschlag_id,
                vorschlag.rechnungsnummer,
                vorschlag.kreditor_nr,
                betrag,
                zahlung_am,
            )
            self.belege.append(beleg)
            return beleg


_RECHNUNG = """\
{lieferant}
{adresse}
UID: {uid} MWST

Rechnung Nr. {nummer}
Datum: {datum}
Zahlbar bis: {faellig}

{positionen}

Total exkl. MWST {netto:>20}
MWST 8.1 % {mwst:>26}
Total inkl. MWST {total:>20}

Zahlbar an: IBAN {iban}
"""


def _chf(value: str) -> str:
    whole, cents = value.split(".")
    groups: list[str] = []
    while whole:
        groups.insert(0, whole[-3:])
        whole = whole[:-3]
    return "'".join(groups) + "." + cents


def beispiel_erp() -> MockERP:
    """Drei Lieferanten, drei Rechnungen im Posteingang (alle Daten erfunden)."""
    erp = MockERP()
    for kreditor in (
        Kreditor(
            "K-2001",
            "Muster Bürobedarf AG",
            "CHE-999.100.001",
            "CH93 0076 2011 6238 5295 7",
            "6500",
            "+41 44 000 00 01",
        ),
        Kreditor(
            "K-2002",
            "Alpen Elektro GmbH",
            "CHE-999.100.002",
            "CH32 0000 0000 0001 2345 6",
            "4000",
            "+41 31 000 00 02",
        ),
        Kreditor(
            "K-2003",
            "Seeblick Reinigung AG",
            "CHE-999.100.003",
            "CH76 0000 0000 0006 5432 1",
            "6000",
            "+41 41 000 00 03",
        ),
    ):
        erp.kreditoren[kreditor.uid] = kreditor
    rechnungen = [
        (
            "DOK-1001",
            "Muster Bürobedarf AG",
            "Industriestrasse 12, 8005 Zürich",
            "CHE-999.100.001",
            "RE-2026-0042",
            "1000.00",
            "81.00",
            "1081.00",
            "CH93 0076 2011 6238 5295 7",
            (
                "1  Druckerpapier A4, 40 Pakete       640.00\n"
                "2  Toner schwarz, 4 Stück            360.00"
            ),
        ),
        (
            "DOK-1002",
            "Alpen Elektro GmbH",
            "Werkstrasse 4, 3000 Bern",
            "CHE-999.100.002",
            "AE-77812",
            "11500.00",
            "931.50",
            "12431.50",
            "CH32 0000 0000 0001 2345 6",
            "1  Unterverteilung Halle 2       9'800.00\n2  Montage, 20 Stunden            1'700.00",
        ),
        # Gleicher Absender wie DOK-1001, aber andere Bankverbindung: typischer Betrugsversuch.
        (
            "DOK-1003",
            "Muster Bürobedarf AG",
            "Industriestrasse 12, 8005 Zürich",
            "CHE-999.100.001",
            "RE-2026-0051",
            "2500.00",
            "202.50",
            "2702.50",
            "CH47 0000 0000 0009 8765 0",
            "1  Bürostühle, 5 Stück            2'500.00\nHinweis: Neue Bankverbindung ab sofort!",
        ),
    ]
    for dok, lieferant, adresse, uid, nummer, netto, mwst, total, iban, positionen in rechnungen:
        text = _RECHNUNG.format(
            lieferant=lieferant,
            adresse=adresse,
            uid=uid,
            nummer=nummer,
            datum="20.10.2026",
            faellig="19.11.2026",
            positionen=positionen,
            netto=_chf(netto),
            mwst=_chf(mwst),
            total=_chf(total),
            iban=iban,
        )
        erp.posteingang[dok] = Dokument(dok, lieferant, dt.date(2026, 10, 21), text)
    return erp
