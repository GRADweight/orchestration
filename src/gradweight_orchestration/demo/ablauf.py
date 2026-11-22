"""Kreditoren-Demo: Rechnung prüfen → Lieferant abgleichen → Buchungsvorschlag →
Freigabe ab Mandatslimit → Buchung im Mock-ERP.

Offline (Standard)::

    gw-orchestration demo

Gegen ein lokales Modell über Ollama (nicht Teil der CI)::

    ollama pull qwen2.5:7b-instruct
    gw-orchestration demo --llm ollama --modell qwen2.5:7b-instruct
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .._json import JSONValue
from ..agent import Agent, RunResult, RunStatus, StepRecord
from ..approvals import ApprovalQueue, ApprovalRequest
from ..audit import AuditLog, Redactor
from ..budget import Budget
from ..governance import AutonomyGrade, Policy, format_amount
from ..llm.client import OpenAICompatClient
from ..llm.protocol import ToolCallingMode
from ..llm.types import ChatModel
from ..tracing import Tracer
from .erp import MockERP, beispiel_erp
from .skript import kreditoren_skript
from .werkzeuge import kreditoren_werkzeuge

__all__ = ["INSTRUKTIONEN", "DemoErgebnis", "kreditoren_demo", "main"]

INSTRUKTIONEN = """\
Du bist Kreditorenbuchhalter:in eines Schweizer KMU. Bearbeite genau das genannte Dokument
aus dem Posteingang, in dieser Reihenfolge:
1. posteingang_lesen
2. rechnung_parsen mit dem vollständigen Text des Dokuments
3. lieferant_abgleichen mit UID und IBAN aus der Rechnung
4. Nur wenn der Lieferant bekannt ist und die IBAN stimmt: buchungsvorschlag_erstellen mit
   Netto, MWST-Satz und dem Standardkonto aus dem Abgleich
5. kreditor_buchen mit dem Bruttobetrag des Vorschlags und dem Fälligkeitsdatum der Rechnung
Übernimm Zahlen und Kennungen exakt aus den Beobachtungen. Hat die Rechnung Befunde oder
weicht die IBAN ab, buche nicht und begründe kurz. Wird eine Buchung abgelehnt, versuche sie
nicht erneut. Antworte am Schluss in einem Satz auf Deutsch."""

DOKUMENTE = ("DOK-1001", "DOK-1002", "DOK-1003")

Entscheid = Callable[[ApprovalRequest], tuple[bool, str, str]]
"""Liefert (freigeben?, entscheidende Person, Notiz/Begründung)."""


def simulierte_freigabe(_request: ApprovalRequest) -> tuple[bool, str, str]:
    return True, "m.keller", "Lieferung und Preis geprüft (simulierte Freigabe)"


def simulierte_ablehnung(_request: ApprovalRequest) -> tuple[bool, str, str]:
    return False, "m.keller", "Betrag nicht budgetiert (simulierte Ablehnung)"


def interaktive_freigabe(request: ApprovalRequest) -> tuple[bool, str, str]:
    antwort = input(f"    Freigeben ({request.id})? [j/N] ").strip().lower()
    person = input("    Ihr Kürzel: ").strip() or "unbekannt"
    if antwort in ("j", "ja", "y", "yes"):
        return True, person, "interaktiv freigegeben"
    return False, person, input("    Begründung: ").strip() or "interaktiv abgelehnt"


@dataclass
class DemoErgebnis:
    laeufe: list[RunResult]
    erp: MockERP
    audit: AuditLog
    tracer: Tracer
    freigaben: ApprovalQueue
    ausgabe_dir: Path
    zeilen: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.status is RunStatus.COMPLETED for r in self.laeufe) and self.audit.verify().ok


def _chf(value: JSONValue) -> str:
    try:
        return format_amount(Decimal(str(value)))
    except (InvalidOperation, ValueError):
        return str(value)


def _liste(value: JSONValue) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _argumente(arguments: dict[str, JSONValue] | None) -> str:
    parts = []
    for key, value in (arguments or {}).items():
        shown = json.dumps(value, ensure_ascii=False)
        if len(shown) > 32:
            shown = shown[:29] + '…"'
        parts.append(f"{key}={shown}")
    return ", ".join(parts)


def _kurz(step: StepRecord) -> str:
    observation = step.observation if isinstance(step.observation, dict) else {}
    if observation.get("ok") is False:
        error = observation.get("error") or {}
        if isinstance(error, dict):
            return f"FEHLER {error.get('type')}: {error.get('message')}"
    result = observation.get("result")
    if not isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False)[:80] if result is not None else ""
    if step.tool == "posteingang_lesen":
        return f"Dokument von {result['absender']}, Eingang {result['eingang']}"
    if step.tool == "rechnung_parsen":
        befunde = "; ".join(_liste(result["befunde"])) or "keine Befunde"
        return (
            f"{result['rechnungsnummer']}: Total {_chf(result['total'])} "
            f"(MWST {result['mwst_satz']} %), fällig {result['faellig_am']} – {befunde}"
        )
    if step.tool == "lieferant_abgleichen":
        if not result["bekannt"]:
            return str(result["hinweis"])
        iban = "IBAN stimmt" if result["iban_stimmt"] else "IBAN WEICHT AB"
        return f"{result['name']} ({result['kreditor_nr']}), {iban}"
    if step.tool == "buchungsvorschlag_erstellen":
        return f"{result['vorschlag_id']}: " + " / ".join(_liste(result["zeilen"]))
    if step.tool == "kreditor_buchen":
        return (
            f"Beleg {result['beleg_nr']}, {_chf(result['betrag'])}, "
            f"Zahlung am {result['zahlung_am']}"
        )
    return json.dumps(result, ensure_ascii=False)[:80]


class _Ausgabe:
    def __init__(self, audit: AuditLog, schreiben: Callable[[str], None]) -> None:
        self.audit = audit
        self.schreiben = schreiben
        self.zeilen: list[str] = []

    def __call__(self, text: str = "") -> None:
        self.zeilen.append(text)
        self.schreiben(text)

    def schritte(self, steps: Sequence[StepRecord], ab: int) -> int:
        details: dict[str, dict[str, Any]] = {}
        for record in self.audit.records():
            call_id = record.data.get("call_id")
            if isinstance(call_id, str):
                details.setdefault(call_id, {})[record.kind] = record.data
        for index, step in enumerate(steps[ab:], start=ab + 1):
            if step.kind == "final":
                continue
            if step.kind == "notice":
                self(f"  {index}. Hinweis: {step.note}")
                continue
            info = details.get(step.call_id or "", {})
            extras = []
            sandbox = (info.get("tool_result") or {}).get("sandbox")
            if isinstance(sandbox, dict):
                netz = "Netz isoliert" if sandbox.get("network_isolated") else "Netz nicht isoliert"
                extras.append(f"Kindprozess, {netz}, CPU {sandbox.get('cpu_seconds')} s")
            nach_freigabe = step.decision in ("approved", "approved_with_changes")
            policy = info.get("policy_decision")
            if policy and step.tool == "kreditor_buchen" and not nach_freigabe:
                extras.append(f"Policy: {policy['reason']}")
            suffix = f"  [{'; '.join(extras)}]" if extras else ""
            if nach_freigabe:
                self(f"  {index}. {step.tool} – Ausführung nach Entscheid")
            else:
                self(f"  {index}. {step.tool}({_argumente(step.arguments)}){suffix}")
            if step.ok is None and step.decision == "require_approval":
                self(f"     → PAUSE: {step.note}")
            else:
                note = f" ({step.note})" if step.note else ""
                self(f"     → {_kurz(step)}{note}")
        return len(steps)


def kreditoren_demo(
    llm: ChatModel,
    *,
    grad: AutonomyGrade = AutonomyGrade.MANDATE,
    limit: Decimal = Decimal(5000),
    entscheid: Entscheid = simulierte_freigabe,
    ausgabe_dir: Path | None = None,
    tool_calling: ToolCallingMode = ToolCallingMode.AUTO,
    dokumente: Sequence[str] = DOKUMENTE,
    schreiben: Callable[[str], None] = print,
) -> DemoErgebnis:
    """Verarbeitet die Dokumente aus dem Posteingang und gibt den Ablauf aus."""
    ziel = ausgabe_dir or Path(tempfile.mkdtemp(prefix="gw-demo-"))
    ziel.mkdir(parents=True, exist_ok=True)
    erp = beispiel_erp()
    audit = AuditLog(ziel / "audit.jsonl", redactor=Redactor())
    freigaben = ApprovalQueue(ziel / "freigaben.sqlite", audit=audit)
    tracer = Tracer()
    policy = Policy(grade=grad, money_limit=limit)
    agent = Agent(
        name="kreditoren",
        llm=llm,
        tools=kreditoren_werkzeuge(erp),
        instructions=INSTRUKTIONEN,
        policy=policy,
        approvals=freigaben,
        audit=audit,
        tracer=tracer,
        budget=Budget(max_steps=12, max_seconds=900),
        tool_calling=tool_calling,
    )
    out = _Ausgabe(audit, schreiben)
    out(
        f"Kreditoren-Demo · Modell: {llm.model_name} · Grad {grad.label} · "
        f"Mandatslimit {format_amount(limit)}"
    )
    out(f"Ausgabe: {ziel}")
    laeufe: list[RunResult] = []
    for dokument in dokumente:
        out()
        out(f"── {dokument} " + "─" * 50)
        result = agent.run(f"Bearbeite {dokument} aus dem Posteingang.", run_id=f"demo-{dokument}")
        gezeigt = out.schritte(result.steps, 0)
        while result.status is RunStatus.PAUSED:
            for request in result.pending_approvals:
                out(f"  FREIGABE NÖTIG {request.id}: {request.tool} – {request.reason}")
                frei, person, notiz = entscheid(request)
                if frei:
                    freigaben.approve(request.id, by=person, note=notiz)
                    out(f"  FREIGEGEBEN durch {person}: {notiz}")
                else:
                    freigaben.reject(request.id, by=person, reason=notiz)
                    out(f"  ABGELEHNT durch {person}: {notiz}")
            result = agent.resume(result.state)
            gezeigt = out.schritte(result.steps, gezeigt)
        status = "Ergebnis" if result.ok else f"Abbruch ({result.reason})"
        out(f"  {status}: {result.answer or result.message}")
        laeufe.append(result)

    out()
    out("── Zusammenfassung " + "─" * 44)
    for beleg in erp.belege:
        out(
            f"  Gebucht: {beleg.rechnungsnummer} {format_amount(beleg.betrag)} "
            f"({beleg.beleg_nr}, Kreditor {beleg.kreditor_nr}, Zahlung am {beleg.zahlung_am})"
        )
    gebucht = {b.rechnungsnummer for b in erp.belege}
    for dokument, result in zip(dokumente, laeufe, strict=True):
        if not any(nummer in (result.answer or "") for nummer in gebucht):
            out(f"  Ohne Buchung ({dokument}): {result.answer or result.message}")
    pruefung = audit.verify()
    zustand = "Kette intakt" if pruefung.ok else f"FEHLER: {pruefung.message}"
    out(f"  Audit-Log: {pruefung.count} Einträge, {zustand}, Head {pruefung.head[:16]}…")
    trace_path = tracer.export_json(ziel / "trace.json")
    out(f"  Trace: {len(tracer.spans)} Spans → {trace_path.name}; Audit → audit.jsonl")
    tokens = sum(r.usage.tokens for r in laeufe)
    geschaetzt = any(r.usage.estimated for r in laeufe)
    out(
        f"  Token: {tokens}{' (geschätzt)' if geschaetzt else ''}, "
        f"Modellaufrufe: {sum(r.usage.steps for r in laeufe)}"
    )
    return DemoErgebnis(laeufe, erp, audit, tracer, freigaben, ziel, out.zeilen)


def _build_llm(args: argparse.Namespace) -> ChatModel:
    if args.llm == "skript":
        return kreditoren_skript()
    base_url = args.base_url or (
        "http://localhost:11434/v1" if args.llm == "ollama" else "https://api.openai.com/v1"
    )
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    return OpenAICompatClient(
        base_url=base_url, model=args.modell, api_key=api_key, timeout_s=args.timeout
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gw-orchestration demo",
        description="Kreditoren-Demo: Rechnung prüfen, Lieferant abgleichen, buchen.",
    )
    parser.add_argument(
        "--llm",
        choices=["skript", "ollama", "openai"],
        default="skript",
        help="skript = deterministisch und offline (Standard)",
    )
    parser.add_argument("--modell", default="qwen2.5:7b-instruct", help="Modellname")
    parser.add_argument("--base-url", help="OpenAI-kompatible URL (Standard je nach --llm)")
    parser.add_argument("--api-key-env", help="Umgebungsvariable mit dem API-Schlüssel")
    parser.add_argument("--timeout", type=float, default=300.0, help="Sekunden pro Modellaufruf")
    parser.add_argument("--grad", type=int, choices=[1, 2, 3, 4, 5], default=4)
    parser.add_argument("--limit", type=Decimal, default=Decimal(5000), help="Mandatslimit CHF")
    parser.add_argument(
        "--freigabe", choices=["simuliert", "ablehnen", "interaktiv"], default="simuliert"
    )
    parser.add_argument(
        "--tool-calling", choices=[m.value for m in ToolCallingMode], default="auto"
    )
    parser.add_argument("--ausgabe", type=Path, help="Ordner für audit.jsonl, trace.json usw.")
    args = parser.parse_args(argv)
    entscheid = {
        "simuliert": simulierte_freigabe,
        "ablehnen": simulierte_ablehnung,
        "interaktiv": interaktive_freigabe,
    }[args.freigabe]
    ergebnis = kreditoren_demo(
        _build_llm(args),
        grad=AutonomyGrade(args.grad),
        limit=args.limit,
        entscheid=entscheid,
        ausgabe_dir=args.ausgabe,
        tool_calling=ToolCallingMode(args.tool_calling),
    )
    ergebnis.freigaben.close()
    if args.llm == "skript" and args.grad == 4 and args.freigabe == "simuliert":
        erwartet = [Decimal("1081.00"), Decimal("12431.50")]
        if sorted(b.betrag for b in ergebnis.erp.belege) != erwartet:
            print("Selbstprüfung fehlgeschlagen: unerwartete Buchungen", file=sys.stderr)
            return 1
    return 0 if ergebnis.ok else 1
