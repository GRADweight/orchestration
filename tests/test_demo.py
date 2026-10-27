"""Kreditoren-Demo: Ablauf, Freigaben, Zielsystem-Prüfungen, HTTP-Ende-zu-Ende."""

from __future__ import annotations

import datetime as dt
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from fake_openai import FakeOpenAI, Reply, chat
from gradweight_orchestration import AutonomyGrade, OpenAICompatClient, ToolCallingMode
from gradweight_orchestration.agent import StepRecord
from gradweight_orchestration.cli import main as cli_main
from gradweight_orchestration.demo import kreditoren_demo
from gradweight_orchestration.demo.ablauf import (
    _kurz,
    interaktive_freigabe,
    simulierte_ablehnung,
)
from gradweight_orchestration.demo.erp import Buchungszeile, ErpFehler, beispiel_erp
from gradweight_orchestration.demo.skript import kreditoren_skript, naechster_schritt
from gradweight_orchestration.demo.werkzeuge import iban_gueltig, rechnung_parsen
from gradweight_orchestration.llm import ScriptContext

quiet: Any = lambda _line: None  # noqa: E731


def test_scripted_demo(tmp_path: Path) -> None:
    result = kreditoren_demo(kreditoren_skript(), ausgabe_dir=tmp_path, schreiben=quiet)
    assert result.ok
    booked = {b.rechnungsnummer: b.betrag for b in result.erp.belege}
    assert booked == {"RE-2026-0042": Decimal("1081.00"), "AE-77812": Decimal("12431.50")}
    text = "\n".join(result.zeilen)
    assert "CHF 0.00" not in text
    assert "über Mandatslimit CHF 5'000.00" in text
    assert "FREIGEGEBEN durch m.keller" in text
    assert "IBAN WEICHT AB" in text
    assert "Kindprozess" in text
    assert "Kette intakt" in text
    # Audit-Log ist geschwärzt: keine IBAN, keine Telefonnummer im Klartext
    raw = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert "CH93 0076" not in raw
    assert "+41 44" not in raw
    assert "[IBAN]" in raw
    trace = json.loads((tmp_path / "trace.json").read_text(encoding="utf-8"))
    assert any(s["name"] == "execute_tool rechnung_parsen" for s in trace["spans"])
    approved = [r for r in result.audit.records() if r.kind == "approval_decided"]
    assert [r.data["status"] for r in approved] == ["approved"]
    result.freigaben.close()


def test_rejected_payment_is_not_booked(tmp_path: Path) -> None:
    result = kreditoren_demo(
        kreditoren_skript(), entscheid=simulierte_ablehnung, ausgabe_dir=tmp_path, schreiben=quiet
    )
    assert [b.rechnungsnummer for b in result.erp.belege] == ["RE-2026-0042"]
    assert "Abgelehnt durch m.keller" in (result.laeufe[1].answer or "")
    result.freigaben.close()


def test_copilot_grade_needs_approval_for_every_write(tmp_path: Path) -> None:
    result = kreditoren_demo(
        kreditoren_skript(), grad=AutonomyGrade.COPILOT, ausgabe_dir=tmp_path, schreiben=quiet
    )
    decided = [r.data["tool"] for r in result.audit.records() if r.kind == "approval_decided"]
    assert decided == ["buchungsvorschlag_erstellen", "kreditor_buchen"] * 2
    assert len(result.erp.belege) == 2
    result.freigaben.close()


def test_assistant_grade_never_writes(tmp_path: Path) -> None:
    result = kreditoren_demo(
        kreditoren_skript(), grad=AutonomyGrade.ASSISTANT, ausgabe_dir=tmp_path, schreiben=quiet
    )
    assert result.erp.belege == []
    assert result.erp.vorschlaege == {}
    assert all("nicht gebucht" in (r.answer or "") for r in result.laeufe)
    result.freigaben.close()


def test_json_protocol_demo(tmp_path: Path) -> None:
    result = kreditoren_demo(
        kreditoren_skript(),
        tool_calling=ToolCallingMode.JSON,
        ausgabe_dir=tmp_path,
        schreiben=quiet,
    )
    assert len(result.erp.belege) == 2
    assert all(r.state.mode == "json" for r in result.laeufe)
    result.freigaben.close()


def test_demo_over_http_with_native_tool_calls(fake_openai: FakeOpenAI, tmp_path: Path) -> None:
    """Der Fake-Server spielt das Modell und antwortet mit echten tool_calls über HTTP."""
    calls = iter(range(1, 1000))

    def model(request: dict[str, Any]) -> Reply:
        names = [t["function"]["name"] for t in request.get("tools", [])]
        turn = naechster_schritt(ScriptContext(request["messages"], names, False))
        if turn.tool_calls:
            name, arguments = turn.tool_calls[0]
            reply = chat(None, [(name, arguments)])
            reply.body["choices"][0]["message"]["tool_calls"][0]["id"] = f"call_{next(calls)}"
            return reply
        return chat(turn.content)

    fake_openai.queue += [model] * 40
    client = OpenAICompatClient(base_url=fake_openai.base_url, model="qwen2.5:7b-instruct")
    result = kreditoren_demo(client, ausgabe_dir=tmp_path, schreiben=quiet)
    assert sorted(b.betrag for b in result.erp.belege) == [Decimal("1081.00"), Decimal("12431.50")]
    assert result.ok
    assert all("tools" in r for r in fake_openai.requests)
    result.freigaben.close()


def test_cli_demo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["demo", "--ausgabe", str(tmp_path / "a")]) == 0
    assert "Zusammenfassung" in capsys.readouterr().out
    assert cli_main(["demo", "--freigabe", "ablehnen", "--ausgabe", str(tmp_path / "b")]) == 0
    with pytest.raises(SystemExit):
        cli_main(["demo", "--grad", "9"])


def test_interactive_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = iter(["j", "cfo", "n", "", "zu teuer"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    request: Any = type("Antrag", (), {"id": "apr_1"})()
    assert interaktive_freigabe(request) == (True, "cfo", "interaktiv freigegeben")
    assert interaktive_freigabe(request) == (False, "unbekannt", "zu teuer")


# --------------------------------------------------------- Zielsystem (ERP) --


def test_erp_enforces_its_own_rules() -> None:
    erp = beispiel_erp()
    zeilen = [
        Buchungszeile("4000", Decimal(100), Decimal(0), "x"),
        Buchungszeile("2000", Decimal(0), Decimal(100), "x"),
    ]
    vorschlag = erp.vorschlag_anlegen("R-1", "K-2001", zeilen)
    assert erp.vorschlag_anlegen("R-1", "K-2001", zeilen) is vorschlag  # idempotent
    with pytest.raises(ErpFehler, match="entspricht nicht"):
        erp.buchen(vorschlag.vorschlag_id, Decimal(99), dt.date(2026, 11, 1))
    erp.buchen(vorschlag.vorschlag_id, Decimal(100), dt.date(2026, 11, 1))
    with pytest.raises(ErpFehler, match="Doppelbuchung"):
        erp.buchen(vorschlag.vorschlag_id, Decimal(100), dt.date(2026, 11, 1))
    with pytest.raises(ErpFehler, match="existiert nicht"):
        erp.buchen("V-9999", Decimal(1), dt.date(2026, 11, 1))
    with pytest.raises(ErpFehler, match="unbekannt"):
        erp.vorschlag_anlegen("R-2", "K-0000", zeilen)
    with pytest.raises(ErpFehler, match="nicht ausgeglichen"):
        erp.vorschlag_anlegen("R-3", "K-2001", zeilen[:1])
    gross = [
        Buchungszeile("4000", Decimal(60000), Decimal(0), "x"),
        Buchungszeile("2000", Decimal(0), Decimal(60000), "x"),
    ]
    big = erp.vorschlag_anlegen("R-4", "K-2001", gross)
    with pytest.raises(ErpFehler, match="API-Limit"):
        erp.buchen(big.vorschlag_id, Decimal(60000), dt.date(2026, 11, 1))


def test_erp_rejection_reaches_the_model(tmp_path: Path) -> None:
    """Auch nach Freigabe gilt das Limit im Zielsystem (zweite Linie)."""
    from gradweight_orchestration.demo import ablauf

    original = ablauf.beispiel_erp

    def tiefes_limit() -> Any:
        erp = original()
        erp.api_limit = Decimal(10000)
        return erp

    ablauf.beispiel_erp = tiefes_limit  # type: ignore[assignment]
    try:
        result = kreditoren_demo(kreditoren_skript(), ausgabe_dir=tmp_path, schreiben=quiet)
    finally:
        ablauf.beispiel_erp = original  # type: ignore[assignment]
    assert [b.rechnungsnummer for b in result.erp.belege] == ["RE-2026-0042"]
    assert "API-Limit" in (result.laeufe[1].answer or "")
    result.freigaben.close()


# ------------------------------------------------------------ Rechnungstext --


def test_parser_detects_arithmetic_problems() -> None:
    erp = beispiel_erp()
    text = erp.posteingang["DOK-1001"].text
    ok = rechnung_parsen(text)
    assert ok.total == Decimal("1081.00")
    assert ok.befunde == []
    falsch = text.replace("MWST 8.1 %", "MWST 7.7 %").replace("1'081.00", "1'091.00")
    befunde = rechnung_parsen(falsch).befunde
    assert any("kein Schweizer Satz" in b for b in befunde)
    assert any("MWST-Betrag" in b for b in befunde)
    assert any("Netto + MWST" in b for b in befunde)
    kaputte_iban = text.replace("CH93 0076", "CH94 0076")
    assert "IBAN-Prüfziffer ungültig" in rechnung_parsen(kaputte_iban).befunde
    with pytest.raises(ValueError, match="Angaben fehlen: Rechnungsnummer"):
        rechnung_parsen(text.replace("Rechnung Nr.", "Beleg"))


def test_script_stops_on_findings_and_unknown_supplier() -> None:
    def ctx(*observations: dict[str, Any]) -> ScriptContext:
        messages = [{"role": "user", "content": "Bearbeite DOK-1 aus dem Posteingang."}]
        messages += [{"role": "tool", "content": json.dumps(o)} for o in observations]
        return ScriptContext(messages, [], False)

    rechnung = {
        "rechnungsnummer": "R-1",
        "befunde": ["MWST-Betrag 1 statt 2"],
        "lieferant_uid": "CHE-999.100.009",
        "iban": "x",
    }
    finding = naechster_schritt(
        ctx({"ok": True, "result": {"text": "t"}}, {"ok": True, "result": rechnung})
    )
    assert finding.content == "R-1 nicht gebucht: MWST-Betrag 1 statt 2"
    rechnung["befunde"] = []
    unknown = naechster_schritt(
        ctx(
            {"ok": True, "result": {"text": "t"}},
            {"ok": True, "result": rechnung},
            {
                "ok": True,
                "result": {"bekannt": False, "iban_stimmt": False, "hinweis": "unbekannt"},
            },
        )
    )
    assert unknown.content == "R-1 nicht gebucht: unbekannt"
    no_doc = naechster_schritt(ScriptContext([{"role": "user", "content": "x"}], [], False))
    assert no_doc.content == "Kein Dokument genannt."


def test_iban_check_and_short_texts() -> None:
    assert iban_gueltig("CH93 0076 2011 6238 5295 7")
    assert not iban_gueltig("CH93 0076 2011 6238 5295 8")
    assert not iban_gueltig("keine IBAN")
    assert _kurz(StepRecord(1, "tool", "x", observation={"ok": True, "result": [1, 2]})) == "[1, 2]"
    assert (
        _kurz(StepRecord(1, "tool", "x", observation={"ok": True, "result": {"a": 1}}))
        == '{"a": 1}'
    )
    assert _kurz(StepRecord(1, "tool", "x", observation=None)) == ""
    unknown = {"ok": True, "result": {"bekannt": False, "hinweis": "neu anlegen"}}
    assert (
        _kurz(StepRecord(1, "tool", "lieferant_abgleichen", observation=unknown)) == "neu anlegen"
    )


def test_unknown_document_and_supplier(tmp_path: Path) -> None:
    result = kreditoren_demo(
        kreditoren_skript(), dokumente=["DOK-9999"], ausgabe_dir=tmp_path, schreiben=quiet
    )
    assert "nicht im Posteingang" in (result.laeufe[0].answer or "")
    result.freigaben.close()
