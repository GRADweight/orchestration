"""Demo: Rechnung lesen → klassifizieren → Bexio → Audit (5 Schritte, M4-M7).

Lauffähig offline (kein Netz, kein 7B nötig). Nutzt die Regel-Planner/
Observer (Testpfad) + StubClient. Für Produktion: setze GW_INFER_URL und
tausche RulePlanner/RuleObserver gegen LLMPlanner/LLMObserver.

Befehl:
    cd path/to/orchestration  # z.B. ~/code/GRADweight/orchestration
    PYTHONPATH=vendor:. python -m examples.demo_rechnung
"""
from __future__ import annotations

import sys
from pathlib import Path

# vendor/ + repo-Root auf sys.path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "vendor"))
sys.path.insert(0, str(_ROOT))

from orch import (  # noqa: E402
    AgentRunner,
    ApprovalGate,
    FeedbackLoop,
    Memory,
    ToolSpec,
)
from gw_shared_lib import audit as gw_audit  # noqa: E402


# ── 5 Beispiel-Tools (s. M4 "5 Beispiel-Tools") ─────────────────────────────
def rechen_ocr(d: dict | None = None) -> str:
    d = d or {}
    return f"OCR: betrag CHF {d.get('betrag', 0):.2f}, mwst {d.get('mwst', 0):.1f}%"


def e_mail_klassifikation(d: dict | None = None) -> str:
    return "Kategorie: verwaltung (Konfidenz 0.9)"


def web_recherche(d: dict | None = None) -> str:
    return "Quelle: interne HR-Reglement, 3 Fundstellen"


def bexio_api(d: dict | None = None) -> str:
    d = d or {}
    return f"Bexio: Buchung RE-{d.get('nr', '0000')} angelegt (CHF {d.get('betrag', 0):.2f})"


def datei_lesen(d: dict | None = None) -> str:
    return "Datei gelesen: 12 Zeilen, gültiges PDF-Struktur"


SPECS = [
    ToolSpec(name="rechnungs-ocr", description="Rechnung per OCR lesen",
             call=rechen_ocr, keywords=("ocr", "rechnung", "les", "betrag")),
    ToolSpec(name="email-triage", description="E-Mail klassifizieren",
             call=e_mail_klassifikation, keywords=("klassifizier", "email", "triage")),
    ToolSpec(name="web-recherche", description="Quellen recherchieren",
             call=web_recherche, keywords=("recherche", "quelle", "fundstelle")),
    ToolSpec(name="bexio-buchen", description="Rechnung in Bexio buchen",
             call=bexio_api, keywords=("bexio", "buchen", "buchung")),
    ToolSpec(name="datei-lesen", description="Datei einlesen und prüfen",
             call=datei_lesen, keywords=("datei", "pdf", "lesen")),
]


def main() -> int:
    import tempfile
    import time

    mandant = "K1"
    prototyp = "02-rechnung"
    # Jeder Lauf nutzt ein frisches temp-Dir (sonst akkumulieren die
    # JSONL-Files und die exakten Zähler-Asserts schlugen beim 2. Lauf.)
    _tmp = tempfile.mkdtemp(prefix="gw-orch-demo-")

    print("=" * 68)
    print("GRADweight orchestration — Demo: Rechnung → Bexio (M4-M7)")
    print("=" * 68)

    # ── 1. M4: Agent-Loop (Planner -> Router -> Executor -> Observer) ──────
    log = gw_audit.AuditLog(f"{_tmp}/demo_{int(time.time())}.jsonl")
    runner = AgentRunner(tool_specs=SPECS, threshold=0.05, max_steps=8,
                         audit_log=log)

    print("\n[M4] Agent-Loop: 3-Schritt-Flow")
    res = runner.run(
        "Rechnung per OCR lesen, dann in Bexio buchen, schliesslich E-Mail-Triage",
        mandant=mandant,
    )
    for i, obs in enumerate(res.observations):
        mark = "OK " if obs.successful else "!! "
        print(f"  {mark} Schritt {i}: {obs.status.value:12s} "
              f"({obs.reason[:52]})")
    print(f"  -> done={res.done}, task={res.task.value}, steps={res.steps_run}")
    assert res.done and res.successful, "M4-Flow schlug fehl"
    # Intention: 3 Schritte, keiner lief per Fallback (Router-Treffer)
    assert res.observations[0].reason.startswith("Erfolgreich"), \
        "Schritt 0 lief nicht via Router-Treffer"

    # ── 2. M6: Memory ──────────────────────────────────────────────────────
    print("\n[M6] Memory: Kurz- + Langzeit")
    mem = Memory()
    mem.add("rechnung ocr gelesen: 1081 CHF", importance=0.9)
    mem.add("bexio buchung erfolgreich")
    got = mem.get("rechnung", k=2)
    print(f"  get('rechnung') -> {[e.event for e in got][:1]}")
    assert got and "rechnung" in got[0].event.lower()

    # ── 3. M7: Human-in-the-Loop (Approval via gw_core) ───────────────────
    print("\n[M7] Approval (Human-in-the-Loop, gw_core.ApprovalQueue)")
    gate = ApprovalGate(mandant=mandant, prototyp=prototyp)
    aid = gate.vorschlag(titel="Rechnung RE-2026-0042 buchen",
                         payload={"betrag": 1081.0, "mwst": 8.1})
    entscheid = gate.freigabe(aid, actor="cfo", anmerkungen="geprüft ok")
    print(f"  vorschlag -> aktion {aid}; freigabe -> "
          f"{entscheid.status} (ausgeführt? {entscheid.ausgefuehrt_aktiv})")
    assert entscheid.ausgefuehrt_aktiv is True

    # Ein Zurückstellen (menschlicher VETO-Kettendurchbruch)
    aid2 = gate.vorschlag(titel="Doppel-Buchung vermeiden",
                          payload={"betrag": 1081.0})
    veto = gate.zurueck(aid2, actor="cfo", begruendung="Bereits gebucht")
    print(f"  veto -> aktion {aid2}: {veto.status} "
          f"(ausgeführt? {veto.ausgefuehrt_aktiv})")
    assert veto.ausgefuehrt_aktiv is False

    # ── 4. M7: Feedback-Loop ──────────────────────────────────────────────
    print("\n[M7] Feedback-Loop (Annotation good/error)")
    fb = FeedbackLoop(f"{_tmp}/feedback_{int(time.time())}.jsonl")
    fb.annotiere(task="ocr", result_summary="1081 CHF ok", label="gut")
    fb.annotiere(task="bexio", result_summary="timeout", label="fehler")
    counts = fb.zähler()
    print(f"  Annotationen: {counts}")
    assert counts["gut"] == 1 and counts["fehler"] == 1

    # ── 5. Audit-Auswertung ───────────────────────────────────────────────
    print("\n[Audit] gw_shared_lib: jede Tool-Aktion protokolliert")
    entries = log.read_all()
    print(f"  {len(entries)} Einträge (tool_ergebnis)")
    print(f"  Einseiter (M7):")
    for line in gate.einseiter().strip().splitlines():
        print(f"    {line}")

    print("\n" + "=" * 68)
    print("STATUS: OK — alle M4-M7 Bausteine lief durch")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
