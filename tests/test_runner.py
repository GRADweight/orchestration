"""5 Tests für den AgentRunner (M4): 3-Schritt-Demo, Timeout, max_steps,
Audit-Trail, Zyklus-Abort."""
from __future__ import annotations

import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_VENDOR / "vendor"))
sys.path.insert(0, str(_VENDOR))

from orch import AgentRunner, ToolSpec  # noqa: E402
from gw_shared_lib import audit as gw_audit  # noqa: E402


def _specs() -> list[ToolSpec]:
    return [
        ToolSpec(name="lesen", description="Rechnung lesen",
                 call=lambda: "gelesen: 1081 CHF",
                 keywords=("rechnung", "lesen")),
        ToolSpec(name="buchen", description="In Bexio buchen",
                 call=lambda: "buchung erfolgreich",
                 keywords=("buchen", "bexio")),
        ToolSpec(name="klassifiziere", description="E-Mail-Triage",
                 call=lambda: "verwaltung",
                 keywords=("klassifizier", "triage")),
    ]


def test_drei_schritt_fluss_erledigt() -> None:
    """Vollständiger 3-Schritt-Flow: alle OK -> done, ERLEDIGT."""
    runner = AgentRunner(tool_specs=_specs(), threshold=0.1, max_steps=6)
    res = runner.run("Rechnung lesen, in Bexio buchen, dann klassifizieren",
                     mandant="K1")
    assert res.done is True
    assert res.successful is True
    assert res.steps_run == 3
    assert len(res.observations) == 3
    assert all(o.successful for o in res.observations)


def test_timeout_bei_langsamer_tool() -> None:
    """Ein zu langsames Tool -> TimeOut -> UERWARTET, nicht ERLEDIGT."""
    import time

    def schlaefen():
        time.sleep(0.3)
        return "spät"

    specs = [ToolSpec(name="schlaefer", description="langsam",
                      call=schlaefen, timeout_s=0.05,
                      keywords=("schlaf",))
             ]
    runner = AgentRunner(tool_specs=specs, threshold=0.01, max_steps=3)
    res = runner.run("ein langer schlaf", mandant="K1")
    assert res.done is False
    assert res.observations and res.observations[0].status.value == "unerwartet"


def test_max_steps_grenzt() -> None:
    """max_steps=1 -> nur 1 Schritt, dann ABGESAGT."""
    runner = AgentRunner(tool_specs=_specs(), threshold=0.1, max_steps=1)
    res = runner.run("a, b, c, d", mandant="K1")
    assert res.done is False
    assert res.task.value == "abgesagt"
    assert res.steps_run <= 1


def test_audit_wird_geschrieben(tmp_path: Path) -> None:
    """Jeder Tool-Aufruf landet im gw_shared_lib Audit-Log."""
    log = gw_audit.AuditLog(str(tmp_path / "audit.jsonl"))
    runner = AgentRunner(tool_specs=_specs(), threshold=0.1, max_steps=6,
                         audit_log=log)
    res = runner.run("Rechnung lesen, in Bexio buchen", mandant="K1")
    assert res.done is True
    entries = log.read_all()
    # 2 Tools, 2 Audit-Einträge
    assert len(entries) == 2
    assert all(e.get("action") == "tool_ergebnis" for e in entries)


def test_zyklus_wird_abgebrochen() -> None:
    """Ein zyklischer Plan -> ABGESAGT, kein Loop-Endlos."""
    runner = AgentRunner(tool_specs=_specs(), threshold=0.1, max_steps=10)
    res = runner.run("Schritt A, Schritt A, Schritt A", mandant="K1")
    assert res.done is False
    assert res.task.value == "abgesagt"
    assert "Zyklus" in res.reason or "zyklisch" in res.reason.lower()


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
