"""5 Tests für den ToolRouter (M4): Konfidenz, Tool-Auswahl, Fallback."""
from __future__ import annotations

import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_VENDOR / "vendor"))
sys.path.insert(0, str(_VENDOR))

from orch import ToolRouter, ToolSpec  # noqa: E402


def _specs() -> list[ToolSpec]:
    return [
        ToolSpec(name="lesen", description="Rechnung lesen", call=lambda: 1,
                 keywords=("rechnung", "lesen")),
        ToolSpec(name="buchen", description="In Bexio buchen", call=lambda: 2,
                 keywords=("buchen", "bexio")),
        ToolSpec(name="klassifiziere", description="E-Mail-Triage", call=lambda: 3,
                 keywords=("klassifizier", "triage")),
    ]


def test_konfidenz_wird_berechnet() -> None:
    router = ToolRouter(_specs(), threshold=0.1)
    decision = router.route("Rechnung lesen")
    assert decision.confidence > 0.0
    assert decision.confident is True


def test_werkzeug_auswahl_lesen() -> None:
    router = ToolRouter(_specs(), threshold=0.1, fallback=_specs()[0])
    decision = router.route("Die Rechnung lesen")
    assert decision.tool.name == "lesen"


def test_werkzeug_auswahl_buchen() -> None:
    router = ToolRouter(_specs(), threshold=0.1, fallback=_specs()[0])
    decision = router.route("Rechnung in Bexio buchen")
    assert decision.tool.name == "buchen"


def test_fallback_bei_null_treffer() -> None:
    specs = _specs()
    router = ToolRouter(specs, threshold=0.1, fallback=specs[2])
    decision = router.route("komplett fremder Text ohne keywords")
    assert decision.tool.name == "klassifiziere"
    assert decision.confidence == 0.0


def test_alternativen_werden_angegeben() -> None:
    router = ToolRouter(_specs(), threshold=0.05)
    decision = router.route("Rechnung lesen, dann buchen")
    # 'buchen' taucht auf -> Alternative existiert
    assert any(a.name == "buchen" for a in decision.alternatives)


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
