"""5 Tests für den RulePlanner (M4): Task → Liste, Liste → leer, Zykluserkennung."""
from __future__ import annotations

import sys
from pathlib import Path

# vendor/ + repo-Root auf sys.path (offline, wie prototypen-Testpfad)
_VENDOR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_VENDOR / "vendor"))
sys.path.insert(0, str(_VENDOR))

from orch import RulePlanner  # noqa: E402


def _planner() -> RulePlanner:
    return RulePlanner()


def test_task_aufgabe_wird_zerlegt() -> None:
    """Aufzählender Text -> 3 Sub-Steps."""
    plan = _planner().plan("Rechnung lesen, in Bexio buchen, dann klassifizieren")
    assert len(plan.subtasks) == 3
    assert "lesen" in plan.subtasks[0].text.lower()
    assert "buchen" in plan.subtasks[1].text.lower()
    assert "klassifizier" in plan.subtasks[2].text.lower()


def test_liste_leer_zurueck() -> None:
    """Leerer Text -> leerer Plan (keine Sub-Steps, nicht cyclic)."""
    plan = _planner().plan("")
    assert plan.subtasks == []
    assert plan.cyclic is False


def test_zykluserkennung() -> None:
    """Zwei identische aufeinanderfolgende Schritte -> cyclic=True."""
    plan = _planner().plan("Schritt A, Schritt A")
    assert plan.cyclic is True


def test_kein_zyklus_bei_verschiedenen() -> None:
    """Zwei verschiedene Schritte -> cyclic=False."""
    plan = _planner().plan("Schritt A, Schritt B")
    assert plan.cyclic is False
    assert len(plan.subtasks) == 2


def test_einziger_schritt_kein_splitter_treffer() -> None:
    """Text ohne Trennzeichen -> ein einzelner Sub-Step."""
    plan = _planner().plan("einsachen Aufgabe ohne trennende Wörter")
    # 'einsachen' etc. wird nicht gesplittet; mindestens 1 Sub-Step
    assert len(plan.subtasks) >= 1


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
