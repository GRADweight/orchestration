"""6 Tests für M6 — Agent-Memory (CANONICAL `gw_shared_lib.memory.Memory`).

S. GAP_ANALYSE.md, M6:

    class Memory:
        add(self, event, importance=float)         # episodic: JSONL
        get(self, query, k=5, threshold=0.7)       # retrieval via shared-lib
        forget(self, event_id)

    Abnahme: `from gw_shared_lib.memory import Memory; m.add('test'); m.get('test', k=1)`

Retrieval ist **bag-of-words** über `StubClient` (kein ML, kein Netz):
- gleiche Tokens → höhere cosine
- exakter Textmatch → cosine = 1.0 (threshold=1.0 filtert alles andere aus)
- kürzerer Treffer mit vollem Overlap rankt höher als langer mit demselben
  Token (Normeffekt: 1/√n)
"""
from __future__ import annotations

import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_VENDOR / "vendor"))
sys.path.insert(0, str(_VENDOR))

from gw_shared_lib.memory import Memory, MemoryEvent  # noqa: E402


# ── 1) add() liefert aufsteigende event_ids ───────────────────────────────────


def test_add_returns_increasing_ids(tmp_path: Path) -> None:
    """add() schreibt append-only JSONL und gibt aufsteigende ids zurück."""
    m = Memory(str(tmp_path / "mem.jsonl"))
    id0 = m.add("erstes ereignis", importance=0.5)
    id1 = m.add("zweites ereignis")
    assert id0 == 0
    assert id1 == 1
    # append-only: Zeilen wachsen
    lines = (tmp_path / "mem.jsonl").read_text().splitlines()
    assert len(lines) == 2


# ── 2) get() rankt relevante Ereignisse (bag-of-words) ───────────────────────


def test_get_ranks_relevant_first(tmp_path: Path) -> None:
    """Ähnlicher Query → relevante Treffer oben, unverbundene hinten."""
    m = Memory(str(tmp_path / "mem.jsonl"))
    m.add("komplette andere sache ohne bezug", importance=1.0)
    m.add("rechnung fällig 31.12", importance=0.5)
    m.add("rechnung gelesen betrag 1234 chf", importance=0.9)

    got = m.get("rechnung", k=3, threshold=0.0)
    # Beide 'rechnung'-Träger ranken vor dem unverbundenen Eintrag
    assert got[0].event.startswith("rechnung")
    assert got[1].event.startswith("rechnung")
    assert got[-1].event == "komplette andere sache ohne bezug"
    assert all(isinstance(e, MemoryEvent) for e in got)


# ── 3) threshold=1.0 → nur exakter Textmatch ──────────────────────────────────


def test_get_threshold_isolates_exact_match(tmp_path: Path) -> None:
    """threshold=1.0 filtert nur den exakten Text (cosine == 1.0)."""
    m = Memory(str(tmp_path / "mem.jsonl"))
    m.add("rechnung gelesen betrag 1234 chf", importance=0.9)
    m.add("komplette andere sache", importance=1.0)

    got = m.get("rechnung gelesen betrag 1234 chf", k=5, threshold=1.0)
    assert len(got) == 1
    assert got[0].event == "rechnung gelesen betrag 1234 chf"

    # Zu hoher threshold → leer
    assert m.get("rechnung", k=5, threshold=0.95) == []


# ── 4) forget() ist non-destruktiv (tombstone) ────────────────────────────────


def test_forget_tombstones_nondestructive(tmp_path: Path) -> None:
    """forget() markiert forgotten=True (tombstone) — der Eintrag bleibt im Dateifussabdruck,
    taucht aber nicht mehr in get() auf."""
    m = Memory(str(tmp_path / "mem.jsonl"))
    a = m.add("zu vergessendes ereignis")
    m.add("bleibendes ereignis")
    m.forget(a)

    got = m.get("vergessendes", k=5, threshold=0.0)
    assert all(e.event_id != a for e in got)
    # Non-destruktiv: Datei existiert noch, alle Zeilen bleiben
    assert len((tmp_path / "mem.jsonl").read_text().splitlines()) == 2
    # Eintrag ist als forgotten markiert
    reloaded = [e for e in m.get("x", k=10, threshold=0.0)]
    # forgotten-Eintrag taucht nicht auf; aber die Datei-Zeile bleibt
    assert "vergessendes" in (tmp_path / "mem.jsonl").read_text()


# ── 5) Persistenz über Reopen + fortlaufende ids ─────────────────────────────


def test_persistence_across_reopen(tmp_path: Path) -> None:
    """Neues Memory(gleicher Pfad) lädt alle bestehenden Ereignisse + ids laufen fort."""
    p = str(tmp_path / "mem.jsonl")
    m1 = Memory(p)
    m1.add("ereignis eins", importance=0.8)
    m1.add("ereignis zwei", importance=0.8)

    m2 = Memory(p)  # reopen
    got = m2.get("ereignis", k=5, threshold=0.0)
    assert {e.event for e in got} == {"ereignis eins", "ereignis zwei"}
    # Neue ids laufen fort (nicht von 0)
    new_id = m2.add("ereignis drei")
    assert new_id == 2


# ── 6) Export + Smoke (CANONICAL-Interface s. Abnahme) ───────────────────────


def test_export_and_smoke(tmp_path: Path) -> None:
    """`from gw_shared_lib import Memory; m.add('test'); m.get('test', k=1)` (M6-Abnahme)."""
    import gw_shared_lib

    m = gw_shared_lib.Memory(str(tmp_path / "mem.jsonl"))
    m.add("test", importance=1.0)
    got = m.get("test", k=1)
    assert got
    assert got[0].event == "test"
    assert "Memory" in gw_shared_lib.__all__


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
