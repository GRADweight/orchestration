"""3 Tests für Memory (M6): Ring-Buff evicts, Langzeit persistiert."""
from __future__ import annotations

import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_VENDOR / "vendor"))
sys.path.insert(0, str(_VENDOR))

from orch import Memory  # noqa: E402


def test_ring_buff_evicts_aeltesten() -> None:
    """Wenn max_tokens überschritten, evictet der Ring die ältesten."""
    mem = Memory(max_tokens=10)
    # 5 Ereignisse à 3 Wörter = 15 > 10
    for i in range(5):
        mem.add(f"ereignis nummer {i}", importance=1.0)
    # Ring muss unter 10 Tokens sein, aber nicht leer
    assert 0 < mem.token_count <= 10
    assert len(mem.ring) >= 1


def test_langzeit_persistiert(tmp_path: Path) -> None:
    """Langzeit-JSONL schreibt wirklich auf Disk."""
    pfad = tmp_path / "memory.jsonl"
    mem = Memory(path=str(pfad))
    mem.add("rechnung gelesen", importance=0.9)
    mem.add("bexio buchung")
    assert pfad.exists()
    inhalt = pfad.read_text(encoding="utf-8")
    assert "rechnung gelesen" in inhalt
    assert "bexio buchung" in inhalt


def test_get_holt_beste_treffer() -> None:
    """bag-of-words: 'rechnung' holt den passenden Eintrag zuerst."""
    mem = Memory()
    mem.add("rechnung gelesen betrag chf", importance=1.0)
    mem.add("komplett andere sache", importance=1.0)
    got = mem.get("rechnung", k=1)
    assert got
    assert "rechnung" in got[0].event


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
