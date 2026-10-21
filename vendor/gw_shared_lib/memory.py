"""M6 — Episodic Memory für alle 6 GRADweight-Pakete.

Add()  : speichert ein Ereignis in einer JSONL-Datei (append-only).
Get()  : holt per Bag-of-Words-Similarität (StubClient, kein Vektor-DB,
         kein GPU) die k besten Ereignisse zurück.
Forget() : markiert ein Ereignis als vergessen (tombstone, non-destructive).

CANONICAL: gw_shared_lib @ f05457b4.
Hinzugefügt: 21.10.2026 für orchestration/M6 (NEU, nicht im CANONICAL-Snapshot).

Offline: gw_shared_lib.inference.StubClient (bag-of-words, kein Netz, kein ML).
Online : Setze GW_INFER_URL → gw_shared_lib.inference.Client
         (echte Embeddings via nomic-embed-text / Ollama / vLLM).

KMU-Sweet: JSONL, kein Vektor-Datenbank. Bei 1000+ Ereignisse:
           pgvector (s. catalog #18, gw-vector Paket, M18).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .inference import StubClient, cosine_similarity


def _jetzt_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class MemoryEvent:
    """Episodisches Ereignis — maschinenlesbar für alle 6 GRADweight-Pakete."""
    event_id: int
    event: str                     # der eigentliche Inhalt (kein PII)
    importance: float = 1.0
    forgotten: bool = False
    timestamp: str = field(default_factory=_jetzt_iso)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls: "type[MemoryEvent]", d: dict) -> "MemoryEvent":
        # Backward-compat: akzeptiert 'text' als alias
        text = d.get("event", d.get("text", ""))
        return cls(
            event_id=int(d.get("event_id", 0)),
            event=text,
            importance=float(d.get("importance", 1.0)),
            forgotten=bool(d.get("forgotten", False)),
            timestamp=d.get("timestamp", _jetzt_iso()),
        )


class Memory:
    """Episodic Memory für alle 6 GRADweight-Pakete (kurzes Interface s. M6).

    Beispiel::

        m = Memory(".audit/memory.jsonl")
        m.add("rechnung gelesen", importance=0.9)
        m.add("bexio Buchung erfolgreich")
        r = m.get("rechnung", k=3)
        assert len(r) >= 1
        assert r[0].event == "rechnung gelesen"
    """

    def __init__(self, path: str | Path = ".audit/memory.jsonl") -> None:
        self._path = Path(path)
        self._next_id = 0
        self._client = StubClient()
        self._load_next_id()

    # ── intern ─────────────────────────────────────────────────────────────

    def _load_next_id(self) -> None:
        if not self._path.exists():
            return
        nxt = 0
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
                eid = int(evt.get("event_id", 0))
                nxt = max(nxt, eid + 1)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        self._next_id = nxt

    def _all_events(self) -> list[MemoryEvent]:
        out: list[MemoryEvent] = []
        if not self._path.exists():
            return out
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(MemoryEvent.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                continue
        return out

    # ── API ────────────────────────────────────────────────────────────────

    def add(self, event: str, importance: float = 1.0) -> int:
        """Speichert ein Ereignis. Gibt event_id zurück."""
        eid = self._next_id
        self._next_id += 1
        evt = MemoryEvent(event_id=eid, event=event, importance=importance)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evt.to_dict(), ensure_ascii=False) + "\n")
        return eid

    def get(self, query: str, k: int = 5, threshold: float = 0.0) -> list[MemoryEvent]:
        """Holt die k besten Treffer per Bag-of-Words-Similarität.

        `threshold`: Mindest-Similarität (0.0 = alle, 0.7+ für ähnliche,
                     1.0 = exakt gleiche Texte). Mit StubClient (bag-of-words):
                     1.0 für exakt identische Texte, 0.3–0.5 für ähnliche.
                     Für Produktion: setze GW_INFER_URL → echte Embeddings.
        """
        q_vec = self._client.embed(query)[0]
        scored: list[tuple[float, MemoryEvent]] = []
        for evt in self._all_events():
            if evt.forgotten:
                continue
            ev_vec = self._client.embed(evt.event)[0]
            sim = cosine_similarity(q_vec, ev_vec)
            if sim >= threshold:
                scored.append((sim, evt))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ev for _, ev in scored[:k]]

    def forget(self, event_id: int) -> None:
        """Markiert Ereignis als vergessen (tombstone, non-destruktiv)."""
        if not self._path.exists():
            return
        lines = self._path.read_text(encoding="utf-8").splitlines()
        updated: list[str] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                evt = MemoryEvent.from_dict(json.loads(line))
                if evt.event_id == event_id:
                    evt.forgotten = True
                updated.append(json.dumps(evt.to_dict(), ensure_ascii=False))
            except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                updated.append(line)
        with self._path.open("w", encoding="utf-8") as f:
            f.write("\n".join(updated) + "\n")
