"""Agent-Memory — kurz + lang (M6).

Kurzzeit: Ring-Buff mit ~1000 "Tokens" (Wörtern) — alte Ereignisse
werden evicted, wenn der Puffer voll ist.
Langzeit: JSONL-Datei (persistiert, nie evictet).

CANONICAL offline: bag-of-words via gw_shared_lib.inference.StubClient.
Für Produktion: gw_infer_url → echtes Embedding.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from gw_shared_lib.inference import StubClient, cosine_similarity  # type: ignore


def asdict(obj) -> dict:
    """dataclass → dict (ohne dataclasses-asdict, für breite Kompat)."""
    return {k: v for k, v in vars(obj).items() if not k.startswith("_")}


@dataclass
class MemoryEntry:
    event: str
    importance: float = 1.0
    event_id: int = 0  # 0 = noch nicht zugewiesen


class Memory:
    """Kurz- + Langzeit-Memory für einen Agenten (M6).

    Kurzzeit: deque (Ring-Buff), maximal `max_tokens` Wörter.
    Langzeit: JSONL-Datei (`path`), nie evictet.
    """

    def __init__(
        self,
        *,
        max_tokens: int = 1000,
        path: str | Path | None = None,
    ) -> None:
        self._max_tokens = max_tokens
        self._ring: deque[MemoryEntry] = deque()
        self._path = Path(path) if path is not None else None
        self._client = StubClient()
        self._event_id = 0

    # ── Kurzzeit (Ring-Buff, evicts) ──────────────────────────────────────

    @property
    def ring(self) -> deque[MemoryEntry]:
        return self._ring

    @property
    def token_count(self) -> int:
        return sum(len(e.event.split()) for e in self._ring)

    def _short_add(self, entry: MemoryEntry) -> None:
        self._ring.append(entry)
        # Evict älteste, bis unter max_tokens
        while self.token_count > self._max_tokens and len(self._ring) > 1:
            self._ring.popleft()

    # ── Langzeit (JSONL, persistiert) ─────────────────────────────────────

    def _long_add(self, entry: MemoryEntry) -> None:
        if self._path is None:
            return
        entry.event_id = self._next_id()
        self._event_id = entry.event_id
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def _next_id(self) -> int:
        self._event_id += 1
        return self._event_id

    # ── API ────────────────────────────────────────────────────────────────

    def add(self, event: str, importance: float = 1.0) -> None:
        """Speichert in Kurz + Lang."""
        entry = MemoryEntry(event=event, importance=importance)
        self._short_add(entry)
        # Lang-Zeit mit eigener ID
        long_entry = MemoryEntry(event=event, importance=importance)
        self._long_add(long_entry)

    def get(self, query: str, k: int = 5, threshold: float = 0.0) -> list[MemoryEntry]:
        """Holt die k besten Treffer aus dem Ring-Buff per
        bag-of-words cosine_similarity (kein Netz, kein Vektor-DB).

        `threshold` = Mindest-Similarität (0.0 = alle; 0.7+ = gute Treffer).
        """
        q_vec = self._client.embed(query)[0]
        scored: list[tuple[float, MemoryEntry]] = []
        for evt in self._ring:
            ev_vec = self._client.embed(evt.event)[0]
            sim = cosine_similarity(q_vec, ev_vec)
            if sim >= threshold:
                scored.append((sim, evt))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ev for _, ev in scored[:k]]

    def forget(self, event: str) -> None:
        """Entfernt ein Ereignis aus Ring-Buff (nicht aus Langzeit-Datei)."""
        target = [e for e in self._ring if e.event == event]
        for e in target:
            self._ring.remove(e)
