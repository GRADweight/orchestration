"""Gedächtnis: Kontextfenster mit Token-Budget und einfacher Langzeitspeicher.

* :class:`ConversationWindow` bestimmt, welche Nachrichten das Modell sieht.
  Systemprompt und Aufgabe bleiben immer; ältere Schritte fallen als Ganzes weg
  (eine Assistenten-Nachricht mit allen zugehörigen Beobachtungen), damit nie
  ein Werkzeugergebnis ohne seinen Aufruf übrig bleibt. Der vollständige
  Verlauf bleibt im Laufzustand und im Audit-Log erhalten.
* :class:`LongTermMemory` speichert Notizen als JSON Lines und findet sie per
  BM25 über Wörter wieder – lexikalisch, ohne Embeddings. Das genügt für
  wiederkehrende Fakten («Lieferant X liefert immer mit 30 Tagen Ziel»), nicht
  für semantische Suche.

Token werden geschätzt (≈ 4 Zeichen pro Token), wenn kein eigener Zähler
übergeben wird; für knappe Kontextfenster einen modellspezifischen Zähler nutzen.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import re
import secrets
import threading
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._json import canonical_json
from .llm.types import Message, estimate_tokens, user_message
from .tools import SideEffect, Tool, tool

__all__ = ["ConversationWindow", "LongTermMemory", "MemoryNote", "memory_tools"]

_OVERHEAD_TOKENS = 4  # Rolle, Trennzeichen o.ä. je Nachricht


def message_tokens(message: Message, counter: Callable[[str], int] = estimate_tokens) -> int:
    return counter(canonical_json(message)) + _OVERHEAD_TOKENS


@dataclass(frozen=True)
class ConversationWindow:
    """Wählt die Nachrichten aus, die in ``max_tokens`` passen."""

    max_tokens: int
    counter: Callable[[str], int] = estimate_tokens

    def __post_init__(self) -> None:
        if self.max_tokens < 64:
            raise ValueError("max_tokens muss ≥ 64 sein")

    def tokens(self, messages: Sequence[Message]) -> int:
        return sum(message_tokens(m, self.counter) for m in messages)

    def select(self, messages: Sequence[Message]) -> list[Message]:
        """Fixierte Nachrichten + die jüngsten vollständigen Schritte, die hineinpassen."""
        pinned_count = 0
        if messages and messages[0].get("role") == "system":
            pinned_count = 1
        if len(messages) > pinned_count and messages[pinned_count].get("role") == "user":
            pinned_count += 1
        pinned = list(messages[:pinned_count])
        groups = _groups(messages[pinned_count:])
        if self.tokens(messages) <= self.max_tokens:
            return list(messages)
        budget = self.max_tokens - self.tokens(pinned)
        kept: list[list[Message]] = []
        for group in reversed(groups):
            cost = self.tokens(group)
            if kept and cost > budget - self._marker_cost(len(groups) - len(kept) - 1):
                break
            kept.insert(0, group)
            budget -= cost
        dropped = sum(len(g) for g in groups[: len(groups) - len(kept)])
        out = pinned
        if dropped:
            out = [*pinned, self._marker(dropped)]
        return out + [m for g in kept for m in g]

    @staticmethod
    def _marker(count: int) -> Message:
        return user_message(
            f"[Hinweis: {count} ältere Nachrichten wurden wegen des Kontextbudgets ausgelassen.]"
        )

    def _marker_cost(self, remaining_groups: int) -> int:
        return message_tokens(self._marker(99), self.counter) if remaining_groups > 0 else 0


def _groups(messages: Sequence[Message]) -> list[list[Message]]:
    """Gruppiert ab jeder Assistenten-Nachricht bis zur nächsten."""
    groups: list[list[Message]] = []
    for message in messages:
        if message.get("role") == "assistant" or not groups:
            groups.append([message])
        else:
            groups[-1].append(message)
    return groups


# ------------------------------------------------------------- Langzeit ---


@dataclass(frozen=True)
class MemoryNote:
    id: str
    text: str
    tags: tuple[str, ...] = ()
    created_at: str = ""
    source: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "source": self.source,
        }


_WORD = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(c for c in folded if not unicodedata.combining(c))
    return [t for t in _WORD.findall(plain) if len(t) > 1]


@dataclass
class LongTermMemory:
    """Notizen in einer JSONL-Datei (oder nur im Speicher), Suche per BM25."""

    path: str | Path | None = None
    k1: float = 1.5
    b: float = 0.75
    clock: Callable[[], str] = field(
        default=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    )

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._notes: list[MemoryNote] = []
        self._file = Path(self.path) if self.path is not None else None
        if self._file is not None and self._file.exists():
            for line in self._file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    raw = json.loads(line)
                    self._notes.append(
                        MemoryNote(
                            raw["id"],
                            raw["text"],
                            tuple(raw["tags"]),
                            raw["created_at"],
                            raw.get("source", ""),
                        )
                    )

    def __len__(self) -> int:
        return len(self._notes)

    def notes(self) -> list[MemoryNote]:
        with self._lock:
            return list(self._notes)

    def add(self, text: str, *, tags: Iterable[str] = (), source: str = "") -> MemoryNote:
        if not text.strip():
            raise ValueError("leere Notiz")
        note = MemoryNote(
            f"note_{secrets.token_hex(5)}", text.strip(), tuple(tags), self.clock(), source
        )
        with self._lock:
            self._notes.append(note)
            if self._file is not None:
                self._file.parent.mkdir(parents=True, exist_ok=True)
                with self._file.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(note.to_json(), ensure_ascii=False) + "\n")
        return note

    def search(
        self, query: str, *, k: int = 3, tags: Iterable[str] | None = None, min_score: float = 0.0
    ) -> list[tuple[MemoryNote, float]]:
        """Die ``k`` besten Notizen (BM25), optional gefiltert nach allen ``tags``."""
        wanted = set(tags or ())
        notes = [n for n in self.notes() if wanted <= set(n.tags)]
        terms = _tokens(query)
        if not notes or not terms:
            return []
        documents = [_tokens(n.text + " " + " ".join(n.tags)) for n in notes]
        average = sum(len(d) for d in documents) / len(documents) or 1.0
        frequency = Counter(t for d in documents for t in set(d))
        total = len(documents)
        scored: list[tuple[MemoryNote, float]] = []
        for note, words in zip(notes, documents, strict=True):
            counts = Counter(words)
            score = 0.0
            for term in terms:
                if term not in counts:
                    continue
                idf = math.log(1 + (total - frequency[term] + 0.5) / (frequency[term] + 0.5))
                tf = counts[term]
                norm = tf + self.k1 * (1 - self.b + self.b * len(words) / average)
                score += idf * tf * (self.k1 + 1) / norm
            if score > min_score:
                scored.append((note, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]


def memory_tools(store: LongTermMemory) -> list[Tool[..., Any]]:
    """Zwei Werkzeuge, mit denen ein Agent Notizen suchen und speichern kann."""

    @tool(name="notizen_suchen", side_effect=SideEffect.READ_ONLY, timeout_s=5)
    def notizen_suchen(suchbegriffe: str, anzahl: int = 3) -> list[str]:
        """Sucht in früheren Notizen (Stichwortsuche, keine Semantik).

        Args:
            suchbegriffe: Wörter, nach denen gesucht wird.
            anzahl: Maximale Anzahl Treffer.
        """
        return [note.text for note, _ in store.search(suchbegriffe, k=max(1, min(anzahl, 10)))]

    @tool(name="notiz_speichern", side_effect=SideEffect.REVERSIBLE, timeout_s=5)
    def notiz_speichern(text: str) -> str:
        """Speichert eine kurze Notiz für spätere Läufe.

        Args:
            text: Inhalt der Notiz (keine Personendaten).
        """
        return store.add(text, source="agent").id

    return [notizen_suchen, notiz_speichern]
