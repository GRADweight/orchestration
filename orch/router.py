"""Router — wählt ein Tool aus einer ToolSpec-Liste (M4).

Konfidenz-basiert: für jeden Sub-Step wird die beste Übereinstimmung
von Tool.keywords mit dem Sub-Step-Text berechnet. Confidence =
max Keyword-Score. `alternatives` liefert weitere Treffer mit
hohen Scores (für Debugging / Fallback).

Fallback: bei null Treffer (confidence < threshold) → `fallback`-Tool
(müsse vorhanden sein, sonst KeyError).
"""
from __future__ import annotations

import re
from typing import Any

from .types import RouterDecision, ToolSpec


def _score(tool: ToolSpec, text: str) -> float:
    """Bag-of-Words Übereinstimmung: wieviele Keywords vorkommen (0..1)."""
    low = text.lower()
    if not tool.keywords:
        return 0.0
    hits = sum(1 for kw in tool.keywords if kw.lower() in low)
    return min(1.0, hits / max(1.0, len(tool.keywords)))


class ToolRouter:
    """Wählt ein Tool aus tool_specs (M4 Router)."""

    def __init__(
        self,
        tool_specs: list[ToolSpec],
        *,
        threshold: float = 0.1,
        fallback: ToolSpec | None = None,
    ) -> None:
        if not tool_specs:
            raise ValueError("tool_specs muss nicht-leer sein")
        self._specs = list(tool_specs)
        self._threshold = threshold
        self._fallback = fallback

    def route(self, subtask_text: str) -> RouterDecision:
        """Bestes Tool + Confidence. `confident` = True wenn >= threshold."""
        scored: list[tuple[float, ToolSpec]] = []
        for spec in self._specs:
            s = _score(spec, subtask_text)
            if s > 0:
                scored.append((s, spec))
        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored or scored[0][0] < self._threshold:
            if self._fallback is None:
                raise KeyError(
                    f"Kein Tool für '{subtask_text}'; kein Fallback gegeben"
                )
            return RouterDecision(tool=self._fallback, confidence=0.0)

        best = scored[0]
        alternatives = [spec for _, spec in scored[1:]]
        return RouterDecision(tool=best[1], confidence=best[0], alternatives=alternatives)
