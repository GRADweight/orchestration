"""Kontextfenster und Langzeitspeicher."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gradweight_orchestration.execution import ToolExecutor
from gradweight_orchestration.llm.types import (
    ToolCall,
    assistant_message,
    system_message,
    tool_message,
    user_message,
)
from gradweight_orchestration.memory import (
    ConversationWindow,
    LongTermMemory,
    memory_tools,
    message_tokens,
)


def history(steps: int, size: int = 200) -> list[dict[str, object]]:
    messages = [system_message("System"), user_message("Aufgabe")]
    for i in range(steps):
        call = ToolCall(f"c{i}", "lesen", {"i": i}, f'{{"i": {i}}}')
        messages.append(assistant_message(None, [call]))
        messages.append(tool_message(f"c{i}", "x" * size))
        if i % 2:
            messages.append(user_message("Hinweis: bitte weiter"))
    return messages  # type: ignore[return-value]


def test_small_history_is_unchanged() -> None:
    messages = history(2, 10)
    assert ConversationWindow(max_tokens=10_000).select(messages) == messages  # type: ignore[arg-type]


def test_old_steps_are_dropped_as_groups() -> None:
    messages = history(10)
    window = ConversationWindow(max_tokens=400)
    selected = window.select(messages)  # type: ignore[arg-type]
    assert selected[:2] == messages[:2]
    assert "ausgelassen" in str(selected[2]["content"])
    assert window.tokens(selected) <= 400
    assert selected[-1] == messages[-1]
    _assert_no_orphans(selected)


def _assert_no_orphans(selected: list[dict[str, object]]) -> None:
    announced: set[str] = set()
    for message in selected:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls", []):  # type: ignore[attr-defined]
                announced.add(call["id"])
        if message.get("role") == "tool":
            assert message["tool_call_id"] in announced


@settings(max_examples=80)
@given(steps=st.integers(0, 15), size=st.integers(1, 400), budget=st.integers(64, 3000))
def test_window_invariants(steps: int, size: int, budget: int) -> None:
    messages = history(steps, size)
    window = ConversationWindow(max_tokens=budget)
    selected = window.select(messages)  # type: ignore[arg-type]
    assert selected[:2] == messages[:2]
    _assert_no_orphans(selected)  # type: ignore[arg-type]
    if len(messages) > 2:
        assert selected[-1] == messages[-1]  # der jüngste Schritt ist immer dabei
        newest_group_start = (
            max(i for i, m in enumerate(messages) if m["role"] == "assistant") if steps else 2
        )
        minimum = window.tokens(messages[:2]) + window.tokens(messages[newest_group_start:])  # type: ignore[arg-type]
        assert window.tokens(selected) <= max(budget, minimum + 64)


def test_window_validation_and_token_count() -> None:
    with pytest.raises(ValueError, match="64"):
        ConversationWindow(max_tokens=10)
    assert message_tokens(user_message("abcd")) > 1


def test_long_term_memory(tmp_path: Path) -> None:
    path = tmp_path / "notizen.jsonl"
    store = LongTermMemory(path, clock=lambda: "2026-10-26T12:00:00+00:00")
    store.add("Lieferant Muster Bürobedarf AG: Zahlungsziel 30 Tage", tags=["lieferant"])
    store.add("Alpen Elektro GmbH verlangt Vorkasse ab CHF 10'000", tags=["lieferant"])
    store.add("Kantine: Menüplan am Montag", tags=["intern"])
    with pytest.raises(ValueError, match="leere"):
        store.add("  ")
    hits = store.search("Zahlungsziel Burobedarf")  # Umlaut-unabhängig
    assert hits[0][0].text.startswith("Lieferant Muster Bürobedarf")
    assert store.search("Menüplan", tags=["lieferant"]) == []
    assert store.search("") == []
    assert store.search("nichtvorhanden") == []
    reopened = LongTermMemory(path)
    assert len(reopened) == 3
    assert reopened.notes()[0].created_at == "2026-10-26T12:00:00+00:00"
    assert reopened.notes()[0].to_json()["tags"] == ["lieferant"]


def test_memory_tools() -> None:
    store = LongTermMemory()
    search, save = memory_tools(store)
    executor = ToolExecutor()
    saved = executor.execute(save, {"text": "Kunde K1 zahlt immer pünktlich"}, call_id="1")
    assert saved.ok
    found = executor.execute(search, {"suchbegriffe": "pünktlich", "anzahl": 50}, call_id="2")
    assert found.output == ["Kunde K1 zahlt immer pünktlich"]
    assert save.side_effect.value == "reversible_write"
