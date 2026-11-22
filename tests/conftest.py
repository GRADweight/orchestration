"""Gemeinsame Fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from fake_openai import FakeOpenAI


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeOpenAI]:
    """Lokaler OpenAI-kompatibler Server; Proxy-Variablen werden für Loopback umgangen."""
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    with FakeOpenAI() as server:
        yield server
