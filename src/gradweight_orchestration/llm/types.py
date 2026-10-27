"""Nachrichten, Werkzeugaufrufe und das ``ChatModel``-Protokoll.

Nachrichten sind schlichte Dicts im Format der OpenAI-kompatiblen Chat-API
(``role``, ``content``, ``tool_calls``, ``tool_call_id``). Damit ist der
Laufzustand ohne Umwege als JSON speicherbar.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias, runtime_checkable

from .._json import JSONObject, JSONValue, canonical_json
from ..errors import OrchestrationError

Message: TypeAlias = dict[str, JSONValue]

__all__ = [
    "ChatModel",
    "LLMError",
    "LLMHTTPError",
    "LLMResponse",
    "LLMResponseError",
    "LLMTransportError",
    "Message",
    "ProtocolError",
    "ToolCall",
    "ToolsNotSupportedError",
    "Usage",
    "assistant_message",
    "estimate_tokens",
    "parse_tool_arguments",
    "system_message",
    "tool_message",
    "user_message",
]


class LLMError(OrchestrationError):
    """Basisklasse für Fehler beim Modellaufruf."""


class LLMTransportError(LLMError):
    """Netzwerkfehler oder Zeitüberschreitung (nach allen Wiederholungen)."""


class LLMHTTPError(LLMError):
    """Der Server hat mit einem Fehlerstatus geantwortet."""

    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status}: {body[:500]}")


class ToolsNotSupportedError(LLMHTTPError):
    """Modell oder Server unterstützt das ``tools``-Feld nicht."""


class LLMResponseError(LLMError):
    """Die Antwort hat nicht das erwartete Format."""


@dataclass(frozen=True)
class Usage:
    """Token-Verbrauch eines oder mehrerer Aufrufe."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated: bool = False
    """``True``, wenn mindestens ein Wert geschätzt statt vom Server gemeldet ist."""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.estimated or other.estimated,
        )


class ProtocolError(LLMError):
    """Das Modell hält das Antwortformat auch nach Korrekturversuchen nicht ein."""

    def __init__(self, message: str, *, raw: str = "", usage: Usage | None = None) -> None:
        self.raw = raw
        self.usage = Usage() if usage is None else usage
        super().__init__(message)


@dataclass(frozen=True)
class ToolCall:
    """Ein vom Modell verlangter Werkzeugaufruf."""

    id: str
    name: str
    arguments: JSONObject | None
    """Geparste Argumente; ``None``, wenn das Modell kein gültiges JSON-Objekt lieferte."""
    raw_arguments: str = "{}"
    error: str | None = None

    def to_message_part(self) -> JSONObject:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.raw_arguments},
        }


@dataclass(frozen=True)
class LLMResponse:
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    model: str | None = None


@runtime_checkable
class ChatModel(Protocol):
    """Alles, was Chat-Nachrichten beantworten kann (HTTP-Client, ``ScriptedLLM`` ...)."""

    @property
    def model_name(self) -> str: ...

    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[JSONObject] | None = None,
        json_mode: bool = False,
    ) -> LLMResponse: ...


def estimate_tokens(text: str) -> int:
    """Grobe Token-Schätzung (≈ 4 Zeichen pro Token). Nur, wo der Server nichts meldet."""
    return math.ceil(len(text) / 4) if text else 0


def _reject_constant(name: str) -> float:
    raise ValueError(f"{name} ist in JSON nicht erlaubt")


def parse_tool_arguments(
    raw: str | Mapping[str, JSONValue] | None,
) -> tuple[JSONObject | None, str, str | None]:
    """Parst Werkzeug-Argumente. Rückgabe: (Objekt oder None, Rohtext, Fehlertext)."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}, "{}", None
    if isinstance(raw, Mapping):
        obj = dict(raw)
        return obj, canonical_json(obj), None
    try:
        value = json.loads(raw, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        return None, raw, f"Argumente sind kein gültiges JSON: {exc}"
    if not isinstance(value, dict):
        return None, raw, "Argumente müssen ein JSON-Objekt sein"
    return value, raw, None


def system_message(text: str) -> Message:
    return {"role": "system", "content": text}


def user_message(text: str) -> Message:
    return {"role": "user", "content": text}


def assistant_message(content: str | None, tool_calls: Sequence[ToolCall] = ()) -> Message:
    message: Message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [call.to_message_part() for call in tool_calls]
    return message


def tool_message(call_id: str, content: str) -> Message:
    return {"role": "tool", "tool_call_id": call_id, "content": content}
