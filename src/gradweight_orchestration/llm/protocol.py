"""Wie Werkzeugaufrufe zwischen Runtime und Modell ausgetauscht werden.

* :class:`NativeToolCalling` nutzt das ``tools``/``tool_calls``-Feld der API.
  Kleine Modelle schreiben den Aufruf manchmal als JSON in den Text statt in
  ``tool_calls``; solche Antworten werden erkannt (``recovered=True``), wenn
  sie exakt einem bekannten Werkzeug entsprechen.
* :class:`JsonToolCalling` ist der Rückfall für Modelle ohne natives
  Tool-Calling: Werkzeuge stehen im Systemprompt, das Modell antwortet im
  JSON-Modus mit genau einem Objekt. Die Antwort wird strikt geprüft; bei
  Verstössen erhält das Modell den Befund und bis zu ``max_repairs``
  Korrekturversuche.
"""

from __future__ import annotations

import enum
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .._json import JSONObject, canonical_json
from ..tools import Tool
from .types import (
    ChatModel,
    Message,
    ProtocolError,
    ToolCall,
    Usage,
    assistant_message,
    tool_message,
    user_message,
)

__all__ = [
    "AssistantTurn",
    "JsonToolCalling",
    "NativeToolCalling",
    "ToolCallingMode",
    "parse_json_turn",
]

_FENCE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)
_ALLOWED_KEYS = {"thought", "tool", "arguments", "final_answer"}

JSON_PROTOCOL = """\
Antwortformat: Antworte IMMER mit genau einem JSON-Objekt und sonst nichts.
Werkzeug aufrufen:  {"thought": "<kurz>", "tool": "<name>", "arguments": {<argumente>}}
Abschliessen:       {"thought": "<kurz>", "final_answer": "<Antwort an den Benutzer>"}
Pro Antwort höchstens ein Werkzeugaufruf. Nach jedem Aufruf erhältst du eine Beobachtung.

Verfügbare Werkzeuge (Argumente als JSON-Schema):
"""

REPAIR_PROMPT = (
    "Deine letzte Antwort war ungültig: {error}\n"
    "Antworte erneut mit genau einem JSON-Objekt gemäss Antwortformat, ohne weiteren Text."
)


class ToolCallingMode(str, enum.Enum):
    AUTO = "auto"
    """Nativ beginnen; meldet der Server fehlende Tool-Unterstützung, auf JSON wechseln."""
    NATIVE = "native"
    JSON = "json"


@dataclass(frozen=True)
class AssistantTurn:
    """Normalisierte Modellantwort, unabhängig vom Austauschformat."""

    content: str | None
    tool_calls: tuple[ToolCall, ...]
    message: Message
    usage: Usage
    model: str | None
    mode: ToolCallingMode
    repairs: int = 0
    recovered: bool = False
    thought: str | None = None


IdFactory = Callable[[], str]


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    match = _FENCE.match(stripped)
    return match.group(1).strip() if match else stripped


def _load_object(text: str) -> Any:
    try:
        return json.loads(_strip_fence(text), parse_constant=_no_constants)
    except json.JSONDecodeError as exc:
        raise ProtocolError(
            f"kein gültiges JSON ({exc.msg}, Position {exc.pos})", raw=text
        ) from exc
    except ValueError as exc:
        raise ProtocolError(str(exc), raw=text) from exc


def _no_constants(name: str) -> float:
    raise ValueError(f"{name} ist in JSON nicht erlaubt")


def parse_json_turn(
    text: str, tool_names: Sequence[str]
) -> tuple[tuple[str, JSONObject] | None, str | None, str | None]:
    """Prüft eine Antwort im JSON-Protokoll.

    Rückgabe: ``((werkzeug, argumente) | None, final_answer | None, thought | None)``.
    Wirft :class:`ProtocolError` mit einer für das Modell verständlichen Begründung.
    """
    obj = _load_object(text)
    if not isinstance(obj, dict):
        raise ProtocolError("die Antwort muss ein JSON-Objekt sein", raw=text)
    unknown = sorted(set(obj) - _ALLOWED_KEYS)
    if unknown:
        raise ProtocolError(
            f"unbekannte Schlüssel {unknown}; erlaubt: thought, tool, arguments, final_answer",
            raw=text,
        )
    thought = obj.get("thought")
    if thought is not None and not isinstance(thought, str):
        raise ProtocolError("'thought' muss ein String sein", raw=text)
    has_tool, has_final = "tool" in obj, "final_answer" in obj
    if has_tool == has_final:
        raise ProtocolError("genau eines von 'tool' oder 'final_answer' angeben", raw=text)
    if has_final:
        if "arguments" in obj:
            raise ProtocolError("'arguments' nur zusammen mit 'tool'", raw=text)
        final = obj["final_answer"]
        if not isinstance(final, str) or not final.strip():
            raise ProtocolError("'final_answer' muss ein nicht-leerer String sein", raw=text)
        return None, final, thought
    name = obj["tool"]
    if not isinstance(name, str) or name not in tool_names:
        raise ProtocolError(
            f"unbekanntes Werkzeug {name!r}; verfügbar: {', '.join(tool_names) or 'keine'}",
            raw=text,
        )
    arguments = obj.get("arguments")
    if not isinstance(arguments, dict):
        raise ProtocolError("'arguments' muss ein JSON-Objekt sein (ggf. {})", raw=text)
    return (name, arguments), None, thought


class NativeToolCalling:
    """Werkzeuge über das ``tools``-Feld der Chat-API."""

    mode = ToolCallingMode.NATIVE

    def __init__(self, *, recover_from_content: bool = True) -> None:
        self._recover = recover_from_content

    def system_addendum(self, tools: Sequence[Tool[..., Any]]) -> str:
        return ""

    def request(
        self,
        model: ChatModel,
        messages: Sequence[Message],
        tools: Sequence[Tool[..., Any]],
        *,
        new_id: IdFactory,
    ) -> AssistantTurn:
        specs = [t.spec() for t in tools]
        response = model.complete(messages, tools=specs or None)
        calls = response.tool_calls
        content = response.content
        recovered = False
        if not calls and content and tools and self._recover:
            found = _recover_call(content, [t.name for t in tools], new_id)
            if found is not None:
                calls, content, recovered = (found,), None, True
        return AssistantTurn(
            content=content,
            tool_calls=calls,
            message=assistant_message(content, calls),
            usage=response.usage,
            model=response.model,
            mode=self.mode,
            recovered=recovered,
        )

    def observation(self, call: ToolCall, payload: str) -> Message:
        return tool_message(call.id, payload)


def _recover_call(content: str, names: Sequence[str], new_id: IdFactory) -> ToolCall | None:
    """Erkennt ``{"name": ..., "arguments": {...}}`` oder das JSON-Protokoll im Text."""
    try:
        obj = json.loads(_strip_fence(content), parse_constant=_no_constants)
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name", obj.get("tool"))
    arguments = obj.get("arguments", obj.get("parameters", {}))
    if name not in names or not isinstance(arguments, dict):
        return None
    if set(obj) - {"name", "tool", "arguments", "parameters", "thought"}:
        return None
    return ToolCall(
        id=new_id(),
        name=str(name),
        arguments=arguments,
        raw_arguments=canonical_json(arguments),
    )


class JsonToolCalling:
    """Werkzeuge über ein JSON-Protokoll im Prompt (für Modelle ohne Tool-Calling)."""

    mode = ToolCallingMode.JSON

    def __init__(self, *, max_repairs: int = 2) -> None:
        if max_repairs < 0:
            raise ValueError("max_repairs darf nicht negativ sein")
        self.max_repairs = max_repairs

    def system_addendum(self, tools: Sequence[Tool[..., Any]]) -> str:
        lines = [JSON_PROTOCOL]
        for item in tools:
            lines.append(f"- {item.name}: {item.description}")
            lines.append(f"  arguments: {canonical_json(item.parameters)}")
        if not tools:
            lines.append("(keine – antworte direkt mit final_answer)")
        return "\n".join(lines)

    def request(
        self,
        model: ChatModel,
        messages: Sequence[Message],
        tools: Sequence[Tool[..., Any]],
        *,
        new_id: IdFactory,
    ) -> AssistantTurn:
        names = [t.name for t in tools]
        conversation = list(messages)
        usage = Usage()
        repairs = 0
        while True:
            response = model.complete(conversation, json_mode=True)
            usage = usage + response.usage
            text = response.content or ""
            try:
                call, final, thought = parse_json_turn(text, names)
            except ProtocolError as exc:
                if repairs >= self.max_repairs:
                    raise ProtocolError(
                        f"Antwortformat nach {repairs} Korrekturversuch(en) verletzt: {exc}",
                        raw=text,
                        usage=usage,
                    ) from exc
                repairs += 1
                conversation += [
                    assistant_message(text),
                    user_message(REPAIR_PROMPT.format(error=exc)),
                ]
                continue
            calls: tuple[ToolCall, ...] = ()
            if call is not None:
                name, arguments = call
                calls = (
                    ToolCall(
                        id=new_id(),
                        name=name,
                        arguments=arguments,
                        raw_arguments=canonical_json(arguments),
                    ),
                )
            return AssistantTurn(
                content=final,
                tool_calls=calls,
                message=assistant_message(_strip_fence(text)),
                usage=usage,
                model=response.model,
                mode=self.mode,
                repairs=repairs,
                thought=thought,
            )

    def observation(self, call: ToolCall, payload: str) -> Message:
        return user_message(f"Beobachtung zu {call.name} (Aufruf {call.id}):\n{payload}")


Strategy = NativeToolCalling | JsonToolCalling


def strategy_for(mode: ToolCallingMode, *, max_repairs: int = 2) -> Strategy:
    """Strategie für einen *wirksamen* Modus (``AUTO`` beginnt nativ)."""
    if mode is ToolCallingMode.JSON:
        return JsonToolCalling(max_repairs=max_repairs)
    return NativeToolCalling()
