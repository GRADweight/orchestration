"""Deterministisches Modell für Tests und Offline-Demos.

Ein Skript ist eine Folge von Zügen. Ein Zug ist entweder fest
(:class:`ScriptedTurn`) oder eine Funktion, die aus dem bisherigen Verlauf den
nächsten Zug berechnet – so lassen sich Argumente aus früheren Beobachtungen
übernehmen, wie es ein echtes Modell tun würde::

    llm = ScriptedLLM([
        ScriptedTurn.call("rechnung_lesen", {"dokument_id": "RE-1"}),
        lambda ctx: ScriptedTurn.call("buchen", {"betrag": ctx.last_result()["betrag"]}),
        ScriptedTurn.final("Gebucht."),
    ])

Im JSON-Modus rendert ``ScriptedLLM`` dieselben Züge im JSON-Protokoll.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Union

from .._json import JSONObject, JSONValue, canonical_json, to_jsonable
from ..errors import OrchestrationError
from .types import (
    LLMResponse,
    Message,
    ToolCall,
    ToolsNotSupportedError,
    Usage,
    estimate_tokens,
)

__all__ = ["ScriptContext", "ScriptExhaustedError", "ScriptedLLM", "ScriptedTurn"]


class ScriptExhaustedError(OrchestrationError):
    """Das Modell wurde öfter aufgerufen, als das Skript Züge hat."""


@dataclass(frozen=True)
class ScriptedTurn:
    content: str | None = None
    tool_calls: tuple[tuple[str, JSONObject], ...] = ()
    raw: str | None = None
    """Unverarbeiteter Antworttext (z.B. absichtlich ungültiges JSON)."""
    thought: str | None = None

    @classmethod
    def call(
        cls, name: str, arguments: Mapping[str, object] | None = None, *, thought: str | None = None
    ) -> ScriptedTurn:
        args = to_jsonable(dict(arguments or {}))
        assert isinstance(args, dict)  # noqa: S101 - dict bleibt dict
        return cls(tool_calls=((name, args),), thought=thought)

    @classmethod
    def calls(cls, *pairs: tuple[str, Mapping[str, object]]) -> ScriptedTurn:
        rendered: list[tuple[str, JSONObject]] = []
        for name, arguments in pairs:
            args = to_jsonable(dict(arguments))
            assert isinstance(args, dict)  # noqa: S101 - dict bleibt dict
            rendered.append((name, args))
        return cls(tool_calls=tuple(rendered))

    @classmethod
    def final(cls, text: str) -> ScriptedTurn:
        return cls(content=text)

    @classmethod
    def text(cls, raw: str) -> ScriptedTurn:
        return cls(raw=raw)


@dataclass
class ScriptContext:
    """Sicht eines Skriptschritts auf den Verlauf."""

    messages: list[Message]
    tools: list[str]
    json_mode: bool

    def observations(self) -> list[JSONValue]:
        """Alle Beobachtungen (geparstes JSON) in Reihenfolge."""
        found: list[JSONValue] = []
        for message in self.messages:
            content = message.get("content")
            if not isinstance(content, str):
                continue
            if message.get("role") == "tool":
                payload = content
            elif message.get("role") == "user" and content.startswith("Beobachtung zu "):
                payload = content.split("\n", 1)[1] if "\n" in content else ""
            else:
                continue
            try:
                found.append(json.loads(payload))
            except json.JSONDecodeError:
                found.append(payload)
        return found

    def last_observation(self) -> JSONValue:
        observations = self.observations()
        if not observations:
            raise LookupError("noch keine Beobachtung im Verlauf")
        return observations[-1]

    def last_result(self) -> JSONValue:
        """``result`` der letzten erfolgreichen Beobachtung."""
        for observation in reversed(self.observations()):
            if isinstance(observation, dict) and observation.get("ok") is True:
                return observation.get("result")
        raise LookupError("keine erfolgreiche Beobachtung im Verlauf")


Step = Union[ScriptedTurn, Callable[[ScriptContext], ScriptedTurn]]  # noqa: UP007


@dataclass(frozen=True)
class ScriptedRequest:
    messages: list[Message]
    tools: list[JSONObject] | None
    json_mode: bool


@dataclass
class ScriptedLLM:
    """Spielt ein Skript ab; protokolliert alle Anfragen in :attr:`requests`."""

    steps: Iterable[Step]
    model_name: str = "scripted"
    supports_tools: bool = True
    requests: list[ScriptedRequest] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._steps = list(self.steps)
        self._calls = 0

    @property
    def remaining(self) -> int:
        return len(self._steps)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[JSONObject] | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        snapshot = json.loads(json.dumps(list(messages)))
        self.requests.append(
            ScriptedRequest(
                messages=snapshot, tools=list(tools) if tools else None, json_mode=json_mode
            )
        )
        if tools and not self.supports_tools:
            raise ToolsNotSupportedError(400, f"model {self.model_name} does not support tools")
        if not self._steps:
            raise ScriptExhaustedError(f"Skript erschöpft nach {self._calls} Zügen")
        step = self._steps.pop(0)
        self._calls += 1
        names = [
            str(spec["function"]["name"])  # type: ignore[call-overload,index]
            for spec in (tools or [])
        ]
        turn = step(ScriptContext(snapshot, names, json_mode)) if callable(step) else step
        if json_mode:
            content: str | None = turn.raw if turn.raw is not None else _render_json(turn)
            calls: tuple[ToolCall, ...] = ()
        else:
            content = turn.raw if turn.raw is not None else turn.content
            calls = tuple(
                ToolCall(
                    id=f"call_{self._calls}_{index}",
                    name=name,
                    arguments=arguments,
                    raw_arguments=canonical_json(arguments),
                )
                for index, (name, arguments) in enumerate(turn.tool_calls)
            )
        completion = (content or "") + "".join(c.raw_arguments for c in calls)
        usage = Usage(
            prompt_tokens=estimate_tokens(canonical_json(snapshot)),
            completion_tokens=estimate_tokens(completion),
            estimated=True,
        )
        return LLMResponse(
            content=content,
            tool_calls=calls,
            finish_reason="tool_calls" if calls else "stop",
            usage=usage,
            model=self.model_name,
        )


def _render_json(turn: ScriptedTurn) -> str:
    if len(turn.tool_calls) > 1:
        raise ValueError("Das JSON-Protokoll erlaubt einen Werkzeugaufruf pro Antwort")
    obj: JSONObject = {}
    if turn.thought:
        obj["thought"] = turn.thought
    if turn.tool_calls:
        name, arguments = turn.tool_calls[0]
        obj["tool"] = name
        obj["arguments"] = arguments
    else:
        obj["final_answer"] = turn.content or ""
    return json.dumps(obj, ensure_ascii=False)
