"""Austauschformate: nativ (mit Erkennung im Text) und JSON-Protokoll mit Korrektur."""

from __future__ import annotations

import pytest

from gradweight_orchestration import SideEffect, tool
from gradweight_orchestration.llm import (
    JsonToolCalling,
    NativeToolCalling,
    ProtocolError,
    ScriptContext,
    ScriptedLLM,
    ScriptedTurn,
    ScriptExhaustedError,
    ToolCallingMode,
    ToolsNotSupportedError,
    parse_json_turn,
)
from gradweight_orchestration.llm.protocol import strategy_for


@tool(side_effect=SideEffect.READ_ONLY)
def kurs(waehrung: str) -> float:
    """Kurs einer Währung in CHF."""
    return 0.94


def ids() -> object:
    counter = iter(range(100))
    return lambda: f"id{next(counter)}"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            '{"tool": "kurs", "arguments": {"waehrung": "EUR"}}',
            (("kurs", {"waehrung": "EUR"}), None, None),
        ),
        ('```json\n{"thought": "fertig", "final_answer": "0.94"}\n```', (None, "0.94", "fertig")),
        ('  {"tool": "kurs", "arguments": {}}  ', (("kurs", {}), None, None)),
    ],
)
def test_parse_valid(text: str, expected: object) -> None:
    assert parse_json_turn(text, ["kurs"]) == expected


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("Ich rufe kurs auf", "kein gültiges JSON"),
        ('{"tool": "kurs", "arguments": {}} Danke!', "kein gültiges JSON"),
        ("[1, 2]", "JSON-Objekt"),
        ('{"tool": "kurs", "final_answer": "x"}', "genau eines"),
        ("{}", "genau eines"),
        ('{"tool": "kurs", "arguments": {}, "extra": 1}', "unbekannte Schlüssel"),
        ('{"tool": "fehlt", "arguments": {}}', "unbekanntes Werkzeug"),
        ('{"tool": "kurs", "arguments": []}', "arguments"),
        ('{"tool": "kurs"}', "arguments"),
        ('{"final_answer": ""}', "nicht-leerer"),
        ('{"final_answer": "x", "arguments": {}}', "nur zusammen"),
        ('{"thought": 1, "final_answer": "x"}', "thought"),
        ('{"tool": "kurs", "arguments": {"x": NaN}}', "NaN"),
    ],
)
def test_parse_invalid(text: str, fragment: str) -> None:
    with pytest.raises(ProtocolError) as info:
        parse_json_turn(text, ["kurs"])
    assert fragment in str(info.value)


def test_json_protocol_repairs_invalid_output() -> None:
    llm = ScriptedLLM(
        [
            ScriptedTurn.text("Klar, ich prüfe den Kurs."),
            ScriptedTurn.call("kurs", {"waehrung": "EUR"}, thought="Kurs holen"),
        ]
    )
    strategy = JsonToolCalling(max_repairs=2)
    turn = strategy.request(llm, [{"role": "user", "content": "Kurs?"}], [kurs], new_id=ids())
    assert turn.repairs == 1
    assert turn.thought == "Kurs holen"
    assert turn.tool_calls[0].name == "kurs"
    assert turn.tool_calls[0].arguments == {"waehrung": "EUR"}
    assert turn.tool_calls[0].id == "id0"
    assert turn.mode is ToolCallingMode.JSON
    repair_request = llm.requests[1].messages
    assert repair_request[-1]["content"].startswith("Deine letzte Antwort war ungültig")
    assert all(r.json_mode for r in llm.requests)
    assert turn.usage.prompt_tokens > 0


def test_json_protocol_gives_up_after_max_repairs() -> None:
    llm = ScriptedLLM([ScriptedTurn.text("nein")] * 3)
    with pytest.raises(ProtocolError) as info:
        JsonToolCalling(max_repairs=2).request(llm, [], [kurs], new_id=ids())
    assert "2 Korrekturversuch" in str(info.value)
    assert info.value.usage.completion_tokens > 0
    with pytest.raises(ValueError, match="negativ"):
        JsonToolCalling(max_repairs=-1)


def test_json_system_addendum_lists_tools() -> None:
    text = JsonToolCalling().system_addendum([kurs])
    assert "- kurs: Kurs einer Währung in CHF." in text
    assert '"waehrung"' in text
    assert "keine" in JsonToolCalling().system_addendum([])
    observation = JsonToolCalling().observation(
        __import__("gradweight_orchestration.llm", fromlist=["ToolCall"]).ToolCall(
            "c1", "kurs", {}
        ),
        '{"ok": true}',
    )
    assert observation["role"] == "user"
    assert str(observation["content"]).startswith("Beobachtung zu kurs (Aufruf c1)")


def test_native_tool_calls_and_recovery_from_content() -> None:
    llm = ScriptedLLM(
        [
            ScriptedTurn.call("kurs", {"waehrung": "USD"}),
            ScriptedTurn.text('{"name": "kurs", "arguments": {"waehrung": "EUR"}}'),
            ScriptedTurn.text('{"name": "unbekannt", "arguments": {}}'),
            ScriptedTurn.text('{"name": "kurs", "arguments": {}, "zusatz": 1}'),
            ScriptedTurn.text("[1]"),
            ScriptedTurn.final("Antwort"),
        ]
    )
    native = NativeToolCalling()
    turn = native.request(llm, [], [kurs], new_id=ids())
    assert turn.tool_calls[0].arguments == {"waehrung": "USD"}
    assert turn.recovered is False
    assert llm.requests[0].tools is not None
    recovered = native.request(llm, [], [kurs], new_id=ids())
    assert recovered.recovered is True
    assert recovered.tool_calls[0].arguments == {"waehrung": "EUR"}
    assert recovered.content is None
    for _ in range(3):
        plain = native.request(llm, [], [kurs], new_id=ids())
        assert plain.tool_calls == ()
        assert plain.recovered is False
    final = native.request(llm, [], [], new_id=ids())
    assert final.content == "Antwort"
    assert llm.requests[-1].tools is None
    assert native.observation(recovered.tool_calls[0], "{}")["role"] == "tool"
    assert native.system_addendum([kurs]) == ""


def test_strategy_factory() -> None:
    assert isinstance(strategy_for(ToolCallingMode.AUTO), NativeToolCalling)
    assert isinstance(strategy_for(ToolCallingMode.NATIVE), NativeToolCalling)
    assert isinstance(strategy_for(ToolCallingMode.JSON, max_repairs=1), JsonToolCalling)


def test_scripted_llm_context_and_errors() -> None:
    seen: list[object] = []

    def step(ctx: ScriptContext) -> ScriptedTurn:
        seen.append(ctx.last_result())
        seen.append(ctx.tools)
        return ScriptedTurn.final("ok")

    llm = ScriptedLLM([step], supports_tools=True)
    messages = [
        {"role": "tool", "tool_call_id": "a", "content": '{"ok": true, "result": 1}'},
        {"role": "user", "content": 'Beobachtung zu x (Aufruf b):\n{"ok": false}'},
        {"role": "tool", "tool_call_id": "c", "content": "kein json"},
    ]
    llm.complete(messages, tools=[kurs.spec()])
    assert seen == [1, ["kurs"]]
    assert llm.remaining == 0
    with pytest.raises(ScriptExhaustedError):
        llm.complete([])
    no_tools = ScriptedLLM([ScriptedTurn.final("x")], supports_tools=False)
    with pytest.raises(ToolsNotSupportedError):
        no_tools.complete([], tools=[kurs.spec()])
    ctx = ScriptContext([{"role": "user", "content": "Beobachtung zu x"}], [], False)
    assert ctx.observations() == [""]
    assert ScriptContext([], [], False).observations() == []
    with pytest.raises(LookupError):
        ScriptContext([], [], False).last_observation()
    with pytest.raises(LookupError):
        ScriptContext([{"role": "tool", "content": '{"ok": false}'}], [], False).last_result()


def test_scripted_json_rendering() -> None:
    llm = ScriptedLLM([ScriptedTurn.calls(("kurs", {}), ("kurs", {}))])
    with pytest.raises(ValueError, match="einen Werkzeugaufruf"):
        llm.complete([], json_mode=True)
    multi = ScriptedLLM([ScriptedTurn.calls(("kurs", {"waehrung": "EUR"}), ("kurs", {}))])
    response = multi.complete([])
    assert [c.id for c in response.tool_calls] == ["call_1_0", "call_1_1"]
    final = ScriptedLLM([ScriptedTurn.final("fertig")]).complete([], json_mode=True)
    assert final.content == '{"final_answer": "fertig"}'
