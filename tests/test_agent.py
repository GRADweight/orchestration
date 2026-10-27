"""Agent-Loop: Werkzeugaufrufe, Neuplanung, Budgets, Policy, Pause/Fortsetzung."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any

import pytest

from fake_openai import FakeOpenAI, Reply, chat
from gradweight_orchestration import (
    Agent,
    ApprovalQueue,
    AuditLog,
    AutonomyGrade,
    Budget,
    ConversationWindow,
    LongTermMemory,
    OpenAICompatClient,
    Param,
    Policy,
    Pricing,
    Redactor,
    RetryPolicy,
    RunState,
    RunStatus,
    ScriptedLLM,
    ScriptedTurn,
    SideEffect,
    ToolCallingMode,
    Tracer,
    tool,
)
from gradweight_orchestration.errors import OrchestrationError
from gradweight_orchestration.llm import ScriptContext

BOOKINGS: list[tuple[str, Decimal]] = []


@dataclass(frozen=True)
class Rechnung:
    nummer: str
    betrag: Decimal
    iban: str


@tool(side_effect=SideEffect.READ_ONLY)
def rechnung_lesen(nummer: str) -> Rechnung:
    """Liest eine Rechnung aus dem Eingang.

    Args:
        nummer: Rechnungsnummer, z.B. RE-1.
    """
    betrag = Decimal("12430.50") if nummer == "RE-GROSS" else Decimal("1081.00")
    return Rechnung(nummer, betrag, "CH93 0076 2011 6238 5295 7")


@tool(side_effect=SideEffect.MONEY, amount_param="betrag")
def buchen(nummer: str, betrag: Annotated[Decimal, Param(ge=0)]) -> str:
    """Bucht die Rechnung im ERP."""
    BOOKINGS.append((nummer, betrag))
    return f"BU-{len(BOOKINGS)}"


@tool(side_effect=SideEffect.READ_ONLY, timeout_s=0.3)
def haengt() -> str:
    """Hängt."""
    time.sleep(5)
    return "nie"


@tool(side_effect=SideEffect.READ_ONLY)
def kaputt(versuch: int) -> str:
    """Schlägt immer fehl."""
    raise RuntimeError(f"defekt {versuch}")


@tool(side_effect=SideEffect.READ_ONLY)
def langer_text() -> str:
    """Liefert viel Text."""
    return "x" * 20_000


TOOLS = [rechnung_lesen, buchen, haengt, kaputt, langer_text]


@pytest.fixture(autouse=True)
def _reset() -> None:
    BOOKINGS.clear()


def book_from_last(ctx: ScriptContext) -> ScriptedTurn:
    """Wie ein Modell: Nummer und Betrag aus der letzten Beobachtung übernehmen."""
    result = ctx.last_result()
    assert isinstance(result, dict)
    return ScriptedTurn.call("buchen", {"nummer": result["nummer"], "betrag": result["betrag"]})


def make_agent(llm: Any, **kwargs: Any) -> Agent:
    options: dict[str, Any] = {
        "name": "kreditoren",
        "llm": llm,
        "tools": TOOLS,
        "policy": Policy(grade=AutonomyGrade.MANDATE, money_limit=Decimal(5000)),
    }
    options.update(kwargs)
    return Agent(**options)


def script(nummer: str = "RE-1") -> list[Any]:
    return [
        ScriptedTurn.call("rechnung_lesen", {"nummer": nummer}),
        book_from_last,
        ScriptedTurn.final("Gebucht."),
    ]


def kinds(audit: AuditLog) -> list[str]:
    return [r.kind for r in audit.records()]


# ------------------------------------------------------------- Grundablauf --


def test_full_loop_arguments_flow_and_audit() -> None:
    audit, tracer = AuditLog(), Tracer()
    llm = ScriptedLLM(script())
    result = make_agent(llm, audit=audit, tracer=tracer).run("Rechnung RE-1 buchen")
    assert result.ok
    assert result.answer == "Gebucht."
    assert [("RE-1", Decimal("1081"))] == BOOKINGS  # Betrag aus Beobachtung, nicht 0.00
    assert [s.kind for s in result.steps] == ["tool", "tool", "final"]
    assert result.steps[1].decision == "allow"
    assert result.usage.steps == 3
    assert result.usage.estimated is True
    assert audit.verify().ok
    assert kinds(audit)[:3] == ["run_started", "llm_request", "llm_response"]
    assert "policy_decision" in kinds(audit)
    assert kinds(audit)[-1] == "run_finished"
    policy = next(
        r for r in audit.records() if r.kind == "policy_decision" and r.data["tool"] == "buchen"
    )
    assert policy.data["rule"] == "mandat:innerhalb"
    names = [s.name for s in tracer.spans]
    assert "invoke_agent kreditoren" in names
    assert names.count("chat scripted") == 3
    root = next(s for s in tracer.spans if s.name == "invoke_agent kreditoren")
    tool_span = next(s for s in tracer.spans if s.name == "execute_tool buchen")
    chat_span = next(s for s in tracer.spans if s.name == "chat scripted")
    assert tool_span.parent_id == root.span_id
    assert chat_span.attributes["gw.prompt_sha256"]
    assert root.attributes["gw.status"] == "completed"
    # Beobachtungen gehen strukturiert als tool-Nachricht zurück
    second = llm.requests[1].messages
    assert second[-1]["role"] == "tool"
    assert json.loads(second[-1]["content"])["result"]["betrag"] == 1081


def test_json_protocol_end_to_end() -> None:
    llm = ScriptedLLM(script())
    result = make_agent(llm, tool_calling=ToolCallingMode.JSON).run("Rechnung RE-1 buchen")
    assert result.ok
    assert [("RE-1", Decimal("1081"))] == BOOKINGS
    assert all(r.json_mode and r.tools is None for r in llm.requests)
    system = llm.requests[0].messages[0]["content"]
    assert "Antwortformat" in system
    assert "- buchen: Bucht die Rechnung im ERP." in system
    assert str(llm.requests[1].messages[-1]["content"]).startswith("Beobachtung zu rechnung_lesen")
    assert result.state.mode == "json"


def test_auto_mode_switches_when_tools_unsupported() -> None:
    audit = AuditLog()
    llm = ScriptedLLM(script(), supports_tools=False)
    result = make_agent(llm, audit=audit).run("Rechnung RE-1 buchen")
    assert result.ok
    assert result.state.mode == "json"
    assert "tool_mode_switched" in kinds(audit)
    assert llm.requests[0].tools is not None
    assert llm.requests[1].json_mode


def test_native_mode_does_not_switch() -> None:
    llm = ScriptedLLM(script(), supports_tools=False)
    result = make_agent(llm, tool_calling=ToolCallingMode.NATIVE).run("x")
    assert result.status is RunStatus.FAILED
    assert result.reason == "llm_error"


def test_against_openai_compatible_server_with_fallback(fake_openai: FakeOpenAI) -> None:
    """Echter HTTP-Pfad: Server ohne Tool-Support → JSON-Protokoll → Buchung."""
    unsupported = Reply(status=400, body={"error": {"message": "model does not support tools"}})

    def booking(request: dict[str, Any]) -> Reply:
        observation = request["messages"][-1]["content"]
        result = json.loads(observation.split("\n", 1)[1])["result"]
        content = json.dumps(
            {
                "tool": "buchen",
                "arguments": {"nummer": result["nummer"], "betrag": result["betrag"]},
            }
        )
        return chat(content)

    fake_openai.queue += [
        unsupported,
        chat('{"thought": "lesen", "tool": "rechnung_lesen", "arguments": {"nummer": "RE-9"}}'),
        booking,
        chat('{"final_answer": "RE-9 gebucht."}'),
    ]
    client = OpenAICompatClient(
        base_url=fake_openai.base_url, model="llama3.2:3b", retry=RetryPolicy(max_attempts=1)
    )
    result = make_agent(client).run("Bitte RE-9 buchen")
    assert result.ok, result.message
    assert [("RE-9", Decimal("1081"))] == BOOKINGS
    assert result.usage.prompt_tokens == 30  # vom Server gemeldet, 3 × 10
    assert fake_openai.requests[1]["response_format"] == {"type": "json_object"}


def test_invalid_and_duplicate_ids_from_server(fake_openai: FakeOpenAI) -> None:
    body = chat().body
    body["choices"][0]["message"]["tool_calls"] = [
        {
            "id": "x",
            "type": "function",
            "function": {"name": "rechnung_lesen", "arguments": '{"nummer": "RE-1"}'},
        },
        {"id": "x", "type": "function", "function": {"name": "buchen", "arguments": "{kaputt"}},
    ]
    fake_openai.queue += [Reply(body=body), chat("Fertig.")]
    client = OpenAICompatClient(base_url=fake_openai.base_url, model="m")
    result = make_agent(client).run("x")
    assert result.ok
    first, second = result.steps[0], result.steps[1]
    assert first.call_id == "x"
    assert second.call_id != "x"
    assert second.ok is False
    error = second.observation["error"]  # type: ignore[index, call-overload]
    assert error["type"] == "validation_error"
    assert "kein gültiges JSON" in error["message"]
    sent = fake_openai.requests[1]["messages"]
    assert [m.get("tool_call_id") for m in sent if m["role"] == "tool"] == ["x", second.call_id]


# ------------------------------------------------------- Fehler, Neuplanung --


def test_validation_error_triggers_replan_and_recovery() -> None:
    llm = ScriptedLLM(
        [
            ScriptedTurn.call("buchen", {"nummer": "RE-1", "betrag": "1081.00"}),
            ScriptedTurn.call("buchen", {"nummer": "RE-1", "betrag": 1081.0}),
            ScriptedTurn.final("ok"),
        ]
    )
    result = make_agent(llm).run("buchen")
    assert result.ok
    assert [("RE-1", Decimal("1081.0"))] == BOOKINGS
    retry_view = llm.requests[1].messages
    observation = json.loads(retry_view[-2]["content"])
    assert observation["error"]["type"] == "validation_error"
    assert observation["error"]["details"][0]["path"] == "$.betrag"
    assert retry_view[-1]["role"] == "user"
    assert "fehlgeschlagen" in retry_view[-1]["content"]
    assert result.state.consecutive_errors == 0


def test_unknown_tool_and_policy_denied() -> None:
    llm = ScriptedLLM(
        [
            ScriptedTurn.call("gibts_nicht", {}),
            ScriptedTurn.call("buchen", {"nummer": "RE-1", "betrag": 1}),
            ScriptedTurn.final("Nicht möglich."),
        ]
    )
    agent = make_agent(llm, policy=Policy(grade=AutonomyGrade.ASSISTANT))
    result = agent.run("x")
    assert result.ok
    assert result.steps[0].observation["error"]["type"] == "unknown_tool"  # type: ignore[index, call-overload]
    assert result.steps[1].decision == "deny"
    assert result.steps[1].observation["error"]["type"] == "policy_denied"  # type: ignore[index, call-overload]
    assert BOOKINGS == []


def test_too_many_consecutive_errors_stop() -> None:
    llm = ScriptedLLM([ScriptedTurn.call("kaputt", {"versuch": i}) for i in range(5)])
    result = make_agent(llm, max_consecutive_errors=3).run("x")
    assert result.status is RunStatus.STOPPED
    assert result.reason == "too_many_errors"
    assert result.usage.steps == 3


def test_identical_calls_are_detected() -> None:
    llm = ScriptedLLM([ScriptedTurn.call("rechnung_lesen", {"nummer": "RE-1"})] * 5)
    result = make_agent(llm, max_identical_calls=2, max_consecutive_errors=10).run("x")
    assert result.status is RunStatus.STOPPED
    assert result.reason == "loop_detected"
    executed = [s for s in result.steps if s.ok]
    assert len(executed) == 2
    assert result.steps[2].observation["error"]["type"] == "duplicate_call"  # type: ignore[index, call-overload]


def test_empty_answer_is_nudged() -> None:
    llm = ScriptedLLM([ScriptedTurn.final(""), ScriptedTurn.final("Jetzt richtig.")])
    result = make_agent(llm).run("x")
    assert result.ok
    assert result.steps[0].kind == "notice"
    assert "leer" in str(llm.requests[1].messages[-1]["content"])
    stubborn = ScriptedLLM([ScriptedTurn.final("")] * 3)
    assert make_agent(stubborn).run("x").reason == "too_many_errors"


def test_tool_timeout_is_an_observation() -> None:
    llm = ScriptedLLM([ScriptedTurn.call("haengt", {}), ScriptedTurn.final("Zeitüberschreitung.")])
    started = time.monotonic()
    result = make_agent(llm).run("x")
    assert time.monotonic() - started < 2.0
    assert result.ok
    assert result.steps[0].observation["error"]["type"] == "timeout"  # type: ignore[index, call-overload]


def test_protocol_and_llm_errors_fail_the_run(fake_openai: FakeOpenAI) -> None:
    broken = ScriptedLLM([ScriptedTurn.text("kein json")] * 3)
    result = make_agent(broken, tool_calling=ToolCallingMode.JSON, max_repairs=2).run("x")
    assert result.status is RunStatus.FAILED
    assert result.reason == "protocol_error"
    assert result.usage.completion_tokens > 0
    fake_openai.queue += [Reply(status=500, body={})]
    client = OpenAICompatClient(
        base_url=fake_openai.base_url, model="m", retry=RetryPolicy(max_attempts=1)
    )
    assert make_agent(client).run("x").reason == "llm_error"


def test_long_errors_keep_their_type() -> None:
    @tool(side_effect=SideEffect.READ_ONLY)
    def laut(x: int) -> str:
        """Fehler mit langem Text."""
        raise ValueError("Ursache " * 500)

    llm = ScriptedLLM([ScriptedTurn.call("laut", {"x": 1}), ScriptedTurn.final("ok")])
    make_agent(llm, tools=[laut], max_observation_chars=400).run("x")
    payload = llm.requests[1].messages[-2]["content"]
    assert len(payload) <= 400
    data = json.loads(payload)
    assert data["error"]["type"] == "tool_exception"
    assert data["error"]["message"].startswith("ValueError: Ursache")


def test_unexpected_errors_still_close_the_audit_run() -> None:
    audit = AuditLog()
    llm = ScriptedLLM([])  # erschöpft sofort: kein LLMError, sondern Programmierfehler
    with pytest.raises(OrchestrationError, match="erschöpft"):
        make_agent(llm, audit=audit).run("x")
    last = audit.records()[-1]
    assert last.kind == "run_aborted"
    assert "ScriptExhaustedError" in str(last.data["error"])
    assert audit.verify().ok


def test_long_observations_are_truncated_validly() -> None:
    llm = ScriptedLLM([ScriptedTurn.call("langer_text", {}), ScriptedTurn.final("ok")])
    make_agent(llm, max_observation_chars=500).run("x")
    payload = llm.requests[1].messages[-1]["content"]
    assert len(payload) <= 500
    assert json.loads(payload)["hinweis"] == "Ergebnis gekürzt"


# ------------------------------------------------------------------ Budgets --


def endless() -> list[ScriptedTurn]:
    return [ScriptedTurn.call("kaputt", {"versuch": i}) for i in range(50)]


def test_max_steps() -> None:
    result = make_agent(
        ScriptedLLM(endless()), budget=Budget(max_steps=4), max_consecutive_errors=100
    ).run("x")
    assert result.reason == "max_steps"
    assert result.usage.steps == 4


def test_token_and_cost_budget() -> None:
    tokens = make_agent(
        ScriptedLLM(endless()), budget=Budget(max_tokens=300), max_consecutive_errors=100
    ).run("x")
    assert tokens.reason == "token_budget"
    pricing = Pricing(input_per_1k=Decimal(1), output_per_1k=Decimal(1))
    cost = make_agent(
        ScriptedLLM(endless()),
        budget=Budget(max_cost=Decimal("0.5"), pricing=pricing),
        max_consecutive_errors=100,
    ).run("x")
    assert cost.reason == "cost_budget"
    assert cost.usage.cost >= Decimal("0.5")


def test_time_budget_counts_only_active_time() -> None:
    now = [0.0]

    def clock() -> float:
        now[0] += 10.0
        return now[0]

    result = make_agent(
        ScriptedLLM(endless()),
        budget=Budget(max_seconds=55),
        max_consecutive_errors=100,
        clock=clock,
    ).run("x")
    assert result.reason == "time_budget"


def test_time_budget_caps_tool_timeout() -> None:
    @tool(side_effect=SideEffect.READ_ONLY, timeout_s=30)
    def langsam() -> str:
        """Langsam."""
        time.sleep(3)
        return "spät"

    llm = ScriptedLLM([ScriptedTurn.call("langsam", {}), ScriptedTurn.final("x")])
    started = time.monotonic()
    result = make_agent(llm, tools=[langsam], budget=Budget(max_seconds=0.5)).run("x")
    assert time.monotonic() - started < 2.0
    assert result.steps[0].observation["error"]["type"] == "timeout"  # type: ignore[index, call-overload]
    assert result.reason == "time_budget"


# ------------------------------------------------------- Freigaben, Pause --


def test_pause_approve_resume_in_new_agent_instance(tmp_path: Path) -> None:
    db, log = tmp_path / "freigaben.sqlite", tmp_path / "audit.jsonl"
    llm = ScriptedLLM(script("RE-GROSS")[:2])
    first = make_agent(llm, approvals=ApprovalQueue(db), audit=AuditLog(log))
    paused = first.run("RE-GROSS buchen", run_id="lauf-1")
    assert paused.status is RunStatus.PAUSED
    assert paused.reason == "approval_pending"
    assert BOOKINGS == []
    request = paused.pending_approvals[0]
    assert request.tool == "buchen"
    assert request.amount is not None
    assert Decimal(request.amount) == Decimal("12430.50")
    assert "über Mandatslimit CHF 5'000.00" in request.reason
    saved = json.dumps(paused.state.to_json())

    # Mensch entscheidet in einem anderen Prozess über die CLI
    cli = [sys.executable, "-m", "gradweight_orchestration", "approvals"]
    listing = subprocess.run(
        [*cli, "list", "--db", str(db)], capture_output=True, text=True, check=True
    )
    assert request.id in listing.stdout
    subprocess.run(
        [
            *cli,
            "approve",
            request.id,
            "--db",
            str(db),
            "--audit",
            str(log),
            "--by",
            "m.keller",
            "--note",
            "Lieferung geprüft",
        ],
        check=True,
        capture_output=True,
    )

    second_llm = ScriptedLLM([ScriptedTurn.final("RE-GROSS nach Freigabe gebucht.")])
    second = make_agent(second_llm, approvals=ApprovalQueue(db), audit=AuditLog(log))
    resumed = second.resume(json.loads(saved))
    assert resumed.ok
    assert [("RE-GROSS", Decimal("12430.5"))] == BOOKINGS
    approved = next(s for s in resumed.steps if s.tool == "buchen" and s.ok)
    assert approved.note == "freigegeben durch m.keller"
    assert approved.decision == "approved"
    assert second_llm.requests[0].messages[-1]["role"] == "tool"
    audit = AuditLog(log)
    assert audit.verify().ok
    sequence = kinds(audit)
    for kind in (
        "approval_requested",
        "run_paused",
        "approval_decided",
        "run_resumed",
        "approval_applied",
        "tool_result",
        "run_finished",
    ):
        assert kind in sequence
    assert sequence.index("approval_decided") < sequence.index("approval_applied")


def test_reject_is_observed_and_not_executed() -> None:
    queue = ApprovalQueue()
    llm = ScriptedLLM([*script("RE-GROSS")[:2], ScriptedTurn.final("Nicht gebucht.")])
    agent = make_agent(llm, approvals=queue)
    paused = agent.run("x")
    queue.reject(paused.pending_approvals[0].id, by="cfo", reason="Doppelte Rechnung")
    result = agent.resume(paused.state)
    assert result.ok
    assert BOOKINGS == []
    rejected = next(s for s in result.steps if s.decision == "rejected")
    assert rejected.observation["error"]["message"] == "Abgelehnt durch cfo: Doppelte Rechnung"  # type: ignore[index, call-overload]


def test_approve_with_changes() -> None:
    queue = ApprovalQueue()
    llm = ScriptedLLM([*script("RE-GROSS")[:2], ScriptedTurn.final("ok")])
    agent = make_agent(llm, approvals=queue)
    paused = agent.run("x")
    queue.approve_with_changes(
        paused.pending_approvals[0].id, by="cfo", arguments={"nummer": "RE-GROSS", "betrag": 12000}
    )
    result = agent.resume(paused.state)
    assert [("RE-GROSS", Decimal("12000"))] == BOOKINGS
    assert any("mit Änderungen" in s.note for s in result.steps)


def test_invalid_human_changes_become_validation_error() -> None:
    queue = ApprovalQueue()
    llm = ScriptedLLM([*script("RE-GROSS")[:2], ScriptedTurn.final("ok")])
    agent = make_agent(llm, approvals=queue)
    paused = agent.run("x")
    queue.approve_with_changes(
        paused.pending_approvals[0].id, by="cfo", arguments={"nummer": "RE-GROSS", "betrag": "viel"}
    )
    result = agent.resume(paused.state)
    assert BOOKINGS == []
    assert any(
        s.observation and s.observation.get("error", {}).get("type") == "validation_error"  # type: ignore[union-attr]
        for s in result.steps
    )


def test_resume_while_still_pending_and_resume_errors() -> None:
    llm = ScriptedLLM(script("RE-GROSS")[:2])
    agent = make_agent(llm)
    paused = agent.run("x")
    again = agent.resume(paused.state)
    assert again.status is RunStatus.PAUSED
    assert llm.remaining == 0  # kein weiterer Modellaufruf
    other = make_agent(llm, name="anderer")
    with pytest.raises(OrchestrationError, match="gehört zu Agent"):
        other.resume(paused.state)
    done = make_agent(ScriptedLLM([ScriptedTurn.final("x")])).run("x")
    with pytest.raises(OrchestrationError, match="nicht pausiert"):
        agent.resume(done.state)
    with pytest.raises(OrchestrationError, match="Zustandsversion"):
        RunState.from_json({"version": 99})


def test_mixed_turn_keeps_call_order() -> None:
    """Ein Schritt mit erlaubtem und freigabepflichtigem Aufruf: Reihenfolge bleibt."""
    queue = ApprovalQueue()
    llm = ScriptedLLM(
        [
            ScriptedTurn.calls(
                ("buchen", {"nummer": "RE-GROSS", "betrag": 12430.5}),
                ("rechnung_lesen", {"nummer": "RE-2"}),
            ),
            ScriptedTurn.final("fertig"),
        ]
    )
    agent = make_agent(llm, approvals=queue)
    paused = agent.run("x")
    assert paused.status is RunStatus.PAUSED
    assert [s.ok for s in paused.steps] == [None, True]
    queue.approve(paused.pending_approvals[0].id, by="cfo")
    result = agent.resume(RunState.from_json(json.loads(json.dumps(paused.state.to_json()))))
    assert result.ok
    sent = llm.requests[1].messages
    assert [m["tool_call_id"] for m in sent if m["role"] == "tool"] == ["call_1_0", "call_1_1"]


def test_approval_does_not_override_a_later_deny() -> None:
    queue = ApprovalQueue()
    llm = ScriptedLLM([*script("RE-GROSS")[:2], ScriptedTurn.final("ok")])
    paused = make_agent(llm, approvals=queue).run("x")
    queue.approve(paused.pending_approvals[0].id, by="cfo")
    stricter = make_agent(
        llm,
        approvals=queue,
        policy=Policy(
            grade=AutonomyGrade.MANDATE, money_limit=Decimal(5000), deny=frozenset({"buchen"})
        ),
    )
    result = stricter.resume(paused.state)
    assert BOOKINGS == []
    denied = next(s for s in result.steps if s.decision == "deny")
    assert "Trotz Freigabe" in denied.observation["error"]["message"]  # type: ignore[index, call-overload]


def test_tool_removed_before_resume() -> None:
    queue = ApprovalQueue()
    llm = ScriptedLLM([*script("RE-GROSS")[:2], ScriptedTurn.final("ok")])
    paused = make_agent(llm, approvals=queue).run("x")
    queue.approve(paused.pending_approvals[0].id, by="cfo")
    smaller = make_agent(llm, approvals=queue, tools=[rechnung_lesen])
    result = smaller.resume(paused.state)
    assert any(
        s.observation and s.observation.get("error", {}).get("type") == "unknown_tool"  # type: ignore[union-attr]
        for s in result.steps
    )


def test_waiting_time_does_not_count(tmp_path: Path) -> None:
    now = [0.0]
    queue = ApprovalQueue()
    llm = ScriptedLLM([*script("RE-GROSS")[:2], ScriptedTurn.final("ok")])
    agent = make_agent(llm, approvals=queue, clock=lambda: now[0], budget=Budget(max_seconds=60))
    paused = agent.run("x")
    now[0] += 86_400  # ein Tag Wartezeit auf die Freigabe
    queue.approve(paused.pending_approvals[0].id, by="cfo")
    result = agent.resume(paused.state)
    assert result.ok
    assert result.usage.active_seconds < 60


# ---------------------------------------------------- Gedächtnis, Schwärzung --


def test_window_and_long_term_memory() -> None:
    store = LongTermMemory()
    store.add("Muster AG: Rechnungen immer mit Skonto 2 % buchen", tags=["kreditoren"])
    audit = AuditLog()
    llm = ScriptedLLM(
        [
            *[ScriptedTurn.call("kaputt", {"versuch": i}) for i in range(6)],
            ScriptedTurn.final("Muster AG erledigt, Kontakt anna@example.com"),
        ]
    )
    agent = make_agent(
        llm,
        window=ConversationWindow(max_tokens=600),
        long_term=store,
        remember=True,
        audit=audit,
        max_consecutive_errors=10,
    )
    result = agent.run("Rechnung der Muster AG buchen")
    assert result.ok
    assert "Skonto 2 %" in str(llm.requests[0].messages[0]["content"])
    dropped = [r.data["dropped"] for r in audit.records() if r.kind == "llm_request"]
    assert max(dropped) > 0  # type: ignore[type-var]
    assert len(store) == 2
    assert "[EMAIL]" in store.notes()[-1].text


def test_audit_redaction_of_arguments() -> None:
    audit = AuditLog(redactor=Redactor())
    llm = ScriptedLLM(
        [
            ScriptedTurn.call("rechnung_lesen", {"nummer": "IBAN CH9300762011623852957"}),
            ScriptedTurn.final("ok"),
        ]
    )
    make_agent(llm, audit=audit).run("x")
    text = json.dumps([r.to_json() for r in audit.records()], ensure_ascii=False)
    assert "CH9300762011623852957" not in text
    assert "[IBAN]" in text
    assert audit.verify().ok


# ------------------------------------------------------------- Validierung --


def test_constructor_validation() -> None:
    llm = ScriptedLLM([])
    with pytest.raises(ValueError, match="name"):
        Agent(name="mit leerzeichen", llm=llm)
    with pytest.raises(ValueError, match="max_"):
        Agent(name="a", llm=llm, max_identical_calls=0)
    child = Agent(name="kind", llm=llm)
    with pytest.raises(ValueError, match="doppelt"):
        Agent(name="a", llm=llm, delegates=[child, child])


def test_agents_are_reusable_and_thread_safe() -> None:
    def run(nummer: str, results: list[bool]) -> None:
        agent = make_agent(ScriptedLLM(script(nummer)))
        results.append(agent.run("x").ok)

    results: list[bool] = []
    threads = [threading.Thread(target=run, args=(f"RE-{i}", results)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == [True] * 4
    assert sorted(n for n, _ in BOOKINGS) == ["RE-0", "RE-1", "RE-2", "RE-3"]
