"""Supervisor-Muster: Delegation an Spezialisten mit eigenen Werkzeugen."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Annotated, Any

import pytest

from gradweight_orchestration import (
    Agent,
    ApprovalQueue,
    AuditLog,
    AutonomyGrade,
    Budget,
    Param,
    Policy,
    RunStatus,
    ScriptedLLM,
    ScriptedTurn,
    SideEffect,
    Supervisor,
    Tracer,
    tool,
)
from gradweight_orchestration.llm import ScriptContext

BOOKED: list[Decimal] = []


@tool(side_effect=SideEffect.READ_ONLY)
def lieferant_pruefen(name: str) -> dict[str, str]:
    """Prüft den Lieferanten in den Stammdaten."""
    return {"name": name, "kreditor": "K-2001", "status": "bekannt"}


@tool(side_effect=SideEffect.MONEY, amount_param="betrag")
def zahlung_buchen(kreditor: str, betrag: Annotated[Decimal, Param(ge=0)]) -> str:
    """Bucht eine Kreditorenzahlung."""
    BOOKED.append(betrag)
    return f"Zahlung {kreditor} {betrag}"


@pytest.fixture(autouse=True)
def _reset() -> None:
    BOOKED.clear()


def specialists(
    queue: ApprovalQueue, audit: AuditLog, tracer: Tracer, buchhaltung_script: list[Any]
) -> tuple[Agent, Agent]:
    pruefung = Agent(
        name="pruefung",
        description="prüft Lieferanten in den Stammdaten",
        llm=ScriptedLLM(
            [
                ScriptedTurn.call("lieferant_pruefen", {"name": "Muster AG"}),
                ScriptedTurn.final("Muster AG ist bekannt, Kreditor K-2001."),
            ]
        ),
        tools=[lieferant_pruefen],
        policy=Policy(grade=AutonomyGrade.ASSISTANT),
        approvals=queue,
        audit=audit,
        tracer=tracer,
    )
    buchhaltung = Agent(
        name="buchhaltung",
        description="bucht Zahlungen an Kreditoren",
        llm=ScriptedLLM(buchhaltung_script),
        tools=[zahlung_buchen],
        policy=Policy(grade=AutonomyGrade.MANDATE, money_limit=Decimal(5000)),
        approvals=queue,
        audit=audit,
        tracer=tracer,
    )
    return pruefung, buchhaltung


def amount_from_task(ctx: ScriptContext) -> ScriptedTurn:
    """Der Spezialist übernimmt den Betrag aus seinem Auftrag."""
    task = str(ctx.messages[1]["content"])
    betrag = task.split("CHF ")[1].split(" ")[0]
    return ScriptedTurn.call("zahlung_buchen", {"kreditor": "K-2001", "betrag": float(betrag)})


def supervisor_script(betrag: str) -> list[Any]:
    def buchen(ctx: ScriptContext) -> ScriptedTurn:
        prior = ctx.last_result()
        assert isinstance(prior, dict)
        assert "K-2001" in str(prior["answer"])  # Ergebnis des ersten Spezialisten
        return ScriptedTurn.call(
            "delegieren_buchhaltung", {"auftrag": f"Zahle CHF {betrag} an Kreditor K-2001."}
        )

    return [
        ScriptedTurn.call("delegieren_pruefung", {"auftrag": "Prüfe Lieferant Muster AG."}),
        buchen,
        ScriptedTurn.final("Lieferant geprüft und Zahlung gebucht."),
    ]


def test_supervisor_delegates_to_specialists() -> None:
    queue, audit, tracer = ApprovalQueue(), AuditLog(), Tracer()
    pruefung, buchhaltung = specialists(
        queue,
        audit,
        tracer,
        [
            amount_from_task,
            ScriptedTurn.final("Gebucht."),
        ],
    )
    llm = ScriptedLLM(supervisor_script("1081.00"))
    boss = Supervisor(
        name="leitung",
        llm=llm,
        specialists=[pruefung, buchhaltung],
        approvals=queue,
        audit=audit,
        tracer=tracer,
    )
    assert [t.name for t in boss.tools] == ["delegieren_pruefung", "delegieren_buchhaltung"]
    assert boss.specialists == [pruefung, buchhaltung]
    result = boss.run("Rechnung der Muster AG über CHF 1081.00 bezahlen", run_id="r1")
    assert result.ok
    assert [Decimal("1081.0")] == BOOKED
    assert [s.kind for s in result.steps] == ["delegation", "delegation", "final"]
    # Der Supervisor sieht nur Delegationen, nie die Werkzeuge der Spezialisten
    offered = {spec["function"]["name"] for spec in llm.requests[0].tools or []}  # type: ignore[index]
    assert offered == {"delegieren_pruefung", "delegieren_buchhaltung"}
    # Gemeinsames Audit-Log mit verschachtelten Lauf-IDs, Kette intakt
    runs = {r.run_id for r in audit.records()}
    assert {"r1", "r1/pruefung-1", "r1/buchhaltung-2"} <= runs
    assert audit.verify().ok
    # Spans der Spezialisten hängen unter dem Delegationsaufruf
    delegation = next(s for s in tracer.spans if s.name == "execute_tool delegieren_buchhaltung")
    child = next(s for s in tracer.spans if s.name == "invoke_agent buchhaltung")
    assert child.parent_id == delegation.span_id


def test_supervisor_cannot_call_specialist_tools_directly() -> None:
    queue, audit, tracer = ApprovalQueue(), AuditLog(), Tracer()
    pruefung, buchhaltung = specialists(queue, audit, tracer, [])
    boss = Supervisor(
        name="leitung",
        llm=ScriptedLLM(
            [
                ScriptedTurn.call("zahlung_buchen", {"kreditor": "K-2001", "betrag": 1}),
                ScriptedTurn.final("Nicht erlaubt."),
            ]
        ),
        specialists=[pruefung, buchhaltung],
    )
    result = boss.run("x")
    assert result.steps[0].observation["error"]["type"] == "unknown_tool"  # type: ignore[index, call-overload]
    assert BOOKED == []


def test_approval_inside_specialist_pauses_whole_run() -> None:
    queue, audit, tracer = ApprovalQueue(), AuditLog(), Tracer()
    pruefung, buchhaltung = specialists(
        queue,
        audit,
        tracer,
        [
            amount_from_task,
            ScriptedTurn.final("Nach Freigabe gebucht."),
        ],
    )
    boss = Supervisor(
        name="leitung",
        llm=ScriptedLLM(supervisor_script("12430.50")),
        specialists=[pruefung, buchhaltung],
        approvals=queue,
        audit=audit,
        tracer=tracer,
    )
    paused = boss.run("Grosse Rechnung bezahlen", run_id="r2")
    assert paused.status is RunStatus.PAUSED
    assert [r.tool for r in paused.pending_approvals] == ["zahlung_buchen"]
    assert paused.pending_approvals[0].run_id == "r2/buchhaltung-2"
    assert [r.id for r in queue.pending(run_id="r2")] == [paused.pending_approvals[0].id]
    assert BOOKED == []
    queue.approve(paused.pending_approvals[0].id, by="cfo")
    state = json.loads(json.dumps(paused.state.to_json()))  # z.B. in einer Datenbank abgelegt
    result = boss.resume(state)
    assert result.ok, result.message
    assert [Decimal("12430.5")] == BOOKED
    assert result.answer == "Lieferant geprüft und Zahlung gebucht."
    assert audit.verify().ok


def test_failed_or_empty_delegation_is_an_observation() -> None:
    queue, audit, tracer = ApprovalQueue(), AuditLog(), Tracer()
    pruefung, _ = specialists(queue, audit, tracer, [])
    stuck = Agent(
        name="stuck", llm=ScriptedLLM([ScriptedTurn.final("")] * 5), budget=Budget(max_steps=1)
    )
    boss = Supervisor(
        name="leitung",
        llm=ScriptedLLM(
            [
                ScriptedTurn.call("delegieren_stuck", {"auftrag": "  "}),
                ScriptedTurn.call("delegieren_stuck", {"auftrag": "Mach etwas."}),
                ScriptedTurn.final("Spezialist gescheitert."),
            ]
        ),
        specialists=[pruefung, stuck],
    )
    result = boss.run("x")
    assert result.ok
    assert result.steps[0].observation["error"]["type"] == "validation_error"  # type: ignore[index, call-overload]
    failed = result.steps[1].observation["error"]  # type: ignore[index, call-overload]
    assert failed["type"] == "delegation_failed"
    assert "stopped" in failed["message"]


def test_supervisor_needs_specialists() -> None:
    with pytest.raises(ValueError, match="Spezialisten"):
        Supervisor(name="leer", llm=ScriptedLLM([]), specialists=[])
