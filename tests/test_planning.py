"""Plan-then-Execute mit Neuplanung und Freigabe-Pause."""

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
    Param,
    Plan,
    PlanAndExecute,
    Planner,
    PlanState,
    Policy,
    RunStatus,
    ScriptedLLM,
    ScriptedTurn,
    SideEffect,
    tool,
)
from gradweight_orchestration.errors import OrchestrationError

DONE: list[str] = []


@tool(side_effect=SideEffect.READ_ONLY)
def rechnung_lesen(nummer: str) -> dict[str, Any]:
    """Liest eine Rechnung."""
    if nummer == "FALSCH":
        raise ValueError("Rechnung nicht gefunden")
    return {"nummer": nummer, "betrag": 12430.5}


@tool(side_effect=SideEffect.MONEY, amount_param="betrag")
def buchen(nummer: str, betrag: Annotated[Decimal, Param(ge=0)]) -> str:
    """Bucht."""
    DONE.append(nummer)
    return "gebucht"


@pytest.fixture(autouse=True)
def _reset() -> None:
    DONE.clear()


def plan_json(*steps: tuple[str, str | None]) -> ScriptedTurn:
    return ScriptedTurn.text(json.dumps({"steps": [{"goal": g, "tool": t} for g, t in steps]}))


def executor(
    llm: ScriptedLLM, queue: ApprovalQueue | None = None, audit: AuditLog | None = None
) -> Agent:
    return Agent(
        name="ausfuehrung",
        llm=llm,
        tools=[rechnung_lesen, buchen],
        policy=Policy(grade=AutonomyGrade.MANDATE, money_limit=Decimal(20_000)),
        approvals=queue,
        audit=audit,
        max_consecutive_errors=1,
    )


def test_plan_then_execute() -> None:
    planner_llm = ScriptedLLM(
        [
            ScriptedTurn.text("Hier ist mein Plan: ..."),  # ungültig → Korrektur
            plan_json(("Rechnung lesen", "werkzeug_gibts_nicht")),  # unbekanntes Werkzeug
            plan_json(("Rechnung RE-1 lesen", "rechnung_lesen"), ("Rechnung buchen", "buchen")),
        ]
    )
    worker_llm = ScriptedLLM(
        [
            ScriptedTurn.call("rechnung_lesen", {"nummer": "RE-1"}),
            ScriptedTurn.final("RE-1 über CHF 12430.50 gelesen."),
            ScriptedTurn.call("buchen", {"nummer": "RE-1", "betrag": 12430.5}),
            ScriptedTurn.final("RE-1 gebucht."),
        ]
    )
    audit = AuditLog()
    runner = PlanAndExecute(Planner(planner_llm), executor(worker_llm, audit=audit))
    result = runner.run("Rechnung RE-1 prüfen und buchen", run_id="p1")
    assert result.ok, result.message
    assert result.answer == "RE-1 gebucht."
    assert [s.goal for s in result.plan.steps] == ["Rechnung RE-1 lesen", "Rechnung buchen"]
    assert len(result.results) == 2
    assert DONE == ["RE-1"]
    second_task = str(worker_llm.requests[2].messages[1]["content"])
    assert "Schritt 2 von 2: Rechnung buchen" in second_task
    assert "RE-1 über CHF 12430.50 gelesen." in second_task  # Ergebnis aus Schritt 1
    assert "plan_created" in [r.kind for r in audit.records()]
    assert result.state.usage.prompt_tokens > 0
    assert all(r.json_mode for r in planner_llm.requests)


def test_replan_after_failed_step() -> None:
    planner_llm = ScriptedLLM(
        [
            plan_json(("Rechnung FALSCH lesen", "rechnung_lesen")),
            plan_json(("Rechnung RE-2 lesen", "rechnung_lesen")),
        ]
    )
    worker_llm = ScriptedLLM(
        [
            ScriptedTurn.call("rechnung_lesen", {"nummer": "FALSCH"}),  # Fehler → Lauf stoppt
            ScriptedTurn.call("rechnung_lesen", {"nummer": "RE-2"}),
            ScriptedTurn.final("RE-2 gelesen."),
        ]
    )
    audit = AuditLog()
    result = PlanAndExecute(Planner(planner_llm), executor(worker_llm, audit=audit)).run("x")
    assert result.ok
    assert result.state.replans == 1
    feedback = str(planner_llm.requests[1].messages[1]["content"])
    assert "gescheitert" in feedback
    assert "plan_revised" in [r.kind for r in audit.records()]


def test_replans_exhausted_and_protocol_errors() -> None:
    planner_llm = ScriptedLLM([plan_json(("Rechnung FALSCH lesen", "rechnung_lesen"))])
    worker_llm = ScriptedLLM([ScriptedTurn.call("rechnung_lesen", {"nummer": "FALSCH"})])
    stopped = PlanAndExecute(Planner(planner_llm), executor(worker_llm), max_replans=0).run("x")
    assert stopped.status is RunStatus.STOPPED
    assert stopped.reason == "plan_failed"

    broken = PlanAndExecute(
        Planner(ScriptedLLM([ScriptedTurn.text("x")] * 3), max_repairs=2), executor(ScriptedLLM([]))
    )
    failed = broken.run("x")
    assert failed.status is RunStatus.FAILED
    assert failed.reason == "protocol_error"

    replan_broken = PlanAndExecute(
        Planner(
            ScriptedLLM([plan_json(("Rechnung FALSCH lesen", None)), ScriptedTurn.text("x")]),
            max_repairs=0,
        ),
        executor(ScriptedLLM([ScriptedTurn.call("rechnung_lesen", {"nummer": "FALSCH"})])),
    )
    assert replan_broken.run("x").reason == "protocol_error"


def test_paused_step_resumes() -> None:
    queue = ApprovalQueue()
    planner_llm = ScriptedLLM([plan_json(("Rechnung buchen", "buchen"))])
    worker_llm = ScriptedLLM(
        [
            ScriptedTurn.call("buchen", {"nummer": "RE-3", "betrag": 25_000}),
            ScriptedTurn.final("RE-3 nach Freigabe gebucht."),
        ]
    )
    runner = PlanAndExecute(Planner(planner_llm), executor(worker_llm, queue))
    paused = runner.run("RE-3 buchen")
    assert paused.status is RunStatus.PAUSED
    assert paused.pending_approvals[0].tool == "buchen"
    queue.approve(paused.pending_approvals[0].id, by="cfo")
    state = PlanState.from_json(json.loads(json.dumps(paused.state.to_json())))
    result = runner.resume(state)
    assert result.ok
    assert DONE == ["RE-3"]
    with pytest.raises(OrchestrationError, match="nicht pausiert"):
        runner.resume(result.state)


def test_validation_of_plan_objects() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        Planner(ScriptedLLM([]), max_steps=0)
    with pytest.raises(ValueError, match="max_replans"):
        PlanAndExecute(Planner(ScriptedLLM([])), executor(ScriptedLLM([])), max_replans=-1)
    with pytest.raises(OrchestrationError):
        Plan.from_json({"steps": "kein array"})
    with pytest.raises(OrchestrationError):
        Plan.from_json({"steps": ["kein objekt"]})
    too_long = Planner(
        ScriptedLLM([plan_json(*[(f"Schritt {i}", None) for i in range(3)])]),
        max_steps=2,
        max_repairs=0,
    )
    with pytest.raises(Exception, match=r"maxItems|höchstens"):
        too_long.plan("x", [])
