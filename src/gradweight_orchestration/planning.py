"""Plan-then-Execute als Alternative zum reinen ReAct-Loop.

Ein :class:`Planner` lässt das Modell zuerst einen kurzen Plan erstellen
(strikt geprüftes JSON, mit Korrekturversuchen). :class:`PlanAndExecute`
arbeitet die Schritte danach einzeln mit einem :class:`~gradweight_orchestration.Agent`
ab; jeder Schritt ist ein eigener Agent-Lauf mit allen Prüfungen (Policy,
Freigaben, Audit). Scheitert ein Schritt, wird der Rest des Plans mit der
Fehlerbeschreibung neu geplant (höchstens ``max_replans``-mal). Pausiert ein
Schritt für eine Freigabe, pausiert der ganze Plan.

Gedacht für längere Abläufe, in denen ein freier Loop den Überblick verlieren
kann; ob das bei einem bestimmten Modell hilft, ist hier nicht gemessen. Der
Preis sind zusätzliche Modellaufrufe.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ._json import JSONObject, JSONValue
from .agent import Agent, RunStatus
from .approvals import ApprovalRequest
from .budget import BudgetState
from .errors import OrchestrationError
from .llm.protocol import REPAIR_PROMPT
from .llm.types import (
    ChatModel,
    ProtocolError,
    Usage,
    assistant_message,
    system_message,
    user_message,
)
from .tools import Tool
from .validation import validate

__all__ = ["Plan", "PlanAndExecute", "PlanRunResult", "PlanState", "PlanStep", "Planner"]

PLANNER_PROMPT = """\
Du planst, bevor gehandelt wird. Zerlege die Aufgabe in höchstens {max_steps} kurze,
überprüfbare Schritte in sinnvoller Reihenfolge. Jeder Schritt nennt ein Ziel und, falls
passend, genau ein Werkzeug. Plane nur, was für die Aufgabe nötig ist.

Verfügbare Werkzeuge:
{tools}

Antworte ausschliesslich mit JSON: {{"steps": [{{"goal": "...", "tool": "<name oder null>"}}]}}"""


@dataclass(frozen=True)
class PlanStep:
    goal: str
    tool: str | None = None


@dataclass(frozen=True)
class Plan:
    steps: tuple[PlanStep, ...]

    def to_json(self) -> JSONObject:
        return {"steps": [{"goal": s.goal, "tool": s.tool} for s in self.steps]}

    @classmethod
    def from_json(cls, raw: JSONObject) -> Plan:
        steps = raw.get("steps")
        if not isinstance(steps, list):
            raise OrchestrationError("Plan ohne Schritte")
        out: list[PlanStep] = []
        for item in steps:
            if not isinstance(item, dict):
                raise OrchestrationError("Planschritt ist kein Objekt")
            tool_name = item.get("tool")
            out.append(PlanStep(str(item["goal"]), None if tool_name is None else str(tool_name)))
        return cls(tuple(out))


def _plan_schema(max_steps: int, tool_names: Sequence[str]) -> JSONObject:
    tool_values: list[JSONValue] = [*tool_names, None]
    return {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "minItems": 1,
                "maxItems": max_steps,
                "items": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string", "minLength": 3},
                        "tool": {"type": ["string", "null"], "enum": tool_values},
                    },
                    "required": ["goal"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["steps"],
        "additionalProperties": False,
    }


class Planner:
    """Lässt das Modell einen Plan im JSON-Modus erstellen; prüft ihn strikt."""

    def __init__(self, llm: ChatModel, *, max_steps: int = 6, max_repairs: int = 2) -> None:
        if max_steps < 1:
            raise ValueError("max_steps muss ≥ 1 sein")
        self.llm = llm
        self.max_steps = max_steps
        self.max_repairs = max_repairs

    def plan(
        self, task: str, tools: Sequence[Tool[..., Any]], *, feedback: str | None = None
    ) -> tuple[Plan, Usage]:
        names = [t.name for t in tools]
        listing = "\n".join(f"- {t.name}: {t.description}" for t in tools) or "(keine)"
        prompt = task if feedback is None else f"{task}\n\n{feedback}"
        messages = [
            system_message(PLANNER_PROMPT.format(max_steps=self.max_steps, tools=listing)),
            user_message(prompt),
        ]
        schema = _plan_schema(self.max_steps, names)
        usage = Usage()
        repairs = 0
        while True:
            response = self.llm.complete(messages, json_mode=True)
            usage = usage + response.usage
            text = (response.content or "").strip()
            error = self._check(text, schema)
            if error is None:
                return Plan.from_json(json.loads(text)), usage
            if repairs >= self.max_repairs:
                raise ProtocolError(
                    f"Plan nach {repairs} Korrekturversuch(en) ungültig: {error}",
                    raw=text,
                    usage=usage,
                )
            repairs += 1
            messages += [assistant_message(text), user_message(REPAIR_PROMPT.format(error=error))]

    @staticmethod
    def _check(text: str, schema: JSONObject) -> str | None:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            return f"kein gültiges JSON ({exc.msg})"
        issues = validate(data, schema)
        if issues:
            return "; ".join(str(i) for i in issues[:5])
        return None


@dataclass
class PlanState:
    run_id: str
    task: str
    plan: Plan
    index: int = 0
    results: list[JSONObject] = field(default_factory=list)
    replans: int = 0
    inner: JSONObject | None = None
    usage: BudgetState = field(default_factory=BudgetState)

    def to_json(self) -> JSONObject:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "plan": self.plan.to_json(),
            "index": self.index,
            "results": list(self.results),
            "replans": self.replans,
            "inner": self.inner,
            "usage": self.usage.to_json(),
        }

    @classmethod
    def from_json(cls, raw: JSONObject) -> PlanState:
        plan = raw.get("plan")
        results = raw.get("results")
        inner = raw.get("inner")
        usage = raw.get("usage")
        return cls(
            run_id=str(raw["run_id"]),
            task=str(raw["task"]),
            plan=Plan.from_json(plan if isinstance(plan, dict) else {}),
            index=int(raw.get("index", 0)),  # type: ignore[arg-type]
            results=[r for r in results if isinstance(r, dict)]
            if isinstance(results, list)
            else [],
            replans=int(raw.get("replans", 0)),  # type: ignore[arg-type]
            inner=inner if isinstance(inner, dict) else None,
            usage=BudgetState.from_json(usage if isinstance(usage, dict) else {}),
        )


@dataclass(frozen=True)
class PlanRunResult:
    status: RunStatus
    reason: str
    message: str
    answer: str | None
    plan: Plan
    results: tuple[JSONObject, ...]
    pending_approvals: tuple[ApprovalRequest, ...]
    state: PlanState

    @property
    def ok(self) -> bool:
        return self.status is RunStatus.COMPLETED


class PlanAndExecute:
    """Plant zuerst, führt dann Schritt für Schritt mit einem Agenten aus."""

    def __init__(self, planner: Planner, executor: Agent, *, max_replans: int = 2) -> None:
        if max_replans < 0:
            raise ValueError("max_replans darf nicht negativ sein")
        self.planner = planner
        self.executor = executor
        self.max_replans = max_replans

    def run(self, task: str, *, run_id: str | None = None) -> PlanRunResult:
        rid = run_id or f"plan_{secrets.token_hex(6)}"
        try:
            plan = self._make_plan(task, rid, feedback=None, usage=None)
        except ProtocolError as exc:
            state = PlanState(rid, task, Plan(()))
            state.usage.add(exc.usage, self.executor.budget.pricing)
            return self._result(state, RunStatus.FAILED, "protocol_error", str(exc))
        state = PlanState(rid, task, plan[0])
        state.usage.add(plan[1], self.executor.budget.pricing)
        return self._continue(state)

    def resume(self, state: PlanState | JSONObject) -> PlanRunResult:
        current = state if isinstance(state, PlanState) else PlanState.from_json(state)
        if current.inner is None:
            raise OrchestrationError("Plan ist nicht pausiert; nichts fortzusetzen")
        return self._continue(current)

    def _make_plan(
        self, task: str, run_id: str, *, feedback: str | None, usage: BudgetState | None
    ) -> tuple[Plan, Usage]:
        tracer, audit = self.executor.tracer, self.executor.audit
        with tracer.span(
            "plan",
            {
                "gen_ai.operation.name": "chat",
                "gw.run_id": run_id,
                "gw.replan": feedback is not None,
            },
        ) as span:
            plan, plan_usage = self.planner.plan(task, self.executor.tools, feedback=feedback)
            span.set("gw.plan_steps", len(plan.steps))
        audit.append(
            "plan_revised" if feedback else "plan_created",
            {"plan": plan.to_json(), "feedback": feedback},
            run_id=run_id,
            agent=self.executor.name,
        )
        if usage is not None:
            usage.add(plan_usage, self.executor.budget.pricing)
        return plan, plan_usage

    @staticmethod
    def _step_prompt(state: PlanState) -> str:
        step = state.plan.steps[state.index]
        lines = [
            f"Gesamtaufgabe: {state.task}",
            f"Aktueller Schritt {state.index + 1} von {len(state.plan.steps)}: {step.goal}",
        ]
        if step.tool:
            lines.append(f"Empfohlenes Werkzeug: {step.tool}")
        if state.results:
            lines.append("Ergebnisse der bisherigen Schritte:")
            lines += [f"- {r['goal']}: {r['answer']}" for r in state.results]
        lines.append(
            "Erledige nur diesen Schritt und antworte mit dessen Ergebnis "
            "(inkl. aller Nummern und Beträge, die spätere Schritte brauchen)."
        )
        return "\n".join(lines)

    def _continue(self, state: PlanState) -> PlanRunResult:
        while state.index < len(state.plan.steps):
            step = state.plan.steps[state.index]
            if state.inner is not None:
                result = self.executor.resume(state.inner)
                state.inner = None
            else:
                result = self.executor.run(
                    self._step_prompt(state),
                    run_id=f"{state.run_id}/schritt-{state.index + 1}-{state.replans}",
                )
            usage = result.usage
            state.usage.prompt_tokens += usage.prompt_tokens
            state.usage.completion_tokens += usage.completion_tokens
            state.usage.cost += usage.cost
            state.usage.steps += usage.steps
            state.usage.estimated = state.usage.estimated or usage.estimated
            if result.status is RunStatus.PAUSED:
                state.inner = result.state.to_json()
                return self._result(
                    state,
                    RunStatus.PAUSED,
                    "approval_pending",
                    f"Schritt {state.index + 1} wartet auf Freigabe",
                    pending=result.pending_approvals,
                )
            if result.ok:
                state.results.append({"goal": step.goal, "answer": result.answer})
                state.index += 1
                continue
            if state.replans >= self.max_replans:
                return self._result(
                    state,
                    RunStatus.STOPPED,
                    "plan_failed",
                    f"Schritt «{step.goal}» gescheitert ({result.reason}); "
                    f"keine Neuplanung mehr übrig",
                )
            state.replans += 1
            done = "; ".join(f"{r['goal']} → {r['answer']}" for r in state.results) or "nichts"
            feedback = (
                f"Bereits erledigt: {done}. Der Schritt «{step.goal}» ist gescheitert: "
                f"{result.message}. Plane nur die verbleibenden Schritte neu."
            )
            try:
                revised, _ = self._make_plan(
                    state.task, state.run_id, feedback=feedback, usage=state.usage
                )
            except ProtocolError as exc:
                return self._result(state, RunStatus.FAILED, "protocol_error", str(exc))
            state.plan = Plan(state.plan.steps[: state.index] + revised.steps)
        answer = state.results[-1].get("answer") if state.results else None
        return self._result(
            state,
            RunStatus.COMPLETED,
            "final_answer",
            "Plan abgeschlossen",
            answer=None if answer is None else str(answer),
        )

    def _result(
        self,
        state: PlanState,
        status: RunStatus,
        reason: str,
        message: str,
        *,
        answer: str | None = None,
        pending: Sequence[ApprovalRequest] = (),
    ) -> PlanRunResult:
        return PlanRunResult(
            status, reason, message, answer, state.plan, tuple(state.results), tuple(pending), state
        )
