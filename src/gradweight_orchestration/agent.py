"""Der Agent: ein ReAct-Loop mit Werkzeugen, Policy, Freigaben, Audit und Trace.

Ablauf pro Schritt:

1. Budget prüfen (Schritte, Token, Kosten, aktive Zeit).
2. Modell aufrufen (nativ oder JSON-Protokoll; bei ``AUTO`` Wechsel auf JSON,
   wenn der Server keine Werkzeuge kann).
3. Keine Werkzeugaufrufe → Antwort, Lauf abgeschlossen.
4. Für jeden Aufruf: Wiederholung erkennen → Werkzeug suchen → Argumente
   strikt validieren → Policy entscheiden → ausführen, ablehnen oder Freigabe
   beantragen. Jede Station landet im Audit-Log und im Trace.
5. Beobachtungen strukturiert zurückgeben; nach Fehlern folgt ein Hinweis zur
   Neuplanung, nach ``max_consecutive_errors`` Fehlschritten wird gestoppt.

Wartet ein Aufruf auf Freigabe, pausiert der Lauf (:attr:`RunStatus.PAUSED`).
:attr:`RunResult.state` ist JSON-serialisierbar; :meth:`Agent.resume` setzt den
Lauf fort, sobald entschieden ist – auch in einem anderen Prozess, wenn die
Freigabe-Warteschlange in einer Datei liegt.

Delegation (Supervisor-Muster): Andere Agenten lassen sich als ``delegates``
übergeben. Sie erscheinen dem Modell als Werkzeug ``delegieren_<name>`` mit dem
Parameter ``auftrag`` und laufen mit *ihren eigenen* Werkzeugen, ihrer Policy und
ihrem Budget. Pausiert ein Spezialist für eine Freigabe, pausiert der ganze Lauf.
"""

from __future__ import annotations

import enum
import re
import secrets
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ._json import JSONObject, JSONValue, canonical_json, dumps, sha256_hex, truncate
from .approvals import ApprovalQueue, ApprovalRequest, ApprovalStatus
from .audit import AuditLog, Redactor
from .budget import Budget, BudgetState
from .errors import OrchestrationError
from .execution import ErrorType, ToolExecutor, ToolResult
from .governance import Decision, Policy
from .llm.protocol import AssistantTurn, ToolCallingMode, strategy_for
from .llm.types import (
    ChatModel,
    LLMError,
    Message,
    ProtocolError,
    ToolCall,
    ToolsNotSupportedError,
    assistant_message,
    system_message,
    user_message,
)
from .memory import ConversationWindow, LongTermMemory
from .tools import SideEffect, Tool, ToolRegistry, tool
from .tracing import Span, Tracer

__all__ = ["Agent", "PendingCall", "RunResult", "RunState", "RunStatus", "StepRecord"]

STATE_VERSION = 1
_AGENT_NAME = re.compile(r"^[A-Za-z0-9_-]{1,40}$")

DEFAULT_INSTRUCTIONS = (
    "Du bist ein sorgfältiger Assistent. Löse die Aufgabe mit den verfügbaren Werkzeugen. "
    "Erfinde keine Werte: übernimm Nummern, Beträge und Kennungen aus den Beobachtungen. "
    "Meldet ein Werkzeug einen Fehler, korrigiere die Argumente oder wähle einen anderen Weg. "
    "Wird eine Aktion abgelehnt, führe sie nicht auf anderem Weg aus. "
    "Antworte am Schluss knapp auf Deutsch."
)
REPLAN_HINT = (
    "Hinweis: Im letzten Schritt ist etwas fehlgeschlagen ({errors}). Prüfe die Beobachtung, "
    "korrigiere die Argumente gemäss Schema oder wähle einen anderen Weg. Wiederhole keinen "
    "identischen Aufruf."
)
EMPTY_HINT = "Hinweis: Deine Antwort war leer. Rufe ein Werkzeug auf oder gib eine Antwort."


class RunStatus(str, enum.Enum):
    COMPLETED = "completed"
    PAUSED = "paused"
    STOPPED = "stopped"
    """Abbruch durch Budget, Schleifenerkennung oder zu viele Fehler."""
    FAILED = "failed"
    """Modell nicht erreichbar oder Antwortformat dauerhaft verletzt."""


@dataclass
class PendingCall:
    """Ein Aufruf, der auf eine Entscheidung wartet (Freigabe oder Spezialist)."""

    call_id: str
    tool: str
    arguments: JSONObject
    approval_id: str | None = None
    delegate: str | None = None
    child: JSONObject | None = None

    def to_json(self) -> JSONObject:
        return {
            "call_id": self.call_id,
            "tool": self.tool,
            "arguments": self.arguments,
            "approval_id": self.approval_id,
            "delegate": self.delegate,
            "child": self.child,
        }

    @classmethod
    def from_json(cls, raw: JSONObject) -> PendingCall:
        arguments = raw.get("arguments")
        child = raw.get("child")
        return cls(
            call_id=str(raw["call_id"]),
            tool=str(raw["tool"]),
            arguments=arguments if isinstance(arguments, dict) else {},
            approval_id=_opt_str(raw.get("approval_id")),
            delegate=_opt_str(raw.get("delegate")),
            child=child if isinstance(child, dict) else None,
        )


def _opt_str(value: JSONValue) -> str | None:
    return None if value is None else str(value)


@dataclass(frozen=True)
class StepRecord:
    """Ein Eintrag im Laufprotokoll (für Menschen und Tests, nicht fürs Modell)."""

    step: int
    kind: str
    """``tool``, ``delegation``, ``final`` oder ``notice``."""
    tool: str | None = None
    call_id: str | None = None
    arguments: JSONObject | None = None
    decision: str | None = None
    ok: bool | None = None
    observation: JSONValue = None
    note: str = ""

    def to_json(self) -> JSONObject:
        return {
            "step": self.step,
            "kind": self.kind,
            "tool": self.tool,
            "call_id": self.call_id,
            "arguments": self.arguments,
            "decision": self.decision,
            "ok": self.ok,
            "observation": self.observation,
            "note": self.note,
        }

    @classmethod
    def from_json(cls, raw: JSONObject) -> StepRecord:
        arguments = raw.get("arguments")
        ok = raw.get("ok")
        return cls(
            step=int(raw["step"]),  # type: ignore[arg-type]
            kind=str(raw["kind"]),
            tool=_opt_str(raw.get("tool")),
            call_id=_opt_str(raw.get("call_id")),
            arguments=arguments if isinstance(arguments, dict) else None,
            decision=_opt_str(raw.get("decision")),
            ok=ok if isinstance(ok, bool) else None,
            observation=raw.get("observation"),
            note=str(raw.get("note", "")),
        )


@dataclass
class RunState:
    """Vollständiger, JSON-serialisierbarer Zustand eines Laufs."""

    run_id: str
    agent: str
    task: str
    messages: list[Message] = field(default_factory=list)
    budget: BudgetState = field(default_factory=BudgetState)
    mode: str = ToolCallingMode.NATIVE.value
    notes: list[str] = field(default_factory=list)
    consecutive_errors: int = 0
    fingerprints: dict[str, int] = field(default_factory=dict)
    call_counter: int = 0
    delegations: int = 0
    turn_calls: list[JSONObject] = field(default_factory=list)
    observations: dict[str, str] = field(default_factory=dict)
    turn_errors: list[str] = field(default_factory=list)
    pending: list[PendingCall] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)
    stop: list[str] = field(default_factory=list)

    def to_json(self) -> JSONObject:
        return {
            "version": STATE_VERSION,
            "run_id": self.run_id,
            "agent": self.agent,
            "task": self.task,
            "messages": list(self.messages),
            "budget": self.budget.to_json(),
            "mode": self.mode,
            "notes": list(self.notes),
            "consecutive_errors": self.consecutive_errors,
            "fingerprints": dict(self.fingerprints),
            "call_counter": self.call_counter,
            "delegations": self.delegations,
            "turn_calls": list(self.turn_calls),
            "observations": dict(self.observations),
            "turn_errors": list(self.turn_errors),
            "pending": [p.to_json() for p in self.pending],
            "steps": [s.to_json() for s in self.steps],
            "stop": list(self.stop),
        }

    @classmethod
    def from_json(cls, raw: JSONObject) -> RunState:
        if raw.get("version") != STATE_VERSION:
            raise OrchestrationError(f"Unbekannte Zustandsversion {raw.get('version')!r}")

        def objects(key: str) -> list[JSONObject]:
            value = raw.get(key, [])
            return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []

        def strings(key: str) -> list[str]:
            value = raw.get(key, [])
            return [str(v) for v in value] if isinstance(value, list) else []

        budget = raw.get("budget")
        fingerprints = raw.get("fingerprints")
        observations = raw.get("observations")
        return cls(
            run_id=str(raw["run_id"]),
            agent=str(raw["agent"]),
            task=str(raw["task"]),
            messages=objects("messages"),
            budget=BudgetState.from_json(budget if isinstance(budget, dict) else {}),
            mode=str(raw.get("mode", ToolCallingMode.NATIVE.value)),
            notes=strings("notes"),
            consecutive_errors=int(raw.get("consecutive_errors", 0)),  # type: ignore[arg-type]
            fingerprints={str(k): int(v) for k, v in fingerprints.items()}  # type: ignore[arg-type]
            if isinstance(fingerprints, dict)
            else {},
            call_counter=int(raw.get("call_counter", 0)),  # type: ignore[arg-type]
            delegations=int(raw.get("delegations", 0)),  # type: ignore[arg-type]
            turn_calls=objects("turn_calls"),
            observations={str(k): str(v) for k, v in observations.items()}
            if isinstance(observations, dict)
            else {},
            turn_errors=strings("turn_errors"),
            pending=[PendingCall.from_json(p) for p in objects("pending")],
            steps=[StepRecord.from_json(s) for s in objects("steps")],
            stop=strings("stop"),
        )


@dataclass(frozen=True)
class RunResult:
    run_id: str
    agent: str
    status: RunStatus
    reason: str
    """Maschinenlesbarer Grund, z.B. ``final_answer``, ``approval_pending``, ``max_steps``."""
    message: str
    answer: str | None
    steps: tuple[StepRecord, ...]
    pending_approvals: tuple[ApprovalRequest, ...]
    usage: BudgetState
    state: RunState

    @property
    def ok(self) -> bool:
        return self.status is RunStatus.COMPLETED


@dataclass
class _Session:
    """Laufzeitdaten eines einzelnen ``run``/``resume``-Aufrufs (nicht persistiert)."""

    state: RunState
    started: float
    span: Span


@dataclass(frozen=True)
class _Outcome:
    status: RunStatus
    reason: str
    message: str
    answer: str | None = None


def _delegate_stub(auftrag: str) -> str:
    """Platzhalter für Delegationswerkzeuge; wird nie direkt aufgerufen."""
    raise NotImplementedError


class Agent:
    """Ein Agent mit Werkzeugen. Instanzen sind wiederverwendbar; der Laufzustand
    steckt vollständig in :class:`RunState`."""

    def __init__(
        self,
        *,
        name: str,
        llm: ChatModel,
        tools: Iterable[Tool[..., Any]] | ToolRegistry = (),
        instructions: str = DEFAULT_INSTRUCTIONS,
        description: str = "",
        policy: Policy | None = None,
        approvals: ApprovalQueue | None = None,
        audit: AuditLog | None = None,
        tracer: Tracer | None = None,
        budget: Budget | None = None,
        tool_calling: ToolCallingMode = ToolCallingMode.AUTO,
        window: ConversationWindow | None = None,
        long_term: LongTermMemory | None = None,
        remember: bool = False,
        executor: ToolExecutor | None = None,
        delegates: Sequence[Agent] = (),
        max_consecutive_errors: int = 3,
        max_identical_calls: int = 2,
        max_observation_chars: int = 4000,
        max_repairs: int = 2,
        replan_hints: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not _AGENT_NAME.match(name):
            raise ValueError("name muss [A-Za-z0-9_-]{1,40} sein")
        if max_consecutive_errors < 1 or max_identical_calls < 1:
            raise ValueError("max_consecutive_errors und max_identical_calls müssen ≥ 1 sein")
        self.name = name
        self.description = description
        self.llm = llm
        self.registry = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        self.instructions = instructions
        self.policy = Policy() if policy is None else policy
        self.approvals = ApprovalQueue() if approvals is None else approvals
        self.audit = AuditLog() if audit is None else audit
        self.tracer = Tracer() if tracer is None else tracer
        self.budget = Budget() if budget is None else budget
        self.tool_calling = tool_calling
        self.window = window
        self.long_term = long_term
        self.remember = remember
        self.executor = ToolExecutor() if executor is None else executor
        self.max_consecutive_errors = max_consecutive_errors
        self.max_identical_calls = max_identical_calls
        self.max_observation_chars = max_observation_chars
        self.max_repairs = max_repairs
        self.replan_hints = replan_hints
        self._clock = clock
        self._delegates: dict[str, Agent] = {}
        self._delegate_tools: list[Tool[..., Any]] = []
        for delegate in delegates:
            self._add_delegate(delegate)

    def _add_delegate(self, delegate: Agent) -> None:
        tool_name = f"delegieren_{delegate.name}"
        if tool_name in self._delegates or tool_name in self.registry or delegate is self:
            raise ValueError(f"Delegation {tool_name!r} ist doppelt oder rekursiv")
        description = (
            f"Beauftragt den Spezialisten «{delegate.name}»"
            + (f": {delegate.description}" if delegate.description else "")
            + ". Gibt dessen Ergebnis zurück. Der Auftrag muss alle nötigen Angaben enthalten."
        )
        stub = tool(name=tool_name, description=description, side_effect=SideEffect.READ_ONLY)(
            _delegate_stub
        )
        self._delegates[tool_name] = delegate
        self._delegate_tools.append(stub)

    @property
    def tools(self) -> list[Tool[..., Any]]:
        """Alle Werkzeuge, die das Modell sieht (inkl. Delegationen)."""
        return [*self.registry, *self._delegate_tools]

    # ------------------------------------------------------------- öffentlich --

    def run(self, task: str, *, run_id: str | None = None) -> RunResult:
        """Startet einen neuen Lauf."""
        state = self._new_state(task, run_id or f"run_{secrets.token_hex(6)}")
        self.audit.append(
            "run_started",
            {
                "task": task,
                "task_sha256": sha256_hex(task),
                "model": self.llm.model_name,
                "policy": self.policy.to_json(),
                "tools": [t.name for t in self.tools],
                "tool_calling": self.tool_calling.value,
            },
            run_id=state.run_id,
            agent=self.name,
        )
        return self._drive(state, operation="invoke_agent")

    def resume(self, state: RunState | JSONObject) -> RunResult:
        """Setzt einen pausierten Lauf fort (nach Entscheid über die Freigaben)."""
        current = state if isinstance(state, RunState) else RunState.from_json(state)
        if current.agent != self.name:
            raise OrchestrationError(
                f"Zustand gehört zu Agent {current.agent!r}, nicht zu {self.name!r}"
            )
        if not current.pending:
            raise OrchestrationError("Lauf ist nicht pausiert; nichts fortzusetzen")
        self.audit.append(
            "run_resumed",
            {"pending": [p.call_id for p in current.pending]},
            run_id=current.run_id,
            agent=self.name,
        )
        return self._drive(current, operation="resume_agent")

    def pending_approvals(self, state: RunState) -> list[ApprovalRequest]:
        """Offene und entschiedene Anträge, auf die der Lauf (rekursiv) wartet."""
        found: list[ApprovalRequest] = []
        for pending in state.pending:
            if pending.approval_id is not None:
                found.append(self.approvals.get(pending.approval_id))
            elif pending.delegate is not None and pending.child is not None:
                delegate = self._delegates[pending.tool]
                found.extend(delegate.pending_approvals(RunState.from_json(pending.child)))
        return [r for r in found if r.status is ApprovalStatus.PENDING]

    # ---------------------------------------------------------------- Ablauf --

    def _new_state(self, task: str, run_id: str) -> RunState:
        initial = (
            ToolCallingMode.JSON
            if self.tool_calling is ToolCallingMode.JSON
            else ToolCallingMode.NATIVE
        )
        notes = []
        if self.long_term is not None:
            notes = [note.text for note, _ in self.long_term.search(task, k=3)]
        return RunState(
            run_id=run_id,
            agent=self.name,
            task=task,
            messages=[user_message(task)],
            mode=initial.value,
            notes=notes,
        )

    def _drive(self, state: RunState, *, operation: str) -> RunResult:
        attributes = {
            "gen_ai.operation.name": operation,
            "gen_ai.agent.name": self.name,
            "gen_ai.request.model": self.llm.model_name,
            "gw.run_id": state.run_id,
            "gw.policy.grade": int(self.policy.grade),
        }
        with self.tracer.span(f"{operation} {self.name}", attributes) as span:
            session = _Session(state=state, started=self._clock(), span=span)
            try:
                outcome = self._loop(session)
            finally:
                state.budget.active_seconds += self._clock() - session.started
            span.update(
                {
                    "gw.status": outcome.status.value,
                    "gw.reason": outcome.reason,
                    "gen_ai.usage.input_tokens": state.budget.prompt_tokens,
                    "gen_ai.usage.output_tokens": state.budget.completion_tokens,
                }
            )
            if outcome.status in (RunStatus.FAILED, RunStatus.STOPPED):
                span.fail(outcome.message)
        kind = "run_paused" if outcome.status is RunStatus.PAUSED else "run_finished"
        self.audit.append(
            kind,
            {
                "status": outcome.status.value,
                "reason": outcome.reason,
                "message": outcome.message,
                "answer": outcome.answer,
                "usage": state.budget.to_json(),
                "pending": [p.approval_id or p.call_id for p in state.pending],
            },
            run_id=state.run_id,
            agent=self.name,
        )
        if outcome.status is RunStatus.COMPLETED and self.remember and self.long_term is not None:
            self.long_term.add(
                Redactor().text(f"Aufgabe: {state.task}\nErgebnis: {outcome.answer}"),
                tags=[self.name],
                source=state.run_id,
            )
        return RunResult(
            run_id=state.run_id,
            agent=self.name,
            status=outcome.status,
            reason=outcome.reason,
            message=outcome.message,
            answer=outcome.answer,
            steps=tuple(state.steps),
            pending_approvals=tuple(self.pending_approvals(state)),
            usage=BudgetState.from_json(state.budget.to_json()),
            state=state,
        )

    def _loop(self, session: _Session) -> _Outcome:
        state = session.state
        if state.pending:
            if not self._resolve_pending(session):
                return _Outcome(
                    RunStatus.PAUSED, "approval_pending", "Wartet auf Entscheid über Freigabe(n)"
                )
            stop = self._finish_turn(state)
            if stop is not None:
                return stop
        while True:
            exceeded = state.budget.exceeded(self.budget, elapsed=self._clock() - session.started)
            if exceeded is not None:
                return _Outcome(RunStatus.STOPPED, exceeded[0], exceeded[1])
            try:
                turn = self._call_model(session)
            except ProtocolError as exc:
                state.budget.add(exc.usage, self.budget.pricing)
                state.budget.steps += 1
                return _Outcome(RunStatus.FAILED, "protocol_error", str(exc))
            except LLMError as exc:
                return _Outcome(RunStatus.FAILED, "llm_error", str(exc))
            state.budget.steps += 1
            step = state.budget.steps
            if not turn.tool_calls:
                state.messages.append(turn.message)
                if turn.content and turn.content.strip():
                    state.steps.append(StepRecord(step, "final", note=turn.content))
                    return _Outcome(
                        RunStatus.COMPLETED,
                        "final_answer",
                        "Antwort erstellt",
                        turn.content.strip(),
                    )
                state.messages.append(user_message(EMPTY_HINT))
                state.steps.append(StepRecord(step, "notice", note="leere Antwort"))
                state.consecutive_errors += 1
                if state.consecutive_errors >= self.max_consecutive_errors:
                    return _Outcome(
                        RunStatus.STOPPED,
                        "too_many_errors",
                        f"{state.consecutive_errors} Fehlschritte in Folge",
                    )
                continue
            turn = self._normalise_ids(state, turn)
            state.messages.append(turn.message)
            state.turn_calls = [{"id": c.id, "name": c.name} for c in turn.tool_calls]
            state.observations = {}
            state.turn_errors = []
            for call in turn.tool_calls:
                self._process_call(session, call, step)
            if state.pending:
                return _Outcome(
                    RunStatus.PAUSED, "approval_pending", "Wartet auf Entscheid über Freigabe(n)"
                )
            stop = self._finish_turn(state)
            if stop is not None:
                return stop

    def _normalise_ids(self, state: RunState, turn: AssistantTurn) -> AssistantTurn:
        """Erzwingt eindeutige, nicht-leere Aufruf-IDs innerhalb eines Schritts."""
        seen: set[str] = set()
        calls: list[ToolCall] = []
        changed = False
        for call in turn.tool_calls:
            if not call.id or call.id in seen:
                call = ToolCall(
                    self._next_id(state), call.name, call.arguments, call.raw_arguments, call.error
                )
                changed = True
            seen.add(call.id)
            calls.append(call)
        if not changed:
            return turn
        message = (
            assistant_message(turn.content, calls)
            if turn.mode is ToolCallingMode.NATIVE
            else turn.message
        )
        return AssistantTurn(
            turn.content,
            tuple(calls),
            message,
            turn.usage,
            turn.model,
            turn.mode,
            turn.repairs,
            turn.recovered,
            turn.thought,
        )

    @staticmethod
    def _next_id(state: RunState) -> str:
        state.call_counter += 1
        return f"call_{state.call_counter}"

    def _system_prompt(self, state: RunState, addendum: str) -> str:
        parts = [self.instructions.strip()]
        if addendum:
            parts.append(addendum)
        if state.notes:
            parts.append(
                "Notizen aus früheren Läufen (können veraltet sein):\n"
                + "\n".join(f"- {n}" for n in state.notes)
            )
        return "\n\n".join(parts)

    def _call_model(self, session: _Session) -> AssistantTurn:
        state = session.state
        try:
            return self._request(session)
        except ToolsNotSupportedError:
            if self.tool_calling is not ToolCallingMode.AUTO or state.mode != "native":
                raise
            state.mode = ToolCallingMode.JSON.value
            self.audit.append(
                "tool_mode_switched",
                {"from": "native", "to": "json"},
                run_id=state.run_id,
                agent=self.name,
            )
            session.span.add_event("tool_mode_switched", to="json")
            return self._request(session)

    def _request(self, session: _Session) -> AssistantTurn:
        state = session.state
        strategy = strategy_for(ToolCallingMode(state.mode), max_repairs=self.max_repairs)
        tools = self.tools
        messages = [
            system_message(self._system_prompt(state, strategy.system_addendum(tools))),
            *state.messages,
        ]
        view = self.window.select(messages) if self.window is not None else messages
        prompt_hash = sha256_hex(
            canonical_json(
                {"messages": list(view), "tools": [t.spec() for t in tools], "mode": state.mode}
            )
        )
        self.audit.append(
            "llm_request",
            {
                "prompt_sha256": prompt_hash,
                "messages": len(view),
                "dropped": len(messages) - len(view),
                "tools": [t.name for t in tools],
                "mode": state.mode,
            },
            run_id=state.run_id,
            agent=self.name,
        )
        attributes = {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": self.llm.model_name,
            "gw.prompt_sha256": prompt_hash,
            "gw.tool_mode": state.mode,
        }
        with self.tracer.span(f"chat {self.llm.model_name}", attributes) as span:
            turn = strategy.request(self.llm, view, tools, new_id=lambda: self._next_id(state))
            state.budget.add(turn.usage, self.budget.pricing)
            span.update(
                {
                    "gen_ai.response.model": turn.model,
                    "gen_ai.usage.input_tokens": turn.usage.prompt_tokens,
                    "gen_ai.usage.output_tokens": turn.usage.completion_tokens,
                    "gw.usage.estimated": turn.usage.estimated,
                    "gw.repairs": turn.repairs,
                    "gw.recovered_tool_call": turn.recovered,
                    "gw.tool_calls": len(turn.tool_calls),
                }
            )
        self.audit.append(
            "llm_response",
            {
                "content": truncate(turn.content or "", 2000),
                "thought": turn.thought,
                "tool_calls": [
                    {"id": c.id, "name": c.name, "arguments": c.arguments, "error": c.error}
                    for c in turn.tool_calls
                ],
                "usage": {
                    "prompt_tokens": turn.usage.prompt_tokens,
                    "completion_tokens": turn.usage.completion_tokens,
                    "estimated": turn.usage.estimated,
                },
                "repairs": turn.repairs,
                "recovered": turn.recovered,
                "mode": turn.mode.value,
            },
            run_id=state.run_id,
            agent=self.name,
        )
        return turn

    # ------------------------------------------------------- Werkzeugaufrufe --

    def _observe(
        self,
        state: RunState,
        call_id: str,
        tool_name: str,
        observation: JSONObject,
        *,
        step: int,
        kind: str = "tool",
        arguments: JSONObject | None = None,
        decision: str | None = None,
        note: str = "",
    ) -> None:
        payload = dumps(observation)
        if len(payload) > self.max_observation_chars:
            payload = dumps(
                {
                    "ok": observation.get("ok"),
                    "result_truncated": truncate(
                        dumps(observation.get("result")), self.max_observation_chars - 120
                    ),
                    "hinweis": "Ergebnis gekürzt",
                }
            )
        state.observations[call_id] = payload
        ok = observation.get("ok") is True
        if not ok:
            error = observation.get("error")
            summary = error.get("type") if isinstance(error, dict) else "Fehler"
            state.turn_errors.append(f"{tool_name}: {summary}")
        state.steps.append(
            StepRecord(step, kind, tool_name, call_id, arguments, decision, ok, observation, note)
        )

    def _process_call(self, session: _Session, call: ToolCall, step: int) -> None:
        state = session.state
        attributes = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": call.name,
            "gen_ai.tool.call.id": call.id,
        }
        with self.tracer.span(f"execute_tool {call.name}", attributes) as span:
            shown = call.raw_arguments if call.arguments is None else canonical_json(call.arguments)
            fingerprint = f"{call.name}:{shown}"
            count = state.fingerprints.get(fingerprint, 0) + 1
            state.fingerprints[fingerprint] = count
            if count > self.max_identical_calls:
                message = (
                    f"Identischer Aufruf zum {count}. Mal – nicht erneut ausgeführt. "
                    "Ändere die Argumente oder schliesse mit einer Antwort ab."
                )
                if count > self.max_identical_calls + 1:
                    state.stop = ["loop_detected", f"Schleife erkannt: {call.name} wiederholt"]
                span.fail("duplicate_call")
                self._observe(
                    state,
                    call.id,
                    call.name,
                    _error(ErrorType.DUPLICATE_CALL, message),
                    step=step,
                    arguments=call.arguments,
                )
                return
            if call.name in self._delegates:
                self._delegate(session, call, step, span)
                return
            found = self.registry.find(call.name)
            if found is None:
                names = ", ".join(t.name for t in self.tools) or "keine"
                span.fail("unknown_tool")
                self._observe(
                    state,
                    call.id,
                    call.name,
                    _error(
                        ErrorType.UNKNOWN_TOOL,
                        f"Werkzeug {call.name!r} gibt es nicht; verfügbar: {names}",
                    ),
                    step=step,
                    arguments=call.arguments,
                )
                return
            issues = found.check_arguments(call.arguments)
            if call.arguments is None or issues:
                details: JSONValue = [{"path": i.path, "message": i.message} for i in issues]
                message = call.error or "Argumente entsprechen nicht dem Schema"
                span.fail("validation_error")
                self.audit.append(
                    "tool_rejected",
                    {
                        "call_id": call.id,
                        "tool": call.name,
                        "raw_arguments": call.raw_arguments,
                        "issues": details,
                    },
                    run_id=state.run_id,
                    agent=self.name,
                )
                self._observe(
                    state,
                    call.id,
                    call.name,
                    _error(ErrorType.VALIDATION, message, details),
                    step=step,
                    arguments=call.arguments,
                )
                return
            assert call.arguments is not None  # noqa: S101 - oben geprüft
            decision = self.policy.decide(found, found.parse_arguments(call.arguments))
            span.update(
                {
                    "gw.side_effect": found.side_effect.value,
                    "gw.policy.decision": decision.decision.value,
                    "gw.policy.rule": decision.rule,
                }
            )
            self.audit.append(
                "policy_decision",
                {
                    "call_id": call.id,
                    "tool": found.name,
                    "arguments": call.arguments,
                    "side_effect": found.side_effect.value,
                    **decision.to_json(),
                },
                run_id=state.run_id,
                agent=self.name,
            )
            if decision.decision is Decision.DENY:
                self._observe(
                    state,
                    call.id,
                    found.name,
                    _error(ErrorType.POLICY_DENIED, decision.reason),
                    step=step,
                    arguments=call.arguments,
                    decision="deny",
                )
                return
            if decision.decision is Decision.REQUIRE_APPROVAL:
                request = self.approvals.submit(
                    run_id=state.run_id,
                    agent=self.name,
                    call_id=call.id,
                    tool=found.name,
                    arguments=call.arguments,
                    side_effect=found.side_effect.value,
                    amount=None if decision.amount is None else str(decision.amount),
                    reason=decision.reason,
                )
                state.pending.append(
                    PendingCall(call.id, found.name, call.arguments, approval_id=request.id)
                )
                self.audit.append(
                    "approval_requested",
                    {
                        "approval_id": request.id,
                        "call_id": call.id,
                        "tool": found.name,
                        "reason": decision.reason,
                    },
                    run_id=state.run_id,
                    agent=self.name,
                )
                span.add_event("approval_requested", approval_id=request.id)
                state.steps.append(
                    StepRecord(
                        step,
                        "tool",
                        found.name,
                        call.id,
                        call.arguments,
                        "require_approval",
                        None,
                        None,
                        f"wartet auf Freigabe {request.id}: {decision.reason}",
                    )
                )
                return
            self._execute(
                session, found, call.id, call.arguments, step=step, decision="allow", span=span
            )

    def _execute(
        self,
        session: _Session,
        item: Tool[..., Any],
        call_id: str,
        arguments: JSONObject,
        *,
        step: int,
        decision: str,
        span: Span,
        note: str = "",
    ) -> None:
        state = session.state
        remaining = state.budget.remaining_seconds(
            self.budget, elapsed=self._clock() - session.started
        )
        if remaining is not None and remaining <= 0:
            result = ToolResult.failure(
                item.name,
                call_id,
                ErrorType.TIMEOUT,
                "Zeitbudget des Laufs erschöpft; nicht ausgeführt",
            )
        else:
            result = self.executor.execute(item, arguments, call_id=call_id, timeout_s=remaining)
        span.update(
            {
                "gw.ok": result.ok,
                "gw.duration_s": round(result.duration_s, 4),
                "gw.isolation": result.isolation,
                "gw.attempts": result.attempts,
            }
        )
        if not result.ok and result.error is not None:
            span.fail(result.error.type.value)
        self.audit.append(
            "tool_result",
            {
                "call_id": call_id,
                "tool": item.name,
                "arguments": arguments,
                "ok": result.ok,
                "output": truncate(dumps(result.output), 4000) if result.ok else None,
                "error": result.error.to_json() if result.error else None,
                "duration_s": round(result.duration_s, 4),
                "attempts": result.attempts,
                "isolation": result.isolation,
                "sandbox": result.sandbox,
                "decision": decision,
            },
            run_id=state.run_id,
            agent=self.name,
        )
        self._observe(
            state,
            call_id,
            item.name,
            result.observation(),
            step=step,
            arguments=arguments,
            decision=decision,
            note=note,
        )

    def _delegate(self, session: _Session, call: ToolCall, step: int, span: Span) -> None:
        state = session.state
        delegate = self._delegates[call.name]
        stub = next(t for t in self._delegate_tools if t.name == call.name)
        issues = stub.check_arguments(call.arguments)
        task = call.arguments.get("auftrag") if call.arguments else None
        if issues or not isinstance(task, str) or not task.strip():
            details: JSONValue = [{"path": i.path, "message": i.message} for i in issues]
            self._observe(
                state,
                call.id,
                call.name,
                _error(ErrorType.VALIDATION, "Auftrag fehlt oder ist leer", details),
                step=step,
                kind="delegation",
                arguments=call.arguments,
            )
            return
        state.delegations += 1
        child_id = f"{state.run_id}/{delegate.name}-{state.delegations}"
        self.audit.append(
            "delegation_started",
            {
                "call_id": call.id,
                "delegate": delegate.name,
                "auftrag": task,
                "child_run_id": child_id,
            },
            run_id=state.run_id,
            agent=self.name,
        )
        span.set("gw.delegate", delegate.name)
        child = delegate.run(task, run_id=child_id)
        self._after_delegation(state, call.id, call.name, call.arguments or {}, child, step)

    def _after_delegation(
        self,
        state: RunState,
        call_id: str,
        tool_name: str,
        arguments: JSONObject,
        child: RunResult,
        step: int,
    ) -> None:
        delegate = self._delegates[tool_name]
        if child.status is RunStatus.PAUSED:
            state.pending.append(
                PendingCall(
                    call_id,
                    tool_name,
                    arguments,
                    delegate=delegate.name,
                    child=child.state.to_json(),
                )
            )
            state.steps.append(
                StepRecord(
                    step,
                    "delegation",
                    tool_name,
                    call_id,
                    arguments,
                    None,
                    None,
                    None,
                    f"{delegate.name} wartet auf Freigabe",
                )
            )
            return
        self.audit.append(
            "delegation_finished",
            {
                "call_id": call_id,
                "delegate": delegate.name,
                "status": child.status.value,
                "reason": child.reason,
            },
            run_id=state.run_id,
            agent=self.name,
        )
        if child.status is RunStatus.COMPLETED:
            observation: JSONObject = {
                "ok": True,
                "result": {"agent": delegate.name, "answer": child.answer},
            }
        else:
            observation = _error(
                ErrorType.DELEGATION, f"{delegate.name}: {child.status.value} ({child.message})"
            )
        self._observe(
            state,
            call_id,
            tool_name,
            observation,
            step=step,
            kind="delegation",
            arguments=arguments,
        )

    def _resolve_pending(self, session: _Session) -> bool:
        """Wendet Entscheide an. ``True``, wenn nichts mehr offen ist."""
        state = session.state
        step = state.budget.steps
        still: list[PendingCall] = []
        for pending in state.pending:
            if pending.delegate is not None:
                delegate = self._delegates[pending.tool]
                child_state = RunState.from_json(pending.child or {})
                child = delegate.resume(child_state)
                if child.status is RunStatus.PAUSED:
                    pending.child = child.state.to_json()
                    still.append(pending)
                else:
                    self._after_delegation(
                        state, pending.call_id, pending.tool, pending.arguments, child, step
                    )
                continue
            assert pending.approval_id is not None  # noqa: S101 - Freigabe oder Delegation
            request = self.approvals.get(pending.approval_id)
            if request.status is ApprovalStatus.PENDING:
                still.append(pending)
                continue
            self.audit.append(
                "approval_applied",
                {
                    "approval_id": request.id,
                    "call_id": pending.call_id,
                    "tool": request.tool,
                    "status": request.status.value,
                    "by": request.decided_by,
                    "note": request.note,
                    "arguments": request.effective_arguments,
                },
                run_id=state.run_id,
                agent=self.name,
            )
            attributes = {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": request.tool,
                "gen_ai.tool.call.id": pending.call_id,
                "gw.approval_id": request.id,
                "gw.approval.status": request.status.value,
            }
            with self.tracer.span(f"execute_tool {request.tool}", attributes) as span:
                approved_tool = self.registry.find(request.tool)
                if request.status.approved and approved_tool is None:
                    span.fail("unknown_tool")
                    self._observe(
                        state,
                        pending.call_id,
                        request.tool,
                        _error(
                            ErrorType.UNKNOWN_TOOL,
                            f"Werkzeug {request.tool!r} nicht mehr registriert",
                        ),
                        step=step,
                        arguments=request.arguments,
                    )
                elif request.status.approved and approved_tool is not None:
                    changed = " mit Änderungen" if request.status is ApprovalStatus.MODIFIED else ""
                    self._execute(
                        session,
                        approved_tool,
                        pending.call_id,
                        request.effective_arguments,
                        step=step,
                        decision=request.status.value,
                        span=span,
                        note=f"freigegeben{changed} durch {request.decided_by}",
                    )
                else:
                    span.fail("rejected_by_human")
                    self._observe(
                        state,
                        pending.call_id,
                        request.tool,
                        _error(
                            ErrorType.REJECTED,
                            f"Abgelehnt durch {request.decided_by}: {request.note}",
                        ),
                        step=step,
                        arguments=request.arguments,
                        decision=request.status.value,
                    )
        state.pending = still
        return not still

    def _finish_turn(self, state: RunState) -> _Outcome | None:
        """Hängt die Beobachtungen in Aufrufreihenfolge an und wertet Fehler aus."""
        strategy = strategy_for(ToolCallingMode(state.mode))
        for call in state.turn_calls:
            call_id = str(call["id"])
            stub = ToolCall(call_id, str(call["name"]), {})
            state.messages.append(strategy.observation(stub, state.observations[call_id]))
        errors = list(state.turn_errors)
        state.turn_calls, state.observations, state.turn_errors = [], {}, []
        if state.stop:
            code, message = state.stop[0], state.stop[1]
            return _Outcome(RunStatus.STOPPED, code, message)
        if not errors:
            state.consecutive_errors = 0
            return None
        state.consecutive_errors += 1
        if state.consecutive_errors >= self.max_consecutive_errors:
            return _Outcome(
                RunStatus.STOPPED,
                "too_many_errors",
                f"{state.consecutive_errors} Fehlschritte in Folge: " + "; ".join(errors),
            )
        if self.replan_hints:
            state.messages.append(user_message(REPLAN_HINT.format(errors="; ".join(errors))))
        return None


def _error(error_type: ErrorType, message: str, details: JSONValue = None) -> JSONObject:
    error: JSONObject = {"type": error_type.value, "message": message}
    if details:
        error["details"] = details
    return {"ok": False, "error": error}
