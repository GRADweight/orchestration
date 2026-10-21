"""AgentRunner — der Loop, der den Agenten von Planner bis Observer führt (M4).

Lauf pro Aufgabe:
  1. Planner.plane  → Sub-Task-Liste (bzw. zyklischer Plan → sofort Abort)
  2. For each Sub-Task:
       Router.route    → welches Tool (Confidence)
       Executor.execute→ Tool ausführen (Timeout + Retry + Audit)
       Observer.observe→ OK / UERWARTET / UNSICHER
  3. Bei allen OK  → done=True, task=ERLEDIGT
     Bei einem UERWARTET → task=FEHLGESCHLAGEN, Loop stoppt
     Bei max_steps / timeout → task=ABGESAGT

CANONICAL offline: Regel-Planner + Regel-Observer + StubClient (kein Netz).
Für Produktion: LLMPlanner + LLMObserver (gw_shared_lib.inference.chat).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .executor import Executor
from .observer import RuleObserver
from .planner import RulePlanner
from .router import ToolRouter
from .types import (
    Observation,
    Plan,
    Task,
    TaskStatus,
    AgentResult,
    RouterDecision,
    ToolResult,
)


@dataclass
class AgentRunner:
    """Führt einen Multi-Agenten-Plan aus (M4).

    Beispiel:
        runner = AgentRunner(tool_specs=[...], threshold=0.1, max_steps=10)
        res = runner.run("erst A, dann B, schliesslich C", mandant="K1")
        assert res.done  # alle OK
    """

    tool_specs: list  # liste von ToolSpec
    threshold: float = 0.1
    max_steps: int = 10
    planner: object = field(default_factory=RulePlanner, repr=False)
    observer: object = field(default_factory=RuleObserver, repr=False)
    executor: Executor | None = None  # None → neue mit default
    audit_log: object = None
    mandant: str = "demo"
    prototyp: str = "orchestration"

    def __post_init__(self) -> None:
        if self.executor is None:
            self.executor = Executor(audit_log=self.audit_log)  # type: ignore
        # Router aufbauen (einer pro Runner, fallback aus specs[0])
        self._router = ToolRouter(
            self.tool_specs,
            threshold=self.threshold,
            fallback=self.tool_specs[0] if self.tool_specs else None,
        )

    # ── intern ─────────────────────────────────────────────────────────────

    def _run_plan(self, plan: Plan) -> AgentResult:
        """Führt alle Sub-Steps aus. Bei einem UERWARTET gestoppt der Loop."""
        obs_list: list[Observation] = []
        current: Task | None = None
        steps = 0
        if getattr(plan, "cyclic", False):
            return AgentResult(
                done=False,
                task=TaskStatus.ABGESAGT,
                observations=[],
                reason="Zykluserkennung: Plan enthält wiederholten Schritt",
            )
        for sub in plan.subtasks:
            if steps >= self.max_steps:
                return AgentResult(
                    done=False,
                    task=TaskStatus.ABGESAGT,
                    observations=obs_list,
                    final_task=current,
                    steps_run=steps,
                    reason=f"max_steps ({self.max_steps}) erreicht",
                )
            task = Task(id=steps, text=sub.text)
            decision = self._router.route(sub.text)
            result = self.executor.execute(
                decision,
                task.text,
                mandant=self.mandant,
                prototyp=self.prototyp,
            )
            obs = self.observer.observe(result, task)
            task = current = Task(
                id=task.id,
                text=task.text,
                status=TaskStatus.ERLEDIGT if obs.successful else TaskStatus.FEHLGESCHLAGEN,
                result=result,
                observation=obs,
            )
            obs_list.append(obs)
            if not obs.successful:
                return AgentResult(
                    done=False,
                    task=TaskStatus.FEHLGESCHLAGEN,
                    observations=obs_list,
                    final_task=task,
                    steps_run=steps + 1,
                    reason=obs.reason,
                )
            steps += 1
        return AgentResult(
            done=True,
            task=TaskStatus.ERLEDIGT,
            observations=obs_list,
            final_task=current,
            steps_run=steps,
            reason="alle Schritte erfolgreich",
        )

    # ── API ────────────────────────────────────────────────────────────────

    def route(self, subtask_text: str) -> RouterDecision:
        return self._router.route(subtask_text)

    def run(self, task_text: str, *, mandant: str | None = None) -> AgentResult:
        """Führt eine Aufgabe (Text) durch den kompletten Loop."""
        if mandant is not None:
            self.mandant = mandant
        plan = self.planner.plan(task_text)
        return self._run_plan(plan)
