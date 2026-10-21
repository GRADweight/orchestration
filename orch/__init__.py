"""orch — GRADweight Multi-Agent-Orchestrator (M4–M7, 21.10.2026).

Struktur (s. GAP_ANALYSE.md, Meilenzeichen M4–M7):

    Planner  →  Router  →  Executor  →  Observer   (M4 Agent-Loop)
    Memory   (M6: kurz + lang, JSONL + Ring-Buff)
    Approve  (M7: Human-in-the-Loop via gw_core.approval.ApprovalQueue)
    Feedback (M7: Annotation gut/fehler → Eval-Set)

CANONICAL offline: Regeler + StubClient (gw_shared_lib), kein Netz.
Für Produktion: LLMPlanner / LLMObserver mit GW_INFER_URL + echtes
Embedding-Modell (nomic-embed-text / Ollama / vLLM).

Öffentliche API:
    from orch import AgentRunner, Planner?, ToolSpec, ...
    # Hier:
    from orch import (
        AgentRunner, RulePlanner, LLMPlanner, ToolRouter,
        Executor, RuleObserver, LLMObserver,
        Memory, FeedbackLoop, ApprovalGate,
        ToolSpec, ToolResult, Observation, AgentResult,
    )
"""
from __future__ import annotations

from .types import (
    AgentResult,
    Observation,
    ObservationStatus,
    Plan,
    RouterDecision,
    SubTask,
    Task,
    TaskStatus,
    ToolResult,
    ToolSpec,
)
from .planner import LLMPlanner, RulePlanner
from .router import ToolRouter, _score
from .observer import LLMObserver, RuleObserver
from .executor import Executor
from .runner import AgentRunner
from .tools import (
    Sandbox,
    TimeLimit,
    ToolRegistry,
    registry,
    tool,
)
from .memory import Memory
from .approve import ApprovalGate, FreigabeEntscheid
from .feedback import Annotation, FeedbackLoop

__all__ = [
    # M4
    "AgentRunner",
    "RulePlanner",
    "LLMPlanner",
    "ToolRouter",
    "Executor",
    "RuleObserver",
    "LLMObserver",
    "Sandbox",
    "ToolRegistry",
    "registry",
    "tool",
    # Typen
    "AgentResult",
    "Observation",
    "ObservationStatus",
    "Plan",
    "RouterDecision",
    "SubTask",
    "Task",
    "TaskStatus",
    "ToolResult",
    "ToolSpec",
    # M6
    "Memory",
    # M7
    "ApprovalGate",
    "FreigabeEntscheid",
    "Annotation",
    "FeedbackLoop",
]

# CANONICAL-Version (für Drift-Guard)
__version__ = "0.1.0"
