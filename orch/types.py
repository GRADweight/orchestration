"""Geteilte Datenstrukturen für den Multi-Agent-Orchestrator.

Rein Daten (kein LLM, kein Netz). Alle Module (Planner → Router →
Executor → Observer) tauschen diese Objekte aus. Lauffähig ohne
jegliche Abhäng (nur stdlib + dataclasses).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class TaskStatus(str, Enum):
    OFFEN = "offen"        # noch nicht bearbeitet
    AKTIV = "aktiv"        # wird vom Executor ausgeführt
    ERLEDIGT = "erledigt"  # Beobachtung: ok
    FEHLGESCHLAGEN = "fehlgeschlagen"  # Beobachtung: unerwartet
    ABGESAGT = "abgesagt"  # max_steps / timeout


class ObservationStatus(str, Enum):
    OK = "ok"             # Ergebnis innerhalb Erwartung
    UERWARTET = "unerwartet"  # Ergebnis ausserhalb Erwartung
    UNSICHER = "unsicher"  # zu wenig Signal, um zu urteilen


# Ein Tool: eine ausführbare Einheit mit Namen, Beschreibung, Timeout-
# und Netz-Flag (Vorbereitung für M5 Sandbox). Der Call-Objekt ist
# die eigentliche Funktion.
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    call: Callable[..., Any]
    timeout_s: float = 5.0
    network: bool = False  # True → braucht Netz (im Testpfad blockiert, M5)
    keywords: tuple[str, ...] = ()  # für Router-Keyword-Matching


@dataclass(frozen=True)
class Task:
    """Ein Plan-Schritt. `id` ist eindeutig innerhalb eines Plans."""
    id: int
    text: str
    status: TaskStatus = TaskStatus.OFFEN
    result: Any = None
    observation: "Observation | None" = None


@dataclass(frozen=True)
class SubTask:
    """Plan-Zwischenstufe: ein Schritt mit optionalen Argumenten."""
    text: str
    args: dict[str, Any] = field(default_factory=dict)
    tool_hint: str | None = None  # optionaler Tool-Name-Hinweis


@dataclass
class Plan:
    text: str
    subtasks: list[SubTask] = field(default_factory=list)
    cyclic: bool = False  # Zykluserkennung (Planner setzt True → Runner stoppt)

    def to_tasks(self) -> list[Task]:
        return [Task(id=i, text=st.text) for i, st in enumerate(self.subtasks)]


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    value: Any = None
    error: str = ""
    tool: str = ""
    retries: int = 0
    timed_out: bool = False


@dataclass(frozen=True)
class Observation:
    # "erfolgreich / falsch / unsicher" — der Observer urteilt über ein
    # ToolResult. status = OK ist der Erfolg-Fall.
    status: ObservationStatus
    confidence: float  # 0.0..1.0 (je höher, desto sicherer in der Beurteilung)
    reason: str
    task_id: int = 0

    @property
    def successful(self) -> bool:
        return self.status == ObservationStatus.OK


@dataclass
class RouterDecision:
    tool: ToolSpec
    confidence: float
    alternatives: list[ToolSpec] = field(default_factory=list)

    @property
    def confident(self) -> bool:
        return self.confidence >= 0.5


@dataclass
class AgentResult:
    """Endresultat eines AgentRunner-Laufs."""
    done: bool
    task: TaskStatus
    observations: list[Observation] = field(default_factory=list)
    final_task: Task | None = None
    steps_run: int = 0
    reason: str = ""  # warum gestoppt (done/max_steps/timeout/abort)

    @property
    def successful(self) -> bool:
        return self.done and self.task == TaskStatus.ERLEDIGT and bool(
            self.observations and all(o.successful for o in self.observations)
        )
