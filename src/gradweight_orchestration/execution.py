"""Ausführung von Werkzeugen mit erzwungenem Zeitlimit und typisiertem Ergebnis.

* ``Isolation.THREAD``: Aufruf in einem eigenen Daemon-Thread. Nach Ablauf des
  Zeitlimits kehrt der Aufrufer sofort zurück. Python kann Threads nicht
  abbrechen – der Thread läuft im Hintergrund weiter, bis das Werkzeug selbst
  endet. Er wird gezählt (:attr:`ToolExecutor.abandoned_threads`). Nur für
  vertrauenswürdigen Code.
* ``Isolation.PROCESS``: :func:`~gradweight_orchestration.sandbox.run_sandboxed`
  mit hartem ``SIGKILL`` und Ressourcenlimits.

Ergebnisse werden gegen den annotierten Rückgabetyp geprüft; das Modell erhält
strukturierte Beobachtungen ``{"ok": true, "result": ...}`` bzw.
``{"ok": false, "error": {"type": ..., "message": ...}}``.
"""

from __future__ import annotations

import enum
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ._json import JSONObject, JSONValue, NotJSONableError, to_jsonable
from .errors import ArgumentValidationError, SandboxUnavailableError, TransientToolError
from .sandbox import SandboxCapabilities, SandboxStatus, run_sandboxed
from .tools import Isolation, Tool
from .validation import from_jsonable, validate

__all__ = ["ErrorType", "ToolError", "ToolExecutor", "ToolResult"]


class ErrorType(str, enum.Enum):
    """Fehlerklassen, die das Modell in Beobachtungen sieht."""

    VALIDATION = "validation_error"
    UNKNOWN_TOOL = "unknown_tool"
    POLICY_DENIED = "policy_denied"
    REJECTED = "rejected_by_human"
    TIMEOUT = "timeout"
    CPU_LIMIT = "cpu_limit"
    MEMORY_LIMIT = "memory_limit"
    OUTPUT_TOO_LARGE = "output_too_large"
    TOOL_EXCEPTION = "tool_exception"
    SANDBOX = "sandbox_error"
    RESULT_TYPE = "result_type_error"
    DELEGATION = "delegation_failed"
    DUPLICATE_CALL = "duplicate_call"


@dataclass(frozen=True)
class ToolError:
    type: ErrorType
    message: str
    details: JSONValue = None

    def to_json(self) -> JSONObject:
        out: JSONObject = {"type": self.type.value, "message": self.message}
        if self.details is not None:
            out["details"] = self.details
        return out


@dataclass(frozen=True)
class ToolResult:
    """Ergebnis eines Werkzeugaufrufs (Erfolg oder strukturierter Fehler)."""

    tool: str
    call_id: str
    ok: bool
    value: Any = None
    """Typisierter Python-Wert (z.B. Dataclass-Instanz)."""
    output: JSONValue = None
    """JSON-Form des Werts – das sieht das Modell."""
    error: ToolError | None = None
    duration_s: float = 0.0
    attempts: int = 1
    isolation: str | None = None
    sandbox: JSONObject | None = None

    @classmethod
    def failure(
        cls,
        tool: str,
        call_id: str,
        error_type: ErrorType,
        message: str,
        details: JSONValue = None,
        **extra: Any,
    ) -> ToolResult:
        return cls(
            tool=tool,
            call_id=call_id,
            ok=False,
            error=ToolError(error_type, message, details),
            **extra,
        )

    def observation(self) -> JSONObject:
        """Strukturierte Beobachtung für das Modell."""
        if self.ok:
            return {"ok": True, "result": self.output}
        assert self.error is not None  # noqa: S101 - Invariante von failure()
        return {"ok": False, "error": self.error.to_json()}


_SANDBOX_ERRORS = {
    SandboxStatus.TIMEOUT: ErrorType.TIMEOUT,
    SandboxStatus.CPU_LIMIT: ErrorType.CPU_LIMIT,
    SandboxStatus.MEMORY_LIMIT: ErrorType.MEMORY_LIMIT,
    SandboxStatus.OUTPUT_TOO_LARGE: ErrorType.OUTPUT_TOO_LARGE,
    SandboxStatus.ERROR: ErrorType.TOOL_EXCEPTION,
    SandboxStatus.KILLED: ErrorType.SANDBOX,
    SandboxStatus.CRASHED: ErrorType.SANDBOX,
    SandboxStatus.SANDBOX_ERROR: ErrorType.SANDBOX,
}


@dataclass
class _Attempt:
    ok: bool
    value: Any = None
    output: JSONValue = None
    error: ToolError | None = None
    transient: bool = False
    sandbox: JSONObject | None = None


class ToolExecutor:
    """Führt Werkzeuge aus; kennt weder Policy noch Audit (das macht der Agent)."""

    def __init__(
        self,
        *,
        capabilities: SandboxCapabilities | None = None,
        sleep: Callable[[float], None] = time.sleep,
        retry_base_delay_s: float = 0.2,
    ) -> None:
        self._capabilities = capabilities
        self._sleep = sleep
        self._retry_base = retry_base_delay_s
        self._lock = threading.Lock()
        self._running: set[threading.Thread] = set()

    @property
    def abandoned_threads(self) -> int:
        """Threads, deren Werkzeug das Zeitlimit überschritten hat und noch läuft."""
        with self._lock:
            return sum(1 for t in self._running if t.is_alive())

    def execute(
        self,
        tool: Tool[..., Any],
        arguments: Mapping[str, JSONValue] | None,
        *,
        call_id: str,
        timeout_s: float | None = None,
    ) -> ToolResult:
        """Validiert, führt aus (mit Wiederholungen) und prüft das Ergebnis."""
        start = time.monotonic()
        try:
            kwargs = tool.parse_arguments(arguments)
        except ArgumentValidationError as exc:
            return ToolResult.failure(
                tool.name,
                call_id,
                ErrorType.VALIDATION,
                "Argumente entsprechen nicht dem Schema",
                details=[{"path": i.path, "message": i.message} for i in exc.issues],
            )
        limit = tool.timeout_s if timeout_s is None else max(0.001, min(timeout_s, tool.timeout_s))
        attempts = 0
        while True:
            attempts += 1
            if tool.isolation is Isolation.PROCESS:
                attempt = self._run_process(tool, dict(arguments or {}), limit)
            else:
                attempt = self._run_thread(tool, kwargs, limit)
            if attempt.ok or not attempt.transient or attempts > tool.retries:
                break
            self._sleep(min(2.0, self._retry_base * (2 ** (attempts - 1))))
        common: dict[str, Any] = {
            "duration_s": time.monotonic() - start,
            "attempts": attempts,
            "isolation": tool.isolation.value,
            "sandbox": attempt.sandbox,
        }
        if not attempt.ok:
            assert attempt.error is not None  # noqa: S101 - Invariante von _Attempt
            return ToolResult(
                tool=tool.name, call_id=call_id, ok=False, error=attempt.error, **common
            )
        return ToolResult(
            tool=tool.name,
            call_id=call_id,
            ok=True,
            value=attempt.value,
            output=attempt.output,
            **common,
        )

    # ------------------------------------------------------------------ #

    def _run_thread(self, tool: Tool[..., Any], kwargs: dict[str, Any], limit: float) -> _Attempt:
        box: dict[str, Any] = {}
        done = threading.Event()

        def runner() -> None:
            try:
                box["value"] = tool.fn(**kwargs)
            except BaseException as exc:  # noqa: BLE001 - Werkzeugfehler werden Beobachtungen
                box["error"] = exc
            finally:
                done.set()

        thread = threading.Thread(target=runner, name=f"tool-{tool.name}", daemon=True)
        with self._lock:
            self._running = {t for t in self._running if t.is_alive()}
            self._running.add(thread)
        thread.start()
        if not done.wait(limit):
            return _Attempt(
                ok=False,
                error=ToolError(
                    ErrorType.TIMEOUT,
                    f"Zeitlimit {limit:g} s überschritten; Aufruf abgebrochen (Thread läuft "
                    "im Hintergrund weiter)",
                ),
            )
        if "error" in box:
            exc = box["error"]
            return _Attempt(
                ok=False,
                transient=isinstance(exc, TransientToolError),
                error=ToolError(ErrorType.TOOL_EXCEPTION, f"{type(exc).__name__}: {exc}"),
            )
        return self._typed(tool, box.get("value"), from_child=False)

    def _run_process(
        self, tool: Tool[..., Any], arguments: dict[str, JSONValue], limit: float
    ) -> _Attempt:
        assert tool.target is not None  # noqa: S101 - von tool() garantiert
        assert tool.sandbox is not None  # noqa: S101 - von tool() garantiert
        try:
            outcome = run_sandboxed(
                tool.target,
                arguments,
                timeout_s=limit,
                limits=tool.sandbox,
                capabilities=self._capabilities,
            )
        except SandboxUnavailableError as exc:
            return _Attempt(ok=False, error=ToolError(ErrorType.SANDBOX, str(exc)))
        report = outcome.report()
        if not outcome.ok:
            error_type = _SANDBOX_ERRORS[outcome.status]
            message = outcome.message
            if outcome.error_type:
                message = f"{outcome.error_type}: {outcome.message}"
            return _Attempt(
                ok=False,
                transient=outcome.error_type == TransientToolError.__name__,
                error=ToolError(error_type, message),
                sandbox=report,
            )
        attempt = self._typed(tool, outcome.value, from_child=True)
        attempt.sandbox = report
        return attempt

    @staticmethod
    def _typed(tool: Tool[..., Any], raw: Any, *, from_child: bool) -> _Attempt:
        """Prüft das Ergebnis gegen den Rückgabetyp und liefert Wert + JSON-Form."""
        try:
            output = raw if from_child else to_jsonable(raw)
        except NotJSONableError as exc:
            return _Attempt(
                ok=False,
                error=ToolError(ErrorType.RESULT_TYPE, f"Ergebnis nicht JSON-fähig: {exc}"),
            )
        if tool.return_schema is not None:
            issues = validate(output, tool.return_schema)
            if issues:
                return _Attempt(
                    ok=False,
                    error=ToolError(
                        ErrorType.RESULT_TYPE,
                        "Ergebnis entspricht nicht dem deklarierten Rückgabetyp",
                        details=[{"path": i.path, "message": i.message} for i in issues],
                    ),
                )
        value = raw
        if from_child:
            try:
                value = from_jsonable(tool.return_type, output)
            except (TypeError, ValueError) as exc:  # pragma: no cover - durch validate abgedeckt
                return _Attempt(ok=False, error=ToolError(ErrorType.RESULT_TYPE, str(exc)))
        return _Attempt(ok=True, value=value, output=output)
