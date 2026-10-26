"""ToolExecutor: erzwungene Zeitlimits, Wiederholungen, typisierte Ergebnisse."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import sandbox_tools
from gradweight_orchestration import (
    ErrorType,
    SideEffect,
    ToolExecutor,
    TransientToolError,
    detect_capabilities,
    tool,
)
from gradweight_orchestration.sandbox import SandboxCapabilities


@dataclass(frozen=True)
class Kurs:
    waehrung: str
    wert: float


@tool(side_effect=SideEffect.READ_ONLY, timeout_s=5)
def kurs(waehrung: str) -> Kurs:
    """Kurs."""
    return Kurs(waehrung, 0.94)


release = threading.Event()


@tool(side_effect=SideEffect.READ_ONLY, timeout_s=0.2)
def haengt() -> str:
    """Hängt, bis der Test ihn freigibt."""
    release.wait(10)
    return "spät"


calls = {"n": 0}


@tool(side_effect=SideEffect.READ_ONLY, retries=2)
def wackelt() -> str:
    """Schlägt zweimal vorübergehend fehl."""
    calls["n"] += 1
    if calls["n"] < 3:
        raise TransientToolError("später")
    return "ok"


@tool(side_effect=SideEffect.READ_ONLY, retries=3)
def kaputt() -> str:
    """Dauerhafter Fehler."""
    raise RuntimeError("defekt")


@tool(side_effect=SideEffect.READ_ONLY)
def falsch() -> int:
    """Falscher Rückgabetyp."""
    return "x"  # type: ignore[return-value]


@tool(side_effect=SideEffect.READ_ONLY)
def nicht_json() -> Any:
    """Nicht JSON-fähig."""
    return object()


def test_success_returns_typed_value_and_json() -> None:
    result = ToolExecutor().execute(kurs, {"waehrung": "EUR"}, call_id="c1")
    assert result.ok
    assert result.value == Kurs("EUR", 0.94)
    assert result.output == {"waehrung": "EUR", "wert": 0.94}
    assert result.observation() == {"ok": True, "result": {"waehrung": "EUR", "wert": 0.94}}
    assert result.isolation == "thread"


def test_validation_error_is_structured() -> None:
    result = ToolExecutor().execute(kurs, {"waehrung": 1, "x": 2}, call_id="c1")
    assert not result.ok
    assert result.error is not None
    assert result.error.type is ErrorType.VALIDATION
    obs = result.observation()
    assert obs["ok"] is False
    details = obs["error"]["details"]  # type: ignore[index, call-overload]
    assert {d["path"] for d in details} == {"$.waehrung", "$.x"}


def test_thread_timeout_returns_promptly() -> None:
    executor = ToolExecutor()
    release.clear()
    started = time.monotonic()
    result = executor.execute(haengt, {}, call_id="c1")
    assert time.monotonic() - started < 1.0
    assert result.error is not None
    assert result.error.type is ErrorType.TIMEOUT
    assert executor.abandoned_threads == 1
    release.set()
    deadline = time.monotonic() + 5
    while executor.abandoned_threads and time.monotonic() < deadline:
        time.sleep(0.01)
    assert executor.abandoned_threads == 0


def test_timeout_override_is_capped_by_tool_limit() -> None:
    release.clear()
    started = time.monotonic()
    result = ToolExecutor().execute(haengt, {}, call_id="c1", timeout_s=60)
    assert time.monotonic() - started < 1.0
    assert result.error is not None
    assert result.error.type is ErrorType.TIMEOUT
    release.set()


def test_transient_errors_are_retried_with_backoff() -> None:
    sleeps: list[float] = []
    calls["n"] = 0
    result = ToolExecutor(sleep=sleeps.append).execute(wackelt, {}, call_id="c1")
    assert result.ok
    assert result.attempts == 3
    assert sleeps == [0.2, 0.4]


def test_permanent_errors_are_not_retried() -> None:
    sleeps: list[float] = []
    result = ToolExecutor(sleep=sleeps.append).execute(kaputt, {}, call_id="c1")
    assert result.error is not None
    assert result.error.type is ErrorType.TOOL_EXCEPTION
    assert result.error.message == "RuntimeError: defekt"
    assert result.attempts == 1
    assert sleeps == []


def test_result_type_is_checked() -> None:
    wrong = ToolExecutor().execute(falsch, {}, call_id="c1")
    assert wrong.error is not None
    assert wrong.error.type is ErrorType.RESULT_TYPE
    opaque = ToolExecutor().execute(nicht_json, {}, call_id="c1")
    assert opaque.error is not None
    assert opaque.error.type is ErrorType.RESULT_TYPE


# ------------------------------------------------------------ Prozessmodus ---

posix_only = pytest.mark.skipif(os.name != "posix", reason="Prozess-Isolation braucht POSIX")


@posix_only
def test_process_tool_returns_typed_dataclass() -> None:
    result = ToolExecutor().execute(sandbox_tools.beleg_parsen, {"text": "B-7;99.95"}, call_id="c1")
    assert result.ok, result.error
    assert result.value == sandbox_tools.Beleg("B-7", sandbox_tools.Decimal("99.95"))
    assert result.isolation == "process"
    assert result.sandbox is not None
    assert result.sandbox["status"] == "ok"


@posix_only
def test_process_tool_timeout() -> None:
    started = time.monotonic()
    result = ToolExecutor().execute(sandbox_tools.isolierter_schlaf, {"seconds": 30}, call_id="c1")
    assert time.monotonic() - started < 3.0
    assert result.error is not None
    assert result.error.type is ErrorType.TIMEOUT


@posix_only
def test_process_tool_transient_retry(tmp_path: Path) -> None:
    sleeps: list[float] = []
    result = ToolExecutor(sleep=sleeps.append).execute(
        sandbox_tools.isolierter_wackler, {"counter": str(tmp_path / "z")}, call_id="c1"
    )
    assert result.ok
    assert result.attempts == 2


@posix_only
def test_process_tool_result_type_checked() -> None:
    result = ToolExecutor().execute(sandbox_tools.falscher_typ, {}, call_id="c1")
    assert result.error is not None
    assert result.error.type is ErrorType.RESULT_TYPE


def test_process_tool_without_posix_fails_closed() -> None:
    caps = SandboxCapabilities(
        platform="win32",
        posix=False,
        is_root=False,
        rlimit_cpu=False,
        rlimit_memory=False,
        rlimit_file_size=False,
        rlimit_nproc=False,
        network_namespace=False,
        pid_namespace=False,
        unshare=None,
    )
    result = ToolExecutor(capabilities=caps).execute(
        sandbox_tools.beleg_parsen, {"text": "B;1"}, call_id="c1"
    )
    assert result.error is not None
    assert result.error.type is ErrorType.SANDBOX


def test_capabilities_are_consistent() -> None:
    caps = detect_capabilities()
    if caps.is_root:
        assert caps.rlimit_nproc is False
