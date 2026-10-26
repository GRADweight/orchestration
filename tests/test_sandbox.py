"""Nachweise für die Subprozess-Isolation: was zugesichert wird, hält.

Tests, deren Voraussetzung die Plattform nicht erfüllt, werden mit Begründung
übersprungen (z.B. RLIMIT_NPROC als root, Netzwerk-Namespace ohne unshare).
"""

from __future__ import annotations

import dataclasses
import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

from gradweight_orchestration import (
    NetworkMode,
    SandboxLimits,
    SandboxStatus,
    SandboxUnavailableError,
    detect_capabilities,
    run_sandboxed,
)
from gradweight_orchestration import _sandbox_child as child
from gradweight_orchestration.sandbox import TargetRef

pytestmark = [
    pytest.mark.sandbox,
    pytest.mark.skipif(os.name != "posix", reason="Prozess-Isolation braucht POSIX"),
]

CAPS = detect_capabilities()
LIMITS = SandboxLimits(network=NetworkMode.DENY_IF_POSSIBLE)


def target(name: str) -> TargetRef:
    return TargetRef(module="sandbox_tools", qualname=name)


def run(name: str, timeout_s: float = 10.0, limits: SandboxLimits = LIMITS, **arguments: object):
    return run_sandboxed(target(name), arguments, timeout_s=timeout_s, limits=limits)  # type: ignore[arg-type]


def test_successful_call_reports_applied_limits() -> None:
    outcome = run("add", a=2, b=3)
    assert outcome.status is SandboxStatus.OK
    assert outcome.value == 5
    assert outcome.applied_limits["cpu_seconds"] == "10"
    assert outcome.applied_limits["memory_bytes"] == str(512 * 1024 * 1024)
    assert outcome.applied_limits["processes"] == "0"
    assert outcome.cpu_seconds >= 0
    assert outcome.report()["status"] == "ok"


def test_timeout_kills_hanging_tool(tmp_path: Path) -> None:
    """Hängendes Werkzeug: Rückkehr kurz nach dem Limit, Prozess schreibt nie mehr."""
    marker = tmp_path / "lebt.txt"
    started = time.monotonic()
    outcome = run("sleep_then_write", timeout_s=1.0, seconds=2.5, path=str(marker))
    elapsed = time.monotonic() - started
    assert outcome.status is SandboxStatus.TIMEOUT
    assert elapsed < 2.0, f"Rückkehr erst nach {elapsed:.2f} s"
    time.sleep(2.5)
    assert not marker.exists(), "Prozess lief nach dem Timeout weiter"


def test_timeout_kills_grandchild_in_same_group(tmp_path: Path) -> None:
    marker = tmp_path / "enkel.txt"
    limits = dataclasses.replace(LIMITS, allow_subprocesses=True)
    outcome = run(
        "spawn_grandchild_then_hang",
        timeout_s=1.0,
        limits=limits,
        path=str(marker),
        new_session=False,
    )
    assert outcome.status is SandboxStatus.TIMEOUT
    time.sleep(2.5)
    assert not marker.exists(), "Enkelprozess überlebte den Abbruch"


@pytest.mark.skipif(
    not CAPS.pid_namespace,
    reason="ohne PID-Namespace kann ein Enkel mit setsid entkommen (dokumentierte Grenze)",
)
def test_pid_namespace_kills_escaped_grandchild(tmp_path: Path) -> None:
    marker = tmp_path / "entkommen.txt"
    limits = dataclasses.replace(LIMITS, allow_subprocesses=True)
    outcome = run(
        "spawn_grandchild_then_hang",
        timeout_s=1.0,
        limits=limits,
        path=str(marker),
        new_session=True,
    )
    assert outcome.status is SandboxStatus.TIMEOUT
    assert outcome.pid_namespace is True
    time.sleep(2.5)
    assert not marker.exists(), "Enkel mit eigener Session überlebte"


def test_cpu_limit_stops_busy_loop() -> None:
    limits = dataclasses.replace(LIMITS, cpu_seconds=1)
    started = time.monotonic()
    outcome = run("burn_cpu", timeout_s=30.0, limits=limits)
    assert outcome.status is SandboxStatus.CPU_LIMIT
    assert time.monotonic() - started < 10.0
    assert outcome.cpu_seconds >= 0.9


@pytest.mark.skipif(not CAPS.rlimit_memory, reason="RLIMIT_AS wird hier nicht zugesichert")
def test_memory_limit() -> None:
    limits = dataclasses.replace(LIMITS, memory_mb=256)
    outcome = run("allocate", limits=limits, mb=1024)
    assert outcome.status is SandboxStatus.MEMORY_LIMIT
    assert outcome.error_type == "MemoryError"
    assert run("allocate", limits=limits, mb=16).status is SandboxStatus.OK


def test_file_size_limit() -> None:
    limits = dataclasses.replace(LIMITS, file_size_mb=1)
    outcome = run("write_file", limits=limits, mb=5)
    assert outcome.status is SandboxStatus.ERROR
    assert "File too large" in outcome.message or "27" in outcome.message


@pytest.mark.skipif(not CAPS.rlimit_nproc, reason="RLIMIT_NPROC wirkt nicht für root")
def test_no_new_processes() -> None:
    outcome = run("spawn_process")
    assert outcome.status is SandboxStatus.ERROR
    assert outcome.error_type in {"BlockingIOError", "PermissionError", "OSError"}
    allowed = dataclasses.replace(LIMITS, allow_subprocesses=True)
    assert run("spawn_process", limits=allowed).status is SandboxStatus.OK


def test_clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GW_GEHEIMNIS", "streng-geheim")
    limits = dataclasses.replace(LIMITS, env=(("GW_ERLAUBT", "1"),))
    outcome = run("environment", limits=limits)
    assert outcome.ok
    env = outcome.value
    assert isinstance(env, dict)
    assert "GW_GEHEIMNIS" not in env
    assert env["GW_ERLAUBT"] == "1"
    assert set(env) <= {
        "PATH",
        "LANG",
        "LC_ALL",
        "HOME",
        "TMPDIR",
        "PYTHONDONTWRITEBYTECODE",
        "GW_ERLAUBT",
        "LC_CTYPE",
    }


def test_temporary_working_directory_is_removed() -> None:
    outcome = run("working_directory", marker="datei.txt")
    assert outcome.ok
    cwd = outcome.value
    assert isinstance(cwd, str)
    assert "gw-sandbox-" in cwd
    assert not Path(cwd).exists()


def test_filesystem_is_not_isolated(tmp_path: Path) -> None:
    """Dokumentierte Grenze: der Kindprozess kann ausserhalb seines cwd schreiben."""
    path = tmp_path / "ausserhalb.txt"
    assert run("write_outside", path=str(path)).ok
    assert path.read_text(encoding="utf-8") == "ausserhalb"


@pytest.fixture
def tcp_server():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    stop = threading.Event()

    def serve() -> None:
        server.settimeout(0.2)
        while not stop.is_set():
            try:
                conn, _ = server.accept()
                conn.close()
            except OSError:
                continue

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    yield server.getsockname()[1]
    stop.set()
    thread.join()
    server.close()


@pytest.mark.skipif(
    not CAPS.network_namespace, reason="kein Netzwerk-Namespace (unshare) verfügbar"
)
def test_network_isolation_blocks_loopback(tcp_server: int) -> None:
    allowed = run(
        "connect", limits=dataclasses.replace(LIMITS, network=NetworkMode.ALLOW), port=tcp_server
    )
    assert allowed.ok, "Kontrolle: ohne Isolation muss die Verbindung klappen"
    assert allowed.network_isolated is False
    denied = run(
        "connect", limits=dataclasses.replace(LIMITS, network=NetworkMode.DENY), port=tcp_server
    )
    assert denied.status is SandboxStatus.ERROR
    assert denied.network_isolated is True
    assert "unreachable" in denied.message.lower() or "refused" in denied.message.lower()


def test_deny_fails_closed_without_namespace() -> None:
    caps = dataclasses.replace(CAPS, network_namespace=False, pid_namespace=False)
    with pytest.raises(SandboxUnavailableError):
        run_sandboxed(
            target("add"),
            {"a": 1, "b": 2},
            timeout_s=5,
            limits=SandboxLimits(network=NetworkMode.DENY),
            capabilities=caps,
        )
    outcome = run_sandboxed(
        target("add"),
        {"a": 1, "b": 2},
        timeout_s=5,
        limits=SandboxLimits(network=NetworkMode.DENY_IF_POSSIBLE),
        capabilities=caps,
    )
    assert outcome.ok
    assert outcome.network_isolated is False


def test_tool_stdout_does_not_break_protocol() -> None:
    outcome = run("noisy", lines=5000)
    assert outcome.ok
    assert outcome.value == "ok"
    assert "Zeile 4999" in outcome.stderr_tail


def test_output_cap() -> None:
    limits = dataclasses.replace(LIMITS, max_output_bytes=10_000)
    outcome = run("big_output", limits=limits, size=1_000_000)
    assert outcome.status is SandboxStatus.OUTPUT_TOO_LARGE


def test_tool_exception_is_reported() -> None:
    outcome = run("fail")
    assert outcome.status is SandboxStatus.ERROR
    assert outcome.error_type == "ValueError"
    assert "absichtlicher Fehler" in outcome.message


def test_unknown_target_is_reported() -> None:
    outcome = run_sandboxed(
        TargetRef("sandbox_tools", "gibt_es_nicht"), {}, timeout_s=5, limits=LIMITS
    )
    assert outcome.status is SandboxStatus.ERROR
    assert outcome.error_type == "AttributeError"


def test_strict_limits_fail_closed() -> None:
    """Kann ein Limit nicht gesetzt werden, läuft das Werkzeug nicht (strict=True).

    ``open_files`` über ``nr_open`` lässt der Kernel auch für root nicht zu.
    """
    impossible = dataclasses.replace(LIMITS, open_files=10**9)
    outcome = run("add", limits=impossible, a=1, b=1)
    assert outcome.status is SandboxStatus.SANDBOX_ERROR
    assert "open_files" in outcome.message
    lenient = dataclasses.replace(impossible, strict=False)
    outcome = run("add", limits=lenient, a=1, b=1)
    assert outcome.ok
    assert outcome.applied_limits["open_files"].startswith("fehlgeschlagen")


def _world_executable(path: str) -> bool:
    """Kann ein fremder Benutzer den Interpreter starten (alle Ordner o+x)?"""
    real = Path(path).resolve()
    return all(p.stat().st_mode & 0o001 for p in [real, *real.parents])


@pytest.mark.skipif(not CAPS.is_root, reason="run_as_user braucht root")
@pytest.mark.skipif(
    not _world_executable(sys.executable),
    reason="Interpreter liegt in einem Ordner, den nobody nicht betreten darf",
)
def test_run_as_unprivileged_user(tmp_path: Path) -> None:
    secret = tmp_path / "nur-root.txt"
    secret.write_text("geheim", encoding="utf-8")
    secret.chmod(0o600)
    plain = dataclasses.replace(
        LIMITS, run_as_user="nobody", network=NetworkMode.ALLOW, pid_namespace=False
    )
    assert run("whoami", limits=plain).value == 65534
    # Mit Namespaces erscheint der Prozess innen als uid 0 (User-Namespace), hat
    # aber ausserhalb nur die Rechte von nobody:
    limits = dataclasses.replace(LIMITS, run_as_user="nobody")
    denied = run("read_file", limits=limits, path=str(secret))
    assert denied.status is SandboxStatus.ERROR
    assert denied.error_type == "PermissionError"
    # Und RLIMIT_NPROC greift wieder, weil der Prozess nicht mehr root ist:
    blocked = run("spawn_process", limits=plain)
    assert blocked.status is SandboxStatus.ERROR
    assert blocked.error_type in {"BlockingIOError", "PermissionError", "OSError"}


def test_run_as_requires_root() -> None:
    caps = dataclasses.replace(CAPS, is_root=False) if CAPS.is_root else CAPS
    with pytest.raises(SandboxUnavailableError):
        run_sandboxed(
            target("add"),
            {"a": 1, "b": 1},
            timeout_s=5,
            limits=dataclasses.replace(LIMITS, run_as_user="nobody"),
            capabilities=caps,
        )


def test_limits_validation() -> None:
    for kwargs in (
        {"cpu_seconds": 0},
        {"memory_mb": 8},
        {"file_size_mb": -1},
        {"open_files": 3},
        {"max_output_bytes": 10},
    ):
        with pytest.raises(ValueError):  # noqa: PT011
            SandboxLimits(**kwargs)  # type: ignore[arg-type]


def test_capabilities_are_reported() -> None:
    data = CAPS.as_dict()
    assert data["platform"] == sys.platform
    assert data["posix"] is True
    assert isinstance(data["notes"], list)
    assert detect_capabilities(refresh=True).platform == sys.platform


# ---------------------------------------------------------------- Kindseite --


def test_child_execute_in_process() -> None:
    """Kindlogik direkt (für Abdeckung ohne Subprozess)."""
    ok = child.execute({"module": "sandbox_tools", "qualname": "add"}, {"a": 1, "b": 2})
    assert ok == {"event": "result", "ok": True, "value": 3}
    failed = child.execute({"module": "sandbox_tools", "qualname": "fail"}, {})
    assert failed["ok"] is False
    assert failed["error_type"] == "ValueError"
    tool_obj = child.execute(
        {"module": "sandbox_tools", "qualname": "beleg_parsen"}, {"text": "B-1;12.50"}
    )
    assert tool_obj["value"] == {"nummer": "B-1", "betrag": 12.5}
    not_callable = child.execute({"module": "sandbox_tools", "qualname": "LIMITS"}, {})
    assert not_callable["error_type"] == "TypeError"


def test_child_resolves_main_script(tmp_path: Path) -> None:
    script = tmp_path / "skript.py"
    script.write_text(
        "def verdoppeln(x: int) -> int:\n    return 2 * x\n"
        "if __name__ == '__main__':\n    raise SystemExit('darf nicht laufen')\n",
        encoding="utf-8",
    )
    result = child.execute(
        {"module": "__main__", "qualname": "verdoppeln", "main_path": str(script)}, {"x": 21}
    )
    assert result == {"event": "result", "ok": True, "value": 42}
    outcome = run_sandboxed(
        TargetRef("__main__", "verdoppeln", str(script)), {"x": 4}, timeout_s=10, limits=LIMITS
    )
    assert outcome.value == 8


def test_child_channel_writes_lines() -> None:
    read_fd, write_fd = os.pipe()
    channel = child._Channel(write_fd)
    channel.emit({"event": "ready", "applied": {}})
    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as fh:
        assert fh.read() == b'{"applied":{},"event":"ready"}\n'


def test_fallback_without_waitid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Plattformen ohne ``waitid``: gleiches Verhalten über ``wait4``."""
    from gradweight_orchestration import sandbox as sandbox_module

    monkeypatch.setattr(sandbox_module, "_HAS_WAITID", False)
    assert run("add", a=1, b=2).value == 3
    marker = tmp_path / "lebt.txt"
    outcome = run("sleep_then_write", timeout_s=0.5, seconds=2.0, path=str(marker))
    assert outcome.status is SandboxStatus.TIMEOUT
    time.sleep(2.0)
    assert not marker.exists()
