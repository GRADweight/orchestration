"""Kindprozess-Seite der Sandbox. Wird nur von :mod:`.sandbox` gestartet.

Ablauf: Protokollkanal sichern (eigener Dateideskriptor, ``stdout`` des
Werkzeugs wird auf ``stderr`` umgeleitet) → Limits setzen → *danach* erst den
Werkzeugcode importieren und aufrufen → Ergebnis als JSON-Zeile melden.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import runpy
import signal
import sys
import traceback
import typing
from collections.abc import Callable, Mapping
from typing import Any, NoReturn

from ._json import JSONObject, JSONValue, canonical_json, to_jsonable

EXIT_CPU = 152


class _Channel:
    """Zeilenbasierter JSON-Kanal zum Elternprozess."""

    def __init__(self, fd: int) -> None:
        self.fd = fd

    def emit(self, message: JSONObject) -> None:
        data = (canonical_json(message) + "\n").encode("utf-8")
        view = memoryview(data)
        while view:
            written = os.write(self.fd, view)
            view = view[written:]


def apply_limits(limits: Mapping[str, Any]) -> dict[str, str]:
    """Setzt die verlangten ``rlimit``-Werte; meldet je Limit Erfolg oder Fehler."""
    import resource

    applied: dict[str, str] = {}

    def _set(name: str, which: int, soft: int, hard: int) -> None:
        try:
            resource.setrlimit(which, (soft, hard))
        except (ValueError, OSError) as exc:
            applied[name] = f"fehlgeschlagen: {exc}"
        else:
            applied[name] = str(soft)

    cpu = int(limits["cpu_seconds"])
    _set("cpu_seconds", resource.RLIMIT_CPU, cpu, cpu + 1)
    if limits.get("memory_mb") is not None:
        size = int(limits["memory_mb"]) * 1024 * 1024
        _set("memory_bytes", resource.RLIMIT_AS, size, size)
    if limits.get("file_size_mb") is not None:
        size = int(limits["file_size_mb"]) * 1024 * 1024
        _set("file_size_bytes", resource.RLIMIT_FSIZE, size, size)
    if limits.get("open_files") is not None:
        count = int(limits["open_files"])
        _set("open_files", resource.RLIMIT_NOFILE, count, count)
    if not limits.get("allow_subprocesses", False):
        _set("processes", resource.RLIMIT_NPROC, 0, 0)
    _set("core_bytes", resource.RLIMIT_CORE, 0, 0)
    return applied


def failed_limits(applied: Mapping[str, str]) -> dict[str, str]:
    """Limits, die nicht gesetzt werden konnten (bei ``strict`` ein Abbruchgrund)."""
    return {k: v for k, v in applied.items() if v.startswith("fehlgeschlagen")}


def resolve_target(target: Mapping[str, Any]) -> Callable[..., Any]:
    """Löst ``modul:qualname`` (oder ein Hauptskript) zu einer Funktion auf."""
    main_path = target.get("main_path")
    obj: Any
    if main_path:
        obj = runpy.run_path(str(main_path), run_name="__gw_sandbox_main__")
        parts = str(target["qualname"]).split(".")
        obj = obj[parts[0]]
        rest = parts[1:]
    else:
        obj = importlib.import_module(str(target["module"]))
        rest = str(target["qualname"]).split(".")
    for part in rest:
        obj = getattr(obj, part)
    fn = getattr(obj, "fn", obj)  # Tool-Objekt -> zugrunde liegende Funktion
    if not callable(fn):
        raise TypeError(f"{target['qualname']} ist nicht aufrufbar")
    return typing.cast(Callable[..., Any], fn)


def execute(target: Mapping[str, Any], arguments: Mapping[str, JSONValue]) -> JSONObject:
    """Ruft das Werkzeug auf und baut die Ergebnisnachricht (fängt alle Fehler)."""
    from .validation import from_jsonable

    try:
        fn = resolve_target(target)
        hints = typing.get_type_hints(fn, include_extras=True)
        kwargs = {
            name: from_jsonable(hints.get(name, Any), value) for name, value in arguments.items()
        }
        value = to_jsonable(fn(**kwargs))
    except BaseException as exc:  # noqa: BLE001 - jeder Fehler wird gemeldet, nicht verschluckt
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return {
            "event": "result",
            "ok": False,
            "error_type": type(exc).__name__,
            "message": str(exc)[:2000],
            "traceback": tb[-4000:],
        }
    return {"event": "result", "ok": True, "value": value}


def main(payload: Mapping[str, Any]) -> NoReturn:  # pragma: no cover - läuft im Kindprozess
    channel = _Channel(os.dup(1))
    os.dup2(2, 1)  # print() im Werkzeug landet auf stderr, nicht im Protokoll

    def _on_cpu_limit(_signum: int, _frame: object) -> None:
        channel.emit({"event": "cpu_limit"})
        os._exit(EXIT_CPU)

    signal.signal(signal.SIGXCPU, _on_cpu_limit)
    limits = payload["limits"]
    applied = apply_limits(limits)
    channel.emit({"event": "ready", "applied": dict(applied)})
    failed = failed_limits(applied)
    if failed and limits.get("strict", True):
        hint = (
            " – auf dieser Plattform nicht durchsetzbar; Limit weglassen (z.B. memory_mb=None) "
            "oder strict=False für bewusst unvollständige Isolation"
        )
        channel.emit({"event": "sandbox_error", "message": f"Limits nicht gesetzt: {failed}{hint}"})
        os._exit(0)
    message = execute(payload["target"], payload["arguments"])
    with contextlib.suppress(OSError, ValueError):  # os._exit leert keine Puffer
        sys.stdout.flush()
        sys.stderr.flush()
    try:
        channel.emit(message)
    except MemoryError:
        channel.emit({"event": "result", "ok": False, "error_type": "MemoryError", "message": ""})
    os._exit(0)
