"""M5 — Sandbox-Worker (Kind-Prozess, ein Tool pro Aufruf).

Wird von ``orch.sandbox.run_tool`` per ``subprocess`` gestartet.
Der Worker ist der **isolations-Träger**: im CHILD-Process (nach fork,
vor exec) setzt die ``preexec_fn`` die CPU-Grenze (RLIMIT_CPU). Der
Worker blockiert dann, je nach Spec:

* ``network=False``         → alle ``socket.*``-Calls werfen
* ``allow_spawn=False``     → ``os.system`` / ``subprocess``-Fork blockiert

Erst DANN wird die Tool-Quelle (via ``inspect.getsource``) in einem
frischen Namespace exec'et und genau einmal aufgerufen. Das Ergebnis
landet in einer Ergebnis-Datei (JSON), damit der Parent es liest,
auch wenn der Kind-Prozess per SIGXCPU/Timeout getötet wurde.

Kein cgroup, kein Namespace (KMU-Sweet-Testpfad, s. M5-M6.md).
Für Produktion: cgroup v2 + ``unshare``/network-namespace.
"""
from __future__ import annotations

import json
import os
import sys

# ── Spezifikation aus env (von Parent gesetzt) ────────────────────────────────
_spec = json.loads(os.environ.get("GW_SB_SPEC", "{}"))


def _block_net(_=None, **__) -> None:
    raise RuntimeError("netzwerk-blockiert (M5, network=False)")


def _block_spawn(_=None, **__) -> None:
    raise RuntimeError("spawn-blockiert (M5, os.system/subprocess/shutil)")


# ── 1. Netzwerk-Isolation (bevor irgendein Tool importiert) ───────────────────
if not _spec.get("network", False):
    import socket

    socket.socket = _block_net           # type: ignore[assignment]
    socket.create_connection = _block_net  # type: ignore[assignment]
    socket.getaddrinfo = _block_net      # type: ignore[assignment]
    socket.gethostbyname = _block_net    # type: ignore[assignment]

# ── 2. Subprocess-/Escape-Block (kein Kind-Fork aus dem Tool) ─────────────────
if not _spec.get("allow_spawn", False):
    import os as _os

    import shutil
    import subprocess as _sp

    _os.system = _block_spawn            # type: ignore[assignment]
    _os.popen = _block_spawn            # type: ignore[assignment]
    for _attr in ("run", "Popen", "call", "check_output", "check_call", "getoutput", "getstatusoutput"):
        if hasattr(_sp, _attr):
            setattr(_sp, _attr, _block_spawn)
    for _attr in ("copy", "rmtree", "move", "copytree", "copy2"):
        if hasattr(shutil, _attr):
            setattr(shutil, _attr, _block_spawn)  # keine Dateimanipulation am Host


def _main() -> int:
    result_path = _spec["result_path"]
    source = _spec["source"]

    fn_name = _spec.get("fn_name", _spec.get("name", "main"))
    _args = list(_spec.get("args", []))
    _kwargs = dict(_spec.get("kwargs", {}))
    ns: dict = {"__name__": "__sandbox_tool__"}
    exec(compile(source, "<sandbox-tool>", "exec"), ns)  # noqa: S102 (isol.)
    fn = ns[fn_name]
    try:
        out = fn(*_args, **_kwargs)
        payload = {"ok": True, "value": out, "error": ""}
    except Exception as e:  # noqa: BLE001 (Tool-Fehler einfangen)
        import sys as _sys

        payload = {"ok": False, "value": None,
                   "error": f"{type(e).__name__}: {e}",
                   "cls": type(e).__name__}
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
