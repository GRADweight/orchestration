"""M5 — Tool-Sandbox (Subprocess-isoliert, stdlib-only).

`Sandbox.run(spec, **kw) -> ToolResult` führt eine `ToolSpec` aus einem
Kind-Prozess aus (`orch.sandbox_worker` — s. dort). Drei Abbruch-Wege:

* **Wall-Timeout** → `subprocess.TimeoutExpired` → `ToolTimeout` (rc -9)
* **CPU-Limit**    → RLIMIT_CPU (preexec_fn, Linux/macOS) → SIGXCPU
                       (rc -24) → `ToolCpuLimit`
* **Escape**       → `socket.*` + `os.system` + `subprocess.*` +
                      `shutil.rmtree/copy` im Kind blockiert →
                      `ToolEscape` (rc -15, oder RuntimeError im Code)

KMU-Sweet (Testpfad): kein cgroup, kein Namespace. Für Produktion:
cgroup v2 / `unshare` / seccomp — dokumentiert in `M5-M6.md`.

Der Worker-Kindprozess ist der Isolationsträger — er importiert
nichts von `orch`, sondern nur den `sandbox_worker.py` Modul-Code.
Die Tool-Funktion (via `inspect.getsource` → `exec`) läuft in einem
frischen Namespace — keine Import-Juggling, kein Pfadproblem.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .types import ToolResult, ToolSpec


# ── Fehlerhierarchie (für saubere Klassifizierung im Aufrufer) ────────────────

class SandboxError(Exception):
    """Basis-Exception für alle Sandbox-Fehler im Tool-Ausführungskontext."""


class ToolTimeout(SandboxError):
    """Wall-Clock-Timeout: das Kind wurde per SIGKILL getötet."""

    def __init__(self, timeout_s: float, tool_name: str = "") -> None:
        self.timeout_s = timeout_s
        self.tool = tool_name
        super().__init__(f"Wall-Timeout nach {timeout_s:.1f}s: {tool_name}")


class ToolCpuLimit(SandboxError):
    """CPU-Limit erreicht: Kind per SIGXCPU getötet."""

    def __init__(self, limit_s: int, tool_name: str = "") -> None:
        self.limit_s = limit_s
        self.tool = tool_name
        super().__init__(f"CPU-Limit {limit_s}s überschritten: {tool_name}")


class ToolNetworkBlock(SandboxError):
    """Netzwerkzugriff im Tool-Code (Socket-API im Kind blockiert)."""

    def __init__(self, tool_name: str = "", where: str = "") -> None:
        self.tool = tool_name
        self.where = where
        msg = f"Netzwerk blockiert (network=False): {tool_name}"
        if where:
            msg += f", {where}"
        super().__init__(msg)


class ToolEscape(SandboxError):
    """Subprocess-/Shell-Escape-Grund (os.system/subprocess/shutil)."""

    def __init__(self, tool_name: str = "", action: str = "") -> None:
        self.tool = tool_name
        self.action = action
        super().__init__(f"Escape-Grund: {tool_name}, {action}")


# ── Ergebnis-Typ ─────────────────────────────────────────────────────────────────


@dataclass
class SandboxReport:
    """Ergebnis-Wrapper für M7-Audit-Trail."""
    tool: str
    ok: bool = False
    value: Any = None
    error: str = ""
    timed_out: bool = False
    cpu_limited: bool = False
    network_blocked: bool = False
    escaped: bool = False
    duration_s: float = 0.0
    cpu_seconds: float = 0.0

    def to_toolresult(self) -> "ToolResult":
        """Converte in `ToolResult` (für M4 Executor-Kompatibilität)."""
        return ToolResult(
            ok=self.ok,
            value=self.value,
            error=self.error,
            retries=0,
            timed_out=self.timed_out,
        )


# ── Sandbox-Orchestrator ─────────────────────────────────────────────────────
#
#  `Sandbox` ist der einzige Ort, an dem ein Tool tatsächlich ausgeführt
#  wird (für den Agent-Runner). In den Tests ruft man `run_function`
#  direkt — `Sandbox.run` nutzt `ToolSpec.call`, `Sandbox.run_function`
#  eine beliebige Funktion.


class Sandbox:
    """Führt eine Funktion aus einem isolierten Kind-Prozess."""

    def __init__(self, *, tmpdir: str | None = None) -> None:
        self._tmpdir = Path(tmpdir) if tmpdir else Path(
            tempfile.mkdtemp(prefix="gw-sbk-", dir="/tmp")
        )
        self._worker = _worker_path()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        spec: ToolSpec,
        *,
        args: list | None = None,
        kwargs: dict | None = None,
    ) -> ToolResult:
        """Führt `spec.call` in der Sandbox aus. Returns `ToolResult`."""
        r = self.run_function(
            spec.call,
            name=spec.name,
            args=tuple(args or ()),
            kwargs=kwargs or {},
            timeout_s=spec.timeout_s,
            network=spec.network,
            limit_cpu=spec.limit_cpu,
            cpu_seconds=spec.cpu_seconds,
        )
        return r.to_toolresult()

    def run_function(
        self,
        fn: Callable[..., Any],
        *,
        name: str = "funktion",
        args: tuple = (),
        kwargs: dict | None = None,
        timeout_s: float = 5.0,
        network: bool = False,
        limit_cpu: bool = True,
        cpu_seconds: float = 5.0,
        allow_spawn: bool = False,
    ) -> SandboxReport:
        """Führt eine beliebige Funktion in der Sandbox aus.

        Die Funktion kommt als **Quell-Text** ins Kind-Prozess
        (via `inspect.getsource`), damit kein Import-Juggling nötig ist.

        Returns ein `SandboxReport` (kein `ToolResult` — die Fehler
        werden im Report als Flags, nicht als Exceptions, klassifiziert).
        """
        src = _tool_source(fn)  # kann SandboxError raise, wenn fn kein mod-objekt

        if isinstance(args, (str, bytes)):
            raise TypeError(
                f"args={type(args).__name__} — erwartete ein Tupel; "
                f"verwende args=({repr(args)},) (einsamen Tupel)"
            )

        payload = {
            "name": name,
            "fn_name": fn.__name__,
            "source": src,
            "args": list(args),
            "kwargs": kwargs or {},
            "result_path": str(_result_path(self._tmpdir, name)),
            "network": network,
            "limit_cpu": limit_cpu,
            "cpu_seconds": int(cpu_seconds),
            "allow_spawn": allow_spawn,
        }
        rep = self._run_child(payload, name, timeout_s)
        return rep

    # ── Intern ────────────────────────────────────────────────────────────────

    def _run_child(
        self,
        payload: dict,
        tool_name: str,
        timeout_s: float,
    ) -> SandboxReport:
        """Startet den Worker-Kindprozess + liest das Ergebnis."""
        result_file = Path(payload["result_path"])
        result_file.parent.mkdir(parents=True, exist_ok=True)
        result_file.write_text("{}", encoding="utf-8")

        # ── Hermetisches Kind: Worker in ein NEUTRALES Temp-Dir kopieren ──
        #
        # Der Worker läuft nie aus `orch/` — dort schattiert `orch/types.py`
        # das stdlib-Modul `types`, und ein Teil von CPythons Bootstrap
        # (`_weakrefset → types.GenericAlias`, versions-abhängig) stürzt dann
        # in einer Importschleife ab. Aus einem neutralen Verzeichnis (tmp)
        # ist `sys.path[0]` leer → stdlib wird sauber aufgelöst, auf jedem
        # OS und jeder Python-Version. (Green-Local / Red-CI-Falle, behoben.)
        child_worker = self._tmpdir / "sandbox_worker.py"
        child_worker.write_bytes(self._worker.read_bytes())

        # ── Hermetisches Kind: sauberes PYTHONPATH ──
        # Das Kind erbt keine `orch`-Pakete vom Parent → kein stdlib-Schatten.
        env = os.environ.copy()
        env["GW_SB_SPEC"] = json.dumps(payload, ensure_ascii=False)
        env.pop("PYTHONPATH", None)
        env["PYTHONPATH"] = ""  # leere Liste: nur stdlib + path[0]
        # SIGXCPU-Signal im Kind abfangen (per preexec_fn gesetzt)
        limit_s = int(payload.get("cpu_seconds", 5))

        start = time.monotonic()
        try:
            p = subprocess.run(
                [sys.executable, str(child_worker)],
                input="",
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout_s,
                preexec_fn=self._preexec(limit_s) if _has_rlimit() else None,
            )
        except subprocess.TimeoutExpired:
            return SandboxReport(
                tool=tool_name,
                ok=False,
                error=f"Wall-Timeout nach {timeout_s:.1f}s",
                timed_out=True,
                duration_s=time.monotonic() - start,
            )
        dur = time.monotonic() - start

        # Ergebnis lesen (Kind könnte per SIGXCPU getötet worden sein,
        #  Ergebnis-Datei könnte leer sein)
        try:
            raw = result_file.read_text(encoding="utf-8")
            res = json.loads(raw) if raw.strip() else {}
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            res = {}

        # Klassifizierung aus Exit-Code + Inhalt
        rc = p.returncode
        err = res.get("error", "") or (p.stderr[-300:] if p.stderr else "")

        if rc == 0 and res:
            if res.get("ok"):
                return SandboxReport(
                    tool=tool_name,
                    ok=True,
                    value=res.get("value"),
                    duration_s=dur,
                )
            # Kind lief, aber Tool selbst ok=False — klassifizieren
            return self._classify_error(res, res.get("error", ""), p, tool_name, dur)

        # rc < 0 → Signal
        if rc < 0:
            signum = -rc
            sig_name = signal.Signals(signum).name if signum < 32 else f"SIG{signum}"
            if sig_name in ("SIGXCPU",):  # signal 24
                return SandboxReport(
                    tool=tool_name,
                    ok=False,
                    error=f"CPU-Limit {limit_s}s: {sig_name}",
                    cpu_limited=True,
                    duration_s=dur,
                )
            if sig_name in ("SIGKILL",):  # signal 9 (wall timeout)
                return SandboxReport(
                    tool=tool_name,
                    ok=False,
                    error=f"Wall-Timeout: {sig_name}",
                    timed_out=True,
                    duration_s=dur,
                )
            # Other signals → generic failure
            return SandboxReport(
                tool=tool_name,
                ok=False,
                error=f"Signal {sig_name}",
                duration_s=dur,
            )

        # rc > 0 oder leer → klassifiziere den Fehler
        return self._classify_error(res, err, p, tool_name, dur)

    @staticmethod
    def _classify_error(
        res: dict,
        err: str,
        p: subprocess.CompletedProcess,
        tool_name: str,
        dur: float,
    ) -> SandboxReport:
        """Klassifiziert einen Kind-Fehler anhand von Fehler-Typ + rc.

        Die Klassifikation nutzt die Marker aus dem Ergebnis-JSON
        (``res``), falls vorhanden; sonst den stderr-Auszug
        (``err``) vom Kindprozess.
        """
        err_str = err or str(p.stderr or "")
        err_lower = err_str.lower()

        if "netzwerk-blockiert" in err_lower or "socket" in err_lower:
            return SandboxReport(
                tool=tool_name, ok=False,
                error=err_str,
                network_blocked=True,
                duration_s=dur,
            )
        if any(m in err_lower for m in ("spawn-blockiert", "os.system", "subprocess", "shutil")):
            return SandboxReport(
                tool=tool_name, ok=False,
                error=err_str,
                escaped=True,
                duration_s=dur,
            )
        return SandboxReport(
            tool=tool_name, ok=False,
            error=err_str or f"rc={p.returncode}",
            duration_s=dur,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _preexec(limit_s: int):
        import resource

        def _pre() -> None:
            resource.setrlimit(resource.RLIMIT_CPU, (limit_s, limit_s))

        return _pre


# ── Module-Level Helpers ─────────────────────────────────────────────────────────


def _has_rlimit() -> bool:
    """True, wenn `resource.setrlimit` verfügbar ist (POSIX)."""
    import sys as _s
    return _s.platform != "win32"


def _worker_path() -> Path:
    return Path(__file__).resolve().parent / "sandbox_worker.py"


def _result_path(tmpdir: Path, name: str) -> Path:
    h = hashlib.sha256(f"{name}-{time.time_ns()}".encode()).hexdigest()[:8]
    return tmpdir / f"result_{h}.json"


def _tool_source(fn: Callable[..., Any]) -> str:
    """Quell-Text der Funktion — für exec im Kind-Prozess.

    Wirft `SandboxError`, wenn `fn` kein `function` ist (z.B. ein
    Modul-Objekt oder ein `lambda`, die kein getsource hat).
    """
    import inspect
    import textwrap

    if not callable(fn) or isinstance(fn, type):
        raise SandboxError(
            f"Nicht-Sandbox-fähig: {type(fn).__name__} — "
            f"nur reguläre `def`-Funktionen sind erlaubt"
        )
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError) as exc:
        raise SandboxError(f"getsource schlug fehl: {exc}") from exc
    return textwrap.dedent(src)


# ── Export (für `orch.sandbox` Import) ────────────────────────────────────────
__all__ = [
    "Sandbox",
    "SandboxError",
    "ToolTimeout",
    "ToolCpuLimit",
    "ToolNetworkBlock",
    "ToolEscape",
    "SandboxReport",
]
