"""Subprozess-Isolation für nicht vertrauenswürdige Werkzeuge (POSIX).

Was hier *erzwungen* wird (sofern :func:`detect_capabilities` es meldet):

* Harter Abbruch nach ``timeout_s``: ``SIGKILL`` an die ganze Prozessgruppe;
  mit PID-Namespace zusätzlich an alle Nachfahren, auch wenn sie ``setsid``
  aufrufen.
* Ressourcenlimits per ``setrlimit`` *vor* dem Import des Werkzeugcodes:
  CPU-Zeit, Adressraum (``RLIMIT_AS``, nur Linux zugesichert), Dateigrösse,
  Anzahl offener Dateien, keine neuen Prozesse (``RLIMIT_NPROC=0``; greift nur
  für Nicht-root), keine Core-Dumps. Harte Limits kann der Kindprozess nicht
  wieder anheben.
* Saubere Umgebung: nur ``PATH``, ``LANG``/``LC_ALL``, ``HOME``/``TMPDIR`` (auf
  das temporäre Arbeitsverzeichnis) und explizit freigegebene Variablen;
  Interpreter im isolierten Modus (``python -I``).
* Temporäres Arbeitsverzeichnis, das nach dem Aufruf gelöscht wird.
* Netzwerk-Isolation nur, wenn ein eigener Netzwerk-Namespace erzeugt werden
  kann (Linux, ``unshare -n`` bzw. ``unshare -rn``). Sonst schlägt
  :attr:`NetworkMode.DENY` geschlossen fehl.

Was hier *nicht* erzwungen wird: Dateisystem-Isolation (der Prozess darf alles
lesen und schreiben, was der Benutzer darf), Systemaufruf-Filter (seccomp),
Schutz vor Kernel-Exploits. Das ist **keine Sicherheitsgrenze** für
feindseligen Code – dafür Container mit gVisor, Firecracker o.ä. verwenden.
"""

from __future__ import annotations

import contextlib
import enum
import functools
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ._json import JSONObject, JSONValue, canonical_json
from .errors import SandboxUnavailableError

__all__ = [
    "NetworkMode",
    "SandboxCapabilities",
    "SandboxLimits",
    "SandboxOutcome",
    "SandboxStatus",
    "TargetRef",
    "detect_capabilities",
    "run_sandboxed",
]

_STDERR_TAIL = 4096
_BOOTSTRAP = (
    "import json,sys\n"
    "_p=json.loads(sys.stdin.buffer.read().decode('utf-8'))\n"
    "sys.path[:0]=[x for x in _p['sys_path'] if x not in sys.path]\n"
    "from gradweight_orchestration._sandbox_child import main\n"
    "main(_p)\n"
)


class NetworkMode(enum.StrEnum):
    """Netzwerkzugriff für den Kindprozess."""

    ALLOW = "allow"
    """Keine Netzwerk-Isolation."""
    DENY = "deny"
    """Eigener Netzwerk-Namespace; ist er nicht verfügbar, wird nicht ausgeführt."""
    DENY_IF_POSSIBLE = "deny_if_possible"
    """Isolieren, wenn möglich; das Ergebnis meldet, ob isoliert wurde."""


@dataclass(frozen=True)
class SandboxLimits:
    """Grenzen für einen isolierten Werkzeugaufruf."""

    cpu_seconds: int = 10
    memory_mb: int | None = 512
    file_size_mb: int | None = 16
    open_files: int | None = 64
    allow_subprocesses: bool = False
    network: NetworkMode = NetworkMode.DENY
    max_output_bytes: int = 1_000_000
    env: tuple[tuple[str, str], ...] = ()
    run_as_user: str | None = None
    pid_namespace: bool = True
    strict: bool = True
    """Schlägt ein verlangtes Limit fehl, wird das Werkzeug nicht ausgeführt."""

    def __post_init__(self) -> None:
        if self.cpu_seconds < 1:
            raise ValueError("cpu_seconds muss ≥ 1 sein")
        if self.memory_mb is not None and self.memory_mb < 32:
            raise ValueError("memory_mb muss ≥ 32 sein (der Interpreter braucht Adressraum)")
        if self.file_size_mb is not None and self.file_size_mb < 0:
            raise ValueError("file_size_mb darf nicht negativ sein")
        if self.open_files is not None and self.open_files < 16:
            raise ValueError("open_files muss ≥ 16 sein")
        if self.max_output_bytes < 1024:
            raise ValueError("max_output_bytes muss ≥ 1024 sein")

    def child_view(self) -> JSONObject:
        """Die Limits, die der Kindprozess selbst setzt."""
        return {
            "cpu_seconds": self.cpu_seconds,
            "memory_mb": self.memory_mb,
            "file_size_mb": self.file_size_mb,
            "open_files": self.open_files,
            "allow_subprocesses": self.allow_subprocesses,
            "strict": self.strict,
        }


@dataclass(frozen=True)
class TargetRef:
    """Importierbare Referenz auf eine Funktion (``modul`` + ``qualname``)."""

    module: str
    qualname: str
    main_path: str | None = None
    """Pfad des Hauptskripts, falls die Funktion in ``__main__`` definiert ist."""


@dataclass(frozen=True)
class SandboxCapabilities:
    """Zur Laufzeit ermittelte Fähigkeiten dieser Plattform."""

    platform: str
    posix: bool
    is_root: bool
    rlimit_cpu: bool
    rlimit_memory: bool
    rlimit_file_size: bool
    rlimit_nproc: bool
    network_namespace: bool
    pid_namespace: bool
    unshare: str | None
    notes: tuple[str, ...] = field(default=())

    def as_dict(self) -> JSONObject:
        return {
            "platform": self.platform,
            "posix": self.posix,
            "is_root": self.is_root,
            "rlimit_cpu": self.rlimit_cpu,
            "rlimit_memory": self.rlimit_memory,
            "rlimit_file_size": self.rlimit_file_size,
            "rlimit_nproc": self.rlimit_nproc,
            "network_namespace": self.network_namespace,
            "pid_namespace": self.pid_namespace,
            "unshare": self.unshare,
            "notes": list(self.notes),
        }


def _probe(cmd: Sequence[str]) -> bool:
    try:
        completed = subprocess.run(  # noqa: S603 - feste Argumente, kein Shell
            list(cmd), capture_output=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


@functools.lru_cache(maxsize=1)
def _detect() -> SandboxCapabilities:
    posix = os.name == "posix"
    try:
        import resource  # noqa: F401
    except ImportError:  # pragma: no cover - nur Windows
        posix = False
    is_root = posix and os.geteuid() == 0
    linux = sys.platform.startswith("linux")
    unshare = shutil.which("unshare") if linux else None
    notes: list[str] = []
    netns = pidns = False
    if unshare:
        base = [unshare, "-n"] if is_root else [unshare, "-r", "-n"]
        netns = _probe([*base, "--", "true"])
        pidns = netns and _probe([*base, "--pid", "--fork", "--kill-child=SIGKILL", "--", "true"])
        if not netns:
            notes.append("unshare vorhanden, aber Namespaces nicht erlaubt (Kernel/AppArmor)")
    elif linux:
        notes.append("unshare (util-linux) nicht gefunden")
    if is_root:
        notes.append("läuft als root: RLIMIT_NPROC wird vom Kernel nicht durchgesetzt")
    if sys.platform == "darwin":
        notes.append("macOS: RLIMIT_AS wird vom Kernel nicht zugesichert; keine Namespaces")
    return SandboxCapabilities(
        platform=sys.platform,
        posix=posix,
        is_root=is_root,
        rlimit_cpu=posix,
        rlimit_memory=posix and linux,
        rlimit_file_size=posix,
        rlimit_nproc=posix and not is_root,
        network_namespace=netns,
        pid_namespace=pidns,
        unshare=unshare,
        notes=tuple(notes),
    )


def detect_capabilities(*, refresh: bool = False) -> SandboxCapabilities:
    """Ermittelt (und cacht), was diese Plattform tatsächlich erzwingen kann."""
    if refresh:
        _detect.cache_clear()
    return _detect()


class SandboxStatus(enum.StrEnum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CPU_LIMIT = "cpu_limit"
    MEMORY_LIMIT = "memory_limit"
    OUTPUT_TOO_LARGE = "output_too_large"
    KILLED = "killed"
    CRASHED = "crashed"
    SANDBOX_ERROR = "sandbox_error"


@dataclass(frozen=True)
class SandboxOutcome:
    """Ergebnis eines isolierten Aufrufs, inklusive was tatsächlich erzwungen wurde."""

    status: SandboxStatus
    value: JSONValue = None
    error_type: str | None = None
    message: str = ""
    duration_s: float = 0.0
    cpu_seconds: float = 0.0
    max_rss_kb: int = 0
    returncode: int | None = None
    network_isolated: bool = False
    pid_namespace: bool = False
    applied_limits: Mapping[str, str] = field(default_factory=dict)
    stderr_tail: str = ""

    @property
    def ok(self) -> bool:
        return self.status is SandboxStatus.OK

    def report(self) -> JSONObject:
        """Kompakter Bericht für Audit und Trace."""
        return {
            "status": self.status.value,
            "duration_s": round(self.duration_s, 4),
            "cpu_seconds": round(self.cpu_seconds, 4),
            "max_rss_kb": self.max_rss_kb,
            "returncode": self.returncode,
            "network_isolated": self.network_isolated,
            "pid_namespace": self.pid_namespace,
            "applied_limits": dict(self.applied_limits),
        }


class _Reader(threading.Thread):
    """Liest eine Pipe in einem Thread, mit Obergrenze bzw. als Endstück."""

    def __init__(self, stream: Any, *, cap: int | None, tail: int | None) -> None:
        super().__init__(daemon=True)
        self._stream = stream
        self._cap = cap
        self._tail = tail
        self.data = bytearray()
        self.overflow = False
        self.overflow_event = threading.Event()

    def run(self) -> None:
        try:
            while True:
                chunk = self._stream.read1(65536)
                if not chunk:
                    break
                self.data.extend(chunk)
                if self._cap is not None and len(self.data) > self._cap:
                    self.overflow = True
                    self.overflow_event.set()
                    break
                if self._tail is not None and len(self.data) > self._tail:
                    del self.data[: len(self.data) - self._tail]
        except (OSError, ValueError):
            return  # Pipe wurde beim Aufräumen geschlossen


def _kill_group(pid: int) -> None:
    """SIGKILL an die Prozessgruppe. Nur aufrufen, solange ``pid`` nicht geerntet ist."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)


_HAS_WAITID = all(hasattr(os, name) for name in ("waitid", "P_PID", "WNOWAIT"))


def _wait_exit(pid: int, deadline: float, stop: threading.Event) -> bool:
    """Wartet, bis der Prozess endet, *ohne* ihn zu ernten (``WNOWAIT``).

    Solange der Zombie nicht geerntet ist, bleiben PID und Prozessgruppe
    reserviert; das anschliessende ``killpg`` trifft garantiert die richtige Gruppe.
    """
    delay = 0.002
    while True:
        info = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        if info is not None and info.si_pid == pid:
            return True
        if stop.is_set() or time.monotonic() >= deadline:
            return False
        time.sleep(delay)
        delay = min(delay * 2, 0.05)


def _reap(proc: subprocess.Popen[bytes], deadline: float, stop: threading.Event | None) -> Any:
    """Erntet den Prozess mit ``wait4``; liefert ``rusage`` oder ``None`` bei Fristablauf."""
    delay = 0.002
    while True:
        pid, status, rusage = os.wait4(proc.pid, os.WNOHANG)
        if pid == proc.pid:
            proc.returncode = os.waitstatus_to_exitcode(status)
            return rusage
        if (stop is not None and stop.is_set()) or time.monotonic() >= deadline:
            return None
        time.sleep(delay)
        delay = min(delay * 2, 0.05)


def _supervise(
    proc: subprocess.Popen[bytes], deadline: float, stop: threading.Event
) -> tuple[bool, Any]:
    """Wartet bis zur Frist, beendet die Prozessgruppe und erntet.

    Rückgabe: (rechtzeitig beendet, ``rusage``). Mit ``waitid(WNOWAIT)`` wird die
    Gruppe beendet, solange die PID noch reserviert ist. Ohne ``waitid`` wird
    zuerst geerntet; dann besteht ein theoretisches Fenster für PID-Wiederverwendung.
    """
    if _HAS_WAITID:
        exited = _wait_exit(proc.pid, deadline, stop)
        _kill_group(proc.pid)
        return exited, _reap(proc, time.monotonic() + 5.0, None)
    rusage = _reap(proc, deadline, stop)
    _kill_group(proc.pid)
    if rusage is not None:
        return True, rusage
    return False, _reap(proc, time.monotonic() + 5.0, None)


def _command_prefix(
    limits: SandboxLimits, caps: SandboxCapabilities
) -> tuple[list[str], bool, bool]:
    """Baut das ``unshare``-Präfix. Rückgabe: (Präfix, netz-isoliert, pid-ns)."""
    want_net = limits.network is not NetworkMode.ALLOW
    if want_net and not caps.network_namespace and limits.network is NetworkMode.DENY:
        raise SandboxUnavailableError(
            "Netzwerk-Isolation verlangt (NetworkMode.DENY), auf dieser Plattform aber "
            "nicht verfügbar. Alternativen: NetworkMode.DENY_IF_POSSIBLE, NetworkMode.ALLOW "
            "oder Ausführung in einem Container."
        )
    net = want_net and caps.network_namespace
    pid = limits.pid_namespace and caps.pid_namespace
    if not (net or pid) or caps.unshare is None:
        return [], False, False
    flags = [caps.unshare]
    if not caps.is_root or limits.run_as_user is not None:
        flags.append("-r")
    if net:
        flags.append("-n")
    if pid:
        flags += ["--pid", "--fork", "--kill-child=SIGKILL"]
    return [*flags, "--"], net, pid


def run_sandboxed(
    target: TargetRef,
    arguments: Mapping[str, JSONValue],
    *,
    timeout_s: float,
    limits: SandboxLimits,
    capabilities: SandboxCapabilities | None = None,
) -> SandboxOutcome:
    """Führt ``target(**arguments)`` in einem isolierten Kindprozess aus.

    Wirft :class:`SandboxUnavailableError`, wenn eine verlangte Isolation nicht
    möglich ist (fail-closed). Alle übrigen Fehler stehen im Ergebnis.
    """
    caps = capabilities or detect_capabilities()
    if not caps.posix:
        raise SandboxUnavailableError("Prozess-Isolation benötigt ein POSIX-System")
    if limits.run_as_user is not None and not caps.is_root:
        raise SandboxUnavailableError("run_as_user setzt voraus, dass der Aufrufer root ist")
    prefix, net_isolated, pid_ns = _command_prefix(limits, caps)

    payload = canonical_json(
        {
            "target": {
                "module": target.module,
                "qualname": target.qualname,
                "main_path": target.main_path,
            },
            "arguments": dict(arguments),
            "sys_path": [os.path.abspath(p) for p in sys.path if p],
            "limits": limits.child_view(),
        }
    ).encode("utf-8")

    with tempfile.TemporaryDirectory(prefix="gw-sandbox-", ignore_cleanup_errors=True) as cwd:
        env = {
            "PATH": os.defpath,
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "HOME": cwd,
            "TMPDIR": cwd,
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        env.update(dict(limits.env))
        group: int | None = None
        if limits.run_as_user is not None:
            import pwd

            try:
                group = pwd.getpwnam(limits.run_as_user).pw_gid
            except KeyError as exc:
                raise SandboxUnavailableError(
                    f"Benutzer {limits.run_as_user!r} existiert nicht"
                ) from exc
            shutil.chown(cwd, user=limits.run_as_user, group=group)
        cmd = [*prefix, sys.executable, "-I", "-c", _BOOTSTRAP]
        start = time.monotonic()
        try:
            proc = subprocess.Popen(  # noqa: S603 - Argumentliste, kein Shell
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                env=env,
                start_new_session=True,
                close_fds=True,
                user=limits.run_as_user,
                group=group,
                extra_groups=[] if group is not None else None,
            )
        except OSError as exc:
            return SandboxOutcome(
                status=SandboxStatus.SANDBOX_ERROR, message=f"Start fehlgeschlagen: {exc}"
            )
        out = _Reader(proc.stdout, cap=limits.max_output_bytes, tail=None)
        err = _Reader(proc.stderr, cap=None, tail=_STDERR_TAIL)
        out.start()
        err.start()
        try:
            assert proc.stdin is not None  # noqa: S101 - durch stdin=PIPE garantiert
            proc.stdin.write(payload)
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass  # Kind ist früh gestorben; Klassifikation folgt unten
        # Immer die ganze Gruppe beenden: bei Fristablauf den Prozess selbst, sonst
        # Nachfahren, die noch laufen und an den Pipes hängen.
        exited, rusage = _supervise(proc, start + timeout_s, out.overflow_event)
        timed_out = not exited and not out.overflow
        duration = time.monotonic() - start
        for reader in (out, err):
            reader.join(timeout=2.0)
        for stream in (proc.stdout, proc.stderr):
            with contextlib.suppress(OSError):
                if stream is not None:
                    stream.close()

    cpu = (rusage.ru_utime + rusage.ru_stime) if rusage is not None else 0.0
    rss = int(rusage.ru_maxrss) if rusage is not None else 0
    if sys.platform == "darwin":  # pragma: no cover - macOS meldet Bytes
        rss //= 1024
    return _classify(
        messages=_parse_messages(bytes(out.data)),
        stderr=bytes(err.data).decode("utf-8", "replace"),
        returncode=proc.returncode,
        timed_out=timed_out,
        overflow=out.overflow,
        timeout_s=timeout_s,
        limits=limits,
        duration=duration,
        cpu=cpu,
        rss=rss,
        net_isolated=net_isolated,
        pid_ns=pid_ns,
    )


def _parse_messages(raw: bytes) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for line in raw.decode("utf-8", "replace").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            messages.append(item)
    return messages


def _classify(
    *,
    messages: list[dict[str, Any]],
    stderr: str,
    returncode: int | None,
    timed_out: bool,
    overflow: bool,
    timeout_s: float,
    limits: SandboxLimits,
    duration: float,
    cpu: float,
    rss: int,
    net_isolated: bool,
    pid_ns: bool,
) -> SandboxOutcome:
    applied: dict[str, str] = {}
    result: dict[str, Any] | None = None
    events = set()
    for message in messages:
        event = message.get("event")
        events.add(event)
        if event == "ready" and isinstance(message.get("applied"), dict):
            applied = {str(k): str(v) for k, v in message["applied"].items()}
        elif event in ("result", "sandbox_error"):
            result = message
    common: dict[str, Any] = {
        "duration_s": duration,
        "cpu_seconds": cpu,
        "max_rss_kb": rss,
        "returncode": returncode,
        "network_isolated": net_isolated,
        "pid_namespace": pid_ns,
        "applied_limits": applied,
        "stderr_tail": stderr[-_STDERR_TAIL:],
    }
    if overflow:
        return SandboxOutcome(
            status=SandboxStatus.OUTPUT_TOO_LARGE,
            message=f"Ausgabe überschreitet {limits.max_output_bytes} Bytes; Prozess beendet",
            **common,
        )
    if timed_out:
        return SandboxOutcome(
            status=SandboxStatus.TIMEOUT,
            message=f"Zeitlimit {timeout_s:g} s überschritten; Prozessgruppe mit SIGKILL beendet",
            **common,
        )
    if "cpu_limit" in events:
        return SandboxOutcome(
            status=SandboxStatus.CPU_LIMIT,
            message=f"CPU-Limit {limits.cpu_seconds} s erreicht (SIGXCPU)",
            **common,
        )
    if result is not None:
        if result.get("event") == "sandbox_error":
            return SandboxOutcome(
                status=SandboxStatus.SANDBOX_ERROR, message=str(result.get("message", "")), **common
            )
        if result.get("ok") is True:
            return SandboxOutcome(status=SandboxStatus.OK, value=result.get("value"), **common)
        error_type = str(result.get("error_type") or "Exception")
        status = (
            SandboxStatus.MEMORY_LIMIT
            if error_type == "MemoryError" and limits.memory_mb is not None
            else SandboxStatus.ERROR
        )
        return SandboxOutcome(
            status=status,
            error_type=error_type,
            message=str(result.get("message", "")),
            **common,
        )
    if cpu >= limits.cpu_seconds - 0.05:
        return SandboxOutcome(
            status=SandboxStatus.CPU_LIMIT,
            message=f"CPU-Limit {limits.cpu_seconds} s erreicht (Kernel-Abbruch)",
            **common,
        )
    if returncode is not None and returncode < 0:
        name = signal.Signals(-returncode).name if -returncode in signal.valid_signals() else ""
        return SandboxOutcome(
            status=SandboxStatus.KILLED,
            message=f"Prozess durch Signal {name or -returncode} beendet",
            **common,
        )
    return SandboxOutcome(
        status=SandboxStatus.CRASHED,
        message=f"Prozess endete ohne Ergebnis (rc={returncode})",
        **common,
    )
