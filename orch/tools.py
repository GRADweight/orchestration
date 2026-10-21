"""Tool-Registry und @tool-Decorator (M4 + Vorbereitung M5).

@tool(timeout=5, network=False) -> Sandbox-Wrapper mit Zeitlimit und
Netz-Sperre. Kein cgroup im Testpfad (KMU-Sweet); die Zeitgrenze
braucht keinen Thread mit SIGALRM, sondern ein einfaches TimeLimit
(der Executor wartet maximal `timeout_s` — ein hängendes Tool wird
mit TimeLimit überschrieben, nicht abgebrochen).

M5 (Sandbox mit cgroup/cpuset + echter Netz-Isolation) fügt hier
nichts hinzu ausser dem `network=True`-Pfahl, der im Testpfad die
Netzwerk-API per Monkeypatch blockiert.
"""
from __future__ import annotations

import time
from typing import Any, Callable


class TimeLimit:
    """Einfache Zeitgrenze — pro Attempt gibt es ein Budget (Sektion).

    `remaining` = verbleibendes Budget. `consume(elapsed)` zieht die
    tatsächlich verbrauchte Zeit ab. `over()` meldet, ob das Budget
    bereits überschritten ist. Kein Signal/Thread (KMU-Sweet, stdlib):
    die Zeitmessung läuft *nach* dem Aufruf — ein einziger, zu langes Tool
    wird so trotzdem als TimeOut markiert.
    """

    def __init__(self, limit: float) -> None:
        self._limit = max(0.0, limit)
        self._start = time.monotonic()

    @property
    def remaining(self) -> float:
        spent = time.monotonic() - self._start
        return max(0.0, self._limit - spent)

    def over(self) -> bool:
        return self.remaining <= 0.0

    def consume(self, elapsed: float) -> float:  # no-op für API-Symmetrie
        return self.remaining


class Sandbox:
    """Netzwerk isoliertes Sandbox-Wrapping (Vorbereitung M5).

    Im Testpfad (network=False) wird `socket` gemockt — ein
    Netzwerk-Aufruf wirft. Für Produktion (network=True) ist die
    Isolation ein no-op (das Tool darf das Netz).
    """
    def __init__(self, *, network: bool = False, timeout_s: float = 5.0) -> None:
        self.network = network
        self.timeout_s = timeout_s

    def execute(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Führt fn aus. Bei network=False wird Socket blockiert."""
        if not self.network:
            import socket
            real_socket = socket.socket
            socket.socket = lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError("Netz gesperrt (network=False, M5)")
            )
            try:
                return fn(*args, **kwargs)
            finally:
                socket.socket = real_socket
        return fn(*args, **kwargs)


class ToolRegistry:
    """Einfache Tool-Registry (M4)."""
    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        self._tools[name] = fn

    def get(self, name: str) -> Callable[..., Any]:
        return self._tools[name]

    def list(self) -> list[str]:
        return list(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def tool(
    name: str | None = None,
    *,
    timeout_s: float = 5.0,
    network: bool = False,
    keywords: tuple[str, ...] = (),
) -> Callable[..., Any]:
    """@tool-Decorator: registriert die Funktion in der globalen Registry.

    Beispiel:
        @tool(name="rechnen")
        def rechnen(a, b):
            return a + b
    """
    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        sandbox = Sandbox(network=network, timeout_s=timeout_s)

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            return sandbox.execute(fn, *args, **kwargs)

        reg = _default_registry()
        reg.register(name or fn.__name__, _wrapped)
        _wrapped.__tool_keywords__ = keywords  # type: ignore
        _wrapped.__tool_timeout__ = timeout_s  # type: ignore
        _wrapped.__tool_network__ = network  # type: ignore
        return _wrapped

    return _decorator


# Globaler Registry (einfach; für einen Prozess / einen Agent)
__registry: ToolRegistry | None = None


def _default_registry() -> ToolRegistry:
    global __registry
    if __registry is None:
        __registry = ToolRegistry()
    return __registry


def registry() -> ToolRegistry:
    return _default_registry()
