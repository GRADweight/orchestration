"""Werkzeugfunktionen für Sandbox-Tests.

Eigenes Modul ohne pytest-Import: der Kindprozess importiert es nach dem Setzen
der Limits; es muss klein und auf Modulebene definiert sein.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from dataclasses import dataclass
from decimal import Decimal

from gradweight_orchestration import Isolation, NetworkMode, SandboxLimits, SideEffect, tool
from gradweight_orchestration.errors import TransientToolError


def add(a: int, b: int) -> int:
    return a + b


def sleep_then_write(seconds: float, path: str) -> str:
    """Schläft und schreibt danach eine Datei – existiert sie, lief der Prozess weiter."""
    time.sleep(seconds)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("lebt")
    return "fertig"


def spawn_grandchild_then_hang(path: str, new_session: bool) -> str:
    """Startet einen Enkelprozess, der später eine Datei schreibt, und hängt dann."""
    subprocess.Popen(
        ["/bin/sh", "-c", f"sleep 2; echo lebt > '{path}'"],
        start_new_session=new_session,
    )
    time.sleep(60)
    return "nie erreicht"


def burn_cpu() -> int:
    total = 0
    while True:
        total += 1


def allocate(mb: int) -> int:
    block = bytearray(mb * 1024 * 1024)
    return len(block)


def write_file(mb: int) -> int:
    with open("gross.bin", "wb") as fh:
        fh.write(b"x" * (mb * 1024 * 1024))
        fh.flush()
    return mb


def spawn_process() -> str:
    return subprocess.run(["true"], check=True).args[0]


def environment() -> dict[str, str]:
    return dict(os.environ)


def working_directory(marker: str) -> str:
    with open(marker, "w", encoding="utf-8") as fh:
        fh.write("x")
    return os.getcwd()


def write_outside(path: str) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("ausserhalb")
    return path


def connect(port: int) -> str:
    with socket.create_connection(("127.0.0.1", port), timeout=2):
        return "verbunden"


def noisy(lines: int) -> str:
    for i in range(lines):
        print(f"Zeile {i}")
    return "ok"


def big_output(size: int) -> str:
    return "x" * size


def fail() -> str:
    raise ValueError("absichtlicher Fehler")


def whoami() -> int:
    return os.getuid()


def read_file(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def flaky(counter: str) -> str:
    """Schlägt beim ersten Aufruf vorübergehend fehl."""
    if not os.path.exists(counter):
        with open(counter, "w", encoding="utf-8") as fh:
            fh.write("1")
        raise TransientToolError("bitte erneut versuchen")
    return "zweiter Versuch"


# Plattformneutral: isolieren, wo möglich (CI auf macOS hat keine Namespaces).
LIMITS = SandboxLimits(network=NetworkMode.DENY_IF_POSSIBLE)


@dataclass(frozen=True)
class Beleg:
    nummer: str
    betrag: Decimal


@tool(side_effect=SideEffect.READ_ONLY, isolation=Isolation.PROCESS, timeout_s=10, sandbox=LIMITS)
def beleg_parsen(text: str) -> Beleg:
    """Zerlegt 'NUMMER;BETRAG' in einen Beleg (läuft isoliert)."""
    nummer, betrag = text.split(";")
    return Beleg(nummer=nummer, betrag=Decimal(betrag))


@tool(
    side_effect=SideEffect.READ_ONLY,
    isolation=Isolation.PROCESS,
    timeout_s=1,
    sandbox=LIMITS,
)
def isolierter_schlaf(seconds: float) -> str:
    """Schläft (zum Testen des Zeitlimits im Prozessmodus)."""
    time.sleep(seconds)
    return "wach"


@tool(
    side_effect=SideEffect.READ_ONLY,
    isolation=Isolation.PROCESS,
    retries=1,
    sandbox=LIMITS,
)
def isolierter_wackler(counter: str) -> str:
    """Schlägt beim ersten Aufruf vorübergehend fehl (Prozessmodus)."""
    return flaky(counter)


@tool(side_effect=SideEffect.READ_ONLY, isolation=Isolation.PROCESS, sandbox=LIMITS)
def falscher_typ() -> int:
    """Liefert absichtlich den falschen Typ."""
    return "keine Zahl"  # type: ignore[return-value]
