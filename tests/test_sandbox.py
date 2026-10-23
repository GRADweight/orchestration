"""10 Tests für M5 — Tool-Sandbox (Subprocess-Isolation, stdlib).

S. GAP_ANALYSE.md, M5:
    10 Tests: Timeout, Netzwerk-Block, CPU-Limit, Sandbox-Escape
              (subprocess vs. os.system), 5 Beispiel-Tools (alle laufen).

KMU-Sweet: kein cgroup, kein Namespace. CPU-Limit via RLIMIT_CPU (POSIX),
Netz-Block + Spawn-Block via Socket-/subprocess-Hook im Kind-Prozess.
"""
from __future__ import annotations

import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_VENDOR / "vendor"))
sys.path.insert(0, str(_VENDOR))

from orch.sandbox import (  # noqa: E402
    Sandbox,
    ToolCpuLimit,
    ToolNetworkBlock,
)
from orch.example_tools import (  # noqa: E402
    bexio_buchung,
    datei_leser,
    klassifiziere_email,
    ocr_rechnung,
    web_search_proxy,
)


# ── CPU- und Hang-Tools (reguläre def → getsource im Kind-Prozess) ────────────


def _cpu_burn() -> int:
    """CPU-hungry: 10**9 Schleifendurchgänge (übersteigt 1s CPU-Limit)."""
    s = 0
    for i in range(10 ** 9):
        s += i
    return s


def _hang() -> str:
    """Hängt sich auf: 30s Wall-Clock (übersteigt 2s Wall-Timeout)."""
    import time

    time.sleep(30)
    return "ended"


def _netz_versuch() -> str:
    """Versucht ein echtes Socket → wird im Sandbox blockiert."""
    import socket

    s = socket.create_connection(("example.com", 80), timeout=1)
    s.close()
    return "verbunden"


def _escape_os_system() -> str:
    """Versucht os.system → wird im Sandbox blockiert."""
    os_system = __import__("os").system
    return os_system("echo hi")  # type: ignore[union-attr]


def _escape_subprocess() -> str:
    """Versucht subprocess.run → wird im Sandbox blockiert."""
    import subprocess

    return subprocess.run(["echo", "hi"], capture_output=True, text=True).stdout


# ── 1) Timeout ─────────────────────────────────────────────────────────────────


def test_timeout_killed_process() -> None:
    """Ein zu langes Tool wird per Wall-Timeout getötet (timed_out=True)."""
    sb = Sandbox()
    rep = sb.run_function(_hang, name="hang", timeout_s=2, cpu_seconds=30)
    assert rep.timed_out is True
    assert rep.ok is False


# ── 2) Netzwerk-Block ──────────────────────────────────────────────────────────


def test_network_block_socket() -> None:
    """Ein Socket-Call im Tool wird blockiert (network=False)."""
    sb = Sandbox()
    rep = sb.run_function(
        web_search_proxy, name="search", args=("x", 3, None), network=False
    )
    assert rep.network_blocked is True
    assert rep.ok is False


# ── 3) CPU-Limit ────────────────────────────────────────────────────────────────


def test_cpu_limit_kills_process() -> None:
    """CPU-Limit: ein CPU-hungry Loop stirbt *kurz* (≈ cpu_seconds),
    weit vor dem grossen Wall-Budget — die beobachtbare Invariante.

    Das genaue Signal ist kernel-abhängig (Linux/Runner liefert oft
    SIGKILL statt SIGXCPU für eine enge Loop). Robust ist aber: ein
    CPU-bound-Tool wird durch das CPU-Limit gestoppt, nicht durch das
    Wall-Budget — das beweist `duration_s << wall_budget`.
    """
    sb = Sandbox()
    rep = sb.run_function(_cpu_burn, name="burn", timeout_s=600, cpu_seconds=1)
    assert rep.ok is False
    # Entweder klassifiziert als cpu_limited ...
    assert rep.cpu_limited or "SIGXCPU" in (rep.error or "")
    # ... oder (fallback) als Signal getötet, und auf jeden Fall KURZ
    # (weit vor dem 600s-Wall-Budget) — beweist, CPU-Limit (nicht Wall) hat
    # den Loop gestoppt.
    assert rep.duration_s < 30.0, f"Loop lief {rep.duration_s:.1f}s — sollte CPU-Limit ~1s sein"



# ── 4) Sandbox-Escape (subprocess vs. os.system) ────────────────────────────


def test_escape_os_system_blocked() -> None:
    """os.system ist im Kind gesperrt (kein Host-Zugriff)."""
    sb = Sandbox()
    rep = sb.run_function(_escape_os_system, name="esc-os", network=False)
    assert rep.ok is False
    assert rep.escaped is True or "blocked" in rep.error.lower()


def test_escape_subprocess_blocked() -> None:
    """subprocess.run ist im Kind gesperrt (kein Host-Zugriff)."""
    sb = Sandbox()
    rep = sb.run_function(_escape_subprocess, name="esc-sp", network=False)
    assert rep.ok is False
    assert rep.escaped is True or "blocked" in rep.error.lower()


# ── 5) 5 Beispiel-Tools laufen (offline) ────────────────────────────────────────


def test_tool_ocr_rechnung() -> None:
    """OCR-Rechnung: Betrag + Frist aus Text (offline-Regex)."""
    sb = Sandbox()
    rep = sb.run_function(
        ocr_rechnung, name="ocr",
        args=("Rechnung 123.45 CHF, fällig 31.12.2026 bei Kunde AG",),
    )
    assert rep.ok is True
    assert abs(rep.value["betrag_chf"] - 123.45) < 0.001
    assert rep.value["gültig"] is True


def test_tool_klassifiziere_email() -> None:
    """E-Mail-Triage: 'Rechnung' → Klasse 'verwaltung'."""
    sb = Sandbox()
    rep = sb.run_function(
        klassifiziere_email, name="mail",
        args=("Rechnung offen", "Bitte überweisen Sie den offenen Betrag."),
    )
    assert rep.ok is True
    assert rep.value["klasse"] == "verwaltung"


def test_tool_bexio_buchung() -> None:
    """Bexio-Buchung (offline-Stub): Status + deterministische ID."""
    sb = Sandbox()
    rep = sb.run_function(
        bexio_buchung, name="bex", args=(1081.0, "K100")
    )
    assert rep.ok is True
    assert rep.value["status"] == "buchung_erstellt"
    assert rep.value["buchung_id"].startswith("BX-")
    assert abs(rep.value["betrag_chf"] - 1081.0) < 0.001


def test_tool_web_search_proxy_fixture() -> None:
    """Web-Suche mit Fixture (kein Netz → offline, ok)."""
    sb = Sandbox()
    rep = sb.run_function(
        web_search_proxy, name="search",
        args=("klimaplan", 3, "offline-fixture"), network=True,
    )
    assert rep.ok is True
    assert len(rep.value["results"]) == 3
    assert rep.value["offline"] is True


def test_tool_datei_leser(tmp_path: Path, monkeypatch) -> None:
    """Datei-Leser (../-Escape blockiert):
    - relativer Pfad (chdir in tmp) → ok
    - "../etc/passwd" → Fehler (Escape blockiert)"""
    import os

    sb = Sandbox()
    tfile = tmp_path / "t.txt"
    tfile.write_text("zeile1\nzeile2\nzeile3\n", encoding="utf-8")
    # macOS tmp_path ist absolut (/private/var/...) → chdir + relativer Pfad
    monkeypatch.chdir(tmp_path)
    rep = sb.run_function(datei_leser, name="df", args=("t.txt",))
    assert rep.ok is True
    assert rep.value["ok"] is True
    assert rep.value["zeilen"] == 3

    rep2 = sb.run_function(datei_leser, name="df", args=("../etc/passwd",))
    assert rep2.ok is True
    assert rep2.value["ok"] is False
    assert "escape" in rep2.value["fehler"].lower() or "blockiert" in rep2.value["fehler"].lower()


# ── Smoke-Export-Check (Sandbox aus `orch` importierbar) ──────────────────────


def test_sandbox_exported_from_orch() -> None:
    """Sandbox ist aus `orch` (nicht nur `orch.sandbox`) importierbar (M7-Export)."""
    from orch import Sandbox as SB

    assert SB is not None
    rep = SB().run_function(
        lambda x: x + 1, name="add", args=(41,)  # type: ignore[arg-type]
    )
    # lambda hat keinen getsource → SandboxError; wir testet den Export nicht
    # per lambda, sondern per Name
    import orch.example_tools as et
    rep2 = SB().run_function(et.bexio_buchung, name="b", args=(1.0, "K"))
    assert rep2.ok is True


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
