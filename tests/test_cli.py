"""Kommandozeile."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from gradweight_orchestration import __version__
from gradweight_orchestration.cli import main


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_sandbox_info(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["sandbox-info"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["platform"] == sys.platform


def test_schema(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["schema", "sandbox_tools:beleg_parsen"]) == 0
    spec = json.loads(capsys.readouterr().out)
    assert spec["function"]["name"] == "beleg_parsen"
    assert main(["schema", "sandbox_tools"]) == 2
    assert main(["schema", "sandbox_tools:add"]) == 2


def test_module_entry_point() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "gradweight_orchestration", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert __version__ in completed.stdout


def test_audit_verify(tmp_path, capsys: pytest.CaptureFixture[str], monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from gradweight_orchestration import AuditLog

    path = tmp_path / "audit.jsonl"
    log = AuditLog(path, key=b"schluessel", fsync=False)
    log.append("event", {"a": 1})
    log.append("event", {"a": 2})
    monkeypatch.setenv("GW_AUDIT_KEY", "schluessel")
    assert main(["audit-verify", str(path), "--key-env", "GW_AUDIT_KEY"]) == 0
    assert "OK: Kette intakt (2 Einträge geprüft)" in capsys.readouterr().out
    assert main(["audit-verify", str(path)]) == 1  # ohne Schlüssel stimmt kein HMAC
    assert "FEHLER" in capsys.readouterr().out
    monkeypatch.delenv("GW_AUDIT_KEY")
    assert main(["audit-verify", str(path), "--key-env", "GW_AUDIT_KEY"]) == 2
    assert "GW_AUDIT_KEY ist nicht gesetzt" in capsys.readouterr().err


def test_approvals_commands(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    from gradweight_orchestration import ApprovalQueue, AuditLog

    db, log = str(tmp_path / "f.sqlite"), str(tmp_path / "a.jsonl")
    assert main(["approvals", "list", "--db", db]) == 0
    assert "Keine offenen Freigaben" in capsys.readouterr().out
    with ApprovalQueue(db, id_factory=iter(["a1", "a2"]).__next__) as queue:
        for _ in range(2):
            queue.submit(
                run_id="r1",
                agent="k",
                call_id="c",
                tool="buchen",
                arguments={"betrag": 12430.5},
                side_effect="money_moving",
                amount="12430.5",
                reason="über Limit",
            )
    assert main(["approvals", "list", "--db", db, "--run", "r1"]) == 0
    out = capsys.readouterr().out
    assert "a1  buchen  Lauf r1, Betrag 12430.5" in out
    assert "Grund: über Limit" in out
    assert main(["approvals", "approve", "a1", "--db", db, "--audit", log, "--by", "cfo"]) == 0
    assert "a1: approved durch cfo" in capsys.readouterr().out
    assert (
        main(["approvals", "reject", "a2", "--db", db, "--by", "cfo", "--reason", "doppelt"]) == 0
    )
    assert main(["approvals", "approve", "a2", "--db", db, "--by", "cfo"]) == 1
    assert "bereits entschieden" in capsys.readouterr().err
    assert AuditLog(log).verify().count == 1


def test_input_errors_exit_with_code_2_without_traceback(
    tmp_path,
    capsys: pytest.CaptureFixture[str],  # type: ignore[no-untyped-def]
) -> None:
    cases = [
        ["schema", "gibt_es_nicht:werkzeug"],
        ["schema", "sandbox_tools:gibt_es_nicht"],
        ["audit-verify", str(tmp_path / "fehlt.jsonl")],
        ["approvals", "list", "--db", str(tmp_path / "kein" / "ordner" / "f.sqlite")],
    ]
    for argv in cases:
        assert main(argv) == 2, argv
        err = capsys.readouterr().err
        assert err.startswith("Fehler:"), argv
        assert "Traceback" not in err
    datei = tmp_path / "datei"
    datei.write_text("x", encoding="utf-8")
    assert main(["demo", "--ausgabe", str(datei)]) == 2
    assert "Fehler:" in capsys.readouterr().err
    with pytest.raises(SystemExit) as info:  # argparse: Exit-Code 2 mit Meldung
        main(["approvals", "approve"])
    assert info.value.code == 2


def test_console_script_reports_errors_cleanly() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "gradweight_orchestration", "audit-verify", "/gibt/es/nicht"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "Traceback" not in completed.stderr
