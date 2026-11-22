"""Hash-verkettetes Audit-Log und Schwärzung."""

from __future__ import annotations

import datetime as dt
import json
import multiprocessing
import re
from decimal import Decimal
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gradweight_orchestration.audit import GENESIS, AuditLog, Redactor


def fill(log: AuditLog, count: int = 5) -> None:
    for i in range(count):
        log.append("tool_result", {"i": i, "betrag": Decimal("10.50")}, run_id="r1", agent="a")


def test_in_memory_chain() -> None:
    log = AuditLog()
    assert log.head == GENESIS
    fill(log)
    records = log.records()
    assert [r.seq for r in records] == [0, 1, 2, 3, 4]
    assert records[0].prev == GENESIS
    assert records[1].prev == records[0].hash
    assert records[0].data == {"i": 0, "betrag": 10.5}
    assert len(log) == 5
    result = log.verify()
    assert result.ok
    assert result.count == 5
    assert result.head == log.head
    assert log.path is None


def test_file_chain_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "audit" / "audit.jsonl"
    fill(AuditLog(path, fsync=False), 3)
    reopened = AuditLog(path)
    fill(reopened, 2)
    assert [r.seq for r in reopened.records()] == [0, 1, 2, 3, 4]
    assert AuditLog.verify_file(path).ok
    assert reopened.path == path


def _tamper(path: Path, index: int, mutate) -> None:  # type: ignore[no-untyped-def]
    lines = path.read_text(encoding="utf-8").splitlines()
    lines = mutate(lines, index)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("mutation", "fragment"),
    [
        (
            lambda lines, i: [*lines[:i], lines[i].replace('"i":2', '"i":99'), *lines[i + 1 :]],
            "Inhalt verändert",
        ),
        (lambda lines, i: lines[:i] + lines[i + 1 :], "Sequenz"),
        (lambda lines, i: [*lines[:i], lines[i + 1], lines[i], *lines[i + 2 :]], "Sequenz"),
        (lambda lines, i: [*lines[:i], "kein json", *lines[i + 1 :]], "nicht lesbar"),
        (lambda lines, i: [*lines[:i], json.dumps({"seq": i}), *lines[i + 1 :]], "nicht lesbar"),
    ],
)
def test_tampering_is_detected(tmp_path: Path, mutation, fragment: str) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "audit.jsonl"
    fill(AuditLog(path, fsync=False))
    _tamper(path, 2, mutation)
    result = AuditLog.verify_file(path)
    assert not result.ok
    assert result.error_seq == 2
    assert fragment in result.message


def test_rehashed_forgery_needs_key(tmp_path: Path) -> None:
    """Ohne Schlüssel lässt sich eine Fälschung neu verketten; mit HMAC-Schlüssel nicht."""
    plain = tmp_path / "plain.jsonl"
    keyed = tmp_path / "keyed.jsonl"
    fill(AuditLog(plain, fsync=False))
    fill(AuditLog(keyed, key=b"geheim", fsync=False))
    for path in (plain, keyed):
        forged = AuditLog(path.with_suffix(".forged"), fsync=False)  # Angreifer ohne Schlüssel
        for record in AuditLog(path).records():
            data = dict(record.data)
            data["i"] = 1000 + int(data["i"])  # type: ignore[call-overload]
            forged.append(record.kind, data, run_id=record.run_id, agent=record.agent)
        path.write_text(path.with_suffix(".forged").read_text(encoding="utf-8"), encoding="utf-8")
    assert AuditLog.verify_file(plain).ok  # ohne Schlüssel: nur Verankerung des Heads hilft
    assert not AuditLog.verify_file(keyed, key=b"geheim").ok
    assert AuditLog(keyed, key=b"geheim").verify().ok is False


def _append_many(path: str, worker: int) -> None:
    log = AuditLog(path, fsync=False)
    for i in range(25):
        log.append("event", {"worker": worker, "i": i})


def test_concurrent_processes_keep_chain_intact(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_append_many, args=(str(path), w)) for w in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(30)
    result = AuditLog.verify_file(path)
    assert result.ok, result.message
    assert result.count == 100


def test_clock_and_redaction_in_log() -> None:
    fixed = dt.datetime(2026, 10, 26, 12, 0, tzinfo=dt.UTC)
    log = AuditLog(redactor=Redactor(), clock=lambda: fixed)
    record = log.append("llm_response", {"text": "IBAN CH93 0076 2011 6238 5295 7", "token": "x"})
    assert record.ts.startswith("2026-10-26T12:00:00")
    assert record.data == {"text": "IBAN [IBAN]", "token": "[ENTFERNT]"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Mail an anna.muster@example.com bitte", "Mail an [EMAIL] bitte"),
        ("IBAN CH93 0076 2011 6238 5295 7.", "IBAN [IBAN]."),
        ("IBAN CH9300762011623852957", "IBAN [IBAN]"),
        ("AHV 756.1234.5678.97", "AHV [AHV]"),
        ("Tel. +41 44 668 18 00", "Tel. [TELEFON]"),
        ("Tel. 044 668 18 00", "Tel. [TELEFON]"),
        ("Karte 4111 1111 1111 1111", "Karte [KARTE]"),
        ("Rechnung RE-2026-0042 über CHF 1'081.00", "Rechnung RE-2026-0042 über CHF 1'081.00"),
        ("Bestellnummer 1234567890123", "Bestellnummer 1234567890123"),  # kein Luhn
    ],
)
def test_redactor_patterns(text: str, expected: str) -> None:
    assert Redactor().text(text) == expected


def test_redactor_hash_mode() -> None:
    with pytest.raises(ValueError, match="key"):
        Redactor(mode="hash")
    redactor = Redactor(mode="hash", key=b"k")
    first = redactor.text("anna@example.com")
    assert re.fullmatch(r"\[EMAIL:[0-9a-f]{10}\]", first)
    assert redactor.text("anna@example.com") == first
    assert redactor.text("bruno@example.com") != first
    assert redactor({"a": ["anna@example.com", 1], "Password": "x"}) == {
        "a": [first, 1],
        "Password": "[ENTFERNT]",
    }


@settings(max_examples=60)
@given(st.text(max_size=200))
def test_redaction_is_idempotent(text: str) -> None:
    redactor = Redactor()
    once = redactor.text(text)
    assert redactor.text(once) == once


@settings(max_examples=40)
@given(
    entries=st.lists(
        st.dictionaries(st.text(max_size=5), st.integers() | st.text(max_size=8), max_size=3),
        min_size=2,
        max_size=6,
    ),
    index=st.integers(min_value=0, max_value=5),
    value=st.integers(),
)
def test_any_content_change_is_detected(
    entries: list[dict[str, object]], index: int, value: int
) -> None:
    log = AuditLog()
    for entry in entries:
        log.append("event", entry)
    rows = [r.to_json() for r in log.records()]
    target = rows[index % len(rows)]
    data = dict(target["data"])  # type: ignore[call-overload]
    data["__manipuliert__"] = value
    target["data"] = data
    from gradweight_orchestration.audit import _verify_rows

    assert not _verify_rows(iter(rows), None).ok
