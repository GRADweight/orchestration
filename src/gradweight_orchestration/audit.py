"""Hash-verkettetes Audit-Log (JSON Lines) und PII-bewusste Schwärzung.

Jeder Eintrag enthält den Hash seines Vorgängers; :meth:`AuditLog.verify` deckt
veränderte, gelöschte, eingefügte oder umsortierte Einträge auf. Mit ``key``
wird HMAC-SHA256 statt SHA-256 verwendet – dann kann nur neu verketten, wer den
Schlüssel kennt.

Grenzen (ehrlich): Die Kette ist *manipulationserkennend*, nicht
*manipulationssicher*. Wer Schreibzugriff hat, kann ohne Schlüssel die ganze
Kette neu berechnen, und das Abschneiden der letzten Einträge fällt nur auf,
wenn der letzte Hash (:attr:`AuditLog.head`) ausserhalb verankert wird
(Bericht, Ticket, WORM-Speicher).

Die Schwärzung arbeitet mit regulären Ausdrücken für E-Mail, IBAN,
AHV-Nummer, Schweizer Telefonnummern und Kartennummern (mit Luhn-Prüfung). Das
ist eine Hilfe zur Datenminimierung, keine Garantie: Namen oder Adressen
erkennt sie nicht.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import json
import os
import re
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Literal

from ._json import JSONObject, JSONValue, canonical_json, to_jsonable

try:  # POSIX: Dateisperre für Anhänge aus mehreren Prozessen
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

__all__ = ["GENESIS", "AuditLog", "AuditRecord", "Redactor", "VerifyResult"]

GENESIS = "0" * 64


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.UTC)


def _luhn(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30}\b")),
    ("AHV", re.compile(r"\b756[.\s]?\d{4}[.\s]?\d{4}[.\s]?\d{2}\b")),
    ("TELEFON", re.compile(r"(?:\+41|0041|\b0)\s?(?:\(0\)\s?)?\d{2}\s?\d{3}\s?\d{2}\s?\d{2}\b")),
    ("KARTE", re.compile(r"\b\d(?:[ -]?\d){12,18}\b")),
)


class Redactor:
    """Schwärzt Personendaten in Texten und JSON-Werten.

    ``mode="mask"`` ersetzt Treffer durch ``[IBAN]`` usw. ``mode="hash"``
    ersetzt durch ein HMAC-Pseudonym (``[IBAN:3f9a…]``), damit gleiche Werte
    korrelierbar bleiben. Dafür ist ein geheimer ``key`` Pflicht: ein Hash ohne
    Schlüssel liesse sich bei kleinen Wertebereichen (AHV-Nummern) durchprobieren.
    """

    SENSITIVE_KEYS = frozenset({"password", "passwort", "api_key", "token", "secret", "geheimnis"})

    def __init__(
        self,
        *,
        mode: Literal["mask", "hash"] = "mask",
        key: bytes | None = None,
        patterns: Sequence[tuple[str, re.Pattern[str]]] | None = None,
        sensitive_keys: frozenset[str] | None = None,
    ) -> None:
        if mode == "hash" and not key:
            raise ValueError("mode='hash' braucht einen geheimen key (Pseudonymisierung)")
        self._mode = mode
        self._key = key or b""
        self._patterns = tuple(patterns) if patterns is not None else _PATTERNS
        self._sensitive = sensitive_keys if sensitive_keys is not None else self.SENSITIVE_KEYS

    def _replacement(self, label: str, value: str) -> str:
        if self._mode == "mask":
            return f"[{label}]"
        digest = hmac.new(self._key, value.encode("utf-8"), hashlib.sha256).hexdigest()[:10]
        return f"[{label}:{digest}]"

    def text(self, value: str) -> str:
        for label, pattern in self._patterns:

            def substitute(match: re.Match[str], label: str = label) -> str:
                found = match.group(0)
                if label == "KARTE" and not _luhn(re.sub(r"\D", "", found)):
                    return found
                return self._replacement(label, found)

            value = pattern.sub(substitute, value)
        return value

    def __call__(self, value: JSONValue) -> JSONValue:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self(item) for item in value]
        if isinstance(value, dict):
            return {
                key: "[ENTFERNT]" if key.lower() in self._sensitive else self(item)
                for key, item in value.items()
            }
        return value


@dataclass(frozen=True)
class AuditRecord:
    seq: int
    ts: str
    run_id: str
    agent: str
    kind: str
    data: JSONObject
    prev: str
    hash: str

    def body(self) -> JSONObject:
        return {
            "seq": self.seq,
            "ts": self.ts,
            "run_id": self.run_id,
            "agent": self.agent,
            "kind": self.kind,
            "data": self.data,
            "prev": self.prev,
        }

    def to_json(self) -> JSONObject:
        out = self.body()
        out["hash"] = self.hash
        return out

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> AuditRecord:
        data = raw["data"]
        if not isinstance(data, dict):
            raise TypeError("data ist kein Objekt")
        return cls(
            seq=int(raw["seq"]),
            ts=str(raw["ts"]),
            run_id=str(raw["run_id"]),
            agent=str(raw["agent"]),
            kind=str(raw["kind"]),
            data=data,
            prev=str(raw["prev"]),
            hash=str(raw["hash"]),
        )


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    count: int
    head: str
    error_seq: int | None = None
    message: str = "Kette intakt"


def _digest(body: JSONObject, key: bytes | None) -> str:
    message = canonical_json(body).encode("utf-8")
    if key:
        return hmac.new(key, message, hashlib.sha256).hexdigest()
    return hashlib.sha256(message).hexdigest()


def _last_line(handle: IO[bytes]) -> bytes | None:
    """Letzte nicht-leere Zeile einer Datei, rückwärts gelesen."""
    handle.seek(0, os.SEEK_END)
    position = handle.tell()
    buffer = b""
    while position > 0:
        step = min(65536, position)
        position -= step
        handle.seek(position)
        buffer = handle.read(step) + buffer
        stripped = buffer.rstrip(b"\n")
        if b"\n" in stripped:
            return stripped.rsplit(b"\n", 1)[1]
    stripped = buffer.rstrip(b"\n")
    return stripped or None


class AuditLog:
    """Append-only, hash-verkettet. ``path=None`` hält die Kette im Speicher."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        key: bytes | None = None,
        redactor: Redactor | None = None,
        clock: Callable[[], _dt.datetime] = _utcnow,
        fsync: bool = True,
    ) -> None:
        self._path = Path(path) if path is not None else None
        self._key = key
        self._redactor = redactor
        self._clock = clock
        self._fsync = fsync
        self._lock = threading.Lock()
        self._memory: list[AuditRecord] = []
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.touch(exist_ok=True)

    @property
    def path(self) -> Path | None:
        return self._path

    def append(
        self, kind: str, data: Mapping[str, Any], *, run_id: str = "", agent: str = ""
    ) -> AuditRecord:
        """Hängt einen Eintrag an (Daten werden JSON-gewandelt und ggf. geschwärzt)."""
        payload = to_jsonable(dict(data))
        if self._redactor is not None:
            payload = self._redactor(payload)
        assert isinstance(payload, dict)  # noqa: S101 - Objekt bleibt Objekt
        timestamp = self._clock().isoformat(timespec="microseconds")

        def build(seq: int, prev: str) -> AuditRecord:
            body: JSONObject = {
                "seq": seq,
                "ts": timestamp,
                "run_id": run_id,
                "agent": agent,
                "kind": kind,
                "data": payload,
                "prev": prev,
            }
            return AuditRecord(
                seq, timestamp, run_id, agent, kind, payload, prev, _digest(body, self._key)
            )

        with self._lock:
            if self._path is None:
                last = self._memory[-1] if self._memory else None
                record = build(last.seq + 1 if last else 0, last.hash if last else GENESIS)
                self._memory.append(record)
                return record
            return self._append_file(build)

    def _append_file(self, build: Callable[[int, str], AuditRecord]) -> AuditRecord:
        assert self._path is not None  # noqa: S101 - nur dateibasiert aufgerufen
        with self._path.open("a+b") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                raw = _last_line(handle)
                if raw is None:
                    record = build(0, GENESIS)
                else:
                    last = AuditRecord.from_json(json.loads(raw))
                    record = build(last.seq + 1, last.hash)
                handle.seek(0, os.SEEK_END)
                handle.write((canonical_json(record.to_json()) + "\n").encode("utf-8"))
                handle.flush()
                if self._fsync:
                    os.fsync(handle.fileno())
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return record

    def records(self) -> list[AuditRecord]:
        if self._path is None:
            with self._lock:
                return list(self._memory)
        return [AuditRecord.from_json(json.loads(line)) for line in self._lines(self._path)]

    def __len__(self) -> int:
        return len(self.records())

    @property
    def head(self) -> str:
        """Hash des letzten Eintrags – zum Verankern ausserhalb des Logs."""
        records = self.records()
        return records[-1].hash if records else GENESIS

    def verify(self) -> VerifyResult:
        if self._path is None:
            with self._lock:
                rows: list[Any] = [r.to_json() for r in self._memory]
            return _verify_rows(iter(rows), self._key)
        return self.verify_file(self._path, key=self._key)

    @classmethod
    def verify_file(cls, path: str | Path, *, key: bytes | None = None) -> VerifyResult:
        def rows() -> Iterator[Any]:
            for line in cls._lines(Path(path)):
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    yield None

        return _verify_rows(rows(), key)

    @staticmethod
    def _lines(path: Path) -> Iterator[str]:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield line


def _verify_rows(rows: Iterator[Any], key: bytes | None) -> VerifyResult:
    previous = GENESIS
    count = 0
    for expected_seq, raw in enumerate(rows):
        try:
            record = AuditRecord.from_json(raw)
        except (KeyError, TypeError, ValueError):
            return VerifyResult(
                False, count, previous, expected_seq, f"Eintrag {expected_seq}: nicht lesbar"
            )
        if record.seq != expected_seq:
            return VerifyResult(
                False,
                count,
                previous,
                expected_seq,
                f"Eintrag {expected_seq}: Sequenz {record.seq} erwartet "
                f"{expected_seq} (gelöscht, eingefügt oder umsortiert)",
            )
        if record.prev != previous:
            return VerifyResult(
                False,
                count,
                previous,
                expected_seq,
                f"Eintrag {expected_seq}: Verkettung gebrochen",
            )
        if not hmac.compare_digest(record.hash, _digest(record.body(), key)):
            return VerifyResult(
                False,
                count,
                previous,
                expected_seq,
                f"Eintrag {expected_seq}: Inhalt verändert (Hash stimmt nicht)",
            )
        previous = record.hash
        count += 1
    return VerifyResult(True, count, previous)
