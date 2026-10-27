"""Strukturierter Trace: Spans pro Lauf, Schritt, Modellaufruf und Werkzeugaufruf.

Ohne Abhängigkeiten; Export als JSON. Attributnamen folgen, wo passend, den
OpenTelemetry-Konventionen für generative KI (``gen_ai.*``, Stand: Entwicklung,
können sich ändern); eigene Attribute tragen das Präfix ``gw.``. Für einen
OpenTelemetry-Export siehe :mod:`gradweight_orchestration.otel`.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import secrets
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

from ._json import JSONObject, JSONValue, canonical_json

__all__ = ["Span", "SpanEvent", "Tracer"]

AttributeValue: TypeAlias = str | int | float | bool


def _attribute(value: Any) -> AttributeValue:
    if isinstance(value, (str, bool, int, float)):
        return value
    try:
        return canonical_json(value)
    except (TypeError, ValueError):
        return repr(value)


@dataclass
class SpanEvent:
    name: str
    time_ns: int
    attributes: dict[str, AttributeValue] = field(default_factory=dict)

    def to_json(self) -> JSONObject:
        return {"name": self.name, "time_ns": self.time_ns, "attributes": dict(self.attributes)}


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    start_ns: int
    end_ns: int | None = None
    attributes: dict[str, AttributeValue] = field(default_factory=dict)
    status: str = "unset"
    status_message: str = ""
    events: list[SpanEvent] = field(default_factory=list)
    _clock: Callable[[], int] = field(default=time.time_ns, repr=False)

    def set(self, key: str, value: Any) -> None:
        if value is not None:
            self.attributes[key] = _attribute(value)

    def update(self, attributes: Mapping[str, Any]) -> None:
        for key, value in attributes.items():
            self.set(key, value)

    def add_event(self, name: str, **attributes: Any) -> None:
        self.events.append(
            SpanEvent(
                name,
                self._clock(),
                {k: _attribute(v) for k, v in attributes.items() if v is not None},
            )
        )

    def fail(self, message: str) -> None:
        self.status = "error"
        self.status_message = message

    @property
    def duration_s(self) -> float | None:
        return None if self.end_ns is None else (self.end_ns - self.start_ns) / 1e9

    def to_json(self) -> JSONObject:
        attributes: dict[str, JSONValue] = dict(self.attributes)
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_s": self.duration_s,
            "status": self.status,
            "status_message": self.status_message,
            "attributes": attributes,
            "events": [e.to_json() for e in self.events],
        }


class Tracer:
    """Sammelt abgeschlossene Spans; Eltern-Kind-Beziehung über ``contextvars``."""

    def __init__(
        self,
        *,
        clock_ns: Callable[[], int] = time.time_ns,
        id_factory: Callable[[int], str] | None = None,
        exporters: Sequence[Callable[[Span], None]] = (),
    ) -> None:
        self._clock = clock_ns
        self._new_id = id_factory or secrets.token_hex
        self._exporters = list(exporters)
        self._current: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
            f"gw_span_{id(self)}", default=None
        )
        self._lock = threading.Lock()
        self._spans: list[Span] = []
        self.export_errors = 0

    def current(self) -> Span | None:
        return self._current.get()

    @contextlib.contextmanager
    def span(self, name: str, attributes: Mapping[str, Any] | None = None) -> Iterator[Span]:
        parent = self._current.get()
        span = Span(
            name=name,
            trace_id=parent.trace_id if parent else self._new_id(16),
            span_id=self._new_id(8),
            parent_id=parent.span_id if parent else None,
            start_ns=self._clock(),
            _clock=self._clock,
        )
        span.update(attributes or {})
        token = self._current.set(span)
        try:
            yield span
        except BaseException as exc:
            span.fail(f"{type(exc).__name__}: {exc}")
            span.add_event("exception", type=type(exc).__name__, message=str(exc))
            raise
        else:
            if span.status == "unset":
                span.status = "ok"
        finally:
            self._current.reset(token)
            span.end_ns = self._clock()
            with self._lock:
                self._spans.append(span)
            for exporter in self._exporters:
                try:
                    exporter(span)
                except Exception:  # noqa: BLE001 - Export darf den Lauf nicht stoppen
                    self.export_errors += 1

    @property
    def spans(self) -> list[Span]:
        with self._lock:
            return list(self._spans)

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()

    def to_json(self) -> JSONObject:
        spans = sorted(self.spans, key=lambda s: s.start_ns)
        return {"spans": [s.to_json() for s in spans]}

    def export_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(
            json.dumps(self.to_json(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target
