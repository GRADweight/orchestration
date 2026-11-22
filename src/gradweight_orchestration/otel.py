"""Optionaler Export der Spans an OpenTelemetry (Extra ``otel``).

Die Runtime hängt nicht von OpenTelemetry ab. Wer es nutzt, überträgt die
fertigen Spans eines Laufs nachträglich, mit Originalzeiten und
Eltern-Kind-Beziehungen::

    from gradweight_orchestration.otel import OpenTelemetryExporter
    OpenTelemetryExporter().export(tracer.spans)

Die OpenTelemetry-Trace- und Span-IDs werden neu vergeben; die eigenen IDs
stehen als ``gw.trace_id`` / ``gw.span_id`` in den Attributen.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from . import __version__
from .tracing import Span

__all__ = ["OpenTelemetryExporter"]


class OpenTelemetryExporter:
    def __init__(
        self, tracer_provider: Any = None, *, instrumentation_name: str = "gradweight_orchestration"
    ) -> None:
        try:
            from opentelemetry import trace
            from opentelemetry.trace import Status, StatusCode
        except ImportError as exc:  # pragma: no cover - nur ohne Extra
            raise ImportError(
                "OpenTelemetryExporter braucht: pip install 'gradweight-orchestration[otel]'"
            ) from exc
        self._trace = trace
        self._status = Status
        self._code = StatusCode
        provider = tracer_provider or trace.get_tracer_provider()
        self._tracer = provider.get_tracer(instrumentation_name, __version__)

    def export(self, spans: Sequence[Span]) -> int:
        """Überträgt abgeschlossene Spans; liefert die Anzahl übertragener Spans."""
        created: dict[str, Any] = {}
        count = 0
        for span in sorted(spans, key=lambda s: s.start_ns):
            if span.end_ns is None:
                continue
            parent = created.get(span.parent_id) if span.parent_id else None
            context = self._trace.set_span_in_context(parent) if parent is not None else None
            attributes = dict(span.attributes)
            attributes["gw.trace_id"] = span.trace_id
            attributes["gw.span_id"] = span.span_id
            otel_span = self._tracer.start_span(
                span.name, context=context, start_time=span.start_ns, attributes=attributes
            )
            for event in span.events:
                otel_span.add_event(
                    event.name, attributes=dict(event.attributes), timestamp=event.time_ns
                )
            if span.status == "error":
                otel_span.set_status(self._status(self._code.ERROR, span.status_message))
            elif span.status == "ok":
                otel_span.set_status(self._status(self._code.OK))
            otel_span.end(end_time=span.end_ns)
            created[span.span_id] = otel_span
            count += 1
        return count
