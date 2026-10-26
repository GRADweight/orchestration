"""Tracing und optionaler OpenTelemetry-Export."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from gradweight_orchestration.tracing import Span, Tracer


def deterministic() -> Tracer:
    counter = iter(range(1, 10_000))
    clock = iter(range(1_000, 10**9, 1_000))
    return Tracer(clock_ns=lambda: next(clock), id_factory=lambda n: f"{next(counter):0{2 * n}x}")


def test_nested_spans_and_export(tmp_path: Path) -> None:
    tracer = deterministic()
    with tracer.span("agent.run", {"gw.run_id": "r1"}) as root:
        assert tracer.current() is root
        with tracer.span("llm.call") as child:
            child.set("gen_ai.usage.input_tokens", 12)
            child.set("ignoriert", None)
            child.set("liste", [1, 2])
            child.add_event("repair", attempt=1, leer=None)
        root.update({"gw.status": "completed"})
    assert tracer.current() is None
    spans = {s.name: s for s in tracer.spans}
    assert spans["llm.call"].parent_id == spans["agent.run"].span_id
    assert spans["llm.call"].trace_id == spans["agent.run"].trace_id
    assert spans["llm.call"].attributes == {"gen_ai.usage.input_tokens": 12, "liste": "[1,2]"}
    assert spans["llm.call"].events[0].attributes == {"attempt": 1}
    assert spans["agent.run"].status == "ok"
    assert spans["agent.run"].duration_s is not None
    data = tracer.to_json()
    assert [s["name"] for s in data["spans"]] == ["agent.run", "llm.call"]  # type: ignore[index, union-attr]
    path = tracer.export_json(tmp_path / "trace.json")
    assert json.loads(path.read_text(encoding="utf-8")) == data
    tracer.clear()
    assert tracer.spans == []


def test_exception_marks_span_and_exporter_errors_are_contained() -> None:
    seen: list[Span] = []

    def broken(_span: Span) -> None:
        raise RuntimeError("Exporter kaputt")

    tracer = Tracer(exporters=[seen.append, broken])
    with pytest.raises(ValueError, match="boom"), tracer.span("tool.call"):
        raise ValueError("boom")
    span = tracer.spans[0]
    assert span.status == "error"
    assert span.status_message == "ValueError: boom"
    assert span.events[0].name == "exception"
    assert seen == [span]
    assert tracer.export_errors == 1


def test_explicit_failure_and_unserialisable_attribute() -> None:
    tracer = Tracer()
    with tracer.span("x") as span:
        span.fail("abgelehnt")
        span.set("obj", object())
    assert tracer.spans[0].status == "error"
    assert str(tracer.spans[0].attributes["obj"]).startswith("<object")


def test_threads_have_separate_context() -> None:
    tracer = Tracer()
    parents: list[str | None] = []

    def worker() -> None:
        with tracer.span("thread") as span:
            parents.append(span.parent_id)

    with tracer.span("main"):
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
    assert parents == [None]


def test_opentelemetry_export() -> None:
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import StatusCode

    from gradweight_orchestration.otel import OpenTelemetryExporter

    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    tracer = deterministic()
    with tracer.span("invoke_agent kreditoren", {"gen_ai.agent.name": "kreditoren"}):
        with tracer.span("execute_tool rechnung_lesen") as tool_span:
            tool_span.add_event("policy", decision="allow")
        with pytest.raises(RuntimeError), tracer.span("chat"):
            raise RuntimeError("Modell weg")
    unfinished = Span("offen", "t", "s", None, 0)
    count = OpenTelemetryExporter(provider).export([*tracer.spans, unfinished])
    assert count == 3
    exported = {s.name: s for s in memory.get_finished_spans()}
    root = exported["invoke_agent kreditoren"]
    child = exported["execute_tool rechnung_lesen"]
    assert child.parent is not None
    assert child.parent.span_id == root.context.span_id
    assert child.events[0].name == "policy"
    assert root.attributes["gen_ai.agent.name"] == "kreditoren"
    assert root.attributes["gw.span_id"] == tracer.spans[-1].span_id
    assert exported["chat"].status.status_code is StatusCode.ERROR
    assert root.status.status_code is StatusCode.OK
    assert (
        child.start_time == next(s for s in tracer.spans if s.name.startswith("execute")).start_ns
    )
