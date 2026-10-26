# gradweight-orchestration

## In English

A small, auditable agent runtime for local LLMs, currently being rebuilt on this
branch. This first layer provides tool declarations with JSON Schemas derived from
type hints, strict argument validation, typed results, enforced timeouts and a
subprocess sandbox with resource limits. The full documentation follows with the
remaining layers.

## Stand dieses Zweigs

Die Runtime wird schichtweise neu aufgebaut. Diese erste Schicht enthält:

- `@tool`: JSON-Schema aus Typannotationen und Docstrings, strikte Validierung
  der Argumente, typisierte Ergebnisse.
- `ToolExecutor`: erzwungene Zeitlimits (Thread oder Kindprozess), Wiederholungen
  nur für lesende oder idempotente Werkzeuge.
- `run_sandboxed`: Kindprozess mit `setrlimit`-Grenzen, sauberer Umgebung,
  temporärem Arbeitsverzeichnis, hartem Abbruch und – wo verfügbar –
  Netzwerk- und PID-Namespace. Das ist keine Sicherheitsgrenze für feindseligen
  Code.

Tests: `python -m pytest`. Kontakt: mail@gradweight.ch. Lizenz: MIT.
