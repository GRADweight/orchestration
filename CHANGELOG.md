# Changelog

Alle relevanten Änderungen an diesem Projekt. Format nach
[Keep a Changelog](https://keepachangelog.com/de/1.1.0/), Versionierung nach
[Semantic Versioning](https://semver.org/lang/de/).

## [Unreleased]

## [1.0.0] – 2026-10-27

Vollständiger Neuaufbau. Version 0.1.0 zerlegte Aufgaben per Regex und rief Werkzeuge ohne
Argumente auf; davon ist nichts übernommen. Ab dieser Version gilt die öffentliche API (alles
in `gradweight_orchestration.__all__`) im Sinne von SemVer; der Status bleibt «Beta», weil
noch keine Produktionserfahrung vorliegt.

### Neu

- `@tool`: JSON-Schema aus Typannotationen und Docstrings (Google, NumPy, reST, deutsche
  Abschnittsnamen); strikte Validierung der Argumente und Prüfung des Rückgabetyps; Seiteneffekt,
  Betrag, Zeitlimit und Isolation pro Werkzeug.
- `OpenAICompatClient` für Ollama, vLLM, LiteLLM und OpenAI über `urllib` (optional `httpx`):
  Zeitlimits, Wiederholungen mit Backoff und Jitter, `Retry-After`, strikte Antwortprüfung.
- Austauschformate `NATIVE`, `JSON` (strikt geprüft, mit Korrekturschleife) und `AUTO` (Wechsel,
  wenn der Server keine Werkzeuge unterstützt); Erkennung von Werkzeugaufrufen im Antworttext.
- `Agent`: ReAct-Loop mit strukturierten Beobachtungen, Neuplanungshinweisen,
  Schleifenerkennung und Budgets für Schritte, Token, Kosten und aktive Laufzeit.
- `Policy` mit Autonomiegrad 01–05 × Seiteneffekt, Mandatslimit (fail-closed) und
  Verschärfungslisten; `ApprovalQueue` (SQLite) mit Pause, Fortsetzung über Prozessgrenzen und
  CLI (`gw-orchestration approvals`).
- Hash-verkettetes `AuditLog` mit `verify()`, optional HMAC, Dateisperre für mehrere Prozesse;
  `Redactor` für E-Mail, IBAN, AHV-Nummer, Telefon und Kartennummern.
- `Tracer` mit Spans und JSON-Export; optionaler `OpenTelemetryExporter` (Extra `otel`).
- Prozess-Isolation mit `setrlimit`, sauberer Umgebung, temporärem Arbeitsverzeichnis,
  `SIGKILL` der Prozessgruppe sowie Netzwerk- und PID-Namespace, wo verfügbar
  (`gw-orchestration sandbox-info`).
- `PlanAndExecute` (Plan zuerst, Neuplanung nach Fehlschlag) und `Supervisor` (Spezialisten mit
  eigenen Werkzeugen, Policy und Budget).
- `ConversationWindow` mit Token-Budget und `LongTermMemory` (JSON Lines, BM25).
- Kreditoren-Demo (`gw-orchestration demo`) mit Mock-ERP, offline oder gegen Ollama.
- CI: Lint, mypy strict, Tests unter Python 3.11, 3.13, 3.14 und macOS, Sandbox-Nachweise mit
  Namespaces und als root, Paket-Smoke, gitleaks, pip-audit; Dependabot.

### Geändert

- Distribution `gradweight-orchestration`, Importname `gradweight_orchestration` (vorher `orch`;
  kein Alias, weil `orch` nie ohne `PYTHONPATH`-Umweg importierbar war).
- Mindestversion Python 3.11.
- Build mit hatchling, Lizenzangabe nach PEP 639.

### Entfernt

- `vendor/` mit Kopien anderer Pakete, `orch/` (Regex-Planer, Keyword-Router, Feedback-Modul),
  interne Notizen.
