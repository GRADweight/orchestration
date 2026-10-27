# Beitragen

## Einrichten

```bash
uv venv && uv pip install -e ".[dev]"      # oder: python -m venv .venv && pip install -e ".[dev]"
```

## Prüfen vor jedem Pull Request

```bash
ruff check . && ruff format --check .
mypy                                       # strict, Konfiguration in pyproject.toml
python -m pytest --cov                     # Abdeckungsschwelle in pyproject.toml
gw-orchestration demo                      # Offline-Demo muss durchlaufen
```

Sandbox-Tests hängen von der Plattform ab und werden mit Begründung übersprungen, wenn eine
Voraussetzung fehlt; `gw-orchestration sandbox-info` zeigt, was auf dem eigenen System gilt.

## Konventionen

- **Keine Laufzeit-Abhängigkeiten im Kern.** Optionales gehört in ein Extra.
- **Kein Netzwerk in Tests.** Modelle über `ScriptedLLM` oder den lokalen Testserver in
  `tests/fake_openai.py` (nur `127.0.0.1`).
- **Jede beworbene Eigenschaft hat einen Test.** Neue Zusagen in README oder Docstrings ohne
  Test werden nicht übernommen; Zahlen in der Doku werden gegen den Code geprüft
  (`tests/test_doku_konsistenz.py`).
- **Fail-closed:** Wo Sicherheit oder Geld betroffen ist, führt Unklarheit zu Ablehnung oder
  Freigabe, nicht zu stiller Ausführung.
- **Sprache:** Bezeichner der Bibliothek englisch; Meldungen, Docstrings und Doku auf
  Schweizer Hochdeutsch (ss statt Eszett). Die Demo verwendet deutsche Fachbegriffe für Werkzeuge
  und Felder, weil diese Namen Teil des Prompts sind, den das Modell sieht.
- **Commits:** kleine, logische Schritte; Nachricht auf Deutsch, erste Zeile ≤ 72 Zeichen.
- **Daten:** nur erfundene oder öffentliche Beispieldaten, keine Secrets.

## Fragen

Issue eröffnen oder mail@gradweight.ch.
