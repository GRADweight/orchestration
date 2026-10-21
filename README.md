# GRADweight — orchestration (M4-M7, 21.10.2026)

**Lauffähiger Multi-Agent-Orchestrator** für GRADweight-Prototypen (#1–20).
Struktur: `Planner -> Router -> Executor -> Observer` (M4), plus
`Memory` (M6) und `Approval` + `Feedback` (M7).

```
orchestration/
  orch/
    types.py     # geteilte Daten: ToolSpec, Task, Observation, Result
    planner.py   # M4: Regel- + LLM-Planner (Zykluserkennung)
    router.py    # M4: Konfidenz-Basierter Tool-Router mit Fallback
    executor.py  # M4: Tool-Ausführung mit Timeout + Retry + Audit
    observer.py  # M4: Ergebnis-Beurteilung (ok / unerwartet / unsicher)
    runner.py    # M4: AgentRunner (vollständiger Loop, max_steps, Timeout)
    tools.py     # M4/M5: @tool-Decorator + Tool-Registry + Sandvorbereitung
    memory.py    # M6: Kurz- (Ring-Buff) + Langzeit (JSONL)
    approve.py   # M7: Human-in-the-Loop (gw_core.approval.ApprovalQueue)
    feedback.py  # M7: Annotation (gut/fehler) -> Eval-Set
  tests/
    test_planner.py   # 5 Tests
    test_router.py    # 5 Tests
    test_memory.py    # 3 Tests
    test_runner.py    # 5 Tests  (3-Schritt-Demo, Timeout, max_steps, Audit, Zyklus)
  examples/
    demo_rechnung.py  # 5-Schritt-Flow: OCR -> Bexio -> M6 -> M7 (freigabe+VETO+Feedback)
  vendor/             # CANONICAL-Snapshots (shared-lib f05457b4, gw_core 8982df27)
  M4-M7.md            # Nicht-Drift-Nachweis und Offene Punkte
```

## Schnellstart (offline, kein Netz)

```bash
cd ~/code/GRADweight/orchestration

# Setup
uv venv --python 3.11 .venv && . .venv/bin/activate
uv pip install pytest

# Tests (18 grün)
PYTHONPATH=vendor:. python -m pytest -q

# Demo
PYTHONPATH=vendor:. python -m examples.demo_rechnung
```

## Bausteine (Kurzbeschreibung)

### M4 — Multi-Agent-Orchestrator
`AgentRunner.run(task_text)` zerlegt eine Aufgabe in Sub-Steps via
`RulePlanner` (oder `LLMPlanner`), wählt für jeden Schritt das bestmögliche
`ToolSpec` via `ToolRouter` (Konfidenz-Threshold + Fallback), führt es aus
(`Executor`, mit Timeout + Retry + Audit), urteilt `RuleObserver` (ok/
unerwartet/unsicher). Bei allen OK —-> `done`, `max_steps`/Timeout ->
`abgesagt`, Zykluserkennung -> `abgesagt`.

**Abnahme (M4):** `pytest` grün, `demo_rechnung.py` zeigt einen
vollständigen 3-Schritt-Flow, kein Netz, kein 7B nötig.

### M6 — Memory
`Memory(max_tokens=1000)`: ein Ring-Buff evictiert ältere Einträge
(wenn `max_tokens` überschritten), JSONL persistiert (nie evictiert).
`get(query, k=5)` holt per Bag-of-Words (StubClient, kein Vektor-DB,
kein GPU).

**Abnahme (M6):** 3 Tests, Ring-Buff evicts, Langzeit persistiert.

### M7 — Human-in-the-Loop + Feedback
`ApprovalGate(mandant, prototyp)` nutzt `gw_core.approval.ApprovalQueue`
(CANONICAL, vendor/gw_core_subset). `freigabe` / `anpassen` / `zurueck`
heben einen Vorschlag von Status `offen` auf `freigegeben` (ausgeführt =
jetzt erlaubt), `zurueck` = `zurueckgestellt` (NIE ausgeführt).
`FeedbackLoop` annotiert gut/fehler und importiert in ein Eval-Set.

**Abnahme (M7):** VETO-Kettendurchbruch (freigabe->True, retour->False),
Annotationen korrekt gezählt, Eval-Set-Import funktioniert.

## Ehrliche Grenzen

- **Testpfad:** `RulePlanner` / `RuleObserver` + `StubClient` (bag-of-words)
  — kein semantisches Embedding. Für Produktion: `GW_INFER_URL` + echtes
  Embedding-Modell (nomic-embed-text via Ollama / vLLM).
- **CI Matrix:** Python 3.11, 3.12. Kein 3.9 (gw_core_subset nutzt
  `datetime.UTC`, 3.11+).
- **Sandbox (M5, nicht in diesem Meilenzeichen):** `Sandbox` ist vorbereitet
  (timeout + Netzwerk-Block), aber cgroup/CPU-Limit kommt erst mit M5.

**Kontakt:** edin@gradweight.ch · gradweight.ch
