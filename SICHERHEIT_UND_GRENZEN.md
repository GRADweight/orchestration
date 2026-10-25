# SICHERHEIT, HAFTUNG UND GRENZEN — GRADweight/orchestration

> **Kurz:** Offene Referenz von GRADweight GmbH. Kein Rechts- oder Beratungsgutachten. Alle Beispiele verwenden erfundene Referenzdaten, keine echten Kundendaten.

| Feld | Wert |
|---|---|
| Repo | `GRADweight/orchestration` |
| Stand | 25.10.2026 |
| Lizenz | `MIT` — ein Satz: Offen nutzbarer Orchestrierungs-Code (35 Tests, ~4900 Zeilen) = Talent-Signal. |
| Status | Intern geprüft, **noch NICHT public** (kein Visibility-Wechsel ohne ausdrückliche Owner-Freigabe). |

## 1. Sicherheitshinweise (Gemeinschaft)
- **Keine echten Kundendaten.** Alle Beispiele verwenden erfundene Referenzdaten (Marken-/Kundenwerte sind Platzhalter). Ueberprueft 25.10.2026: Grep nach AG/GmbH/Kundenamen lieferte nur Fachbegriffe.
- **Kein Rechts- oder Beratungsgutachten.** Dieses Repo ist eine technische Referenz, kein rechtsverbindliches Dokument. Vor einem konkreten Einsatzzweck eigene Pruefung / Fachberatung.
- **Versioniert, aber ohne Garantie.** Schildert den aktuellen Stand; keine SLA oder Aktualisierungspflicht. Vor Public erneut gegen HEAD pruefen.

## 2. Bekannte Fehler / offene Gaps (Repo-spezifisch)
- **Intern docs M4-M7.md, M5-M6.md (lokal entfernt)..** Vor Public im Remote ebenfalls nicht pushen.
- **~/code/ + /tmp/smoke Pfad-Leaks (lokal fix -> <repo>/)..** Im Remote-Commit übernehmen.
- **35 Tests + 4864 Zeilen, kein Changelog..** Optional v0.1.0 Changelog.

## 3. Vor Public (Checkliste)
- [ ] Gegen HEAD pruefen: keine internen Pfade (/Users, ~/, /tmp), keine echten Kundendaten, keine Preise/Marge im Public-Text
- [ ] Interne Docs entfernt (OFFENE_PUNKTE, PROJEKT, ROADMAP, M-Module)
- [ ] Lizenz korrekt + ein-Satz-Erklaerung im README (s. oben)
- [ ] Sichtbarkeit erst nach **ausdruecklicher Owner-Freigabe** (kein automatischer Visibility-Wechsel)

_Grundlage: interner Forensic-Scan 25.10.2026; Werte aus /tmp/gwscan; nichts geschoben._
