# Sicherheit

## Schwachstellen melden

Bitte keine öffentlichen Issues für Sicherheitsprobleme. Meldewege:

- GitHub Private Vulnerability Reporting: Reiter «Security» → «Report a vulnerability»
- E-Mail an mail@gradweight.ch

Hilfreich sind Version bzw. Commit, Plattform (Linux/macOS, root oder nicht), ein minimales
Beispiel und die erwartete gegenüber der beobachteten Wirkung. Wir bestätigen den Eingang
innert fünf Arbeitstagen und stimmen eine Veröffentlichung mit der meldenden Person ab.

## Unterstützte Versionen

| Version | Unterstützt |
|---|---|
| 1.x | ja |
| 0.x | nein |

## Was als Schwachstelle gilt

- Eine im README als erzwungen beschriebene Eigenschaft hält nicht, z.B. ein Werkzeug im
  Prozess-Modus überlebt das Zeitlimit, ein gesetztes Limit wirkt nicht, die Policy erlaubt
  einen Aufruf, den die Matrix verbietet, eine Freigabe lässt sich ohne Entscheid ausführen,
  oder `AuditLog.verify()` meldet eine veränderte Kette als intakt.
- Geheimnisse (API-Schlüssel) erscheinen in Logs, Audit-Einträgen oder Traces.

## Was ausdrücklich keine Garantie ist

Die Prozess-Isolation ist keine Sicherheitsgrenze gegen feindseligen Code: kein
Dateisystem-Schutz, kein Systemaufruf-Filter, und Netzwerk-Isolation nur mit verfügbaren
Namespaces. Das Audit-Log ist ohne HMAC-Schlüssel und externe Verankerung des letzten Hashs
manipulationserkennend, nicht manipulationssicher. Die Schwärzung ist musterbasiert. Siehe
Abschnitt «Grenzen» im README.
