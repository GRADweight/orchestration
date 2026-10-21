"""M5 — 5 Beispiel-Tools für die Sandbox (s. GAP_ANALYSE.md, M5).

Alle Tools sind **offline + stdlib** (KMU-Sweet). Jedes ist eine
reguläre `def`-Funktion (damit `inspect.getsource → exec` im
Kind-Prozess funktioniert), mit Imports im Körper (selbst-contained).

    1. ocr_rechnung        — Betrag/Frist aus einer Rechnung (Regex)
    2. klassifiziere_email — E-Mail-Triage (Verwaltung/Angebot/Sonstiges)
    3. bexio_buchung       — Buchung in Bexio simulieren (Offline-Stub,
                             kein Netz — für die API: GW_BEXIO_URL setzen)
    4. web_search_proxy    — Suche via Proxy; ohne Fixture → echtes
                             Socket → wird im Sandbox blockiert (network=False)
    5. datei_leser         — sicherer Datei-Leser (Pfad-Sandbox, keine ../)

`web_search_proxy` ist der einzige Tool, der Netz *versucht* — damit
zeigt er, wie der Netzwerk-Block wirkt (s. test_network_block).
"""
from __future__ import annotations

import re
from typing import Any


# ── 1. Rechnung-OCR (offline, Regex statt Vision) ─────────────────────────────


def ocr_rechnung(text: str) -> dict:
    """Holt Betrag (CHF) und Fälligkeit aus einer Rechnung (Text).

    Offline-Ersatz für echtes Vision-OCR (s. catalog): die Text-Extraktion
    (z.B. pdf-to-text) ist vorgelagert; hier der Parsing-Teil.
    """
    import re
    m_amount = re.search(r"(\d+[.,]\d{2})\s*(?:CHF|Fr\.?|\s*EUR)", text, re.I)
    m_due = re.search(r"(?:fällig|fälligkeit|due).{0,20}?(\d{1,2}\.\d{1,2}\.\d{2,4})",
                      text, re.I)
    return {
        "betrag_chf": float(m_amount.group(1).replace(",", ".")) if m_amount else None,
        "faellig_am": m_due.group(1) if m_due else None,
        "gültig": bool(m_amount),
    }


# ── 2. E-Mail-Klassifikation (Keyword-Triage, offline) ────────────────────────


def klassifiziere_email(betreff: str, korper: str = "") -> dict:
    """Triage: Verwaltung / Angebot / Sonstiges (Keyword-Dichten)."""
    text = f"{betreff} {korper}".lower()
    cats = {
        "verwaltung": ("rechnung", "buchung", "kündigung", "vertrag", "rechnung",
                       "miete", "versicherung"),
        "angebot": ("angebot", "quote", "preisanfrage", "discount", "angebot",
                    "prospekt"),
        "sonstiges": (),
    }
    scores = {
        c: sum(1 for kw in kws if kw in text) for c, kws in cats.items()
    }
    best = max(scores, key=scores.get)
    # 0 → Sonstiges
    if scores[best] == 0:
        best = "sonstiges"
    return {"klasse": best, "scores": scores, "begründung": f"'{best}' (Dichte {scores[best]})"}


# ── 3. Bexio-API (Offline-Stub; für Produktion: GW_BEXIO_URL) ─────────────────


def bexio_buchung(betrag_chf: float, konto: str, *args: Any, **kwargs: Any) -> dict:
    """Simuliert eine Buchung in Bexio — offline-deterministisch.

    KEIN echtes Netz: der Stub bildet den API-Response ab (Status + ID).
    Für Produktion: setze `GW_BEXIO_URL` und ersetze durch einen
    HTTP-Call (über den Web-Search-Proxy-Pattern / mit network=True).
    """
    import hashlib

    # Deterministische Buchung-ID (aus Betrag + Konto)
    sig = hashlib.sha256(f"{betrag_chf:.2f}-{konto}".encode()).hexdigest()[:10]
    return {
        "status": "buchung_erstellt",
        "buchung_id": f"BX-{sig}",
        "betrag_chf": round(float(betrag_chf), 2),
        "konto": konto,
        "offline_stub": True,
    }


# ── 4. Web-Search-Proxy (versucht Netz; ohne Fixture blockiert) ───────────────


def web_search_proxy(query: str, max_results: int = 3, fixture: str | None = None,
                     *args: Any, **kwargs: Any) -> dict:
    """Sucht über einen Proxy. Mit `fixture` → offline (canned), sonst
    echter Socket-Call (→ im Sandbox mit `network=False` blockiert)."""
    if fixture is not None:
        return {
            "query": query,
            "results": [f"offline-{i}-{query}" for i in range(min(3, max_results))],
            "offline": True,
        }
    import socket  # realer Netzversuch (im Sandbox blockiert)
    # Kein Proxy im Testpfad → wir versuchen eine Verbindung.
    s = socket.create_connection(("proxy.gw.local", 80), timeout=2)
    s.close()
    return {"query": query, "results": []}


# ── 5. Dateilser (Pfad-Sandbox, ../-Escape blockiert) ─────────────────────────


def datei_leser(pfad: str, *args: Any, **kwargs: Any) -> dict:
    """Liest eine Datei — blockiert `../`-Escape und absolute Pfade
    ausserhalb des geworbenen Roots. Returns eine Zeilen-Zähleung."""
    from pathlib import Path

    p = Path(pfad)
    # ../-Escape blockieren und absolut machen
    if ".." in pfad or p.is_absolute():
        return {"fehler": "Pfad-Escape blockiert (../ oder absolut)", "ok": False}
    if not p.exists():
        return {"fehler": f"Datei nicht gefunden: {pfad}", "ok": False}
    text = p.read_text(encoding="utf-8", errors="replace")
    return {
        "ok": True,
        "zeilen": len(text.splitlines()),
        "zeichen": len(text),
        "pfad": str(p),
    }


# ── Registrierung (für die M4 ToolRegistry / ToolRouter) ───────────────────────


from .types import ToolSpec  # noqa: E402


def build_specs() -> list[ToolSpec]:
    """Die 5 Beispiel-Tools als ToolSpec (offline default, network=False)."""
    return [
        ToolSpec("ocr_rechnung", "Betrag+ Frist aus Rechnung (OCR-Text)",
                 ocr_rechnung, keywords=("rechnung", "ocr", "betrag")),
        ToolSpec("klassifiziere_email", "E-Mail-Triage (VW/Angebot/Sonstiges)",
                 klassifiziere_email, keywords=("email", "triage", "klassifizier")),
        ToolSpec("bexio_buchung", "Buchung in Bexio (offline-Stub)",
                 bexio_buchung, timeout_s=3.0,
                 keywords=("buchen", "bexio", "konto")),
        ToolSpec("web_search_proxy", "Suche via Proxy (offline mit Fixture)",
                 web_search_proxy, timeout_s=3.0,
                 keywords=("suche", "search", "web")),
        ToolSpec("datei_leser", "Sichere Datei-Leser (../ blockiert)",
                 datei_leser, keywords=("datei", "lesen", "lesen")),
    ]


__all__ = [
    "ocr_rechnung",
    "klassifiziere_email",
    "bexio_buchung",
    "web_search_proxy",
    "datei_leser",
    "build_specs",
]
