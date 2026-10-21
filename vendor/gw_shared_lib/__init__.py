"""GRADweight — shared-lib: das gemeinsame technische Fundament.

Dieses Paket bündelt die Bausteine, aus denen alle GRADweight-Prototypen
(1–20, s. internal-kb/katalog) und Prototypen-Repos aufgebaut werden.
Ziel: ein KMU soll mit `pip install gw-shared-lib` einen lauffähigen
Grundstock bekommen, der **lokal** läuft und keine Cloud-Datenübertragung
verlangt (DSG / SOVERAE-konform).

Sechs Bausteine (siehe README):

  1. inference  - OpenAI-kompatibler Client (embeddings + chat) für
                  Ollama / vLLM / LiteLLM-Gateway. Dazu ein StubClient
                  für Offline-Tests (kein Netz, deterministisch).
  2. guard      - Signatur-Schicht (deterministisch, kein Netz) +
                  pluggbarer Judge-Interface + GuardPipeline. CANONICAL:
                  die 16 Injection-Signaturen gibt es nur hier einmal.
  3. rag        - Chunk + retrieve (cosine) + RAGPipeline
                  (Guard -> Retrieve -> quelle-zitierte Antwort). Verwendet
                  inference, damit keine zweite Inferenz-Schicht existiert.
  4. parsers    - Text/Markdown immer; PDF (pdfplumber, optional) und
                  DOCX (python-docx, optional) mit graceful Degradation.
  5. config     - .env- / OS-basierter Config-Loader (keine Secrets in
                  Code; alles via OS.environ / .env).
  6. audit      - Audit-Trail: wer / was / wann / Grund als JSON-lines.
                  KMU-Grundbaustein (s. gw_core, aber self-contained).

Einzige harte Abhängigkeit: keine (nur stdlib). numpy ist optional
(dann schnellere cosine-Sim). pdfplumber / python-docx sind optional
und werden gracefully ausgelastet, falls nicht installiert.

Versionierung: SemVer. Die Signaturliste ist CANONICAL — andere
Repos (guardrails, rag) synchronisieren hierher, nicht umgekehrt.
"""
from __future__ import annotations

__version__ = "0.1.0"

# Kern-Importe (funktionieren ohne jede optionale Abhängigkeit) —
# die hier gezeigten Symbole sind die öffentliche API.
from . import inference, guard, rag, parsers, config, audit
from .inference import Client, StubClient, embed_texts, chat
from .guard import (
    INJECTION_SIGNATURES, INJECTION_CATEGORIES,
    GuardResult, Judge, DummyJudge, GuardPipeline,
    analyse, detect,
)
from .rag import Chunk, RAGPipeline, RAGResult
from .config import Config, load_config
from .audit import AuditLog, audit_action
from .parsers import parse_text, parse_file, to_chunks

__all__ = [
    "__version__",
    # inference
    "Client", "StubClient", "embed_texts", "chat",
    # guard
    "INJECTION_SIGNATURES", "INJECTION_CATEGORIES",
    "GuardResult", "Judge", "DummyJudge", "GuardPipeline",
    "analyse", "detect",
    # rag
    "Chunk", "RAGPipeline", "RAGResult",
    # config
    "Config", "load_config",
    # audit
    "AuditLog", "audit_action",
    # parsers
    "parse_text", "parse_file", "to_chunks",
    # submodules
    "inference", "guard", "rag", "parsers", "config", "audit",
]
