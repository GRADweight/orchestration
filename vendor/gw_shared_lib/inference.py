"""Schicht 0: Inferenz-Client (OpenAI-kompatibel).

CANONICAL-Inferenzschicht für den ganzen GRADweight-Stack.
Ersetzt die zersplitterten Versionen (rag/retrieve.ollama_embed,
platforms/stack/minimal/rag.py embed/chat) im ganzen Org.

Zwei Client-Klassen:
  - Client       : HTTP-Client gegen einen OpenAI-kompatiblen
                   Endpoint (/v1/embeddings, /v1/chat/completions).
                   Lauffähig gegen Ollama, vLLM, LiteLLM-Gateway.
  - StubClient   : Offline-Client für Tests und Demo (deterministisch).
                   Kein Netz, kein Crash. Liefert bag-of-words-Vektor
                   via SHA-256-Hash-basierte Token-Segregation.

Wenn die HTTP-Calls fehlschlagen, fällt `embed_texts` bzw. `chat`
auf `StubClient` zurück — fail-soft, keine Erfindung.
"""
from __future__ import annotations

import json
import os
import math
import re
import hashlib
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional

DEFAULT_URL = os.environ.get("GW_INFER_URL", "http://localhost:11434")
DEFAULT_EMBED = os.environ.get("GW_INFER_EMBED", "nomic-embed-text")
DEFAULT_CHAT = os.environ.get("GW_INFER_CHAT", "llama3.2:3b")


@dataclass
class Client:
    """OpenAI-kompatibler HTTP-Client.

    `url`       : Basis-URL (z.B. http://localhost:11434/v1)
    `embed_model` : Standard-Embedding-Modell
    `chat_model`  : Standard-Chat-Modell
    `timeout`    : HTTP-Timeout in Sekunden
    """
    url: str = DEFAULT_URL
    embed_model: str = DEFAULT_EMBED
    chat_model: str = DEFAULT_CHAT
    timeout: float = 120.0

    def _post(self, path: str, payload: dict) -> dict:
        full = self.url.rstrip("/") + path
        req = urllib.request.Request(
            full,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def embed(self, texts, model: Optional[str] = None) -> list[list[float]]:
        """Embeddings für `texts` (str oder list[str]).

        Rückgabe: Liste von Vektoren, immer in derselben Reihenfolge.
        """
        model = model or self.embed_model
        single = isinstance(texts, str)
        inputs = [texts] if single else list(texts)
        out: list[list[float]] = []
        for t in inputs:
            body = {"model": model, "input": t}
            for ep in ("embeddings", "embed"):
                try:
                    if ep == "embeddings" and "/v1" not in self.url:
                        # Ollama /api/embeddings
                        data = self._post(f"/api/{ep}", body)
                        vecs = (data or {}).get("embeddings")
                        if vecs:
                            out.append(vecs[0])
                            if single:
                                return out
                            continue
                    if ep == "embeddings" and "/v1" in self.url:
                        # OpenAI /v1/embeddings
                        data = self._post(f"/api/embeddings", body)
                        ds = (data or {}).get("data")
                        if ds and len(ds) > 0:
                            out.append(sorted(ds, key=lambda x: x.get("index", 0))[0]["embedding"])
                            if single:
                                return out
                            continue
                    if ep == "embed":
                        data = self._post(f"/api/{ep}", body)
                        emb = (data or {}).get("embedding")
                        if emb:
                            out.append(emb)
                            if single:
                                return out
                            continue
                except Exception:
                    break
        return out if out else ([] if not single else [[]])

    def chat(self, messages: list[dict], model: Optional[str] = None,
             temperature: float = 0.2, max_tokens: int = 400) -> str:
        """Eine Chat-Request über /chat/completions. Gibt den Text
        zurück. Fail-soft: '' bei Fehler (der Aufrufer muss
        die Antwort validieren — RAG-Pipeline tut es).
        """
        model = model or self.chat_model
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        for ep in ("/api/chat", "/v1/chat/completions"):
            try:
                data = self._post(ep, body)
                choices = data.get("choices") or []
                if choices:
                    return (choices[0].get("message", {}) or {}).get(
                        "content", "")
            except Exception:
                continue
        return ""


@dataclass
class StubClient(Client):
    """Deterministischer Offline-Client für Tests und Demos.

    Kein HTTP, kein Netz. Liefert:
      - embed()   : bag-of-words-Vektor via Token-Hashing
                    (gleiche Text -> gleiche Vektor, ähnliche Texte
                     haben höheren Übereinstimmungs-Wert —
                      kein ML, kein Netz, aber ehrlicher als ein
                      SHA-256-Whole-Text-Hash, der keine
                      Semantik versteht).
      - chat()    : ein fester Antwort-Text
                    "Offline (StubClient)."
    """
    embed_model: str = "stub"
    chat_model: str = "stub"
    _dim: int = 128
    _chat_reply: str = "Offline (StubClient). Für Produktion: setze GW_INFER_URL."

    def embed(self, texts, model: Optional[str] = None) -> list[list[float]]:
        single = isinstance(texts, str)
        inputs = [texts] if single else list(texts)
        out = [_stub_vec(t, self._dim) for t in inputs]
        return out[0:1] if single else out

    def chat(self, messages, model=None, temperature=0.2,
             max_tokens=400) -> str:
        return self._chat_reply


def _stub_vec(text: str, dim: int = 128) -> list[float]:
    """Offlin-Ersetzer für einen Embedding-Server.

    **Ehrlich:** Dies ist KEIN semantisches Embedding.
    Es ist ein bag-of-words-Vektor: die Tokens des Textes
    werden per SHA-256 auf `dim` Dimensionen gehasht und
    gezählt. Ähnlichkeit = Übereinstimmung der Tokens (Wort-
    Overlap, nicht Bedeutung-Ähnlichkeit).

    Das ist **genug** für die Retrievall-Logik (welche Chunk
    ist am nächsten?), aber **nicht** für echte semantische
    Ähnlichkeit (dafür braucht man ein echtes Embedding-Modell,
    z.B. nomic-embed-text). Der Aufrufer muss wissen, dass
    `StubClient.cosine(a,b)` ≈ 0 bedeutet "keine Wort-Über-
    einstimmung", aber nicht "verschiedene Bedeutung".

    Kein externer Abhäng (nur stdlib re + hashlib).
    """
    dim = max(dim, 32)
    vec = [0.0] * dim
    tokens = [t for t in re.findall(r"\w+", text.lower())
              if len(t) > 1]  # keine 1-zeiger
    n = len(tokens)
    if n == 0:
        return [0.0] * dim
    for tok in tokens:
        h = hashlib.sha256(tok.encode()).digest()
        idx = int.from_bytes(h[:4], "big") % dim
        vec[idx] += 1.0
    norm = sum(x * x for x in vec) ** 0.5
    if norm > 1e-9:
        vec = [x / norm for x in vec]
    return vec


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Deterministische cosine-Similarität (reines Python, kein
    numpy nötig). 0..1 (nur positive, wenn beide Vektoren
    identisch) oder -1..1 für generelle Vektoren."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def embed_texts(texts, *, client: Optional[Client] = None,
                fallback_stub: bool = True) -> list[list[float]]:
    """Bequemer Wrapper: versucht HTTP-Client, fällt auf StubClient
    zurück wenn `fallback_stub=True` und HTTP fehlschlägt.

    Für RAG-Integrationen: wenn der Embedding-Server (Ollama/vLLM)
    nicht erreichbar ist, kein Crash, sondern ein "kein Treffer"
    Zustand, der von der RAG-Pipeline als "keine Angaben" behandelt
    wird.
    """
    client = client or Client()
    try:
        out = client.embed(texts)
    except Exception:
        if fallback_stub:
            out = StubClient().embed(texts)
        else:
            raise
    return out


def chat(messages, *, client: Optional[Client] = None,
         fallback_stub: bool = True) -> str:
    client = client or Client()
    try:
        return client.chat(messages)
    except Exception:
        if fallback_stub:
            return StubClient().chat(messages)
        raise
