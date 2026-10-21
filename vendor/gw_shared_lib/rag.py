"""Schicht 3: RAG-Pipeline über der shared-lib.

RAGPipeline:
  1. Guard-Gate     — optional, GuardPipeline.run()
  2. Retrieval      — cosine-Similarität über Chunk-Embeddings
  3. RAG-Antwort    — quell-zitiert, mit Schwellunterschied:
                       - kein Treffer   -> "keine Angaben" (kein Halluzinieren)
                       - Treffer        -> Antwort mit Cites
  4. Guard          — optional

Der Retrieval ist in-Memory-cosine. Für einen Vektorstore (pgvector,
Qdrant) siehe `platforms/stack/minimal/` und ein
`RAGPipeline(vector_backend=...)` Extension (noch nicht in dieser
Version).

Der Client ist ein `gw_shared_lib.inference.Client` —
**diese** RAG-Schicht nutzt **diesen** Canonical, keine zweite
Inferenz-Schicht.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .guard import analyse, GuardPipeline
from .inference import (
    Client, StubClient, cosine_similarity, embed_texts, DEFAULT_EMBED,
    DEFAULT_URL,
)


@dataclass
class Chunk:
    content: str
    source: str = ""
    confidence: float = 0.0          # Quelle-Confidence, nicht Retrieval-Score
    access_level: str = "public"     # public | internal | confidential
    embedding: list[float] = field(default_factory=list)
    retrieval_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "content": (self.content[:150] + "…") if len(self.content) > 150
                       else self.content,
            "confidence": self.confidence,
            "access_level": self.access_level,
            "retrieval_score": round(self.retrieval_score, 4),
        }


@dataclass
class RAGResult:
    question: str = ""
    answer: str = ""
    citations: list[dict] = field(default_factory=list)
    guard_blocked: bool = False
    guard_reason: str = ""
    confidence: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        out = {
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "guard_blocked": self.guard_blocked,
            "confidence": self.confidence,
        }
        if self.guard_reason:
            out["guard_reason"] = self.guard_reason
        if self.error:
            out["error"] = self.error
        return out


def _embed_one(text: str, client: Client, embed_model: str,
               use_stub: bool = False) -> list[float]:
    out = embed_texts(text, client=client, fallback_stub=use_stub)
    return out[0] if out else []


class RAGPipeline:
    """Guard-Gate -> Retrieval -> Antwort-Konstruktion."""

    def __init__(self, *,
                 corpus: Optional[list[Chunk]] = None,
                 guard: Optional[GuardPipeline] = None,
                 top_k: int = 3,
                 min_score: float = 0.12,
                 ollama_url: Optional[str] = None,
                 embed_model: Optional[str] = None,
                 use_stub: bool = True,
                 level_filter: str | int | None = None,
                 ) -> None:
        self.corpus = corpus or []
        self.guard = guard
        self.top_k = top_k
        self.min_score = min_score
        self.url = ollama_url or DEFAULT_URL
        self.embed_model = embed_model or DEFAULT_EMBED
        self.use_stub = use_stub
        self.level_filter = level_filter

    def _pre_embed(self) -> None:
        client = StubClient() if self.use_stub else Client(
            url=self.url, embed_model=self.embed_model)
        for c in self.corpus:
            if not c.embedding:
                c.embedding = _embed_one(c.content, client,
                                          self.embed_model,
                                          use_stub=self.use_stub)

    def answer(self, question: str) -> RAGResult:
        # 1. Guard-Gate
        if self.guard is not None:
            gr = self.guard.run(question)
            if gr.blocked:
                return RAGResult(
                    question=question,
                    answer="(Frage aufgrund von Prompt-Injection abgewiesen)",
                    guard_blocked=True,
                    guard_reason=gr.reason,
                )

        # 2. Retrieval
        self._pre_embed()
        client = StubClient() if self.use_stub else Client(
            url=self.url, embed_model=self.embed_model)
        q_emb = _embed_one(question, client, self.embed_model,
                           use_stub=self.use_stub)
        if not q_emb:
            return RAGResult(question=question, answer="",
                             error="Embedding nicht verfügbar",
                             confidence=0.0)

        scored: list[Chunk] = []
        for c in self.corpus:
            if not c.embedding:
                continue
            if self.level_filter is not None:
                # level-filter: access_level in ("public","internal","confidential")
                if str(self.level_filter).lower() != c.access_level.lower():
                    continue
            sim = cosine_similarity(q_emb, c.embedding)
            if sim >= self.min_score:
                c.retrieval_score = sim
                scored.append(c)
        scored.sort(key=lambda c: c.retrieval_score, reverse=True)
        scored = scored[:self.top_k]

        if not scored:
            return RAGResult(
                question=question,
                answer=(f'Zu „{question}" habe ich in den hinterlegten '
                       "Quellen keine Angaben gefunden. Bitte prüfe die "
                       "Quelle direkt oder wende dich an den Ansprechpartner."),
                confidence=0.0,
            )

        # 3. Quelle-zitierte Antwort
        lines = [f'Zu „{question}"']
        for i, m in enumerate(scored, 1):
            lines.append(f"[{i}] Quelle: {m.source or 'unknown'} "
                         f"(Conf: {m.confidence*100:.0f}%, "
                         f"Retrieval: {m.retrieval_score:.3f})")
            snippet = m.content[:180] + ("…" if len(m.content) > 180 else "")
            lines.append(f'    „{snippet}"')
        max_conf = max(m.confidence for m in scored)
        return RAGResult(
            question=question,
            answer="\n".join(lines),
            citations=[
                {"source": m.source, "confidence": m.confidence,
                 "retrieval_score": m.retrieval_score,
                 "content": m.content[:120]}
                for m in scored
            ],
            confidence=max_conf,
        )

    @staticmethod
    def is_confidential(citations: list[dict]) -> bool:
        for c in citations:
            al = c.get("access_level", "")
            if al in ("confidential", "vertraulich", "secret"):
                return True
        return False
