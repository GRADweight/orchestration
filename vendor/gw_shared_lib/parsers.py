"""Schicht 2: Doku-Parser (Text/Markdown, PDF, DOCX).

Zurückgabe: list[Chunk], wobei jeder Chunk aus einer Text-Quelle
mit Source-Metadaten kommt. Für RAG: 1 Text -> N Chunks (pro
Sektion/Headline im Dokument).

Text/Markdown:    immer (stdlib)
PDF              : pdfplumber (optional)
DOCX             : python-docx (optional)

Wenn die optionale Dependency nicht installiert ist, fällt der
Parser graceful auf "Text extrahieren, kein Chunking" zurück (fail-soft).
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from .rag import Chunk


@dataclass
class ParseResult:
    chunks: list[Chunk] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source: str = ""

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "chunks": [c.to_dict() for c in self.chunks],
            "warnings": self.warnings,
        }


def _split_sections(text: str, source: str,
                    default_conf: float = 0.8,
                    max_chunk: int = 1500) -> list[Chunk]:
    """Teilt in Abschnitte (## / # als Grenze) bzw. in max_chunk.

    Kein ML. Nur Headline-Struktur. Für KMU-Doku gut genug.
    """
    if not text.strip():
        return []
    lines = text.splitlines()
    out: list[Chunk] = []
    current_lines: list[str] = []
    current_src = source

    def flush(current_lines, current_src, conf):
        if not current_lines:
            return None
        body = "\n".join(current_lines).strip()
        if not body:
            return None
        if len(body) > max_chunk:
            # grobes Splitting: max chunk size
            for i in range(0, len(body), max_chunk):
                out.append(Chunk(
                    content=body[i:i + max_chunk],
                    source=f"{current_src}#{i//max_chunk + 1}",
                    confidence=conf,
                    access_level="internal",
                ))
            return
        out.append(Chunk(
            content=body,
            source=current_src,
            confidence=conf,
            access_level="internal",
        ))

    # Split auf "##" Headlines
    sections: list[tuple[str, list[str]]] = []
    header = ""
    buf: list[str] = []
    for line in lines:
        if line.startswith("## ") and not line.startswith("### "):
            if buf:
                sections.append((header, buf))
            header = line[3:].strip()
            buf = []
        else:
            buf.append(line)
    if buf:
        sections.append((header, buf))
    if not sections:
        # Kein "##" in Text — nur ein einziger Chunk
        out.append(Chunk(content=text, source=source,
                          confidence=default_conf,
                          access_level="internal"))
        return out
    for h, buf in sections:
        body = "\n".join(buf).strip()
        if not body:
            continue
        chunk_src = f"{source}#{h[:80]}" if h else source
        out.append(Chunk(content=body,
                          source=chunk_src,
                          confidence=default_conf,
                          access_level="internal"))
    return out


def parse_text(text: str, *, source: str = "inline",
               access_level: str = "internal",
               confidence: float = 0.8,
               max_chunk: int = 1500) -> list[Chunk]:
    out = _split_sections(text, source, default_conf=confidence,
                          max_chunk=max_chunk)
    for c in out:
        c.access_level = access_level
    return out


def parse_file(path: str | Path, *, access_level: str = "internal",
               confidence: float = 0.8,
               max_chunk: int = 1500) -> ParseResult:
    p = Path(path)
    if not p.exists():
        return ParseResult(warnings=[f"Datei nicht gefunden: {path}"],
                           source=str(p))
    ext = p.suffix.lower()
    if ext in (".txt", ".md", ".markdown"):
        text = p.read_text(encoding="utf-8")
        chunks = _split_sections(text, str(p), default_conf=confidence,
                                max_chunk=max_chunk)
        for c in chunks:
            c.access_level = access_level
        return ParseResult(chunks=chunks, source=str(p))

    if ext == ".pdf":
        chunks = _parse_pdf(p, confidence=confidence, max_chunk=max_chunk)
        for c in chunks:
            c.access_level = access_level
        if not chunks:
            return ParseResult(
                warnings=[
                    "PDF-Parser nicht verfügbar (pdfplumber nicht "
                    "installiert). Nur Datei-Header extrahiert."
                ],
                source=str(p),
            )
        return ParseResult(chunks=chunks, source=str(p))

    if ext in (".docx",):
        chunks = _parse_docx(p, confidence=confidence, max_chunk=max_chunk)
        for c in chunks:
            c.access_level = access_level
        if not chunks:
            return ParseResult(
                warnings=[
                    "DOCX-Parser nicht verfügbar (python-docx nicht "
                    "installiert)."
                ],
                source=str(p))
        return ParseResult(chunks=chunks, source=str(p))

    return ParseResult(warnings=[f"Unbekantes Format: {ext} (nur .txt/.md, "
                                  ".pdf, .docx unterstützt)"],
                       source=str(p))


def _parse_pdf(p: Path, confidence: float, max_chunk: int) -> list[Chunk]:
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        return []
    out: list[Chunk] = []
    with pdfplumber.open(p) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            for sub in _split_sections(text, source=f"{p.name}#{i}",
                                       default_conf=confidence * 0.9,
                                       max_chunk=max_chunk):
                out.append(sub)
    return out


def _parse_docx(p: Path, confidence: float, max_chunk: int) -> list[Chunk]:
    try:
        from docx import Document  # type: ignore
    except ImportError:
        return []
    doc = Document(p)
    out: list[Chunk] = []
    for para in doc.paragraphs:
        text = (para.text or "").strip()
        if text:
            if para.style and para.style.name and \
               para.style.name.startswith("Heading"):
                out.append(Chunk(
                    content=text,
                    source=f"{p.name}#{para.style.name}",
                    confidence=confidence * 0.9,
                    access_level="internal",
                ))
            else:
                out.append(Chunk(
                    content=text,
                    source=str(p.name),
                    confidence=confidence * 0.7,
                    access_level="internal",
                ))
    return out


def to_chunks(files, *, access_level: str = "internal",
              confidence: float = 0.8,
              max_chunk: int = 1500) -> tuple[list[Chunk], list[str]]:
    """Lädt eine Liste von Dateien in eine Chunk-Liste. Gibt
    (chunks, warnings) zurück."""
    warnings: list[str] = []
    out: list[Chunk] = []
    for f in files:
        res = parse_file(f, access_level=access_level,
                          confidence=confidence, max_chunk=max_chunk)
        out.extend(res.chunks)
        warnings.extend(res.warnings)
    return out, warnings
