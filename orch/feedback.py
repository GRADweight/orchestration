"""Feedback-Loop — Annotation + Import ins Eval-Set (M7).

Nach jeder Agent-Handlung darf ein Mensch das Ergebnis annotieren
(gut / fehler / neutral). Annotationen landen als JSONL und können
in ein Eval-Set importiert werden (Regression-Test: alt → neu).

CANONICAL: kein Netz, kein ML — nur JSONL + stdlib.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# CANONICAL: die 3 Bewertungs-Klassen eines Menschen (gut / schlecht / neutral)
_LABELS = ("gut", "fehler", "neutral")


def _jetzt_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Annotation:
    """Eine menschliche Annotation eines Agent-Ergebnisses."""

    annot_id: int
    task: str
    result_summary: str
    label: str            # "gut" | "fehler" | "neutral"
    annotator: str = "mensch"
    timestamp: str = field(default_factory=_jetzt_iso)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Annotation":
        return cls(
            annot_id=int(d.get("annot_id", 0)),
            task=d.get("task", ""),
            result_summary=d.get("result_summary", ""),
            label=d.get("label", "neutral"),
            annotator=d.get("annotator", "mensch"),
            timestamp=d.get("timestamp", _jetzt_iso()),
        )


def asdict(obj: object) -> dict:
    return {k: v for k, v in vars(obj).items() if not k.startswith("_")}


class FeedbackLoop:
    """Sammelt Annotationen (gut/fehler) und importiert sie ins Eval-Set.

    Beispiel:
        fb = FeedbackLoop(".audit/feedback.jsonl")
        fb.annotiere(task="Rechnung lesen", result_summary="betrag=1081", label="gut")
        fb.annotiere(task="Bexio buchen", result_summary="timeout", label="fehler")
        fb.import_eval_set(".audit/eval.jsonl")
    """

    def __init__(self, path: str | Path = ".audit/feedback.jsonl") -> None:
        self._path = Path(path)
        self._next_id = self._load_next_id()

    def _load_next_id(self) -> int:
        if not self._path.exists():
            return 1
        nxt = 1
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                nxt = max(nxt, int(json.loads(line).get("annot_id", 0)) + 1)
            except (json.JSONDecodeError, ValueError):
                pass
        return nxt

    def _read_all(self) -> list[Annotation]:
        out: list[Annotation] = []
        if not self._path.exists():
            return out
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(Annotation.from_dict(json.loads(line)))
            except (json.JSONDecodeError, ValueError, KeyError):
                continue
        return out

    def annotiere(
        self,
        *,
        task: str,
        result_summary: str,
        label: str = "neutral",
        annotator: str = "mensch",
    ) -> int:
        """Speichert eine Annotation. Liefert annot_id zurück."""
        if label not in _LABELS:
            raise ValueError(
                f"Ungültiges Label {label!r}; erlaubt: {', '.join(_LABELS)}"
            )
        aid = self._next_id
        self._next_id += 1
        ann = Annotation(
            annot_id=aid,
            task=task,
            result_summary=result_summary,
            label=label,
            annotator=annotator,
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ann.to_dict(), ensure_ascii=False) + "\n")
        return aid

    def import_eval_set(
        self,
        ziel: str | Path,
        *,
        nur_fehler: bool = False,
    ) -> int:
        """Importiert (optional nur 'fehler'-) Annotationen in ein Eval-Set.

        Regression-Test: 'gut'-Fälle prüfen, dass ein neu trainiertes Modell
        weiterhin gut abschneidet; 'fehler'-Fälle prüfen, dass er nicht
        wieder auftritt.
        """
        aus = self._read_all()
        if nur_fehler:
            aus = [a for a in aus if a.label == "fehler"]
        ziel = Path(ziel)
        ziel.parent.mkdir(parents=True, exist_ok=True)
        with ziel.open("a", encoding="utf-8") as f:
            for ann in aus:
                f.write(json.dumps(ann.to_dict(), ensure_ascii=False) + "\n")
        return len(aus)

    def zähler(self) -> dict[str, int]:
        """Kurz-KPI: wieviele pro Label."""
        counts = {"gut": 0, "fehler": 0, "neutral": 0}
        for a in self._read_all():
            counts[a.label] = counts.get(a.label, 0) + 1
        return counts
