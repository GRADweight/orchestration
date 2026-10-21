"""Schicht 0: Config.

Alles Laufzeit-Parametrize kommt aus der Umgebung (OS.environ) oder —
optional — aus einer .env-Datei. Keinen Secret im Code, keine Secrets in
Logs. (s. CLAUDE.md: "Keine Geheimisse in Dateien, Commits, Logs".)

Nutzen:
    cfg = load_config()                     # liest OS.environ
    cfg.ollama_url                          # -> str
    cfg.ollama_emb_model                    # -> str
    cfg.require("SOME_SECRET_KEY")          # raise wenn fehlt (fail-closed)

Ein .env wird nur geladen, wenn `GW_DOTENV=1` gesetzt ist ODER eine
.env-Datei neben der Anwendung liegt und `GW_DOTENV=1`. Kein
PyDepend on python-dotenv (kein harten Dep).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Defaults — bewusst für eine KMU-Lokalo-Default-Situation gesetzt.
_DEFAULTS = {
    "OLLAMA_URL": "http://localhost:11434",
    "OLLAMA_EMBED_MODEL": "nomic-embed-text",
    "OLLAMA_CHAT_MODEL": "llama3.2:3b",
    "AUDIT_DIR": "./.audit",
    "RAG_TOP_K": "3",
    "RAG_MIN_SCORE": "0.12",
}


@dataclass
class Config:
    """Ein geöffne Config aus OS.environ + optional .env + Defaults.

    Zugang per Attribut: `cfg.ollama_url`. Unbekannte Keys werfen
    `KeyError` (fail-closed).
    """
    data: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.data, dict):
            raise TypeError("Config.data muss ein dict sein")
        # Defaults einfüllen, nur wenn im Dict nicht gesetzt
        for k, v in _DEFAULTS.items():
            self.data.setdefault(k, v)

    def __getattr__(self, name: str) -> str:
        # dataclass __getattr__ only called when normal lookup fails
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self.data[name.upper()].__str__()
        except KeyError:
            # auch lowercase versuchen
            try:
                return self.data[name].__str__()
            except KeyError:
                raise AttributeError(
                    f"Config: '{name}' nicht gesetzt. "
                    f"Setze es in OS.environ oder pass es an load_config()."
                ) from None

    def get(self, key: str, default: str | None = None) -> str | None:
        """Soft-Lookup: gibt `default` zurück wenn nicht gesetzt."""
        return self.data.get(key.upper(), self.data.get(key, default))

    def require(self, key: str) -> str:
        """Fail-closed: raise wenn nicht gesetzt."""
        v = self.get(key)
        if v is None or v == "":
            raise KeyError(
                f"Config: '{key}' ist nötig, aber nicht gesetzt. "
                f"Exportiere es in OS.environ. (Kein Default, weil Secret.)"
            )
        return v  # type: ignore[return-value]

    def as_env(self) -> dict[str, str]:
        return dict(self.data)


def _load_dotenv(path: Path) -> dict[str, str]:
    """Minimal .env-Parser (kein python-dotenv). Nur KEY=VALUE,
    Kommentar mit #. Fallback ist leer. Fail-soft bei Parse-Fehler."""
    out: dict[str, str] = {}
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k:
                out[k] = v
    except OSError:
        pass
    return out


def load_config(*, dotenv_path: str | os.PathLike[str] | None = None,
                auto_dotenv: bool = True) -> Config:
    """Lädt Config aus:
      1. Defaults
      2. .env wenn auto_dotenv=True und GW_DOTENV=1 oder
         .env liegt neben CWD
      3. OS.environ (hat immer Vorrang)
    """
    if auto_dotenv and dotenv_path is None:
        candidate = Path.cwd() / ".env"
        if os.environ.get("GW_DOTENV") == "1" and candidate.exists():
            dotenv_path = candidate
    out = {k: v for k, v in _DEFAULTS.items()}
    if dotenv_path is not None:
        p = Path(dotenv_path)
        if p.exists():
            env = _load_dotenv(p)
            out.update(env)
    # OS.environ hat Vorrang
    for k in _DEFAULTS:
        if k in os.environ:
            out[k] = os.environ[k]
    # Auch nicht-Default-Keys aus .env mitnehmen, wenn vorhanden
    if dotenv_path is not None:
        out.update(_load_dotenv(Path(dotenv_path)))
    for k, v in os.environ.items():
        # gross-gehaltene Keys mitnehmen, nur wenn nicht schon aus .env
        if "_" in k or k.isupper():
            out.setdefault(k, v)
    return Config(data=out)
