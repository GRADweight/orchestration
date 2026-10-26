"""JSON-Hilfsfunktionen: Typen, kanonische Serialisierung, Umwandlung von Python-Werten.

Alles, was die Runtime an das LLM, ins Audit-Log oder in einen Subprozess gibt,
läuft durch :func:`to_jsonable` und :func:`canonical_json`. Damit sind Hashes
reproduzierbar und Werkzeug-Ergebnisse eindeutig typisiert.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, TypeAlias

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject: TypeAlias = dict[str, JSONValue]


class NotJSONableError(TypeError):
    """Ein Wert lässt sich nicht verlustfrei als JSON darstellen."""


def to_jsonable(value: Any, *, _path: str = "$") -> JSONValue:
    """Wandelt einen Python-Wert in einen JSON-Wert um.

    Unterstützt: ``None``, ``bool``, ``int``, ``float`` (endlich), ``str``,
    ``Decimal`` (als Zahl), ``date``/``datetime`` (ISO 8601), ``Enum`` (Wert),
    Dataclasses, Mappings mit String-Schlüsseln sowie Listen und Tupel.
    Alles andere löst :class:`NotJSONableError` aus – bewusst streng, damit
    typisierte Werkzeug-Ergebnisse nicht stillschweigend zu ``repr``-Text werden.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, enum.Enum):
        return to_jsonable(value.value, _path=_path)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NotJSONableError(f"{_path}: nicht-endliche Zahl {value!r}")
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise NotJSONableError(f"{_path}: nicht-endliche Zahl {value!r}")
        exponent = value.as_tuple().exponent
        integral = isinstance(exponent, int) and exponent >= 0
        return int(value) if integral else float(value)
    if isinstance(value, _dt.datetime):
        return value.isoformat()
    if isinstance(value, _dt.date):
        return value.isoformat()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: to_jsonable(getattr(value, f.name), _path=f"{_path}.{f.name}")
            for f in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        out: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise NotJSONableError(f"{_path}: Schlüssel {key!r} ist kein String")
            out[key] = to_jsonable(item, _path=f"{_path}.{key}")
        return out
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [to_jsonable(item, _path=f"{_path}[{i}]") for i, item in enumerate(value)]
    raise NotJSONableError(f"{_path}: Typ {type(value).__name__} ist nicht JSON-fähig")


def canonical_json(value: JSONValue) -> str:
    """Deterministische, kompakte JSON-Darstellung (sortierte Schlüssel, UTF-8)."""
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def sha256_hex(text: str) -> str:
    """SHA-256 eines Textes (UTF-8) als Hex-String."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dumps(value: JSONValue) -> str:
    """Lesbare, stabile JSON-Darstellung für Beobachtungen an das LLM."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def truncate(text: str, max_chars: int) -> str:
    """Kürzt Text auf ``max_chars`` Zeichen und markiert die Kürzung sichtbar."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = f" …[gekürzt, {len(text) - max_chars} Zeichen ausgelassen]"
    return text[: max(0, max_chars - len(marker))] + marker
