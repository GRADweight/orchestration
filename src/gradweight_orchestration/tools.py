"""Werkzeuge: ``@tool``-Decorator, Seiteneffekt-Klassen und Registry.

Ein Werkzeug ist eine normale, typannotierte Python-Funktion. Der Decorator
leitet daraus das JSON-Schema für das Modell ab, validiert Argumente strikt
und hält die Governance-Angaben fest (Seiteneffekt, Betrag, Isolation)::

    @tool(side_effect=SideEffect.READ_ONLY)
    def wechselkurs(waehrung: Literal["EUR", "USD"]) -> float:
        '''Liefert den Referenzkurs einer Währung in CHF.'''
        ...

Die dekorierte Funktion bleibt direkt aufrufbar (``wechselkurs("EUR")``).
"""

from __future__ import annotations

import enum
import re
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Generic, ParamSpec, TypeVar, overload

from ._json import JSONObject, JSONValue
from .errors import ArgumentValidationError, ToolDefinitionError, ValidationIssue
from .sandbox import SandboxLimits, TargetRef
from .schema import function_schema
from .validation import from_jsonable, validate

__all__ = ["Isolation", "SideEffect", "Tool", "ToolRegistry", "tool"]

P = ParamSpec("P")
R = TypeVar("R")

_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class SideEffect(enum.StrEnum):
    """Wirkung eines Werkzeugs nach aussen – Grundlage der Freigabe-Policy."""

    READ_ONLY = "read_only"
    """Liest nur; keine Zustandsänderung ausserhalb des Prozesses."""
    REVERSIBLE = "reversible_write"
    """Schreibt, lässt sich aber vollständig rückgängig machen (z.B. Entwurf, Storno)."""
    IRREVERSIBLE = "irreversible"
    """Nicht rückholbare Wirkung (z.B. E-Mail an Kunden, Löschung)."""
    MONEY = "money_moving"
    """Bewegt Geld oder verpflichtet zu Zahlungen."""

    @property
    def rank(self) -> int:
        return _RANK[self]


_RANK = {
    SideEffect.READ_ONLY: 0,
    SideEffect.REVERSIBLE: 1,
    SideEffect.IRREVERSIBLE: 2,
    SideEffect.MONEY: 3,
}


class Isolation(enum.StrEnum):
    """Wie ein Werkzeug ausgeführt wird."""

    THREAD = "thread"
    """Im eigenen Prozess, in einem Thread mit Zeitlimit. Für vertrauenswürdigen Code."""
    PROCESS = "process"
    """In einem Kindprozess mit Ressourcenlimits und hartem Abbruch."""


@dataclass(frozen=True, eq=False)
class Tool(Generic[P, R]):
    """Ein registrierbares Werkzeug. Instanzen entstehen über :func:`tool`."""

    fn: Callable[P, R]
    name: str
    description: str
    parameters: JSONObject
    param_types: Mapping[str, Any]
    return_type: Any
    return_schema: JSONObject | None
    side_effect: SideEffect
    timeout_s: float
    isolation: Isolation
    sandbox: SandboxLimits | None
    amount_param: str | None
    idempotent: bool
    retries: int
    target: TargetRef | None = field(default=None)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.fn(*args, **kwargs)

    def spec(self) -> JSONObject:
        """Eintrag für das ``tools``-Feld der OpenAI-kompatiblen Chat-API."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def check_arguments(self, raw: Mapping[str, JSONValue] | None) -> list[ValidationIssue]:
        """Validiert Roh-Argumente (JSON) gegen das Schema; leer = gültig."""
        if raw is None:
            return [ValidationIssue("$", "Argumente fehlen oder sind kein JSON-Objekt")]
        return validate(dict(raw), self.parameters)

    def parse_arguments(self, raw: Mapping[str, JSONValue] | None) -> dict[str, Any]:
        """Validiert und wandelt in Python-Typen um; wirft :class:`ArgumentValidationError`."""
        issues = self.check_arguments(raw)
        if issues:
            raise ArgumentValidationError(self.name, issues)
        assert raw is not None  # noqa: S101 - durch check_arguments garantiert
        try:
            return {key: from_jsonable(self.param_types[key], value) for key, value in raw.items()}
        except (TypeError, ValueError) as exc:  # pragma: no cover - Schema und Typ widersprüchlich
            raise ArgumentValidationError(self.name, [ValidationIssue("$", str(exc))]) from exc

    def amount(self, arguments: Mapping[str, Any]) -> Decimal | None:
        """Betrag des Aufrufs (für Geld-Limits) oder ``None``, wenn unbekannt."""
        if self.amount_param is None:
            return None
        value = arguments.get(self.amount_param)
        if value is None or isinstance(value, bool):
            return None
        try:
            amount = Decimal(str(value))
        except (InvalidOperation, ValueError):
            return None
        return amount if amount.is_finite() else None


def _target_for(fn: Callable[..., Any]) -> TargetRef:
    qualname = getattr(fn, "__qualname__", "")
    module = getattr(fn, "__module__", "")
    if not qualname or "<locals>" in qualname or "<lambda>" in qualname:
        raise ToolDefinitionError(
            f"{qualname or fn!r}: Isolation.PROCESS braucht eine Funktion auf Modulebene "
            "(keine Closures oder Lambdas), damit der Kindprozess sie importieren kann"
        )
    main_path: str | None = None
    if module == "__main__":
        main_file = getattr(sys.modules.get("__main__"), "__file__", None)
        if not main_file:
            raise ToolDefinitionError(
                f"{qualname}: Isolation.PROCESS in einer interaktiven Sitzung nicht möglich"
            )
        main_path = str(main_file)
    return TargetRef(module=module, qualname=qualname, main_path=main_path)


def _build(
    fn: Callable[P, R],
    *,
    name: str | None,
    description: str | None,
    side_effect: SideEffect,
    timeout_s: float,
    isolation: Isolation,
    sandbox: SandboxLimits | None,
    amount_param: str | None,
    idempotent: bool,
    retries: int,
) -> Tool[P, R]:
    schema = function_schema(fn, name=name, description=description)
    where = schema.name
    if not _NAME.match(schema.name):
        raise ToolDefinitionError(f"Werkzeugname {schema.name!r} muss [A-Za-z0-9_-]{{1,64}} sein")
    if timeout_s <= 0:
        raise ToolDefinitionError(f"{where}: timeout_s muss > 0 sein")
    if retries < 0:
        raise ToolDefinitionError(f"{where}: retries darf nicht negativ sein")
    if retries > 0 and not (idempotent or side_effect is SideEffect.READ_ONLY):
        raise ToolDefinitionError(
            f"{where}: Wiederholungen nur für lesende oder idempotente Werkzeuge "
            "(sonst droht doppelte Wirkung)"
        )
    if isolation is Isolation.THREAD and sandbox is not None:
        raise ToolDefinitionError(
            f"{where}: sandbox-Limits wirken nur mit Isolation.PROCESS; im Thread sind sie "
            "nicht durchsetzbar"
        )
    if amount_param is not None:
        prop = schema.parameters["properties"]
        assert isinstance(prop, dict)  # noqa: S101 - von function_schema garantiert
        if amount_param not in prop:
            raise ToolDefinitionError(f"{where}: amount_param {amount_param!r} ist kein Parameter")
        entry = prop[amount_param]
        kind = entry.get("type") if isinstance(entry, dict) else None
        kinds = set(kind) if isinstance(kind, list) else {kind}
        if not kinds & {"number", "integer"}:
            raise ToolDefinitionError(f"{where}: amount_param {amount_param!r} muss eine Zahl sein")
    target: TargetRef | None = None
    limits = sandbox
    if isolation is Isolation.PROCESS:
        target = _target_for(fn)
        limits = SandboxLimits() if sandbox is None else sandbox
    return Tool(
        fn=fn,
        name=schema.name,
        description=schema.description,
        parameters=schema.parameters,
        param_types=schema.param_types,
        return_type=schema.return_type,
        return_schema=schema.return_schema,
        side_effect=side_effect,
        timeout_s=float(timeout_s),
        isolation=isolation,
        sandbox=limits,
        amount_param=amount_param,
        idempotent=idempotent,
        retries=retries,
        target=target,
    )


@overload
def tool(fn: Callable[P, R], /) -> Tool[P, R]: ...


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    side_effect: SideEffect = SideEffect.IRREVERSIBLE,
    timeout_s: float = 30.0,
    isolation: Isolation = Isolation.THREAD,
    sandbox: SandboxLimits | None = None,
    amount_param: str | None = None,
    idempotent: bool = False,
    retries: int = 0,
) -> Callable[[Callable[P, R]], Tool[P, R]]: ...


def tool(
    fn: Callable[P, R] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    side_effect: SideEffect = SideEffect.IRREVERSIBLE,
    timeout_s: float = 30.0,
    isolation: Isolation = Isolation.THREAD,
    sandbox: SandboxLimits | None = None,
    amount_param: str | None = None,
    idempotent: bool = False,
    retries: int = 0,
) -> Tool[P, R] | Callable[[Callable[P, R]], Tool[P, R]]:
    """Macht aus einer typannotierten Funktion ein :class:`Tool`.

    Args:
        name: Werkzeugname für das Modell (Standard: Funktionsname).
        description: Beschreibung (Standard: Docstring ohne Abschnitte).
        side_effect: Wirkung nach aussen. Standard ist bewusst
            ``IRREVERSIBLE`` (fail-closed): wer nichts angibt, braucht je nach
            Autonomiegrad eine Freigabe.
        timeout_s: Hartes Zeitlimit pro Aufruf.
        isolation: ``THREAD`` (vertrauenswürdig) oder ``PROCESS`` (Kindprozess).
        sandbox: Limits für ``PROCESS`` (Standard: :class:`SandboxLimits`).
        amount_param: Name des Betrags-Parameters (für Geld-Limits der Policy).
        idempotent: Mehrfaches Ausführen hat dieselbe Wirkung wie einmaliges.
        retries: Wiederholungen bei :class:`TransientToolError` (nur lesend/idempotent).
    """

    def decorate(func: Callable[P, R]) -> Tool[P, R]:
        return _build(
            func,
            name=name,
            description=description,
            side_effect=side_effect,
            timeout_s=timeout_s,
            isolation=isolation,
            sandbox=sandbox,
            amount_param=amount_param,
            idempotent=idempotent,
            retries=retries,
        )

    if fn is not None:
        return decorate(fn)
    return decorate


class ToolRegistry:
    """Menge eindeutig benannter Werkzeuge."""

    def __init__(self, tools: Iterable[Tool[..., Any]] = ()) -> None:
        self._tools: dict[str, Tool[..., Any]] = {}
        for item in tools:
            self.register(item)

    def register(self, item: Tool[..., Any]) -> None:
        if not isinstance(item, Tool):
            raise ToolDefinitionError(f"{item!r} ist kein Tool; mit @tool dekorieren")
        if item.name in self._tools:
            raise ToolDefinitionError(f"Werkzeug {item.name!r} ist bereits registriert")
        self._tools[item.name] = item

    def find(self, name: str) -> Tool[..., Any] | None:
        return self._tools.get(name)

    def __getitem__(self, name: str) -> Tool[..., Any]:
        return self._tools[name]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool[..., Any]]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[JSONValue]:
        return [t.spec() for t in self._tools.values()]
