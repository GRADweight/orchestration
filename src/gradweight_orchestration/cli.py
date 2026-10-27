"""Kommandozeile ``gw-orchestration``.

Befehle:

* ``sandbox-info`` – was diese Plattform isolieren kann
* ``schema MODUL:NAME`` – Werkzeug-Schema, wie es das Modell sieht
* ``audit-verify PFAD`` – Hash-Kette eines Audit-Logs prüfen
* ``approvals list|approve|reject`` – Freigaben bearbeiten (SQLite-Datei)
* ``demo`` – Kreditoren-Demo (offline mit Skript-Modell oder gegen Ollama)
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections.abc import Sequence

from . import __version__
from .approvals import ApprovalQueue
from .audit import AuditLog
from .errors import ApprovalError
from .sandbox import detect_capabilities
from .tools import Tool


def _key(env_name: str | None) -> bytes | None:
    if not env_name:
        return None
    value = os.environ.get(env_name)
    if not value:
        raise SystemExit(f"Umgebungsvariable {env_name} ist nicht gesetzt")
    return value.encode("utf-8")


def _cmd_sandbox_info(_args: argparse.Namespace) -> int:
    print(json.dumps(detect_capabilities().as_dict(), indent=2, ensure_ascii=False))
    return 0


def _cmd_schema(args: argparse.Namespace) -> int:
    module_name, _, attr = str(args.target).partition(":")
    if not attr:
        print("Erwartet MODUL:NAME, z.B. meinpaket.werkzeuge:rechnung_lesen", file=sys.stderr)
        return 2
    obj = importlib.import_module(module_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, Tool):
        print(f"{args.target} ist kein mit @tool dekoriertes Werkzeug", file=sys.stderr)
        return 2
    print(json.dumps(obj.spec(), indent=2, ensure_ascii=False))
    return 0


def _cmd_audit_verify(args: argparse.Namespace) -> int:
    result = AuditLog.verify_file(args.path, key=_key(args.key_env))
    status = "OK" if result.ok else "FEHLER"
    print(f"{status}: {result.message} ({result.count} Einträge geprüft)")
    print(f"Letzter gültiger Hash: {result.head}")
    return 0 if result.ok else 1


def _queue(args: argparse.Namespace) -> ApprovalQueue:
    audit = AuditLog(args.audit, key=_key(args.key_env)) if args.audit else None
    return ApprovalQueue(args.db, audit=audit)


def _cmd_approvals_list(args: argparse.Namespace) -> int:
    with _queue(args) as queue:
        pending = queue.pending(run_id=args.run)
        if not pending:
            print("Keine offenen Freigaben.")
        for request in pending:
            amount = f", Betrag {request.amount}" if request.amount else ""
            print(f"{request.id}  {request.tool}  Lauf {request.run_id}{amount}")
            print(f"    Grund: {request.reason}")
            print(f"    Argumente: {json.dumps(request.arguments, ensure_ascii=False)}")
    return 0


def _cmd_approvals_decide(args: argparse.Namespace) -> int:
    with _queue(args) as queue:
        try:
            if args.action == "approve":
                request = queue.approve(args.id, by=args.by, note=args.note)
            else:
                request = queue.reject(args.id, by=args.by, reason=args.reason)
        except ApprovalError as exc:
            print(f"Fehler: {exc}", file=sys.stderr)
            return 1
    print(f"{request.id}: {request.status.value} durch {request.decided_by}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gw-orchestration", description="Kleine, prüfbare Agent-Runtime für lokale LLMs."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("sandbox-info", help="zeigt, was diese Plattform isolieren kann")
    info.set_defaults(func=_cmd_sandbox_info)

    schema = sub.add_parser("schema", help="zeigt das Tool-Schema, wie es das Modell sieht")
    schema.add_argument("target", help="MODUL:NAME eines @tool-Werkzeugs")
    schema.set_defaults(func=_cmd_schema)

    verify = sub.add_parser("audit-verify", help="prüft die Hash-Kette eines Audit-Logs")
    verify.add_argument("path")
    verify.add_argument("--key-env", help="Name der Umgebungsvariable mit dem HMAC-Schlüssel")
    verify.set_defaults(func=_cmd_audit_verify)

    approvals = sub.add_parser("approvals", help="Freigaben anzeigen und entscheiden")
    actions = approvals.add_subparsers(dest="action", required=True)
    for name, helptext in (
        ("list", "offene Freigaben"),
        ("approve", "freigeben"),
        ("reject", "ablehnen"),
    ):
        action = actions.add_parser(name, help=helptext)
        action.add_argument("--db", required=True, help="SQLite-Datei der Freigaben")
        action.add_argument("--audit", help="Audit-Log, in das Entscheide geschrieben werden")
        action.add_argument("--key-env", help="Umgebungsvariable mit dem HMAC-Schlüssel")
        if name == "list":
            action.add_argument("--run", help="nur dieser Lauf (inkl. Unterläufe)")
            action.set_defaults(func=_cmd_approvals_list)
            continue
        action.add_argument("id")
        action.add_argument("--by", required=True, help="entscheidende Person")
        if name == "approve":
            action.add_argument("--note", default="")
        else:
            action.add_argument("--reason", required=True)
        action.set_defaults(func=_cmd_approvals_decide)

    sub.add_parser(
        "demo", help="Kreditoren-Demo (Optionen: gw-orchestration demo -h)", add_help=False
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["demo"]:  # eigene Optionen; argparse.REMAINDER schluckt "--x" nicht
        from .demo import main as demo_main

        return demo_main(arguments[1:])
    args = build_parser().parse_args(arguments)
    code: int = args.func(args)
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
