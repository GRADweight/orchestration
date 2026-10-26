"""Kommandozeile ``gw-orchestration``."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Sequence

from . import __version__
from .sandbox import detect_capabilities
from .tools import Tool


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
