"""Meldet Testergebnisse aus einer JUnit-XML-Datei als GitHub-Annotationen.

Ohne Option: jeder Fehlschlag als ``::error``; Rückgabe 1, wenn es welche gibt.
Mit ``--skipped``: übersprungene Tests samt Grund als eine ``::notice`` (Nachweis, was auf
welcher Plattform nicht geprüft werden konnte).
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET


def _escape(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(text: str) -> str:
    """Eigenschaften wie title= trennen GitHub-Befehle an ':' und ','."""
    return _escape(text).replace(":", "%3A").replace(",", "%2C")


def main(path: str, *, skipped: bool, title: str) -> int:
    root = ET.parse(path).getroot()  # noqa: S314 - eigene, lokal erzeugte Datei
    failures = 0
    skips: list[str] = []
    for case in root.iter("testcase"):
        name = f"{case.get('classname', '')}.{case.get('name', '')}"
        for kind in ("failure", "error"):
            node = case.find(kind)
            if node is not None and not skipped:
                detail = (node.get("message") or "") + "\n" + (node.text or "")[-1500:]
                print(f"::error title={_escape_property(name)}::{_escape(detail.strip())}")
                failures += 1
        node = case.find("skipped")
        if node is not None:
            skips.append(f"{case.get('name', '')}: {node.get('message', '')}")
    if skipped:
        text = "\n".join(skips) if skips else "keine"
        print(f"::notice title={_escape_property(title)}::{_escape(text)}")
        return 0
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    args = sys.argv[1:]
    want_skipped = "--skipped" in args
    rest = [a for a in args if a != "--skipped"]
    raise SystemExit(
        main(
            rest[0], skipped=want_skipped, title=rest[1] if len(rest) > 1 else "Übersprungene Tests"
        )
    )
