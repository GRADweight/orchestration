"""Meldet fehlgeschlagene Tests aus einer JUnit-XML-Datei als GitHub-Annotationen."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET


def _escape(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main(path: str) -> int:
    root = ET.parse(path).getroot()  # noqa: S314 - eigene, lokal erzeugte Datei
    count = 0
    for case in root.iter("testcase"):
        for kind in ("failure", "error"):
            node = case.find(kind)
            if node is None:
                continue
            title = f"{case.get('classname', '')}.{case.get('name', '')}"
            detail = (node.get("message") or "") + "\n" + (node.text or "")[-1500:]
            print(f"::error title={_escape(title)}::{_escape(detail.strip())}")
            count += 1
    return 0 if count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
