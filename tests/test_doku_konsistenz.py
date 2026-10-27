"""Doku gegen Code: README-Beispiele laufen, Tabellen und Auszüge stimmen, Hygiene.

Prüft nur Allgemeines (keine Liste interner Begriffe – die gehört nicht in ein
öffentliches Repo): lokale Pfade, E-Mail-Adressen, Links auf GRADweight-Repos.
"""

from __future__ import annotations

import contextlib
import io
import re
import subprocess
from pathlib import Path

import pytest

import gradweight_orchestration
from gradweight_orchestration import Policy, __version__

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
FENCE = re.compile(r"^```([^\n]*)\n(.*?)^```", re.MULTILINE | re.DOTALL)
PUBLIC_REPOS = {
    "guardrails",
    "autonomy",
    "finops",
    "orchestration",
    "rag",
    "shared-lib",
    "gradweight",
    "contactcenter",
    "platforms",
}
RESERVED_MAIL_DOMAINS = ("example.com", "example.org", "example.net")  # RFC 2606
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".txt", ".cfg", ".ini", ".json"}
SKIP_DIRS = {
    ".git",
    ".venv",
    "dist",
    "build",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".hypothesis",
    "htmlcov",
}


def blocks(info: str) -> list[str]:
    return [body for lang, body in FENCE.findall(README) if lang.strip() == info]


def project_files() -> list[Path]:
    try:
        listed = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.splitlines()
        files = [ROOT / name for name in listed]
    except (OSError, subprocess.CalledProcessError):  # z.B. im sdist ohne Git
        files = [
            p
            for p in ROOT.rglob("*")
            if p.is_file()
            and not set(p.relative_to(ROOT).parts) & SKIP_DIRS
            and not p.relative_to(ROOT).parts[0].startswith(".venv")
        ]
    return [p for p in files if p.suffix in TEXT_SUFFIXES and p.exists()]


def test_readme_python_blocks_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Alle ```python-Blöcke laufen nacheinander in einem Namensraum."""
    code = blocks("python")
    assert len(code) >= 2
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {"__name__": "readme"}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        for block in code:
            exec(compile(block, "README.md", "exec"), namespace)  # noqa: S102 - eigene Doku
    assert "CHF 1'081.00 über Mandatslimit CHF 500.00" in output.getvalue()


def test_policy_table_matches_code() -> None:
    assert Policy.matrix_markdown() in README


def test_demo_excerpt_matches_real_output(tmp_path: Path) -> None:
    from gradweight_orchestration.demo import kreditoren_demo
    from gradweight_orchestration.demo.skript import kreditoren_skript

    (excerpt,) = blocks("text demo")
    result = kreditoren_demo(kreditoren_skript(), ausgabe_dir=tmp_path, schreiben=lambda _: None)
    actual = "\n".join(result.zeilen)
    for line in excerpt.strip("\n").splitlines():
        assert line in actual, f"README-Zeile nicht in der Demo-Ausgabe: {line!r}"
    result.freigaben.close()


def test_referenced_tests_exist() -> None:
    names = set(re.findall(r"`(test_[a-z0-9_]+)`", README))
    assert names
    source = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "tests").glob("test_*.py"))
    missing = sorted(n for n in names if f"def {n}(" not in source)
    assert missing == []


def test_versions_are_consistent() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    released = re.findall(r"^## \[(\d+\.\d+\.\d+)\]", changelog, re.MULTILINE)
    assert released[0] == __version__
    assert gradweight_orchestration.__version__ == __version__


def test_readme_has_no_test_counts_or_buzzwords() -> None:
    assert not re.search(r"\b\d+\s+(Tests|passed|grün)\b", README)
    for word in ("Production-ready", "produktionsreif", "Wow", "genial"):
        assert word not in README


def test_documentation_language() -> None:
    for name in ("README.md", "CHANGELOG.md", "CONTRIBUTING.md", "SECURITY.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "ß" not in text, name  # Schweizer Hochdeutsch
        assert "!!" not in text, name
    assert "## In English" in README
    assert "## Grenzen" in README


def test_no_local_paths_and_only_allowed_contacts() -> None:
    path_pattern = re.compile(r"/(?:Users|home)/|(?<![\w.])~/")
    mail_pattern = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
    link_pattern = re.compile(r"github\.com/GRADweight/([\w.-]+)")
    problems: list[str] = []
    for path in project_files():
        if path.resolve() == Path(__file__).resolve():
            continue  # diese Datei enthält die Suchmuster selbst
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(ROOT)
        if path_pattern.search(text):
            problems.append(f"{rel}: lokaler Pfad")
        for mail in mail_pattern.findall(text):
            domain = mail.rsplit("@", 1)[1].lower()
            if mail != "mail@gradweight.ch" and not domain.endswith(RESERVED_MAIL_DOMAINS):
                problems.append(f"{rel}: E-Mail {mail}")
        for repo in link_pattern.findall(text):
            if repo.removesuffix(".git") not in PUBLIC_REPOS:
                problems.append(f"{rel}: Link auf {repo}")
    assert problems == []


def test_license_is_plain_mit() -> None:
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("MIT License\n\nCopyright (c) 2026 GRADweight GmbH\n")
    assert text.rstrip().endswith("OTHER DEALINGS IN THE\nSOFTWARE.")
