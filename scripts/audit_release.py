"""Fail when tracked release files contain private artifacts, local paths, or likely secrets."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).parents[1]
FORBIDDEN_SUFFIXES: Final = {".db", ".docx", ".epub", ".log", ".pdf", ".sqlite", ".zip"}
FORBIDDEN_NAMES: Final = {".env", ".env.local", ".env.production"}
PATTERNS: Final = {
    "local_user_path": re.compile(rb"(?:/" rb"Users/|[A-Za-z]:\\Users\\)"),
    "openai_key": re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "anthropic_key": re.compile(rb"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    "github_token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "google_key": re.compile(rb"\bAIza[A-Za-z0-9_-]{30,}\b"),
    "slack_token": re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def tracked_files() -> tuple[Path, ...]:
    """Return repository files selected for publication."""
    git = shutil.which("git")
    if git is None:
        msg = "git is required for the release audit"
        raise RuntimeError(msg)
    completed = subprocess.run(  # noqa: S603 - executable resolved from the local PATH
        [git, "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return tuple(ROOT / raw.decode() for raw in completed.stdout.split(b"\0") if raw)


def audit() -> tuple[str, ...]:
    """Return actionable privacy or secret findings."""
    findings: list[str] = []
    for path in tracked_files():
        relative = path.relative_to(ROOT)
        if path.name in FORBIDDEN_NAMES or path.suffix.casefold() in FORBIDDEN_SUFFIXES:
            findings.append(f"forbidden_artifact: {relative}")
            continue
        if not path.is_file():
            continue
        content = path.read_bytes()
        findings.extend(
            f"{name}: {relative}" for name, pattern in PATTERNS.items() if pattern.search(content)
        )
    return tuple(findings)


def main() -> None:
    """Run the release audit as a command-line quality gate."""
    findings = audit()
    if findings:
        print("Release audit failed:")
        for finding in findings:
            print(f"- {finding}")
        raise SystemExit(1)
    print("Release audit passed: no private artifacts, local user paths, or likely secrets found.")


if __name__ == "__main__":
    main()
