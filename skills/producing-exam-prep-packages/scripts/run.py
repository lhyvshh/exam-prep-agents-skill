"""Portable entry point bundled inside the agent skill folder."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    """Run the bundled Typer application."""
    from exam_prep_skill.cli import app  # noqa: PLC0415

    app()


if __name__ == "__main__":
    main()
