"""Install the bundled Agent Skill into a supported personal skill directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Final

SKILL_NAME: Final = "producing-exam-prep-packages"
TARGET_ROOTS: Final = {
    "codex": Path(".codex/skills"),
    "claude": Path(".claude/skills"),
    "gemini": Path(".gemini/skills"),
    "copilot": Path(".copilot/skills"),
    "agents": Path(".agents/skills"),
}


def install_skill(target: str, *, home: Path, force: bool = False) -> Path:
    """Copy the complete skill into one agent host's user-level directory."""
    if target not in TARGET_ROOTS:
        msg = f"Unsupported target: {target}"
        raise ValueError(msg)
    source = Path(__file__).parents[1] / "skills" / SKILL_NAME
    destination = home.expanduser().resolve() / TARGET_ROOTS[target] / SKILL_NAME
    if destination.exists():
        if not force:
            msg = f"Skill already exists at {destination}; pass --force to replace it"
            raise FileExistsError(msg)
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(".venv", "__pycache__", ".DS_Store", "*.pyc"),
    )
    return destination


def main() -> None:
    """Install one or all supported host targets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=(*TARGET_ROOTS, "all"),
        default="agents",
        help="Agent host to install for (default: shared ~/.agents/skills).",
    )
    parser.add_argument("--home", type=Path, default=Path.home())
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    targets = tuple(TARGET_ROOTS) if args.target == "all" else (args.target,)
    for target in targets:
        destination = install_skill(target, home=args.home, force=args.force)
        print(f"Installed {target}: {destination}")


if __name__ == "__main__":
    main()
