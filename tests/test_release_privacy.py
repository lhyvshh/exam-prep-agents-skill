from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_release_audit_passes_for_tracked_repository() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "audit_release.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Release audit passed" in completed.stdout
