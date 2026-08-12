from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from exam_prep_skill.cli import app

runner = CliRunner()


def test_help_lists_portable_workflow_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "init" in result.stdout
    assert "next" in result.stdout
    assert "submit" in result.stdout
    assert "render" in result.stdout


def test_init_creates_private_state_inside_selected_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "private-course"

    result = runner.invoke(app, ["init", str(workspace)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "initialized"
    assert Path(payload["workspace"]) == workspace.resolve()
    assert (workspace / ".exam-prep" / "state").is_dir()
    assert (workspace / ".exam-prep" / "output").is_dir()
    assert not (Path.cwd() / ".exam-prep").exists()


def test_status_returns_machine_readable_error_for_unknown_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "missing"

    result = runner.invoke(app, ["status", str(workspace)])

    assert result.exit_code == 2
    assert json.loads(result.stdout) == {
        "code": "workspace_not_initialized",
        "message": f"No exam-prep workspace at {workspace.resolve()}",
        "status": "error",
    }
