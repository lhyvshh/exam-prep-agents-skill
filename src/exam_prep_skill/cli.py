"""JSON-first command line boundary for agent hosts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from exam_prep_skill.models import WorkspaceConfig

app = typer.Typer(
    add_completion=False,
    help="Build validated offline flashcard and mock-exam packages with your active agent.",
    no_args_is_help=True,
)
HumanOption = Annotated[bool, typer.Option("--human", help="Print concise human-readable output.")]


def _emit(payload: dict[str, str], *, human: bool) -> None:
    if human:
        typer.echo(payload.get("message", payload.get("status", "ok")))
        return
    typer.echo(json.dumps(payload, sort_keys=True))


def _config_path(workspace: Path) -> Path:
    return workspace.resolve() / ".exam-prep" / "config.json"


@app.command("init")
def init_workspace(
    workspace: Annotated[Path, typer.Argument(help="Private project workspace.")],
    preset: Annotated[str, typer.Option(help="Curriculum preset.")] = "generic",
    human: HumanOption = False,
) -> None:
    """Initialize private state outside the installed skill."""
    root = workspace.expanduser().resolve()
    state_dir = root / ".exam-prep" / "state"
    output_dir = root / ".exam-prep" / "output"
    state_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = WorkspaceConfig(
        workspace=root,
        state_dir=state_dir,
        output_dir=output_dir,
        preset=preset,
    )
    _config_path(root).write_text(config.model_dump_json(indent=2), encoding="utf-8")
    _emit(
        {
            "status": "initialized",
            "workspace": str(root),
            "message": f"Initialized private workspace at {root}",
        },
        human=human,
    )


@app.command("status")
def status(
    workspace: Annotated[Path, typer.Argument(help="Private project workspace.")],
    human: HumanOption = False,
) -> None:
    """Report resumable workspace state."""
    root = workspace.expanduser().resolve()
    config_path = _config_path(root)
    if not config_path.is_file():
        _emit(
            {
                "status": "error",
                "code": "workspace_not_initialized",
                "message": f"No exam-prep workspace at {root}",
            },
            human=human,
        )
        raise typer.Exit(code=2)
    _emit(
        {
            "status": "ready",
            "workspace": str(root),
            "message": "Workspace is initialized",
        },
        human=human,
    )


def _not_ready(command: str, human: bool) -> None:
    _emit(
        {
            "status": "error",
            "code": "workflow_not_prepared",
            "message": f"Run source preparation before '{command}'",
        },
        human=human,
    )
    raise typer.Exit(code=2)


@app.command("next")
def next_item(human: HumanOption = False) -> None:
    """Return the next bounded work item."""
    _not_ready("next", human)


@app.command("submit")
def submit(human: HumanOption = False) -> None:
    """Submit a candidate to deterministic validation."""
    _not_ready("submit", human)


@app.command("render")
def render(human: HumanOption = False) -> None:
    """Render a fully accepted learner package."""
    _not_ready("render", human)
