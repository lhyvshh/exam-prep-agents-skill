from __future__ import annotations

import json
import zipfile
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


def test_help_lists_source_preparation_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "prepare-flashcards" in result.stdout
    assert "prepare-exam" in result.stdout


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


def test_flashcard_workflow_prepares_submits_and_renders_learner_only_package(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "private-course"
    book = Path(__file__).parent / "fixtures" / "frm_style_book.txt"

    prepared = runner.invoke(
        app,
        [
            "prepare-flashcards",
            str(workspace),
            str(book),
            "--preset",
            "frm-part-1",
            "--title",
            "Synthetic FRM Cards",
            "--target-per-objective",
            "1",
        ],
    )

    assert prepared.exit_code == 0, prepared.stdout
    prepared_payload = json.loads(prepared.stdout)
    assert prepared_payload["total_items"] == 1

    next_result = runner.invoke(app, ["next", str(workspace)])
    assert next_result.exit_code == 0
    item = json.loads(next_result.stdout)["item"]
    evidence_pages = [source["page"] for source in item["evidence"]]
    payload = {
        "item_id": item["item_id"],
        "card_id": "card-001",
        "objective_id": item["objective_id"],
        "slot": item["slot"],
        "prompt": "How is expected loss calculated across possible outcomes?",
        "answer": ("Multiply each possible loss by its probability and add the weighted losses."),
        "difficulty": "exam",
        "source_pages": [evidence_pages[0]],
    }
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    submitted = runner.invoke(app, ["submit", str(workspace), str(candidate)])
    assert submitted.exit_code == 0, submitted.stdout
    assert json.loads(submitted.stdout)["accepted"] is True

    rendered = runner.invoke(app, ["render", str(workspace)])
    assert rendered.exit_code == 0, rendered.stdout
    output = Path(json.loads(rendered.stdout)["output"])
    package = Path(json.loads(rendered.stdout)["package"])
    assert output.is_file()
    assert package.is_file()
    assert output.suffix == ".html"
    assert "Synthetic FRM Cards" in output.read_text(encoding="utf-8")

    with zipfile.ZipFile(package) as archive:
        assert archive.namelist() == [output.name]
    assert not any(
        path.suffix.casefold() == ".pdf" or "validation" in path.name.casefold()
        for path in (workspace / ".exam-prep" / "output").iterdir()
    )


def test_mock_exam_workflow_preserves_blueprint_and_renders_explanations(tmp_path: Path) -> None:
    workspace = tmp_path / "private-exam"
    fixtures = Path(__file__).parent / "fixtures"
    prepared = runner.invoke(
        app,
        [
            "prepare-exam",
            str(workspace),
            str(fixtures / "frm_style_book.txt"),
            "--source-exam",
            str(fixtures / "source_exam.txt"),
            "--preset",
            "frm-part-1",
            "--title",
            "Synthetic Mock Exam",
        ],
    )
    assert prepared.exit_code == 0, prepared.stdout
    assert json.loads(prepared.stdout)["total_items"] == 2

    prompts = (
        (
            "A risk committee reviews several possible loss outcomes. Which approach correctly "
            "computes the portfolio's expected monetary loss?"
        ),
        (
            "A position has a 25% chance of losing 12 units and a 75% chance of no loss. What is "
            "the expected loss?"
        ),
    )
    for index, prompt in enumerate(prompts, start=1):
        next_result = runner.invoke(app, ["next", str(workspace)])
        item = json.loads(next_result.stdout)["item"]
        payload = {
            "item_id": item["item_id"],
            "question_id": f"generated-{index}",
            "prompt": prompt,
            "choices": [
                "Weight every stated loss by its probability",
                "Use only the largest possible loss",
                "Ignore outcomes below one-half probability",
                "Replace expected loss with the loss dispersion",
            ],
            "correct_choice": "A",
            "explanation": (
                "Choice A is correct because expected loss weights every possible outcome by "
                "its probability. The other choices omit outcomes or confuse the mean with "
                "dispersion."
            ),
            "verification": (
                "Recomputed from the stated probabilities and checked against the cited book "
                "definition."
            ),
            "source_pages": [item["evidence"][-1]["page"]],
            "objective_code": "LO 1.a",
            "question_type": "calculation" if index == 2 else "conceptual",
            "difficulty": "exam",
        }
        candidate = tmp_path / f"question-{index}.json"
        candidate.write_text(json.dumps(payload), encoding="utf-8")
        submitted = runner.invoke(app, ["submit", str(workspace), str(candidate)])
        assert submitted.exit_code == 0, submitted.stdout
        assert json.loads(submitted.stdout)["accepted"] is True

    rendered = runner.invoke(app, ["render", str(workspace)])
    assert rendered.exit_code == 0, rendered.stdout
    html = Path(json.loads(rendered.stdout)["output"]).read_text(encoding="utf-8")
    assert "Synthetic Mock Exam" in html
    assert "answer-explanation" in html
    assert "source-pages" in html
