from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]
SKILL = ROOT / "skills" / "producing-exam-prep-packages"


def test_skill_folder_is_self_contained_for_standard_agent_installers() -> None:
    required = (
        SKILL / "SKILL.md",
        SKILL / "agents" / "openai.yaml",
        SKILL / "pyproject.toml",
        SKILL / "scripts" / "run.py",
        SKILL / "scripts" / "exam_prep_skill" / "cli.py",
        SKILL / "assets" / "question_quality_classifier.pt",
    )

    assert all(path.is_file() for path in required)


def test_skill_metadata_names_the_portable_workflow_without_host_lock_in() -> None:
    content = (SKILL / "SKILL.md").read_text(encoding="utf-8")

    assert "name: producing-exam-prep-packages" in content
    assert "Codex" not in content.split("---", 2)[1]
    assert "Claude" not in content.split("---", 2)[1]
    assert "Gemini" not in content.split("---", 2)[1]
    assert "prepare-flashcards" in content
    assert "prepare-exam" in content
    assert "next" in content
    assert "submit" in content
    assert "render" in content


def test_skill_runtime_keeps_source_material_outside_the_installed_folder() -> None:
    forbidden_suffixes = {".pdf", ".docx", ".epub"}

    assert not any(
        path.suffix.casefold() in forbidden_suffixes for path in SKILL.rglob("*") if path.is_file()
    )
