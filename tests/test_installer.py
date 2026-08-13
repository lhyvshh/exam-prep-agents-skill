from __future__ import annotations

from pathlib import Path

import pytest
from scripts.install_skill import install_skill


@pytest.mark.parametrize(
    ("target", "relative_destination"),
    [
        ("codex", ".codex/skills/producing-exam-prep-packages"),
        ("claude", ".claude/skills/producing-exam-prep-packages"),
        ("gemini", ".gemini/skills/producing-exam-prep-packages"),
        ("copilot", ".copilot/skills/producing-exam-prep-packages"),
        ("agents", ".agents/skills/producing-exam-prep-packages"),
    ],
)
def test_installer_copies_complete_skill_without_local_caches(
    tmp_path: Path,
    target: str,
    relative_destination: str,
) -> None:
    source = Path(__file__).parents[1] / "skills" / "producing-exam-prep-packages"
    (source / ".venv").mkdir(exist_ok=True)

    destination = install_skill(target, home=tmp_path)

    assert destination == tmp_path / relative_destination
    assert (destination / "SKILL.md").is_file()
    assert (destination / "scripts" / "run.py").is_file()
    assert (destination / "assets" / "question_quality_classifier.pt").is_file()
    assert (destination / "assets" / "question_quality_classifier.json").is_file()
    assert not (destination / ".venv").exists()
    assert not any(path.name == "__pycache__" for path in destination.rglob("*"))


def test_installer_refuses_to_replace_existing_skill_without_force(tmp_path: Path) -> None:
    destination = tmp_path / ".codex" / "skills" / "producing-exam-prep-packages"
    destination.mkdir(parents=True)

    with pytest.raises(FileExistsError):
        install_skill("codex", home=tmp_path)
