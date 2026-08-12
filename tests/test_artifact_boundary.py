from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from exam_prep_skill.render import write_learner_zip


def test_learner_zip_contains_only_html(tmp_path: Path) -> None:
    destination = tmp_path / "package.zip"

    write_learner_zip(destination, "flashcards.html", "<!doctype html><title>Cards</title>")

    with zipfile.ZipFile(destination) as archive:
        assert archive.namelist() == ["flashcards.html"]
        assert archive.read("flashcards.html").startswith(b"<!doctype html>")


def test_learner_zip_rejects_non_html_output_name(tmp_path: Path) -> None:
    destination = tmp_path / "package.zip"

    with pytest.raises(ValueError, match="HTML"):
        write_learner_zip(destination, "manifest.json", "{}")
