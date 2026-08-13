from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from exam_prep_skill.extraction import (
    parse_book_pages,
    parse_exam_text,
    parse_page_fixture,
    parse_pdf,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_frm_hierarchy_correlates_learning_objective_quiz_and_answer() -> None:
    pages = parse_page_fixture((FIXTURES / "frm_style_book.txt").read_text())

    curriculum = parse_book_pages("Synthetic FRM Book", pages, preset="frm-part-1")

    assert curriculum.preset == "frm-part-1"
    assert curriculum.domain_weights == (20, 20, 30, 30)
    assert len(curriculum.modules) == 1
    module = curriculum.modules[0]
    assert module.title == "Measuring Risk"
    assert module.start_page == 2
    assert module.objectives[0].code == "LO 1.a"
    assert module.objectives[0].title == "Explain expected loss and unexpected loss."
    assert "probability-weighted" in module.key_concepts
    assert "Which statement" in module.module_quiz
    assert "1. B." in module.answer_key
    assert curriculum.excluded_pages == (1, 4)


def test_generic_hierarchy_does_not_inherit_frm_rules() -> None:
    pages = parse_page_fixture((FIXTURES / "generic_book.txt").read_text())

    curriculum = parse_book_pages("Statistics Notes", pages)

    assert curriculum.preset == "generic"
    assert curriculum.domain_weights == ()
    assert [module.title for module in curriculum.modules] == ["Probability", "Dispersion"]
    assert curriculum.modules[0].objectives[0].title == "Calculate and interpret expected value."


def test_exam_parser_preserves_question_order_choices_and_answer_explanations() -> None:
    source = (FIXTURES / "source_exam.txt").read_text()

    exam = parse_exam_text("Synthetic Exam", source)

    assert [question.position for question in exam.questions] == [1, 2]
    assert all(len(question.choices) == 4 for question in exam.questions)
    assert exam.questions[0].objective_code == "LO 1.a"
    assert exam.questions[0].correct_choice == "B"
    assert "probability-weighted" in exam.questions[0].explanation
    assert exam.questions[1].question_type == "calculation"


def test_pdf_extraction_preserves_physical_page_numbers(tmp_path: Path) -> None:
    pdf_path = tmp_path / "book.pdf"
    script = (
        "import pymupdf as fitz,sys; d=fitz.open(); "
        "p=d.new_page(); p.insert_text((72,72),'MODULE 4.1: INTRODUCTION'); "
        "p=d.new_page(); p.insert_text((72,72),'LO 4.a: Explain the concept.'); "
        "d.save(sys.argv[1]); d.close()"
    )
    subprocess.run([sys.executable, "-c", script, str(pdf_path)], check=True)

    pages = parse_pdf(pdf_path)

    assert [page.physical_page for page in pages] == [1, 2]
    assert pages[1].text.startswith("LO 4.a")


def test_pdf_worker_uses_current_pymupdf_import_without_stdout_deprecation() -> None:
    worker = (
        Path(__file__).parents[1]
        / "skills"
        / "producing-exam-prep-packages"
        / "scripts"
        / "exam_prep_skill"
        / "pdf_worker.py"
    ).read_text(encoding="utf-8")

    assert "import pymupdf as fitz" in worker
    assert "\nimport fitz" not in worker
