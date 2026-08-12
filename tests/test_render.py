from __future__ import annotations

import json

from exam_prep_skill.models import (
    LearnerExam,
    LearnerExamQuestion,
    LearnerFlashcard,
    LearnerFlashcardDeck,
)
from exam_prep_skill.render import render_flashcards, render_mock_exam


def _deck() -> LearnerFlashcardDeck:
    return LearnerFlashcardDeck(
        package_id="deck-1",
        title="Risk Foundations",
        cards=(
            LearnerFlashcard(
                card_id="c1",
                objective_id="lo-1a",
                objective_code="LO 1.a",
                objective_title="Explain expected and unexpected loss",
                module_title="Measuring Risk",
                prompt="What is expected loss?",
                answer="The probability-weighted average loss across all possible outcomes.",
                slot="definition",
                difficulty="medium",
                source_pages=(2,),
            ),
            LearnerFlashcard(
                card_id="c2",
                objective_id="lo-1b",
                objective_code="LO 1.b",
                objective_title="Interpret loss dispersion",
                module_title="Measuring Risk",
                prompt="What does unexpected loss capture?",
                answer="Dispersion around expected loss rather than the probability-weighted mean.",
                slot="interpretation",
                difficulty="medium",
                source_pages=(3,),
            ),
        ),
    )


def _exam() -> LearnerExam:
    return LearnerExam(
        package_id="exam-1",
        title="Risk Foundations Mock Exam",
        duration_minutes=90,
        questions=(
            LearnerExamQuestion(
                question_id="q1",
                position=1,
                prompt="Which method correctly computes expected loss across several outcomes?",
                choices=(
                    "Weight each possible loss by its probability",
                    "Use only the largest loss",
                    "Use the loss standard deviation",
                    "Ignore outcomes below fifty percent probability",
                ),
                correct_choice="A",
                explanation=(
                    "Choice A is correct because expected loss is probability weighted. "
                    "The other choices omit valid outcomes or substitute dispersion."
                ),
                objective_code="LO 1.a",
                objective_title="Explain expected and unexpected loss",
                source_pages=(2,),
                difficulty="exam",
            ),
        ),
    )


def test_flashcard_html_contains_accessible_multi_select_and_direct_navigation() -> None:
    html = render_flashcards(_deck())

    assert 'data-app="flashcards"' in html
    assert "objective-filter" in html
    assert "Select visible" in html
    assert "card-queue" in html
    assert 'data-action="previous"' in html
    assert 'data-action="next"' in html
    assert 'aria-live="polite"' in html
    assert "localStorage" in html
    assert "exportProgress" in html
    assert "import-progress" in html
    assert "fetch(" not in html
    assert "http://" not in html
    assert "https://" not in html


def test_flashcard_payload_escapes_script_terminator() -> None:
    deck = _deck().model_copy(
        update={"title": "Risk </script><script>alert('x')</script> Foundations"}
    )

    html = render_flashcards(deck)

    assert "</script><script>alert" not in html
    assert "\\u003c\\/script\\u003e" in html


def test_mock_exam_html_contains_timer_grading_and_explanations() -> None:
    html = render_mock_exam(_exam())

    assert 'data-app="mock-exam"' in html
    assert "exam-timer" in html
    assert "question-navigator" in html
    assert "Flag question" in html
    assert "Submit exam" in html
    assert "answer-explanation" in html
    assert "source-pages" in html
    assert "localStorage" in html
    assert "fetch(" not in html


def test_embedded_payload_is_valid_json() -> None:
    html = render_mock_exam(_exam())
    payload = html.split('<script id="package-data" type="application/json">', 1)[1].split(
        "</script>", 1
    )[0]

    parsed = json.loads(payload)

    assert parsed["package_id"] == "exam-1"
    assert parsed["questions"][0]["correct_choice"] == "A"
