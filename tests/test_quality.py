from __future__ import annotations

import json
from pathlib import Path

from exam_prep_skill.extraction import parse_exam_text
from exam_prep_skill.models import CandidateSubmission, WorkItem
from exam_prep_skill.quality import QualityGate
from exam_prep_skill.queue import QueueStore, build_exam_items

FIXTURES = Path(__file__).parent / "fixtures"
CHECKPOINT = (
    Path(__file__).parents[1]
    / "skills"
    / "producing-exam-prep-packages"
    / "assets"
    / "question_quality_classifier.pt"
)


def _item() -> WorkItem:
    exam = parse_exam_text("Synthetic Exam", (FIXTURES / "source_exam.txt").read_text())
    return build_exam_items(exam)[0]


def _submission(
    item_id: str, *, prompt: str, choices: list[str] | None = None
) -> CandidateSubmission:
    payload = {
        "item_id": item_id,
        "question_id": f"q-{abs(hash(prompt))}",
        "prompt": prompt,
        "choices": choices
        or [
            "Weight every possible loss by its probability",
            "Select only the most severe observed loss",
            "Average the two largest losses without probabilities",
            "Treat dispersion as the expected monetary loss",
        ],
        "correct_choice": "A",
        "explanation": (
            "Choice A applies the probability-weighted definition. Each alternative omits "
            "probability information or incorrectly substitutes a dispersion measure."
        ),
        "verification": "Checked against the cited expected-loss definition on the source page.",
        "source_pages": [1],
        "objective_code": "LO 1.a",
        "question_type": "conceptual",
        "difficulty": "exam",
    }
    return CandidateSubmission(item_id=item_id, payload_json=json.dumps(payload))


def test_gate_rejects_wrong_choice_count_before_pytorch(tmp_path: Path) -> None:
    item = _item()
    store = QueueStore.create(tmp_path, (item,), QualityGate(CHECKPOINT))
    candidate = _submission(
        item.item_id, prompt="Which method correctly measures expected loss?", choices=["A", "B"]
    )

    result = store.submit(candidate)

    assert not result.accepted
    assert result.score is None
    assert "choice_count" in {issue.code for issue in result.issues}


def test_gate_rejects_source_question_copy(tmp_path: Path) -> None:
    item = _item()
    store = QueueStore.create(tmp_path, (item,), QualityGate(CHECKPOINT))
    candidate = _submission(item.item_id, prompt="Which statement best describes expected loss?")

    result = store.submit(candidate)

    assert not result.accepted
    assert "source_copy" in {issue.code for issue in result.issues}


def test_gate_rejects_duplicate_across_generated_exam_registry(tmp_path: Path) -> None:
    first, second = build_exam_items(
        parse_exam_text("Synthetic Exam", (FIXTURES / "source_exam.txt").read_text())
    )
    store = QueueStore.create(tmp_path, (first, second), QualityGate(CHECKPOINT))
    prompt = "A risk officer compares several outcomes. Which method produces expected loss?"

    assert store.submit(_submission(first.item_id, prompt=prompt)).accepted
    duplicate = _submission(second.item_id, prompt=prompt)
    result = store.submit(duplicate)

    assert not result.accepted
    assert "duplicate_question" in {issue.code for issue in result.issues}


def test_gate_exhausts_item_without_lowering_threshold(tmp_path: Path) -> None:
    item = _item()
    store = QueueStore.create(tmp_path, (item,), QualityGate(CHECKPOINT), max_attempts=3)
    weak = _submission(item.item_id, prompt="Which applies there?")

    for _ in range(3):
        result = store.submit(weak)

    assert not result.accepted
    assert store.next_item() is None
    assert store.items[0].status.value == "exhausted"
