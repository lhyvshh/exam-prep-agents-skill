from __future__ import annotations

import json
from pathlib import Path

from exam_prep_skill.extraction import (
    CurriculumRecord,
    SourceExamRecord,
    parse_book_pages,
    parse_exam_text,
    parse_page_fixture,
)
from exam_prep_skill.models import CandidateSubmission, WorkItemStatus
from exam_prep_skill.quality import QualityGate
from exam_prep_skill.queue import (
    CARD_SLOTS,
    QueueStore,
    build_exam_items,
    build_flashcard_items,
)

FIXTURES = Path(__file__).parent / "fixtures"
CHECKPOINT = (
    Path(__file__).parents[1]
    / "skills"
    / "producing-exam-prep-packages"
    / "assets"
    / "question_quality_classifier.pt"
)


def _curriculum() -> CurriculumRecord:
    pages = parse_page_fixture((FIXTURES / "frm_style_book.txt").read_text())
    return parse_book_pages("Synthetic FRM Book", pages, preset="frm-part-1")


def _exam() -> SourceExamRecord:
    return parse_exam_text("Synthetic Exam", (FIXTURES / "source_exam.txt").read_text())


def _good_question(item_id: str, prompt: str) -> CandidateSubmission:
    payload = {
        "item_id": item_id,
        "question_id": f"generated-{item_id}",
        "prompt": prompt,
        "choices": [
            "Use the probability-weighted loss across all stated outcomes",
            "Use only the largest loss in the scenario",
            "Ignore outcomes whose probability is below one-half",
            "Subtract the standard deviation from the largest loss",
        ],
        "correct_choice": "A",
        "explanation": (
            "Choice A is correct because expected loss weights every stated loss by its "
            "probability. The other choices omit outcomes or confuse expected loss with dispersion."
        ),
        "verification": (
            "Recomputed from the stated outcomes and checked against the cited definition."
        ),
        "source_pages": [1],
        "objective_code": "LO 1.a",
        "question_type": "conceptual",
        "difficulty": "exam",
    }
    return CandidateSubmission(item_id=item_id, payload_json=json.dumps(payload))


def test_flashcard_queue_allocates_ten_distinct_slots_per_objective() -> None:
    items = build_flashcard_items(_curriculum(), target_per_objective=10)

    assert len(items) == 10
    assert tuple(item.slot for item in items) == CARD_SLOTS
    assert all(item.objective_id == items[0].objective_id for item in items)


def test_exam_queue_preserves_one_for_one_blueprint_order() -> None:
    exam = _exam()

    items = build_exam_items(exam)

    assert [item.blueprint_position for item in items] == [1, 2]
    blueprints = [json.loads(item.blueprint_json or "{}") for item in items]
    assert [len(blueprint["choices"]) for blueprint in blueprints] == [4, 4]
    assert blueprints[1]["question_type"] == "calculation"


def test_rejected_candidate_remains_pending_for_targeted_retry(tmp_path: Path) -> None:
    item = build_exam_items(_exam())[0]
    store = QueueStore.create(tmp_path, (item,), QualityGate(CHECKPOINT))
    weak = _good_question(item.item_id, "Which statement correctly applies there?")

    result = store.submit(weak)

    assert not result.accepted
    assert "vague_prompt" in {issue.code for issue in result.issues}
    assert store.next_item() is not None
    assert store.next_item().attempts == 1
    assert not list((tmp_path / "accepted").glob("*.json"))


def test_queue_accepts_grounded_question_with_pytorch_provenance(tmp_path: Path) -> None:
    item = build_exam_items(_exam())[0]
    store = QueueStore.create(tmp_path, (item,), QualityGate(CHECKPOINT))
    candidate = _good_question(
        item.item_id,
        "A risk manager has several possible loss outcomes. Which method correctly computes "
        "expected loss?",
    )

    result = store.submit(candidate)

    assert result.accepted
    assert result.model_source == "pytorch_checkpoint"
    assert result.checkpoint_sha256
    assert result.score is not None
    assert result.score >= 0.70
    assert store.next_item() is None
    assert store.items[0].status is WorkItemStatus.ACCEPTED


def test_queue_resume_preserves_pending_position(tmp_path: Path) -> None:
    items = build_exam_items(_exam())
    QueueStore.create(tmp_path, items, QualityGate(CHECKPOINT))

    reopened = QueueStore.open(tmp_path, QualityGate(CHECKPOINT))

    assert reopened.next_item() == items[0]
