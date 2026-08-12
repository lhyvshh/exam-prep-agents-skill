"""Resumable host-agent work queue."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict, Field

from exam_prep_skill.models import (
    CandidateSubmission,
    PackageKind,
    SourceRef,
    ValidationResult,
    WorkItem,
    WorkItemStatus,
)

if TYPE_CHECKING:
    from exam_prep_skill.extraction import CurriculumRecord, SourceExamRecord
    from exam_prep_skill.quality import QualityGate

CARD_SLOTS: Final = (
    "definition",
    "intuition",
    "formula",
    "interpretation",
    "application",
    "scenario",
    "comparison",
    "common-trap",
    "worked-decision",
    "synthesis",
)


class QueueState(BaseModel):
    """Immutable persisted queue state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    items: tuple[WorkItem, ...]
    registry_prompts: tuple[str, ...] = ()
    max_attempts: int = Field(default=3, ge=1)


def build_flashcard_items(
    curriculum: CurriculumRecord,
    *,
    target_per_objective: int = 10,
) -> tuple[WorkItem, ...]:
    """Create balanced card slots for every normalized learning objective."""
    slots = CARD_SLOTS[:target_per_objective]
    items: list[WorkItem] = []
    for module in curriculum.modules:
        for objective in module.objectives:
            concept_source = SourceRef(
                source_id=curriculum.source_id,
                title=module.title,
                page=module.start_page,
                excerpt=module.key_concepts or objective.title,
            )
            for slot in slots:
                item_id = f"card-{objective.objective_id}-{slot}"
                items.append(
                    WorkItem(
                        item_id=item_id,
                        kind=PackageKind.FLASHCARDS,
                        objective_id=objective.objective_id,
                        slot=slot,
                        instructions=(
                            f"Create one {slot} flashcard for {objective.code}: {objective.title}. "
                            "Use only the supplied evidence and cite its physical page."
                        ),
                        evidence=(*objective.sources, concept_source),
                    )
                )
    return tuple(items)


def build_exam_items(
    exam: SourceExamRecord,
    curriculum: CurriculumRecord | None = None,
) -> tuple[WorkItem, ...]:
    """Create one generated-question task for every source-exam position."""
    items: list[WorkItem] = []
    objective_evidence: dict[str, tuple[SourceRef, ...]] = {}
    if curriculum is not None:
        for module in curriculum.modules:
            for objective in module.objectives:
                objective_evidence[objective.code.casefold()] = (
                    *objective.sources,
                    SourceRef(
                        source_id=curriculum.source_id,
                        title=module.title,
                        page=module.start_page,
                        excerpt=module.key_concepts or objective.title,
                    ),
                )
    for question in exam.questions:
        item_id = f"exam-{exam.exam_id}-{question.position:03d}"
        source = SourceRef(
            source_id=exam.exam_id,
            title=exam.title,
            page=1,
            excerpt=(
                f"{question.prompt}\nCorrect answer: {question.correct_choice}. "
                f"{question.explanation}"
            ),
        )
        items.append(
            WorkItem(
                item_id=item_id,
                kind=PackageKind.MOCK_EXAM,
                objective_id=question.objective_code or None,
                blueprint_position=question.position,
                blueprint_json=question.model_dump_json(),
                instructions=(
                    "Generate a new question with the same objective, cognitive operation, format, "
                    "difficulty, and choice count, but a different scenario, facts, and "
                    "reasoning angle."
                ),
                evidence=(*objective_evidence.get(question.objective_code.casefold(), ()), source),
            )
        )
    return tuple(items)


class QueueStore:
    """Persist queue transitions and accepted candidates atomically."""

    def __init__(self, root: Path, state: QueueState, gate: QualityGate | None) -> None:
        """Create a queue around already parsed state."""
        self.root = root
        self.state = state
        self.gate = gate

    @property
    def items(self) -> tuple[WorkItem, ...]:
        """Return current work-item snapshots."""
        return self.state.items

    @classmethod
    def create(
        cls,
        root: Path,
        items: tuple[WorkItem, ...],
        gate: QualityGate | None = None,
        *,
        max_attempts: int = 3,
    ) -> QueueStore:
        """Create a new resumable queue."""
        root.mkdir(parents=True, exist_ok=True)
        state = QueueState(items=items, max_attempts=max_attempts)
        store = cls(root, state, gate)
        store._persist()
        return store

    @classmethod
    def open(cls, root: Path, gate: QualityGate | None = None) -> QueueStore:
        """Resume an existing queue without regenerating accepted material."""
        state = QueueState.model_validate_json((root / "queue.json").read_text(encoding="utf-8"))
        return cls(root, state, gate)

    def next_item(self) -> WorkItem | None:
        """Return the first pending item in deterministic order."""
        return next(
            (item for item in self.state.items if item.status is WorkItemStatus.PENDING), None
        )

    def submit(self, submission: CandidateSubmission) -> ValidationResult:
        """Validate one candidate and persist only accepted learner content."""
        if self.gate is None:
            msg = "A quality gate is required to submit candidates"
            raise RuntimeError(msg)
        index = next(
            (
                position
                for position, item in enumerate(self.state.items)
                if item.item_id == submission.item_id
            ),
            None,
        )
        if index is None:
            return ValidationResult(
                accepted=False,
                item_id=submission.item_id,
                issues=(),
            )
        item = self.state.items[index]
        result, accepted_prompt = self.gate.validate(item, submission, self.state.registry_prompts)
        attempts = item.attempts + 1
        status = WorkItemStatus.ACCEPTED if result.accepted else WorkItemStatus.PENDING
        if not result.accepted and attempts >= self.state.max_attempts:
            status = WorkItemStatus.EXHAUSTED
        updated_item = item.model_copy(update={"attempts": attempts, "status": status})
        items = list(self.state.items)
        items[index] = updated_item
        registry = self.state.registry_prompts
        if result.accepted and accepted_prompt is not None:
            registry = (*registry, accepted_prompt)
            accepted_dir = self.root / "accepted"
            accepted_dir.mkdir(parents=True, exist_ok=True)
            (accepted_dir / f"{item.item_id}.json").write_text(
                submission.payload_json,
                encoding="utf-8",
            )
        self.state = self.state.model_copy(
            update={"items": tuple(items), "registry_prompts": registry}
        )
        self._persist()
        return result

    def _persist(self) -> None:
        (self.root / "queue.json").write_text(
            self.state.model_dump_json(indent=2), encoding="utf-8"
        )
