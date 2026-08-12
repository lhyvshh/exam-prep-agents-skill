"""Typed trust-boundary models shared by the command workflow."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class FrozenModel(BaseModel):
    """Base for immutable JSON records."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class PackageKind(StrEnum):
    """Learner package variants."""

    FLASHCARDS = "flashcards"
    MOCK_EXAM = "mock_exam"


class WorkItemStatus(StrEnum):
    """Resumable work-item states."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EXHAUSTED = "exhausted"


class WorkspaceConfig(FrozenModel):
    """Private project workspace locations."""

    workspace: Path
    state_dir: Path
    output_dir: Path
    preset: str = "generic"


class SourceRef(FrozenModel):
    """Grounding evidence from a private source document."""

    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    page: int = Field(ge=1)
    excerpt: str = Field(min_length=1)


class LearningObjective(FrozenModel):
    """Normalized curriculum objective."""

    objective_id: str = Field(min_length=1)
    code: str = Field(min_length=1)
    title: str = Field(min_length=1)
    module_id: str = Field(min_length=1)
    sources: tuple[SourceRef, ...] = ()


class WorkItem(FrozenModel):
    """Bounded semantic task for the active host agent."""

    item_id: str = Field(min_length=1)
    kind: PackageKind
    objective_id: str | None = None
    blueprint_position: int | None = Field(default=None, ge=1)
    blueprint_json: str | None = None
    slot: str | None = None
    instructions: str = Field(min_length=1)
    evidence: tuple[SourceRef, ...] = ()
    status: WorkItemStatus = WorkItemStatus.PENDING
    attempts: int = Field(default=0, ge=0)


class CandidateSubmission(FrozenModel):
    """Host-agent answer submitted to deterministic gates."""

    item_id: str = Field(min_length=1)
    payload_json: str = Field(min_length=2)


class ValidationIssue(FrozenModel):
    """Machine-readable rejection reason."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ValidationResult(FrozenModel):
    """Outcome returned to the host after submission."""

    accepted: bool
    item_id: str = Field(min_length=1)
    issues: tuple[ValidationIssue, ...] = ()
    score: float | None = Field(default=None, ge=0, le=1)
    model_source: str | None = None
    checkpoint_sha256: str | None = None


class FlashcardCandidate(FrozenModel):
    """Structured flashcard candidate supplied by the host agent."""

    item_id: str = Field(min_length=1)
    card_id: str = Field(min_length=1)
    objective_id: str = Field(min_length=1)
    slot: str = Field(min_length=1)
    prompt: str = Field(min_length=8)
    answer: str = Field(min_length=8)
    difficulty: str = Field(min_length=1)
    source_pages: tuple[int, ...] = Field(min_length=1)


class ExamQuestionCandidate(FrozenModel):
    """Structured mock-exam candidate supplied by the host agent."""

    item_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    prompt: str = Field(min_length=8)
    choices: tuple[str, ...] = Field(min_length=2)
    correct_choice: str = Field(pattern=r"^[A-Z]$")
    explanation: str = Field(min_length=20)
    verification: str = Field(min_length=12)
    source_pages: tuple[int, ...] = Field(min_length=1)
    objective_code: str
    question_type: str = Field(min_length=1)
    difficulty: str = Field(min_length=1)


class PackageRecord(FrozenModel):
    """Private record for a rendered learner artifact."""

    package_id: str = Field(min_length=1)
    kind: PackageKind
    title: str = Field(min_length=1)
    output_path: Path
    item_count: int = Field(ge=1)
    created_at: str = Field(min_length=1)


class LearnerFlashcard(FrozenModel):
    """Accepted card exposed to a learner package."""

    card_id: str
    objective_id: str
    objective_code: str
    objective_title: str
    module_title: str
    prompt: str
    answer: str
    slot: str
    difficulty: str
    source_pages: tuple[int, ...]


class LearnerFlashcardDeck(FrozenModel):
    """Self-contained flashcard package payload."""

    package_id: str
    title: str
    cards: tuple[LearnerFlashcard, ...] = Field(min_length=1)


class LearnerExamQuestion(FrozenModel):
    """Verified question exposed to a learner package."""

    question_id: str
    position: int = Field(ge=1)
    prompt: str
    choices: tuple[str, ...] = Field(min_length=2)
    correct_choice: str
    explanation: str
    objective_code: str
    objective_title: str
    source_pages: tuple[int, ...]
    difficulty: str


class LearnerExam(FrozenModel):
    """Self-contained mock-exam package payload."""

    package_id: str
    title: str
    duration_minutes: int = Field(ge=1)
    questions: tuple[LearnerExamQuestion, ...] = Field(min_length=1)
