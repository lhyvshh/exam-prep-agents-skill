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


class PackageRecord(FrozenModel):
    """Private record for a rendered learner artifact."""

    package_id: str = Field(min_length=1)
    kind: PackageKind
    title: str = Field(min_length=1)
    output_path: Path
    item_count: int = Field(ge=1)
    created_at: str = Field(min_length=1)
