"""Deterministic and PyTorch-backed learner artifact quality gates."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import warnings
from pathlib import Path
from types import TracebackType
from typing import Final, Literal, Protocol, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

from exam_prep_skill.models import (
    CandidateSubmission,
    ExamQuestionCandidate,
    FlashcardCandidate,
    PackageKind,
    ValidationIssue,
    ValidationResult,
    WorkItem,
)

QUALITY_THRESHOLD: Final = 0.70
SOURCE_COPY_THRESHOLD: Final = 0.82
DUPLICATE_THRESHOLD: Final = 0.90
MIN_PROMPT_WORDS: Final = 8
MIN_EXPLANATION_WORDS: Final = 15
EXPLANATION_FEATURE_WORDS: Final = 12
TOKEN_PATTERN: Final = re.compile(r"[a-z0-9]+")
VAGUE_PATTERN: Final = re.compile(r"\b(?:there|this thing|which applies)\b", re.IGNORECASE)
GRAMMAR_PATTERN: Final = re.compile(
    r"\b(?:it are|they is|does not correctly applies)\b",
    re.IGNORECASE,
)
PORTABLE_EXPORT_FORMAT: Final = "exam-prep-pytorch-linear-v1"


class _ScoreTensor(Protocol):
    def item(self) -> float: ...


class _TorchModel(Protocol):
    def eval(self) -> object: ...

    def __call__(self, features: object) -> _ScoreTensor: ...


class _TorchJit(Protocol):
    def load(self, path: str, *, map_location: str) -> _TorchModel: ...


class _NoGradContext(Protocol):
    def __enter__(self) -> object: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


class _TorchRuntime(Protocol):
    float32: object
    jit: _TorchJit

    def tensor(self, data: list[float], *, dtype: object) -> object: ...

    def no_grad(self) -> _NoGradContext: ...


class _Scorer(Protocol):
    def score(self, features: list[float]) -> float: ...


class _PortableExport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["exam-prep-pytorch-linear-v1"]
    source_checkpoint_sha256: str
    weights: tuple[float, float, float, float, float]
    bias: float


class _PytorchScorer:
    def __init__(self, runtime: _TorchRuntime, model: _TorchModel) -> None:
        self.runtime = runtime
        self.model = model
        self.model.eval()

    def score(self, features: list[float]) -> float:
        feature_tensor = self.runtime.tensor(features, dtype=self.runtime.float32)
        with self.runtime.no_grad():
            return float(self.model(feature_tensor).item())


class _PortableScorer:
    def __init__(self, export: _PortableExport) -> None:
        self.weights = np.asarray(export.weights, dtype=np.float32)
        self.bias = np.float32(export.bias)

    def score(self, features: list[float]) -> float:
        feature_array = np.asarray(features, dtype=np.float32)
        logit = np.float32(np.dot(feature_array, self.weights) + self.bias)
        return float(1.0 / (1.0 + math.exp(-float(logit))))


def _load_torch_runtime() -> _TorchRuntime | None:
    try:
        module = importlib.import_module("torch")
    except (ImportError, OSError):
        return None
    if not all(hasattr(module, attribute) for attribute in ("float32", "jit", "no_grad", "tensor")):
        return None
    return cast("_TorchRuntime", module)


def _torch_jit_load(runtime: _TorchRuntime, checkpoint_path: Path) -> _TorchModel:
    return runtime.jit.load(str(checkpoint_path), map_location="cpu")


def _portable_scorer(checkpoint_path: Path, checkpoint_sha256: str) -> _PortableScorer:
    export_path = checkpoint_path.with_suffix(".json")
    export = _PortableExport.model_validate_json(export_path.read_text(encoding="utf-8"))
    if export.source_checkpoint_sha256 != checkpoint_sha256:
        msg = "portable quality export does not match the bundled PyTorch checkpoint"
        raise RuntimeError(msg)
    return _PortableScorer(export)


class QualityGate:
    """Reject invalid content before applying the bundled PyTorch scorer."""

    def __init__(self, checkpoint_path: Path) -> None:
        """Load a local allowlisted checkpoint and record its content hash."""
        self.checkpoint_path = checkpoint_path
        checkpoint_bytes = checkpoint_path.read_bytes()
        self.checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
        runtime = _load_torch_runtime()
        if runtime is None:
            self.scorer: _Scorer = _portable_scorer(checkpoint_path, self.checkpoint_sha256)
            self.model_source = "pytorch_portable_export"
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"`torch\.jit\.load` is deprecated\..*",
                    category=DeprecationWarning,
                )
                model = _torch_jit_load(runtime, checkpoint_path)
            self.scorer = _PytorchScorer(runtime, model)
            self.model_source = "pytorch_checkpoint"

    def validate(
        self,
        item: WorkItem,
        submission: CandidateSubmission,
        registry_prompts: tuple[str, ...],
        registry_responses: tuple[str, ...] = (),
    ) -> tuple[ValidationResult, str | None, str | None]:
        """Validate a candidate and return its prompt for accepted deduplication."""
        match item.kind:
            case PackageKind.FLASHCARDS:
                return self._validate_card(item, submission, registry_prompts, registry_responses)
            case PackageKind.MOCK_EXAM:
                return self._validate_question(
                    item, submission, registry_prompts, registry_responses
                )

    def _validate_card(
        self,
        item: WorkItem,
        submission: CandidateSubmission,
        registry_prompts: tuple[str, ...],
        registry_responses: tuple[str, ...],
    ) -> tuple[ValidationResult, str | None, str | None]:
        try:
            candidate = FlashcardCandidate.model_validate_json(submission.payload_json)
        except ValidationError as error:
            return _schema_failure(item, error), None, None
        issues: list[ValidationIssue] = []
        if candidate.item_id != item.item_id:
            issues.append(_issue("item_mismatch", "Candidate item identifier does not match."))
        if candidate.objective_id != item.objective_id or candidate.slot != item.slot:
            issues.append(
                _issue("coverage_mismatch", "Card does not match its assigned objective and slot.")
            )
        if not set(candidate.source_pages).issubset(_evidence_pages(item)):
            issues.append(
                _issue("invalid_evidence", "Card cites a page outside its work-item evidence.")
            )
        if any(
            _similarity(candidate.prompt, existing) >= DUPLICATE_THRESHOLD
            for existing in registry_prompts
        ):
            issues.append(_issue("duplicate_card", "Card is too similar to accepted material."))
        if any(
            _similarity(candidate.answer, existing) >= DUPLICATE_THRESHOLD
            for existing in registry_responses
        ):
            issues.append(_issue("duplicate_answer", "Card answer repeats accepted material."))
        if _language_issue(candidate.prompt + " " + candidate.answer):
            issues.append(_issue("language_quality", "Card contains malformed or vague language."))
        if issues:
            return _rejected(item, issues), None, None
        score = self._score(candidate.prompt, candidate.answer)
        if score < QUALITY_THRESHOLD:
            return _quality_rejection(item, score, self), None, None
        return _accepted(item, score, self), candidate.prompt, candidate.answer

    def _validate_question(
        self,
        item: WorkItem,
        submission: CandidateSubmission,
        registry_prompts: tuple[str, ...],
        registry_responses: tuple[str, ...],
    ) -> tuple[ValidationResult, str | None, str | None]:
        try:
            candidate = ExamQuestionCandidate.model_validate_json(submission.payload_json)
        except ValidationError as error:
            return _schema_failure(item, error), None, None
        blueprint = json.loads(item.blueprint_json or "{}")
        issues = _question_issues(item, candidate, blueprint, registry_prompts, registry_responses)
        if issues:
            return _rejected(item, issues), None, None
        score = self._score(candidate.prompt, candidate.explanation)
        if score < QUALITY_THRESHOLD:
            return _quality_rejection(item, score, self), None, None
        return _accepted(item, score, self), candidate.prompt, "\n".join(candidate.choices)

    def _score(self, prompt: str, explanation: str) -> float:
        prompt_words = len(TOKEN_PATTERN.findall(prompt.casefold()))
        explanation_words = len(TOKEN_PATTERN.findall(explanation.casefold()))
        features = [
            min(prompt_words / 12, 1),
            min(explanation_words / 20, 1),
            float(prompt.rstrip().endswith("?")),
            float(not _language_issue(prompt)),
            float(explanation_words >= EXPLANATION_FEATURE_WORDS),
        ]
        return self.scorer.score(features)


def _question_issues(
    item: WorkItem,
    candidate: ExamQuestionCandidate,
    blueprint: dict[str, str | list[str] | int],
    registry_prompts: tuple[str, ...],
    registry_responses: tuple[str, ...],
) -> list[ValidationIssue]:
    issues = _structural_question_issues(item, candidate, blueprint)
    issues.extend(
        _similarity_question_issues(candidate, blueprint, registry_prompts, registry_responses)
    )
    issues.extend(_language_question_issues(candidate))
    return issues


def _structural_question_issues(
    item: WorkItem,
    candidate: ExamQuestionCandidate,
    blueprint: dict[str, str | list[str] | int],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    expected_choices = blueprint.get("choices", [])
    if not isinstance(expected_choices, list) or len(candidate.choices) != len(expected_choices):
        issues.append(_issue("choice_count", "Choice count does not match the source blueprint."))
    if candidate.correct_choice not in _choice_labels(len(candidate.choices)):
        issues.append(
            _issue("answer_choice", "Correct choice does not identify a delivered option.")
        )
    if len({_normalize(choice) for choice in candidate.choices}) != len(candidate.choices):
        issues.append(_issue("duplicate_choice", "Answer choices must be distinct."))
    if candidate.item_id != item.item_id:
        issues.append(_issue("item_mismatch", "Candidate item identifier does not match."))
    expected_objective = str(blueprint.get("objective_code", "")).strip()
    if expected_objective and candidate.objective_code.casefold() != expected_objective.casefold():
        issues.append(
            _issue("objective_mismatch", "Question does not match its source blueprint objective.")
        )
    expected_type = str(blueprint.get("question_type", "")).strip()
    if expected_type and candidate.question_type.casefold() != expected_type.casefold():
        issues.append(
            _issue(
                "question_type_mismatch",
                "Question does not match its source blueprint cognitive format.",
            )
        )
    if not set(candidate.source_pages).issubset(_evidence_pages(item)):
        issues.append(_issue("invalid_evidence", "Question cites a page outside its evidence."))
    return issues


def _similarity_question_issues(
    candidate: ExamQuestionCandidate,
    blueprint: dict[str, str | list[str] | int],
    registry_prompts: tuple[str, ...],
    registry_responses: tuple[str, ...],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    source_prompt = str(blueprint.get("prompt", ""))
    if _similarity(candidate.prompt, source_prompt) >= SOURCE_COPY_THRESHOLD:
        issues.append(_issue("source_copy", "Question copies the source exam too closely."))
    if any(
        _similarity(candidate.prompt, existing) >= DUPLICATE_THRESHOLD
        for existing in registry_prompts
    ):
        issues.append(
            _issue("duplicate_question", "Question overlaps accepted generated material.")
        )
    answer_set = "\n".join(candidate.choices)
    if any(
        _similarity(answer_set, existing) >= DUPLICATE_THRESHOLD for existing in registry_responses
    ):
        issues.append(_issue("duplicate_answers", "Answer set repeats accepted material."))
    return issues


def _language_question_issues(candidate: ExamQuestionCandidate) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if VAGUE_PATTERN.search(candidate.prompt) or len(_tokens(candidate.prompt)) < MIN_PROMPT_WORDS:
        issues.append(_issue("vague_prompt", "Question prompt is vague or underspecified."))
    if _language_issue(candidate.prompt + " " + " ".join(candidate.choices)):
        issues.append(
            _issue("language_quality", "Question contains malformed or incomplete language.")
        )
    if len(_tokens(candidate.explanation)) < MIN_EXPLANATION_WORDS:
        issues.append(
            _issue("explanation_depth", "Explanation must justify the answer and distractors.")
        )
    return issues


def _schema_failure(item: WorkItem, error: ValidationError) -> ValidationResult:
    return _rejected(item, [_issue("schema", error.errors()[0]["msg"])])


def _rejected(
    item: WorkItem,
    issues: list[ValidationIssue],
    *,
    score: float | None = None,
    gate: QualityGate | None = None,
) -> ValidationResult:
    return ValidationResult(
        accepted=False,
        item_id=item.item_id,
        issues=tuple(issues),
        score=score,
        model_source=gate.model_source if gate else None,
        checkpoint_sha256=gate.checkpoint_sha256 if gate else None,
    )


def _quality_rejection(item: WorkItem, score: float, gate: QualityGate) -> ValidationResult:
    return _rejected(
        item,
        [_issue("pytorch_quality", "Candidate did not meet the PyTorch quality threshold.")],
        score=score,
        gate=gate,
    )


def _accepted(item: WorkItem, score: float, gate: QualityGate) -> ValidationResult:
    return ValidationResult(
        accepted=True,
        item_id=item.item_id,
        score=score,
        model_source=gate.model_source,
        checkpoint_sha256=gate.checkpoint_sha256,
    )


def _evidence_pages(item: WorkItem) -> set[int]:
    return {source.page for source in item.evidence}


def _language_issue(text: str) -> bool:
    normalized = text.strip()
    return bool(
        GRAMMAR_PATTERN.search(normalized)
        or re.search(r"\b(?:a|an|the|to|of)$", normalized, re.IGNORECASE)
    )


def _similarity(left: str, right: str) -> float:
    left_tokens = set(_tokens(left))
    right_tokens = set(_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_PATTERN.findall(text.casefold()))


def _normalize(text: str) -> str:
    return " ".join(_tokens(text))


def _choice_labels(count: int) -> set[str]:
    return {chr(ord("A") + index) for index in range(count)}


def _issue(code: str, message: str) -> ValidationIssue:
    return ValidationIssue(code=code, message=message)
