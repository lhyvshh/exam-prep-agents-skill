"""Provider-neutral source extraction with stable physical page anchors."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from exam_prep_skill.models import LearningObjective, SourceRef

PAGE_MARKER: Final = re.compile(r"^--- PAGE (?P<page>\d+) ---$", re.MULTILINE)
MODULE_MARKER: Final = re.compile(
    r"^MODULE\s+(?P<code>[\w.]+):\s*(?P<title>.+)$", re.IGNORECASE | re.MULTILINE
)
LESSON_MARKER: Final = re.compile(r"^LESSON:\s*(?P<title>.+)$", re.IGNORECASE | re.MULTILINE)
OBJECTIVE_MARKER: Final = re.compile(
    r"^(?:(?P<code>LO\s+[\w.]+)|Learning goal):\s*(?P<title>.+)$", re.IGNORECASE | re.MULTILINE
)
EXCLUDED_HEADINGS: Final = ("contents", "index", "preface", "copyright")
FRM_WEIGHTS: Final = (20, 20, 30, 30)


class ExtractionModel(BaseModel):
    """Immutable normalized extraction record."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class DocumentPage(ExtractionModel):
    """Text attached to its physical PDF page."""

    physical_page: int = Field(ge=1)
    text: str


class ModuleRecord(ExtractionModel):
    """Normalized module or generic lesson."""

    module_id: str
    code: str
    title: str
    start_page: int = Field(ge=1)
    objectives: tuple[LearningObjective, ...]
    key_concepts: str
    module_quiz: str = ""
    answer_key: str = ""


class CurriculumRecord(ExtractionModel):
    """Provider-neutral curriculum hierarchy."""

    source_id: str
    title: str
    preset: str
    domain_weights: tuple[int, ...]
    modules: tuple[ModuleRecord, ...]
    excluded_pages: tuple[int, ...]


class ExamQuestionRecord(ExtractionModel):
    """Source-exam blueprint record."""

    position: int = Field(ge=1)
    prompt: str
    choices: tuple[str, ...]
    correct_choice: str
    explanation: str
    objective_code: str = ""
    question_type: str


class SourceExamRecord(ExtractionModel):
    """Ordered source exam and answers."""

    exam_id: str
    title: str
    questions: tuple[ExamQuestionRecord, ...]


def parse_page_fixture(text: str) -> tuple[DocumentPage, ...]:
    """Parse synthetic page markers used by provider-neutral tests."""
    matches = tuple(PAGE_MARKER.finditer(text))
    pages: list[DocumentPage] = []
    for index, marker in enumerate(matches):
        start = marker.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        pages.append(
            DocumentPage(
                physical_page=int(marker.group("page")),
                text=text[start:end].strip(),
            )
        )
    return tuple(pages)


def parse_pdf(path: Path) -> tuple[DocumentPage, ...]:
    """Extract PDF text in an isolated process to contain native-library failures."""
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "exam_prep_skill.pdf_worker", str(path.resolve())],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    return tuple(DocumentPage.model_validate(item) for item in payload)


def parse_book_pages(
    title: str,
    pages: tuple[DocumentPage, ...],
    *,
    preset: str = "generic",
) -> CurriculumRecord:
    """Normalize explicit provider hierarchy, then fall back to generic lessons."""
    included = tuple(page for page in pages if not _excluded(page.text))
    excluded_pages = tuple(page.physical_page for page in pages if _excluded(page.text))
    source_id = _stable_id(title + "\n" + "\n".join(page.text for page in pages))
    modules = _parse_explicit_modules(source_id, included)
    if not modules:
        modules = _parse_generic_lessons(source_id, included)
    return CurriculumRecord(
        source_id=source_id,
        title=title,
        preset=preset,
        domain_weights=FRM_WEIGHTS if preset == "frm-part-1" else (),
        modules=modules,
        excluded_pages=excluded_pages,
    )


def parse_exam_text(title: str, text: str) -> SourceExamRecord:
    """Separate ordered multiple-choice questions from their answer key."""
    body, _, answers = text.partition("ANSWER KEY")
    question_pattern = re.compile(r"(?ms)^\s*(?P<number>\d+)\.\s*(?P<block>.*?)(?=^\s*\d+\.|\Z)")
    answer_pattern = re.compile(
        r"(?ms)^\s*(?P<number>\d+)\.\s*(?P<choice>[A-Z])\.\s*(?P<explanation>.*?)(?=^\s*\d+\.|\Z)"
    )
    answer_map = {
        int(match.group("number")): (match.group("choice"), match.group("explanation").strip())
        for match in answer_pattern.finditer(answers)
    }
    questions: list[ExamQuestionRecord] = []
    for match in question_pattern.finditer(body):
        position = int(match.group("number"))
        block = match.group("block").strip()
        choice_matches = tuple(re.finditer(r"(?m)^([A-Z])\.\s*(.+)$", block))
        if not choice_matches:
            continue
        prompt = block[: choice_matches[0].start()].strip()
        objective_match = re.search(r"\[(LO\s+[\w.]+)]", prompt, re.IGNORECASE)
        prompt = re.sub(r"\s*\[LO\s+[\w.]+]\s*", " ", prompt, flags=re.IGNORECASE).strip()
        choices = tuple(choice.group(2).strip() for choice in choice_matches)
        correct_choice, explanation = answer_map.get(position, ("", ""))
        question_type = (
            "calculation"
            if re.search(r"\d+%|\bcalculate\b|\bwhat is\b", prompt, re.IGNORECASE)
            else "conceptual"
        )
        questions.append(
            ExamQuestionRecord(
                position=position,
                prompt=prompt,
                choices=choices,
                correct_choice=correct_choice,
                explanation=explanation,
                objective_code=objective_match.group(1) if objective_match else "",
                question_type=question_type,
            )
        )
    return SourceExamRecord(
        exam_id=_stable_id(title + text), title=title, questions=tuple(questions)
    )


def _parse_explicit_modules(
    source_id: str,
    pages: tuple[DocumentPage, ...],
) -> tuple[ModuleRecord, ...]:
    joined = "\n".join(f"\n[[PAGE:{page.physical_page}]]\n{page.text}" for page in pages)
    matches = tuple(MODULE_MARKER.finditer(joined))
    module_headers = tuple(
        match for match in matches if "quiz" not in match.group("code").casefold()
    )
    modules: list[ModuleRecord] = []
    for index, match in enumerate(module_headers):
        end = module_headers[index + 1].start() if index + 1 < len(module_headers) else len(joined)
        block = joined[match.start() : end]
        code = match.group("code")
        page = _page_before(joined, match.start())
        modules.append(_module_from_block(source_id, code, match.group("title"), page, block))
    return tuple(modules)


def _parse_generic_lessons(
    source_id: str,
    pages: tuple[DocumentPage, ...],
) -> tuple[ModuleRecord, ...]:
    modules: list[ModuleRecord] = []
    for page in pages:
        matches = tuple(LESSON_MARKER.finditer(page.text))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(page.text)
            block = page.text[match.start() : end]
            code = f"generic-{len(modules) + 1}"
            modules.append(
                _module_from_block(source_id, code, match.group("title"), page.physical_page, block)
            )
    return tuple(modules)


def _module_from_block(
    source_id: str, code: str, title: str, page: int, block: str
) -> ModuleRecord:
    module_id = _stable_id(f"{source_id}:{code}:{title}")
    objective_matches = tuple(OBJECTIVE_MARKER.finditer(block))
    objectives = tuple(
        LearningObjective(
            objective_id=_stable_id(f"{module_id}:{match.group(0)}"),
            code=(match.group("code") or f"Goal {index + 1}"),
            title=match.group("title").strip(),
            module_id=module_id,
            sources=(
                SourceRef(
                    source_id=source_id, title=title.title(), page=page, excerpt=match.group(0)
                ),
            ),
        )
        for index, match in enumerate(objective_matches)
    )
    quiz_match = re.search(r"(?ms)^MODULE QUIZ[^\n]*\n(?P<text>.*?)(?=^ANSWER KEY|\Z)", block)
    answer_match = re.search(
        r"(?ms)^ANSWER KEY FOR MODULE QUIZZES.*?^MODULE QUIZ[^\n]*\n(?P<text>.*)$", block
    )
    key_start = re.search(r"(?mi)^KEY CONCEPTS\s*$", block)
    key_end = re.search(r"(?mi)^MODULE QUIZ", block)
    key_concepts = ""
    if key_start:
        end = key_end.start() if key_end else len(block)
        key_concepts = block[key_start.end() : end].strip()
    elif objectives:
        key_concepts = block[objective_matches[-1].end() :].strip()
    return ModuleRecord(
        module_id=module_id,
        code=code,
        title=title.strip().title(),
        start_page=page,
        objectives=objectives,
        key_concepts=key_concepts,
        module_quiz=quiz_match.group("text").strip() if quiz_match else "",
        answer_key=answer_match.group("text").strip() if answer_match else "",
    )


def _excluded(text: str) -> bool:
    first = next((line.strip().casefold() for line in text.splitlines() if line.strip()), "")
    return any(first.startswith(heading) for heading in EXCLUDED_HEADINGS)


def _page_before(text: str, position: int) -> int:
    pages = tuple(re.finditer(r"\[\[PAGE:(\d+)]]", text[:position]))
    return int(pages[-1].group(1)) if pages else 1


def _stable_id(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]
