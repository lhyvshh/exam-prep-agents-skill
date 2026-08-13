"""JSON-first command line boundary for agent hosts."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from exam_prep_skill.extraction import (
    CurriculumRecord,
    DocumentPage,
    parse_book_pages,
    parse_exam_text,
    parse_page_fixture,
    parse_pdf,
)
from exam_prep_skill.models import (
    CandidateSubmission,
    ExamQuestionCandidate,
    FlashcardCandidate,
    GenerationRegistry,
    LearnerExam,
    LearnerExamQuestion,
    LearnerFlashcard,
    LearnerFlashcardDeck,
    PackageKind,
    WorkItemStatus,
)
from exam_prep_skill.queue import QueueStore, build_exam_items, build_flashcard_items
from exam_prep_skill.render import render_flashcards, render_mock_exam, write_learner_zip
from exam_prep_skill.workspace import WorkflowState, WorkspaceStore

if TYPE_CHECKING:
    from exam_prep_skill.quality import QualityGate

app = typer.Typer(
    add_completion=False,
    help="Build validated offline flashcard and mock-exam packages with your active agent.",
    no_args_is_help=True,
)
HumanOption = Annotated[bool, typer.Option("--human", help="Print concise human-readable output.")]
BookArgument = Annotated[list[Path], typer.Argument(help="Owned exam book PDF or text file.")]


def _emit(payload: dict[str, Any], *, human: bool) -> None:
    if human:
        typer.echo(str(payload.get("message", payload.get("status", "ok"))))
        return
    typer.echo(json.dumps(payload, sort_keys=True, default=str))


def _config_path(workspace: Path) -> Path:
    return workspace.resolve() / ".exam-prep" / "config.json"


def _require_workspace(workspace: Path) -> WorkspaceStore:
    root = workspace.expanduser().resolve()
    if not _config_path(root).is_file():
        _emit(
            {
                "status": "error",
                "code": "workspace_not_initialized",
                "message": f"No exam-prep workspace at {root}",
            },
            human=False,
        )
        raise typer.Exit(code=2)
    return WorkspaceStore.open(root)


def _checkpoint_path() -> Path:
    packaged = Path(__file__).with_name("assets") / "question_quality_classifier.pt"
    if packaged.is_file():
        return packaged
    return Path(__file__).parents[2] / "assets" / "question_quality_classifier.pt"


def _quality_gate() -> QualityGate:
    from exam_prep_skill.quality import QualityGate  # noqa: PLC0415

    return QualityGate(_checkpoint_path())


def _read_pages(path: Path) -> tuple[DocumentPage, ...]:
    resolved = path.expanduser().resolve()
    if resolved.suffix.casefold() == ".pdf":
        return parse_pdf(resolved)
    text = resolved.read_text(encoding="utf-8")
    pages = parse_page_fixture(text)
    return pages or (DocumentPage(physical_page=1, text=text),)


def _read_exam_text(path: Path) -> str:
    pages = _read_pages(path)
    return "\n".join(page.text for page in pages)


def _prepare_curriculum(
    books: list[Path], preset: str, title: str
) -> tuple[bytes, CurriculumRecord]:
    if not books:
        msg = "At least one exam book is required"
        raise typer.BadParameter(msg)
    records: list[CurriculumRecord] = []
    source_bytes: list[bytes] = []
    for book in books:
        resolved = book.expanduser().resolve()
        content = resolved.read_bytes()
        source_bytes.append(content)
        records.append(parse_book_pages(resolved.stem, _read_pages(resolved), preset=preset))
    merged_bytes = b"\n\n".join(source_bytes)
    merged_id = hashlib.sha256(merged_bytes).hexdigest()[:16]
    return merged_bytes, CurriculumRecord(
        source_id=merged_id,
        title=title,
        preset=preset,
        domain_weights=(20, 20, 30, 30) if preset == "frm-part-1" else (),
        modules=tuple(module for record in records for module in record.modules),
        excluded_pages=tuple(page for record in records for page in record.excluded_pages),
    )


def _initialize_prepared_workspace(workspace: Path, preset: str) -> WorkspaceStore:
    store = WorkspaceStore.initialize(workspace, preset=preset)
    if store.queue_dir.exists():
        shutil.rmtree(store.queue_dir)
    return store


@app.command("init")
def init_workspace(
    workspace: Annotated[Path, typer.Argument(help="Private project workspace.")],
    preset: Annotated[str, typer.Option(help="Curriculum preset.")] = "generic",
    human: HumanOption = False,
) -> None:
    """Initialize private state outside the installed skill."""
    store = WorkspaceStore.initialize(workspace, preset=preset)
    _emit(
        {
            "status": "initialized",
            "workspace": str(store.config.workspace),
            "message": f"Initialized private workspace at {store.config.workspace}",
        },
        human=human,
    )


@app.command("prepare-flashcards")
def prepare_flashcards(
    workspace: Annotated[Path, typer.Argument(help="Private project workspace.")],
    books: BookArgument,
    title: Annotated[str, typer.Option(help="Learner deck title.")] = "Study Flashcards",
    preset: Annotated[str, typer.Option(help="Use generic or frm-part-1.")] = "generic",
    target_per_objective: Annotated[
        int, typer.Option(min=1, max=10, help="Cards assigned to every learning objective.")
    ] = 10,
    human: HumanOption = False,
) -> None:
    """Extract books and create a balanced flashcard work queue."""
    store = _initialize_prepared_workspace(workspace, preset)
    content, curriculum = _prepare_curriculum(books, preset, title)
    cache_key = store.save_curriculum(content, curriculum)
    items = build_flashcard_items(curriculum, target_per_objective=target_per_objective)
    if not items:
        _emit(
            {
                "status": "error",
                "code": "no_learning_objectives",
                "message": "No source-backed learning objectives were found in the selected books.",
            },
            human=human,
        )
        raise typer.Exit(code=2)
    registry = store.load_generation_registry()
    QueueStore.create(
        store.queue_dir,
        items,
        registry=registry,
    )
    store.save_workflow(
        WorkflowState(
            kind=PackageKind.FLASHCARDS,
            title=title,
            curriculum_cache_key=cache_key,
        )
    )
    _emit(
        {
            "status": "prepared",
            "kind": PackageKind.FLASHCARDS.value,
            "total_items": len(items),
            "objectives": len({item.objective_id for item in items}),
            "message": f"Prepared {len(items)} balanced flashcard work items.",
        },
        human=human,
    )


@app.command("prepare-exam")
def prepare_exam(
    workspace: Annotated[Path, typer.Argument(help="Private project workspace.")],
    books: BookArgument,
    source_exam: Annotated[Path, typer.Option(help="Owned sample exam with answer key.")],
    title: Annotated[str, typer.Option(help="Learner exam title.")] = "New Mock Exam",
    preset: Annotated[str, typer.Option(help="Use generic or frm-part-1.")] = "generic",
    duration_minutes: Annotated[int, typer.Option(min=1)] = 240,
    human: HumanOption = False,
) -> None:
    """Extract books and source exam into a one-for-one question queue."""
    store = _initialize_prepared_workspace(workspace, preset)
    content, curriculum = _prepare_curriculum(books, preset, title)
    cache_key = store.save_curriculum(content, curriculum)
    exam_path = source_exam.expanduser().resolve()
    source = parse_exam_text(exam_path.stem, _read_exam_text(exam_path))
    items = build_exam_items(source, curriculum)
    if not items:
        _emit(
            {
                "status": "error",
                "code": "no_exam_questions",
                "message": "No complete multiple-choice questions and answer key were found.",
            },
            human=human,
        )
        raise typer.Exit(code=2)
    registry = store.load_generation_registry()
    QueueStore.create(
        store.queue_dir,
        items,
        registry=registry,
    )
    store.save_workflow(
        WorkflowState(
            kind=PackageKind.MOCK_EXAM,
            title=title,
            curriculum_cache_key=cache_key,
            duration_minutes=duration_minutes,
        )
    )
    _emit(
        {
            "status": "prepared",
            "kind": PackageKind.MOCK_EXAM.value,
            "total_items": len(items),
            "message": f"Prepared {len(items)} one-for-one exam work items.",
        },
        human=human,
    )


@app.command("status")
def status(
    workspace: Annotated[Path, typer.Argument(help="Private project workspace.")],
    human: HumanOption = False,
) -> None:
    """Report resumable workspace state."""
    root = workspace.expanduser().resolve()
    if not _config_path(root).is_file():
        _emit(
            {
                "status": "error",
                "code": "workspace_not_initialized",
                "message": f"No exam-prep workspace at {root}",
            },
            human=human,
        )
        raise typer.Exit(code=2)
    store = WorkspaceStore.open(root)
    queue_path = store.queue_dir / "queue.json"
    if not queue_path.is_file():
        _emit(
            {"status": "ready", "workspace": str(root), "message": "Workspace is initialized"},
            human=human,
        )
        return
    queue = QueueStore.open(store.queue_dir)
    counts = {
        state.value: sum(item.status is state for item in queue.items) for state in WorkItemStatus
    }
    _emit(
        {
            "status": "complete" if counts[WorkItemStatus.PENDING.value] == 0 else "in_progress",
            "workspace": str(root),
            "counts": counts,
            "total_items": len(queue.items),
            "message": f"{counts[WorkItemStatus.ACCEPTED.value]} of {len(queue.items)} accepted.",
        },
        human=human,
    )


@app.command("next")
def next_item(
    workspace: Annotated[Path, typer.Argument(help="Private project workspace.")],
    human: HumanOption = False,
) -> None:
    """Return the next bounded work item."""
    store = _require_workspace(workspace)
    if not (store.queue_dir / "queue.json").is_file():
        _workflow_not_prepared("next", human)
    item = QueueStore.open(store.queue_dir).next_item()
    if item is None:
        _emit(
            {"status": "complete", "item": None, "message": "All available items are resolved."},
            human=human,
        )
        return
    _emit(
        {"status": "pending", "item": item.model_dump(mode="json"), "message": item.instructions},
        human=human,
    )


@app.command("submit")
def submit(
    workspace: Annotated[Path, typer.Argument(help="Private project workspace.")],
    candidate: Annotated[Path, typer.Argument(help="Candidate JSON file generated by the host.")],
    human: HumanOption = False,
) -> None:
    """Submit a candidate to deterministic and PyTorch validation."""
    store = _require_workspace(workspace)
    if not (store.queue_dir / "queue.json").is_file():
        _workflow_not_prepared("submit", human)
    payload_json = candidate.expanduser().resolve().read_text(encoding="utf-8")
    payload = json.loads(payload_json)
    item_id = str(payload.get("item_id", ""))
    queue = QueueStore.open(store.queue_dir, _quality_gate())
    result = queue.submit(CandidateSubmission(item_id=item_id, payload_json=payload_json))
    if result.accepted:
        store.save_generation_registry(
            GenerationRegistry(
                prompts=queue.state.registry_prompts,
                responses=queue.state.registry_responses,
            )
        )
    body = result.model_dump(mode="json")
    body["status"] = "accepted" if result.accepted else "rejected"
    body["message"] = (
        "Candidate accepted."
        if result.accepted
        else "Candidate rejected; correct every issue before continuing."
    )
    _emit(body, human=human)


@app.command("render")
def render(
    workspace: Annotated[Path, typer.Argument(help="Private project workspace.")],
    human: HumanOption = False,
) -> None:
    """Render a fully accepted learner package."""
    store = _require_workspace(workspace)
    if not (store.queue_dir / "queue.json").is_file():
        _workflow_not_prepared("render", human)
    queue = QueueStore.open(store.queue_dir)
    unresolved = tuple(item for item in queue.items if item.status is not WorkItemStatus.ACCEPTED)
    if unresolved:
        _emit(
            {
                "status": "error",
                "code": "quality_gate_incomplete",
                "unresolved_items": len(unresolved),
                "message": "Every item must be accepted before rendering.",
            },
            human=human,
        )
        raise typer.Exit(code=2)
    workflow = store.load_workflow()
    curriculum = store.load_curriculum(workflow.curriculum_cache_key)
    accepted = [
        (store.queue_dir / "accepted" / f"{item.item_id}.json").read_text(encoding="utf-8")
        for item in queue.items
    ]
    package_id = hashlib.sha256("\n".join(accepted).encode()).hexdigest()[:16]
    slug = _slug(workflow.title)
    output = store.output_dir / f"{slug}.html"
    package = store.output_dir / f"{slug}.zip"
    if workflow.kind is PackageKind.FLASHCARDS:
        html = render_flashcards(_build_deck(package_id, workflow.title, accepted, curriculum))
    else:
        html = render_mock_exam(
            _build_exam(
                package_id,
                workflow.title,
                workflow.duration_minutes,
                accepted,
                queue,
                curriculum,
            )
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    write_learner_zip(package, output.name, html)
    _emit(
        {
            "status": "rendered",
            "output": str(output),
            "package": str(package),
            "item_count": len(accepted),
            "message": f"Rendered offline learner package at {output}",
        },
        human=human,
    )


def _build_deck(
    package_id: str,
    title: str,
    accepted: list[str],
    curriculum: CurriculumRecord,
) -> LearnerFlashcardDeck:
    objective_map = {
        objective.objective_id: (objective, module)
        for module in curriculum.modules
        for objective in module.objectives
    }
    cards: list[LearnerFlashcard] = []
    for raw in accepted:
        candidate = FlashcardCandidate.model_validate_json(raw)
        objective, module = objective_map[candidate.objective_id]
        cards.append(
            LearnerFlashcard(
                card_id=candidate.card_id,
                objective_id=candidate.objective_id,
                objective_code=objective.code,
                objective_title=objective.title,
                module_title=module.title,
                prompt=candidate.prompt,
                answer=candidate.answer,
                slot=candidate.slot,
                difficulty=candidate.difficulty,
                source_pages=candidate.source_pages,
            )
        )
    return LearnerFlashcardDeck(package_id=package_id, title=title, cards=tuple(cards))


def _build_exam(
    package_id: str,
    title: str,
    duration_minutes: int,
    accepted: list[str],
    queue: QueueStore,
    curriculum: CurriculumRecord,
) -> LearnerExam:
    objective_titles = {
        objective.code.casefold(): objective.title
        for module in curriculum.modules
        for objective in module.objectives
    }
    questions: list[LearnerExamQuestion] = []
    for item, raw in zip(queue.items, accepted, strict=True):
        candidate = ExamQuestionCandidate.model_validate_json(raw)
        questions.append(
            LearnerExamQuestion(
                question_id=candidate.question_id,
                position=item.blueprint_position or len(questions) + 1,
                prompt=candidate.prompt,
                choices=candidate.choices,
                correct_choice=candidate.correct_choice,
                explanation=candidate.explanation,
                objective_code=candidate.objective_code,
                objective_title=objective_titles.get(
                    candidate.objective_code.casefold(), candidate.objective_code
                ),
                source_pages=candidate.source_pages,
                difficulty=candidate.difficulty,
            )
        )
    return LearnerExam(
        package_id=package_id,
        title=title,
        duration_minutes=duration_minutes,
        questions=tuple(questions),
    )


def _workflow_not_prepared(command: str, human: bool) -> None:
    _emit(
        {
            "status": "error",
            "code": "workflow_not_prepared",
            "message": f"Run source preparation before '{command}'",
        },
        human=human,
    )
    raise typer.Exit(code=2)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "study-package"
