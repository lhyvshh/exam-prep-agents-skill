from __future__ import annotations

from pathlib import Path

from exam_prep_skill.extraction import parse_book_pages, parse_page_fixture
from exam_prep_skill.workspace import WorkspaceStore, content_hash


def test_workspace_persists_normalized_curriculum_by_content_hash(tmp_path: Path) -> None:
    source = "--- PAGE 1 ---\nMODULE 1.1: BASICS\nLO 1.a: Explain a basic concept."
    curriculum = parse_book_pages("Private Notes", parse_page_fixture(source))
    store = WorkspaceStore.initialize(tmp_path)

    cache_key = store.save_curriculum(source.encode(), curriculum)

    assert cache_key == content_hash(source.encode())
    assert store.load_curriculum(cache_key) == curriculum
    assert str(store.state_dir).startswith(str(tmp_path.resolve()))


def test_workspace_initialization_is_resumable(tmp_path: Path) -> None:
    first = WorkspaceStore.initialize(tmp_path, preset="generic")

    second = WorkspaceStore.open(tmp_path)

    assert second.config == first.config
    assert second.state_dir.is_dir()
