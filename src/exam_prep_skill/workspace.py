"""Private resumable workspace persistence."""

from __future__ import annotations

import hashlib
from pathlib import Path

from exam_prep_skill.extraction import CurriculumRecord
from exam_prep_skill.models import WorkspaceConfig


def content_hash(content: bytes) -> str:
    """Return a stable source-content key."""
    return hashlib.sha256(content).hexdigest()


class WorkspaceStore:
    """Persist private state outside the installed skill repository."""

    def __init__(self, config: WorkspaceConfig) -> None:
        """Create a store from an already parsed workspace config."""
        self.config = config

    @property
    def state_dir(self) -> Path:
        """Return the private state directory."""
        return self.config.state_dir

    @classmethod
    def initialize(cls, workspace: Path, *, preset: str = "generic") -> WorkspaceStore:
        """Create or reopen a workspace without deleting accepted state."""
        root = workspace.expanduser().resolve()
        state_dir = root / ".exam-prep" / "state"
        output_dir = root / ".exam-prep" / "output"
        state_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        config = WorkspaceConfig(
            workspace=root,
            state_dir=state_dir,
            output_dir=output_dir,
            preset=preset,
        )
        (root / ".exam-prep" / "config.json").write_text(
            config.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return cls(config)

    @classmethod
    def open(cls, workspace: Path) -> WorkspaceStore:
        """Load a previously initialized workspace."""
        config_path = workspace.expanduser().resolve() / ".exam-prep" / "config.json"
        return cls(WorkspaceConfig.model_validate_json(config_path.read_text(encoding="utf-8")))

    def save_curriculum(self, content: bytes, curriculum: CurriculumRecord) -> str:
        """Cache normalized curriculum records by private content hash."""
        cache_key = content_hash(content)
        path = self.state_dir / "sources" / f"{cache_key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(curriculum.model_dump_json(indent=2), encoding="utf-8")
        return cache_key

    def load_curriculum(self, cache_key: str) -> CurriculumRecord:
        """Load a cached normalized curriculum record."""
        path = self.state_dir / "sources" / f"{cache_key}.json"
        return CurriculumRecord.model_validate_json(path.read_text(encoding="utf-8"))
