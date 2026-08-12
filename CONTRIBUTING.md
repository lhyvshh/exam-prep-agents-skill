# Contributing

Contributions are welcome for provider-neutral parsing, deterministic quality gates, accessibility, and offline learner workflows.

1. Create a focused branch.
2. Add or update synthetic tests before implementation changes.
3. Run `uv run pytest -q`, `uv run ruff check .`, `uv run basedpyright`, and `uv run python scripts/audit_release.py`.
4. Keep source books, exams, answers, caches, generated packages, credentials, and personal information out of commits.
5. Open a pull request that explains the learner impact and verification performed.

Do not contribute copyrighted provider content. Fixtures must be short, original, synthetic examples.
