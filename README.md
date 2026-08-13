# Exam Prep Agents Skill

Turn exam books and sample exams you are authorized to use into validated, interactive study packages with the AI coding agent you already have.

The same bundled engine works with **Codex, Claude Code, Gemini CLI, GitHub Copilot CLI, and other Agent Skills-compatible hosts**. It generates balanced flashcards and new one-for-one mock exams, then exports self-contained HTML that runs offline on a phone, tablet, or computer.

## What It Produces

- **Flashcard packages:** ten semantic card slots per learning objective by default, grouped by full LO title with multi-select, search, jump navigation, keyboard controls, local progress, and progress import/export.
- **Mock exam packages:** one generated question for each source-exam position, preserving LO, format, cognitive operation, difficulty, and choice count while changing the scenario and reasoning angle.
- **Learner-only downloads:** one interactive `.html` file, optionally wrapped in a ZIP containing only that HTML. No manifest, validation report, source document, or cache is included.

Every candidate passes schema, source-page, coverage, language, source-copy, duplicate, blueprint, and bundled PyTorch quality gates. Supported platforms run the current PyTorch runtime; Intel Macs use a hash-verified portable export of the same float32 classifier. Rejected content is regenerated instead of appearing in the learner package.

## Install

Requirements: Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/lhyvshh/exam-prep-agents-skill.git
cd exam-prep-agents-skill
python3 scripts/install_skill.py --target codex
```

Choose `codex`, `claude`, `gemini`, `copilot`, `agents`, or `all`. The shared `agents` target installs to `~/.agents/skills`, which is recognized by hosts that support the common Agent Skills location.

You can also ask Codex to install the skill directly from:

```text
https://github.com/lhyvshh/exam-prep-agents-skill/tree/main/skills/producing-exam-prep-packages
```

Native host options are documented by [Claude Code](https://code.claude.com/docs/en/skills), [Gemini CLI](https://geminicli.com/docs/cli/using-agent-skills/), and [GitHub Copilot CLI](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/add-skills). Restart or reload the host's skill list after installation.

## Use

Attach or point your agent to the books and, for mock exams, a sample exam with its answer key. Then ask naturally:

```text
Build an offline flashcard package from these CFA books. Use ten cards per learning objective.
```

```text
Build a new 100-question mock exam from these books and this sample exam. Match each source position one for one, reject repeats, and return the interactive HTML.
```

The skill drives a resumable command loop. Long jobs can stop and continue without accepting partial or low-quality output:

```text
prepare-flashcards / prepare-exam -> next -> submit -> render
```

The `frm-part-1` preset preserves the FRM Part I 20/20/30/30 topic weighting fixture. The default `generic` preset keeps an open path for CFA and other exams by deriving structure from source headings instead of inventing a curriculum.

## Privacy

- Books, source exams, answers, generated candidates, and progress remain in the private workspace selected by the user and are ignored by Git.
- The skill repository contains no uploaded books, personal caches, API keys, or generated learner packages.
- No separate model API key is required. Generation uses the active agent subscription and model selected in Codex, Claude Code, Gemini CLI, Copilot CLI, or another host.
- Source text sent to a model is governed by that host's account, privacy settings, and terms. Review those settings before processing confidential material.
- Finished HTML packages make no network requests and need no local app server.

## Development

```bash
uv sync --locked
uv run pytest -q
uv run ruff check .
uv run basedpyright
uv run python scripts/audit_release.py
```

The repository uses synthetic fixtures only. See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution rules and [SECURITY.md](SECURITY.md) for private vulnerability reporting.

## License

[MIT](LICENSE)
