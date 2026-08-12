---
name: producing-exam-prep-packages
description: Use when a user wants to turn owned exam books or sample exams into grounded offline flashcards or new mock exams, especially when coverage balance, source-backed explanations, non-repetition, resumability, or interactive HTML downloads matter.
---

# Producing Exam Prep Packages

Build learner-ready artifacts from files the user is authorized to use. Keep source files and all working state in the user-selected private workspace. Never copy books, exams, answer keys, caches, or candidate JSON into this skill directory or the final ZIP.

## Runtime

Set `SKILL_DIR` to the directory containing this file. Run commands with:

```bash
uv run --project "$SKILL_DIR" python "$SKILL_DIR/scripts/run.py" COMMAND
```

Use `--help` for exact arguments. The runtime returns JSON by default so the workflow is consistent across agent hosts.

## Flashcards

1. Run `prepare-flashcards WORKSPACE BOOK... --title TITLE --preset PRESET`.
2. Run `next WORKSPACE` and read the returned work item, instructions, and evidence.
3. Generate exactly one card matching its assigned LO and semantic slot. Use only the evidence. Submit the raw candidate JSON with `submit WORKSPACE CANDIDATE.json`.
4. If rejected, correct every returned issue and retry. Never lower a threshold or bypass an exhausted item.
5. Repeat `next` until it returns `complete`, then run `render WORKSPACE`.

The default is ten distinct cards per learning objective. Preserve the full LO title and physical source-page references.

## Mock Exams

1. Run `prepare-exam WORKSPACE BOOK... --source-exam EXAM --title TITLE --preset PRESET`.
2. For each `next` item, match the source question's LO, cognitive operation, format, choice count, and difficulty while changing the scenario, facts, and reasoning angle.
3. Make every distractor plausible and independently explain why the keyed answer is correct. Verify calculations and claims against the returned book evidence.
4. Submit with `submit`, fix all gate failures, and continue until complete.
5. Run `render WORKSPACE` only after every position is accepted.

Never repeat or lightly paraphrase source questions, generated questions, choices, or scenarios. The runtime maintains the accepted-question registry and applies deterministic checks before its bundled PyTorch checkpoint.

## Presets

- Use `frm-part-1` only for FRM Part I. It preserves the 20/20/30/30 curriculum weighting fixture.
- Use `generic` for CFA, other credentials, and provider-neutral exam books. Infer hierarchy only from source headings; do not invent courses or learning objectives.

## Deliverables

Return the generated `.html` or `.zip` path from `render`. The ZIP must contain exactly one self-contained interactive HTML file. It must work without a server, network request, or API key and provide local progress, import/export, keyboard navigation, and source-page references.
