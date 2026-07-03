# CLAUDE.md

Claude may pre-stage work-package specifications in `.specs/WP*.md`. Codex
implements exactly one assigned work package at a time from those specifications;
the one-at-a-time rule applies to execution, not specification authoring.

Sonnet subagents may review focused slices of the implementation, but each
delegated task must cover only one work package. Implementers must not commit
`results_*/`, `data/`, `.npz`, or other generated experiment output directories.
