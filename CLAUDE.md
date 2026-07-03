# CLAUDE.md

Claude writes one work-package specification at a time in `.specs/WP*.md`.
Codex implements exactly one assigned work package from that specification.

Sonnet subagents may review focused slices of the implementation, but each
delegated task must cover only one work package. Implementers must not commit
`results_*/`, `data/`, `.npz`, or other generated experiment output directories.
