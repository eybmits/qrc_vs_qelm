# qrc_vs_qelm

`qrc_vs_qelm` is a pre-registered, resource-matched research package comparing
Quantum Reservoir Computing, modeled as a recurrent dissipative Lindblad spin
reservoir with intrinsic memory, against Quantum Extreme Learning Machines,
modeled as static memoryless quantum channels applied to classical
delay-embedded inputs.

## Repo Layout

- `configs/` - smoke and full experiment configuration stubs.
- `src/qrc_vs_qelm/` - Python package source.
- `tests/` - scaffold and regression tests.
- `PROTOCOL.md` - protocol skeleton for WP7 completion.
- `PREREGISTRATION.md` - preregistration skeleton for WP7 completion.
- `AGENTS.md` - implementation governance for agents.
- `CLAUDE.md` - work-package workflow notes.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
```

## Work-Package Index

| WP | Scope | Status |
| --- | --- | --- |
| WP0 | Scaffold and governance | Complete |
| WP1 | Lindblad core port | Pending |
| WP2 | Task construction | Pending |
| WP3 | QRC/QELM feature pipelines | Pending |
| WP4 | Resource matching and readouts | Pending |
| WP5 | Experiment runners | Pending |
| WP6 | Analysis and reporting | Pending |
| WP7 | Protocol and preregistration content | Pending |
