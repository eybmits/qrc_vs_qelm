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
| [WP0](.specs/WP0.md) | Scaffold and governance | Complete |
| [WP1](.specs/WP1.md) | Lindblad core port | Complete |
| [WP2](.specs/WP2.md) | QELM model | Pending |
| [WP3](.specs/WP3.md) | QRC/QELM feature pipelines | Pending |
| [WP4](.specs/WP4.md) | Resource matching and readouts | Pending |
| [WP5](.specs/WP5.md) | Experiment runners | Pending |
| [WP6](.specs/WP6.md) | Analysis and reporting | Pending |
| [WP7](.specs/WP7.md) | Protocol and preregistration content | Pending |
