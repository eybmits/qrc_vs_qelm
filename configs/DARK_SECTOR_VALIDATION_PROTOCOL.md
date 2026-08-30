# Frozen dark-sector validation protocol

Frozen on 2026-08-28 before generation of the full validation output.

The primary claim is deliberately existential and architecture-specific.  In a
two-qubit QRC with permutation-symmetric Hamiltonian terms and collective loss,
the entangled singlet projector is conserved.  Two old histories are generated
from `|00>` by either doing nothing or applying the addressable pulse
`exp(-i X_0/2)`.  Both branches then receive the same 80-symbol suffix.  The
registered target is the singlet-projector expectation, measurable from the
`XX`, `YY`, and `ZZ` correlators.

The primary result passes only if all 12 independently seeded coupling/suffix
roots satisfy the five numerical checks frozen in `dark_sector_validation.json`.
The exact predicted target gap is `sin(1/2)^2/2`; the balanced suffix-only MSE
floor is one quarter of its square.  These checks establish a concrete QRC/task
for which no deterministic reset-on-window QELM, regardless of its internal
feature count, can uniformly emulate all histories at any fixed finite window.

The claim does **not** say that every QRC defeats every QELM, that a stateful
student fails, or that the construction gives quantum computational advantage.
Matched local loss and asymmetric collective jumps are frozen as falsification
controls: they must remove exact protection, and the latter must show the exact
quadratic initial leakage predicted from the jump operator.
