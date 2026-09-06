# Prospective five-qubit, long-context emulation follow-up

This is a separate prospective amendment designed after the N=2,3,4 studies
were opened. It neither changes their frozen protocols nor joins their root
denominators. The parent investigator approved this exact scope before any
new follow-up target outcomes were inspected on 6 September 2026.

## Question and fixed design

Does the history/representation distinction remain observable for a larger
recurrent target and windows beyond six inputs, when every supplied lag is
actually encoded by the reset emulator?

- Target: five-qubit input-driven TFIM Lindblad QRC, local loss, with rates
  `gamma={1,4}` and inherited `dt=.5,h=.5,h_in=1,J_s=1`.
- Outputs: all single-site and same-axis two-site X,Y,Z expectations (45).
- Windows: exactly `d={2,4,6,8,10}`, the nested tails of one length-ten suffix.
- Emulator: fixed ten-qubit unitary TFIM reset-QELM, one input per distinct
  qubit, unused qubits in `|0>`, fixed independent dynamics. No old lag is
  discarded, recycled, or silently padded into a six-qubit input register.
- Single-site X,Y,Z features at 1,2,4 nested checkpoints from the four-point
  grid through evolution time .5: exactly 30,60,120 features. The register is
  fixed at ten qubits throughout this follow-up; these budgets are not the
  six-qubit budgets from the original experiment.
- Twelve new roots `long_n5_01` through `long_n5_12`; all role seeds are
  explicit in `configs/canonical_emulation_long_n5.yaml` and checked against
  existing structured seed records before freezing.
- Independently generated train/validation/test streams with 480/160/320
  scored rows and 128 burn-in inputs; rows within a stream remain correlated.
  32 common-suffix groups with four independent old prefixes each. Inputs
  use the inherited 33-symbol alphabet. Newest suffix symbols are unique;
  shorter windows are exact tails, preserving paired comparisons.
- Controls are unchanged: same-physics reset replay, constant, linear delay,
  full quadratic delay, and random tanh features matched to each feature
  budget. Every model receives the identical windows and row splits.
- Ridge grid, resolved-subspace cutoff, training-only standardization,
  validation-only selection, and exact suffix risk decomposition are inherited
  from the unchanged canonical runner. No held-out result selects a feature
  budget, loss rate, window, root, or reported cell.

The losses select one inherited slower-forgetting condition and one fast-loss
anchor; no fresh outcomes informed this choice. The study tests two loss
conditions, not a dense phase boundary or an asymptotic size law.

## Numerical backend and pre-opening evidence

The target remains the original dense Lindblad implementation with cached
propagators. The reset emulator uses identical dense Hamiltonians and unitary
checkpoints. A separate backend computes the same product-state encoding and
local Pauli expectations using basis-index permutations instead of dense
observable contractions. This is an algebraic computational optimization,
not a change to the feature map, training, or resources.

The original canonical sources remain untouched. A scoped factory override
is recorded in the new runner; it is restored when execution exits. Before
opening any `long_n5_*` root, deterministic tests must compare optimized
features with the original dense implementation for randomized N<=4 cases
and independent ten-qubit probes. Maximum feature disagreement must be at
most 1e-11. These tests and runtime preflights use development seeds, not
follow-up seeds. Two executions of the separate development smoke profile
must have distinct execution IDs/directories and identical hashes for the
stream, suffix, system and summary tables.

The freeze binds this protocol, the exact configuration, all canonical
numerical source files, the new runner/backend/analysis/audit files and their
tests, plus the prior-seed registry. Figure, manuscript and changing visual
test files are deliberately excluded. The test command snapshots numerical
material before and after executing the complete fixed test-file list,
checks that JUnit contains the complete collected case inventory, and binds
the zero-failure raw report to that material. Raw JUnit is rechecked during
freeze verification; stale or partial reports cannot be rebound by a separate
post-hoc command. Fresh execution refuses
missing/mismatched freeze evidence or a nonempty output directory.

## Prespecified engineering and scientific summaries

Engineering checks:

1. Complete grid: 24 target systems, 120 root-loss-window cells, 360 QELM
   root-budget cells and 1,200 rows in each stream/suffix table. Each target
   system must be live at the inherited variance cutoff 1e-8.
2. Suffix decomposition error <=1e-10, identical-window prediction agreement,
   and all original sampled suffix bounds hold. Recompute normalized errors
   from MSE and target variance rather than trusting stored labels.
3. Every cell contains replay, constant, linear, quadratic and the three
   QELM/tanh budgets. History terms agree across models. All checks retain
   failures rather than filtering unsuccessful roots.
4. Same-material duplicate smoke evidence, backend equivalence, fresh seeds
   and artifact hashes verify before fresh execution and independently at
   aggregation. Runtime is descriptive, not a success criterion.

Prespecified paired history contrasts use each root as the independent unit:

- For each loss separately, report relative history-NRMSE reduction from d=2
  to d=10, the median, a two-sided 95% root bootstrap interval, and the number
  of positive roots out of the fixed 12. The directional prediction is at
  least 10 positive roots for each loss; no minimum effect size is imposed.
- At gamma=1, also report the descriptive d=6 to d=10 contrast and counts of
  roots crossing the 0.10 history target, without an additional pass gate.
  This is the longer-than-original-window check. These contrasts compare
  the inherited nested-suffix diagnostics; target endpoint distributions and
  normalization variances can differ with window length. They are not a
  same-trajectory causal context intervention.
- Bootstrap whole roots 10,000 times with the explicit analysis seed. No
  individual suffix row, feature budget, or loss/window cell is an independent
  replicate. A failed liveness or nonfinite contrast is reported as failed
  rather than silently omitted from the denominator.

For all ten aggregate loss-window cells, report root median/min/max and
10th/90th percentiles of history error, with exact history-success counts.
History is robustly sufficient when at least 10/12 roots have error <=0.10.
For each feature budget report QELM total and representation errors, root
regimes, and the number of successes. The smallest successful tested budget
is descriptive; retain cases in which larger windows or budgets fail.

Report all stream and suffix errors for every control. The explicit
root-level classical comparison is fixed B=120 QELM versus the quadratic
delay model at each loss/window; provide medians, root bootstrap intervals,
and positive-root counts. Additional controls and budgets remain descriptive,
not a per-test-cell oracle used to advertise an advantage. No QELM success,
classical win, or overall scientific pass is required for inclusion.

## Output and claim boundary

All generated results remain under the new ignored directory
`results_n5_long_context_v1/`: `smoke_a`, `smoke_b`, `run`, and `analysis`.
The aggregate source and root-contrast tables are auditable plot sources;
the original frozen figure data are never overwritten or pooled with them.

This follow-up can establish measured extension to five target qubits and
ten explicit inputs within this local-loss family. It cannot establish a
universal boundary, efficient asymptotic QELM scaling, hardware validity,
quantum advantage, or that one particular reset-QELM is universally dense.

The larger ten-qubit feature register is an explicit resource change from
the earlier six-qubit study. Within-follow-up window and budget comparisons
hold it fixed; cross-study improvements cannot be attributed to context
alone. All original G0-G6 and secondary decisions remain unchanged.
