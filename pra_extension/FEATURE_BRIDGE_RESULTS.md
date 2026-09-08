# Intermediate features: paired exploratory extension

The extension resolves where the measured decline between B=18 and B=36
occurs. These are new model fits using exact quantum expectations, not
interpolated plot points or new finite-shot observations. The same 12 existing
roots per target size are reused; this is not independent confirmation.

## Reviewed execution

- 36 target/root cells, with target sizes 2, 3, and 4 and a fixed six-qubit QELM.
- 72 genuinely new fits at B=24/30; 72 repeated endpoint controls at B=18/36.
- All 36 reconstructed input/target/feature array dictionaries hash-match
  the original run. All 72 endpoint fits match the original metrics and fitting
  metadata within the protocol tolerance; no failed or omitted production cells.
- 27 development tests passed before the local source lock; two separate
  development smokes produced identical scientific hashes and matching anchors.
- Feature subsets were fixed before intermediate results were generated, use
  every qubit, balance Pauli axes, and never use validation/test outcomes to
  choose observables. Training-only scaling and validation-selected ridge are
  inherited unchanged from the parent study.

## Measured common-suffix NRMSE

Root medians, in percent of the target standard deviation:

| Target QRC | B=18 | B=24 | B=30 | B=36 |
|---|---:|---:|---:|---:|
| 2 qubits | 22.35 | 10.95 | 6.75 | 5.07 |
| 3 qubits | 17.22 | 8.43 | 5.86 | 5.43 |
| 4 qubits | 16.73 | 9.15 | 6.93 | 5.38 |

Thus the observed median decline is already substantial at 24 features and
continues through 30 to 36; it is not a jump that occurs only upon acquiring
the complete second 18-observable bank. Individual-root error need not decrease
at every step. The summary JSON records paired changes and 10th--90th root
percentiles, not confidence intervals or new confirmatory significance tests.

The comparison concerns these nested, balanced feature banks. Feature identity
and checkpoint access matter as well as count. Fitted errors still include
finite-data estimation, regularization and diagnostic-panel sampling, and do
not isolate a universal representation floor or demonstrate quantum advantage.

## Artifacts

- Configuration: `configs/feature_bridge.json`.
- Design and selection rule: `pra_extension/FEATURE_BRIDGE_PROTOCOL.md`.
- Local pre-execution source lock: `configs/feature_bridge_freeze.json`.
- Execution and hashes: `results_feature_bridge_v1/manifest.json`.
- All 144 fit rows: `results_feature_bridge_v1/metrics.csv`.
- Descriptive summary: `results_feature_bridge_v1/summary.json`.

Figure 2b appends only the 72 new B=24/30 rows. Original endpoint rows, original
54/72/108/144 results and all original classical controls remain unchanged.
The complete figure export contains 984 rows: 504 history, 288 features and
192 shot rows. The displayed feature axis shows 18/24/30/36/54/72; 108/144 remain
available in the full source. Both captions disclose the exploratory addition.

Final verification: 777 repository tests passed, including frozen-source,
deterministic replication, fitting, data-to-plot and legend-clearance checks.
All 55 source-lock files also hash-match the archived source snapshot.
Both manuscript versions retain 12 pages; the anonymous source archive
rebuilds byte-identically. Rendered differences are limited to journal pages
5/6 and conference pages 5/7, inspected without overlap or clipping. Fonts,
figure sizes and proofs were not reduced. These checks do not imply venue
policy compliance, author approval or submission readiness.
