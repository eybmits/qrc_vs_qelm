# Separate long-context follow-up display

Specified before the new N5 outcomes are opened. This display is not a
replacement for the frozen N2--N4 figures and does not pool their roots.

- Question: does longer context remove missing history while a fixed feature
  map can still fail to emulate the resulting window function?
- Two panels, loss rates 1 and 4; five-qubit target, ten-qubit reset emulator,
  and all five windows 2, 4, 6, 8, 10 in both panels.
- Three series on the identical held-out common-suffix panel: old-history
  NRMSE, fixed B=120 QELM total NRMSE, and quadratic-delay total NRMSE.
  B=120 is specified in advance, not chosen by test performance. All other
  budgets and stream controls remain in the numerical analysis.
- Curves show root medians; capped whiskers span the 10th--90th percentiles
  of all twelve roots, not confidence intervals. No individual-root lines,
  smoothing, fitted trends, dropped roots, or clipped uncertainty.
- Consistent Helvetica-family type, square axes, panel letters inside, no
  plot titles, and compact in-panel legends. History: black; reset QELM:
  reference green (#00A14B); quadratic control: reference purple (#A93BFF).
  A gray dashed line denotes 10% NRMSE, not a significance boundary.
- Both panels use the same percentage/logarithmic y scale and real numeric
  window coordinates. Limits cover every displayed percentile and median.
- Every plotted root value is exported with model, root, loss, window,
  metric, and source provenance. The renderer checks upstream integrity
  and hashes before consuming the fresh study.
- The final caption must state the larger feature register and separate
  prospective cohort. Negative or nonmonotone outcomes are retained.

Final pagination may place this display as a compact fourth figure; if it
cannot fit the unchanged twelve-page budget, the manuscript will report the
complete prespecified aggregate result in prose and retain the plot as a
repository artifact rather than shrinking labels below the common style.

## Executed display

The complete prespecified display fits as Figure 4 on page 7 of the
twelve-page ICAART manuscript; the journal-format version is also twelve
pages. All 360 root-series values are retained. The final shared key is
inside panel a only, leaving the small-error whiskers in b unobscured.
Extra upper clearance avoids a panel-letter collision. No scales, labels,
windows, series or roots were selected to favor the QELM.
