# Figure 2 renderer

`render_resource_atlas.py` builds the paper's history, feature-capacity, and
measurement-budget panels from the recorded experiment results. It writes
PDF/PNG figures, CSV source tables, and a JSON provenance ledger.

Run from the repository root:

```sh
MPLCONFIGDIR=/tmp/qrc-mpl-cache .venv/bin/python paper/figure2_candidates/render_resource_atlas.py --publication --output paper/figures/resource_atlas
.venv/bin/python -m pytest -q paper/figure2_candidates/test_resource_atlas.py
```

Tests verify source coverage, aggregation, validation-choice frequencies,
uncertainty intervals, layout, and deterministic rendering. To inspect a separate
preview, supply another `--output` directory.
