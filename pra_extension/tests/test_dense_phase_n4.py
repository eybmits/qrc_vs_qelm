from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dense_phase_n4_common import (  # noqa: E402
    CONFIG_PATH,
    FREEZE_PATH,
    _csv_structured_seeds,
    assert_seed_disjointness,
    dense_n4_seed_map,
    root_set_sha256,
    validate_dense_n4_configuration,
    verify_dense_n4_freeze,
)
from analyze_dense_phase_n4 import aggregate, validate_results  # noqa: E402


ROOT_SET_SHA256 = "fe8dfffb532ba6bbec8ba439c774a06f466bbbf40b74a45214c45cbbcca5afb3"
RESULTS = ROOT / "results_dense_phase_n4_v1"
ANALYSIS = ROOT / "results_dense_phase_n4_v1_analysis"


def test_dense_n4_configuration_and_seed_namespace_are_exact() -> None:
    assert validate_dense_n4_configuration(CONFIG_PATH) == {
        "passed": True,
        "systems": 132,
        "target_live_systems": 132,
        "stream_rows": 7_920,
        "suffix_rows": 7_920,
        "aggregate_cells": 66,
        "history_root_cells": 792,
        "ptd_root_budget_cells": 2_376,
        "scaling_root_fits": 12,
        "positive_loss_cells_per_fit": 60,
        "root_set_sha256": ROOT_SET_SHA256,
    }
    assert root_set_sha256(CONFIG_PATH) == ROOT_SET_SHA256
    seeds = dense_n4_seed_map(CONFIG_PATH)
    assert len(seeds) == 96
    assert min(seeds) == 94_010_001
    assert max(seeds) == 94_080_012
    record = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    audit = assert_seed_disjointness(
        CONFIG_PATH,
        record["prior_seed_source_files"].keys(),
    )
    assert audit["passed"] is True
    assert audit["n4_seed_count"] == 96
    assert audit["prior_seed_registry_sha256"] == record["prior_seed_registry_sha256"]


def test_dense_n4_material_and_prior_registry_were_frozen() -> None:
    freeze = verify_dense_n4_freeze(CONFIG_PATH, FREEZE_PATH)
    assert freeze["passed"] is True
    assert all(freeze["checks"].values())


def test_dense_n4_configuration_drift_fails_closed(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["common"]["teacher"]["gammas"][1] = 0.25
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="gamma/window grid"):
        validate_dense_n4_configuration(changed)


def test_structured_csv_seed_parser_detects_lists_but_ignores_large_metrics(
    tmp_path: Path,
) -> None:
    source = tmp_path / "seed_fixture.csv"
    source.write_text(
        "root_seed,input_seeds_kept,large_metric\n11,12|13,94010001\n",
        encoding="utf-8",
    )
    assert _csv_structured_seeds(source, "seed_fixture.csv") == [
        (11, "seed_fixture.csv:2:root_seed[0]"),
        (12, "seed_fixture.csv:2:input_seeds_kept[0]"),
        (13, "seed_fixture.csv:2:input_seeds_kept[1]"),
    ]


def test_tampered_dense_n4_freeze_record_fails_closed(tmp_path: Path) -> None:
    record = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    record["n4_replication_results_previously_opened"] = True
    changed = tmp_path / "freeze.json"
    changed.write_text(json.dumps(record), encoding="utf-8")
    audit = verify_dense_n4_freeze(CONFIG_PATH, changed)
    assert audit["passed"] is False
    assert audit["checks"]["n4_replication_results_previously_opened"] is False


def test_dense_n4_passes_every_prospectively_frozen_gate() -> None:
    validated = validate_results(RESULTS, CONFIG_PATH)
    cells, scaling, summary = aggregate(validated, CONFIG_PATH)
    assert summary["decision"] == "PASSES_DENSE_N4_EXTRAPOLATION"
    assert summary["integrity_passed"] is True
    assert summary["joint_negative_slope_and_r_squared_roots"] == 12
    assert summary["frontier_d_star_censored_at_7"] == {
        "0": 7,
        "0.2": 7,
        "0.5": 7,
        "0.75": 7,
        "1": 7,
        "1.25": 6,
        "1.5": 5,
        "2": 4,
        "2.5": 3,
        "3": 3,
        "4": 2,
    }
    assert summary["frontier_finite_loss_rates"] == 6
    assert summary["anchor_emulated_roots_B72"] == 12
    assert summary["cell_state_counts"] == {
        "FEASIBLE": 12,
        "HISTORY_MISSING": 47,
        "UNRESOLVED": 7,
    }
    assert len(cells) == 66
    assert len(scaling) == 12
    assert scaling["n_cells"].eq(60).all()
    assert scaling["joint_direction_fit_passed"].all()
    assert scaling["r_squared"].min() == pytest.approx(0.944257622567)
    assert cells["median_history_nrmse"].min() == pytest.approx(0.00186083438979)
    assert cells["median_history_nrmse"].max() == pytest.approx(0.871123279582)


def test_committed_dense_n4_analysis_matches_fresh_validation() -> None:
    validated = validate_results(RESULTS, CONFIG_PATH)
    cells, scaling, summary = aggregate(validated, CONFIG_PATH)
    committed_cells = pd.read_csv(ANALYSIS / "dense_n4_phase_cells.csv")
    committed_scaling = pd.read_csv(ANALYSIS / "dense_n4_root_scaling.csv")
    committed_summary = json.loads(
        (ANALYSIS / "dense_n4_summary.json").read_text(encoding="utf-8")
    )
    pd.testing.assert_frame_equal(
        committed_cells,
        cells,
        check_dtype=False,
        rtol=1e-10,
        atol=1e-12,
    )
    pd.testing.assert_frame_equal(
        committed_scaling,
        scaling,
        check_dtype=False,
        rtol=1e-10,
        atol=1e-12,
    )
    assert committed_summary == summary
