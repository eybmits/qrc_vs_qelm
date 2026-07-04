from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

import qrc_vs_qelm.gates as gates_module
from qrc_vs_qelm.gates import (
    GATE_DEFS,
    evaluate_stage1_gates,
    evaluate_stage2_gates,
    evaluate_stage3_gates,
    gates_markdown_table,
    stage_verdict,
)


def _stage1_frame(labels, *, qelm_trivial=False):
    rows = []
    for family_idx, label in enumerate(labels):
        for seed in range(8):
            jitter = 0.001 * seed
            if label == "QRC":
                cap_qrc = 0.72 + jitter
                cap_qelm = 0.52 + jitter
            elif label == "QELM":
                cap_qrc = 0.48 + jitter
                cap_qelm = 0.70 + jitter
            else:
                cap_qrc = 0.60 + jitter
                cap_qelm = 0.595 + jitter
            linear_delay = cap_qelm + 0.02 if qelm_trivial else 0.40
            rows.append(
                {
                    "cell_id": seed,
                    "family_cell": f"tau{family_idx}_p2",
                    "tau": family_idx,
                    "p": 2,
                    "capacity_qrc": cap_qrc,
                    "capacity_qelm": cap_qelm,
                    "capacity_linear_delay": linear_delay,
                    "delta": cap_qrc - cap_qelm,
                }
            )
    return pd.DataFrame(rows)


def _record(records, gate_id):
    matches = [record for record in records if record.gate_id == gate_id]
    assert matches
    return matches[-1]


def _stage1_linear_delay_frame(labels, qelm_minus_linear_by_family):
    rows = []
    for family_idx, label in enumerate(labels):
        deltas = qelm_minus_linear_by_family[family_idx]
        for seed, qelm_minus_linear in enumerate(deltas):
            if label == "QRC":
                cap_qrc = 0.72
                cap_qelm = 0.52
            elif label == "QELM":
                cap_qrc = 0.48
                cap_qelm = 0.70
            else:
                cap_qrc = 0.60
                cap_qelm = 0.595
            rows.append(
                {
                    "cell_id": seed,
                    "family_cell": f"tau{family_idx}_p2",
                    "tau": family_idx,
                    "p": 2,
                    "capacity_qrc": cap_qrc,
                    "capacity_qelm": cap_qelm,
                    "capacity_linear_delay": cap_qelm - float(qelm_minus_linear),
                }
            )
    return pd.DataFrame(rows)


def _stage1_significance_filter_frame():
    rows = []
    qrc_straddling = [0.04] * 12 + [-0.20]
    cells = [
        ("QRC_straddle", qrc_straddling),
        ("QRC_straddle", qrc_straddling),
        ("QRC_straddle", qrc_straddling),
        ("QELM_significant", [-0.08] * 13),
        ("QELM_significant", [-0.08] * 13),
    ]
    for family_idx, (label, deltas) in enumerate(cells):
        for seed, delta in enumerate(deltas):
            cap_qelm = 0.60
            rows.append(
                {
                    "cell_id": seed,
                    "family_cell": f"tau{family_idx}_p2",
                    "tau": family_idx,
                    "p": 2,
                    "capacity_qrc": cap_qelm + float(delta),
                    "capacity_qelm": cap_qelm,
                    "capacity_linear_delay": 0.40,
                    "expected_label": label,
                }
            )
    return pd.DataFrame(rows)


def _stage2_frame(shot_values, *, shift):
    rows = []
    inf_shots, finite_shots = shot_values
    for cell_id in range(8):
        tiers = [
            (inf_shots, 1),
            (finite_shots, 3 if shift else 1),
        ]
        for shot_value, favored_count in tiers:
            for family_idx in range(4):
                delta = 0.10 if family_idx < favored_count else -0.10
                rows.append(
                    {
                        "cell_id": cell_id,
                        "family_cell": f"tau{family_idx}_p2",
                        "p": 2,
                        "shots": shot_value,
                        "capacity_qrc": 0.50 + delta,
                        "capacity_qelm": 0.50,
                    }
                )
    return pd.DataFrame(rows)


def _stage3_frame(*, crossover, monotone):
    rows = []
    gammas = [0.0, 0.1, 0.2, 0.3, 0.4]
    mc_values = (
        [1.0, 0.8, 0.6, 0.4, 0.2]
        if monotone
        else [0.2, 0.4, 0.6, 0.8, 1.0]
    )
    nc_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    for gamma, mc, nc in zip(gammas, mc_values, nc_values, strict=True):
        for family_idx in range(4):
            qelm_catches_qrc = crossover and gamma >= 0.2 and family_idx < 2
            cap_qrc = 0.50 if qelm_catches_qrc else 0.60
            cap_qelm = 0.55 if qelm_catches_qrc else 0.50
            rows.append(
                {
                    "gamma": gamma,
                    "family_cell": f"tau{family_idx}_p2",
                    "p": 2,
                    "capacity_qrc": cap_qrc,
                    "capacity_qelm": cap_qelm,
                    "mc": mc,
                    "nc": nc,
                }
            )
    return pd.DataFrame(rows)


def test_stage1_clean_two_regime_frame_passes_g1b_and_stage():
    records = evaluate_stage1_gates(
        _stage1_frame(["QRC", "QRC", "QRC", "QELM", "QELM", "QELM"]),
        n_boot=128,
    )

    g1b = _record(records, "G1b")
    assert g1b.passed
    assert g1b.value["qrc_fraction"] == pytest.approx(0.5)
    assert g1b.value["qelm_fraction"] == pytest.approx(0.5)
    assert stage_verdict(records, stage=1)["verdict"] == "PASS"


def test_stage1_all_qrc_frame_is_single_regime():
    records = evaluate_stage1_gates(
        _stage1_frame(["QRC", "QRC", "QRC", "QRC", "QRC"]),
        n_boot=128,
    )

    g1b = _record(records, "G1b")
    verdict = stage_verdict(records, stage=1)

    assert not g1b.passed
    assert g1b.verdict == "SINGLE_REGIME"
    assert verdict["verdict"] == "SINGLE_REGIME"


def test_stage1_qelm_not_beating_linear_delay_triggers_kill_rule():
    records = evaluate_stage1_gates(
        _stage1_frame(["QRC", "QRC", "QELM", "QELM"], qelm_trivial=True),
        n_boot=128,
    )

    g1c = _record(records, "G1c")
    verdict = stage_verdict(records, stage=1)

    assert not g1c.passed
    assert g1c.verdict == "QELM_TRIVIAL"
    assert verdict["verdict"] == "QELM_TRIVIAL"
    assert verdict["failed_gates"] == ["G1c"]


def test_stage1_g1c_rejects_raw_mean_linear_delay_flukes():
    noisy = [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
    records = evaluate_stage1_gates(
        _stage1_linear_delay_frame(
            ["QRC", "QRC", "QELM", "QELM"],
            [noisy, noisy, noisy, noisy],
        ),
        n_boot=512,
    )

    g1c = _record(records, "G1c")
    verdict = stage_verdict(records, stage=1)

    assert not g1c.passed
    assert g1c.verdict == "QELM_TRIVIAL"
    assert verdict["verdict"] == "QELM_TRIVIAL"


def test_stage1_g1c_accepts_significant_qelm_over_linear_delay_floor():
    noisy = [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
    records = evaluate_stage1_gates(
        _stage1_linear_delay_frame(
            ["QRC", "QRC", "QELM", "QELM"],
            [[0.08] * 8, noisy, noisy, noisy],
        ),
        n_boot=512,
    )

    g1c = _record(records, "G1c")

    assert g1c.passed
    assert g1c.value == pytest.approx(0.25)


def test_stage1_g1b_excludes_ci_straddling_direction_cells():
    records = evaluate_stage1_gates(_stage1_significance_filter_frame(), n_boot=256)

    g1b = _record(records, "G1b")
    verdict = stage_verdict(records, stage=1)

    assert not g1b.passed
    assert g1b.verdict == "SINGLE_REGIME"
    assert g1b.value["qrc_fraction"] == pytest.approx(0.0)
    assert g1b.value["qelm_fraction"] == pytest.approx(0.4)
    assert "direction_labels: QRC=3" in g1b.notes
    assert verdict["verdict"] == "SINGLE_REGIME"


def test_stage1_wired_tie_band_changes_labeling(monkeypatch):
    df = _stage1_frame(["QRC"])
    default_records = gates_module.evaluate_stage1_gates(df, n_boot=128)
    assert "label=QRC" in _record(default_records, "G1a").notes

    monkeypatch.setattr(gates_module, "TIE_BAND", 0.30)
    wide_records = gates_module.evaluate_stage1_gates(df, n_boot=128)

    assert "label=tie" in _record(wide_records, "G1a").notes


@pytest.mark.parametrize("shot_values", [(np.inf, 10000.0), ("inf", "1e4")])
def test_stage2_g2a_passes_when_qrc_favored_count_shifts_by_shot_tier(shot_values):
    records = evaluate_stage2_gates(_stage2_frame(shot_values, shift=True), n_boot=128)

    g2a = _record(records, "G2a")

    assert g2a.passed
    assert g2a.verdict == "PASS"
    assert g2a.ci_low > 0.0


def test_infinite_shot_detection_keeps_finite_large_tiers_finite():
    values = pd.Series(["∞", "inf", "infinity", np.inf, "1e5", "100000", 100000])

    mask = gates_module._is_infinite_shots(values)

    assert mask.tolist() == [True, True, True, True, False, False, False]


@pytest.mark.parametrize(
    ("shift", "expected_verdict", "expected_value"),
    [(True, "PASS", 2.0), (False, "SHOT_INSENSITIVE", 0.0)],
)
def test_stage2_g2a_accepts_unicode_infinity_shot_tier(
    shift,
    expected_verdict,
    expected_value,
):
    records = evaluate_stage2_gates(
        _stage2_frame(("∞", 10000), shift=shift),
        n_boot=128,
    )

    g2a = _record(records, "G2a")

    assert g2a.verdict == expected_verdict
    assert g2a.value == expected_value
    assert "requires both S=infinity and S=1e4 rows" not in g2a.notes


def test_stage2_g2a_shot_insensitive_when_counts_do_not_move():
    records = evaluate_stage2_gates(
        _stage2_frame((np.inf, 10000.0), shift=False),
        n_boot=128,
    )

    g2a = _record(records, "G2a")

    assert not g2a.passed
    assert g2a.verdict == "SHOT_INSENSITIVE"


def test_stage2_g2a_missing_shot_column_returns_failure_verdict():
    df = _stage2_frame((np.inf, 10000.0), shift=True).drop(columns=["shots"])

    records = evaluate_stage2_gates(df, n_boot=128)
    g2a = _record(records, "G2a")

    assert not g2a.passed
    assert g2a.verdict == "SHOT_INSENSITIVE"
    assert "missing shot-count column" in g2a.notes


def test_stage3_g3a_passes_for_monotone_tradeoff_with_crossover():
    records = evaluate_stage3_gates(
        _stage3_frame(crossover=True, monotone=True),
        n_boot=128,
    )

    g3a = _record(records, "G3a")

    assert g3a.passed
    assert g3a.verdict == "PASS"
    assert g3a.value["gamma_star"] == pytest.approx(0.2)
    assert g3a.value["crossover_fraction"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("crossover", "monotone"),
    [
        (False, True),
        (True, False),
    ],
)
def test_stage3_g3a_no_tradeoff_without_crossover_or_monotone_trend(crossover, monotone):
    records = evaluate_stage3_gates(
        _stage3_frame(crossover=crossover, monotone=monotone),
        n_boot=128,
    )

    g3a = _record(records, "G3a")

    assert not g3a.passed
    assert g3a.verdict == "NO_TRADEOFF"


def test_gate_definitions_are_deeply_frozen():
    assert isinstance(GATE_DEFS, MappingProxyType)
    with pytest.raises(TypeError):
        GATE_DEFS["new"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        GATE_DEFS["G1b"]["threshold"]["min_qrc_fraction"] = 0.30  # type: ignore[index]


def test_gates_markdown_table_contains_every_gate_id():
    table = gates_markdown_table()

    assert table.startswith("| Gate | Stage |")
    for gate_id in GATE_DEFS:
        assert gate_id in table
