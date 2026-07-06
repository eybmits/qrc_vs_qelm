from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IPC_PILOT = ROOT / "scripts" / "ipc_pilot.py"

spec = importlib.util.spec_from_file_location("ipc_pilot", IPC_PILOT)
ipc_pilot = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = ipc_pilot
spec.loader.exec_module(ipc_pilot)


def _fake_score_row(
    *,
    T: int,
    cell,
    gamma: float,
    window_k: int,
    model: str,
    ipc_total: float,
) -> dict:
    raw_total = float(ipc_total) * 0.8
    return {
        "T": int(T),
        "washout": max(1, int(T) // 10),
        "cell_id": str(cell["cell_id"]),
        "input_seed": int(cell["seed_cell"].input_seed),
        "coupling_seed": int(cell["seed_cell"].coupling_seed),
        "shot_seed": int(cell["seed_cell"].shot_seed),
        "window_k": int(window_k),
        "gamma": float(gamma),
        "dissipator": ipc_pilot.PILOT_DISSIPATOR,
        "n_qubits": 4,
        "D": 8,
        "n_surrogates": 1,
        "basis_max_delay": 4,
        "basis_max_degree": 2,
        "basis_max_total_degree": 2,
        "basis_max_functions": 16,
        "split_rows": {"train": 10, "val": 5, "test": 5, "first": 0, "last": T - 1},
        "model": model,
        "features_dead": False,
        "liveness_failure": None,
        "ipc_total": float(ipc_total),
        "ipc_mc": float(ipc_total) * 0.5,
        "ipc_nc": float(ipc_total) * 0.5,
        "raw_total_ipc": raw_total,
        "effective_rank": 10,
        "rank_clamp_fraction": raw_total / 10.0,
        "n_significant": 3,
        "threshold_saturated": False,
        "threshold": 0.01,
        "basis_edge_capacity": 0.01,
        "surrogate_retained_capacity_max": 0.0,
        "surrogate_retained_capacity_mean": 0.0,
        "n_basis": 8,
        "truncated": False,
    }


def test_main_smoke_writes_well_formed_pilot_json(monkeypatch, tmp_path: Path) -> None:
    def fake_score_grid(
        *,
        base_config,
        out_dir,
        cells,
        t_value,
        windows,
        gammas,
        profile,
    ):
        rows = []
        for gamma in gammas:
            for window_k in windows:
                for cell in cells:
                    t_factor = 1.0 if int(t_value) < 96 else 1.02
                    qrc_total = t_factor * (0.7 + float(gamma))
                    delta = -0.08 if float(gamma) < 0.1 else 0.12
                    qelm_total = qrc_total - delta + 0.01 / int(window_k)
                    rows.append(
                        _fake_score_row(
                            T=int(t_value),
                            cell=cell,
                            gamma=float(gamma),
                            window_k=int(window_k),
                            model="qrc",
                            ipc_total=qrc_total,
                        )
                    )
                    rows.append(
                        _fake_score_row(
                            T=int(t_value),
                            cell=cell,
                            gamma=float(gamma),
                            window_k=int(window_k),
                            model="qelm",
                            ipc_total=qelm_total,
                        )
                    )
        return rows

    monkeypatch.setattr(ipc_pilot, "_score_grid", fake_score_grid)
    monkeypatch.setattr(
        ipc_pilot,
        "_spectral_gap_by_gamma",
        lambda **kwargs: [
            {
                "gamma": float(gamma),
                "s_value": 0.5,
                "n_couplings": 1,
                "coupling_seeds": [987654],
                "spectral_gap_mean": float(gamma) + 0.01,
                "spectral_gap_median": float(gamma) + 0.01,
                "spectral_gap_values": [float(gamma) + 0.01],
                "errors": [],
            }
            for gamma in kwargs["gammas"]
        ],
    )

    rc = ipc_pilot.main(["--profile", "smoke", "--out", str(tmp_path)])

    assert rc == 0
    payload = json.loads((tmp_path / "pilot.json").read_text(encoding="utf-8"))
    assert payload["profile"] == "smoke"
    assert payload["pilot_only"] is True
    assert payload["confirmatory_seeds_touched"] is False
    assert {"part_a", "part_b", "part_c"} <= set(payload)
    assert payload["part_a"]["selection"]["found"] is True
    assert payload["part_a"]["selection"]["t_star"] == 96
    assert payload["part_b"]["signed_delta_table"]
    assert payload["part_b"]["delta_infty"]
    assert "go_no_go" in payload["part_c"]


def test_t_reliability_logic_picks_stabilized_T_on_synthetic_fixture() -> None:
    rows = []
    for T, qrc, qelm in [
        (500, 1.0, 0.80),
        (1000, 1.30, 1.00),
        (2000, 1.32, 1.02),
    ]:
        rows.extend(
            [
                {
                    "T": T,
                    "series_id": "QRC",
                    "ipc_total_mean": qrc,
                    "rank_clamp_fraction_max": 0.50,
                    "basis_edge_capacity_max": 0.01,
                },
                {
                    "T": T,
                    "series_id": "QELM(k=4)",
                    "ipc_total_mean": qelm,
                    "rank_clamp_fraction_max": 0.55,
                    "basis_edge_capacity_max": 0.02,
                },
            ]
        )
    selection = ipc_pilot.select_stabilized_T(
        pd.DataFrame(rows),
        tolerance=0.05,
        rank_clamp_fraction_max=0.98,
        basis_edge_max=0.05,
    )

    assert selection["found"] is True
    assert selection["t_star"] == 2000


def test_sign_crossover_detection_fires_on_synthetic_delta_infty() -> None:
    delta_infty = pd.DataFrame(
        [
            {"gamma": 0.02, "window_k": 12, "delta_mean": -0.20},
            {"gamma": 0.10, "window_k": 12, "delta_mean": -0.10},
            {"gamma": 0.30, "window_k": 12, "delta_mean": 0.10},
            {"gamma": 0.50, "window_k": 12, "delta_mean": 0.20},
        ]
    )

    crossover = ipc_pilot.detect_sign_crossover(delta_infty)

    assert crossover["both_signs_present"] is True
    assert crossover["gamma_star"] is not None
    assert 0.10 < crossover["gamma_star"] < 0.30
