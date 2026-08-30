from qrc_vs_qelm.canonical_resources import (
    canonical_resource_report,
    expected_observable_contract,
    observable_names_sha256,
)
from qrc_vs_qelm.core import readout


def _profile() -> dict:
    return {
        "study": "canonical_qrc_to_parallel_qelm_emulation",
        "profile": "smoke",
        "teacher_n_qubits": [2, 3],
        "teacher": {
            "target_observables": {"axes": ["x", "y", "z"], "max_weight": 2}
        },
        "student": {
            "n_qubits": 6,
            "observables": {"axes": ["x", "y", "z"], "max_weight": 1},
        },
        "windows": [1, 2, 4, 6],
        "snapshot_budgets": [1, 2, 4],
        "max_snapshots": 4,
        "suffix_groups": 4,
        "prefix_branches": 3,
        "prefix_burnin": 16,
        "rows": {"train": 64, "val": 24, "test": 40},
    }


def test_expected_observable_contract_matches_production_order():
    for n_qubits, expected_count in ((2, 9), (3, 18)):
        observables = readout.pauli_observables(
            n_qubits,
            axes=("x", "y", "z"),
            max_weight=2,
        )
        contract = expected_observable_contract(
            n_qubits,
            {"axes": ["x", "y", "z"], "max_weight": 2},
        )

        assert contract["count"] == expected_count
        assert contract["names_sha256"] == observable_names_sha256(
            item.name for item in observables
        )


def test_resource_report_separates_runtime_from_deterministic_proxies():
    report = canonical_resource_report(_profile(), wall_clock_seconds=1.25)

    assert report["decision_bearing"] is False
    assert report["runtime"]["wall_clock_seconds"] == 1.25
    assert report["state_width"]["teacher_density_matrix_complex_entries"] == {
        "2": 16,
        "3": 64,
    }
    assert report["state_width"]["student_statevector_complex_amplitudes"] == 64
    assert report["circuit_depth_proxies"]["student_parallel_encoding_layers"] == 1
    assert report["measurement_setting_proxies"]["student_features_per_budget"] == {
        "1": 18,
        "2": 36,
        "4": 72,
    }
    assert report["measurement_setting_proxies"][
        "student_global_basis_settings_per_budget"
    ] == {"1": 3, "2": 6, "4": 12}
