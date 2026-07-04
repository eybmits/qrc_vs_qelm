from pathlib import Path

import json

import yaml

import qrc_vs_qelm


REQUIRED_CONFIG_KEYS = {
    "profile",
    "n_qubits",
    "T",
    "washout",
    "hp_grids",
    "seed_grid",
    "shot_tiers",
    "gamma_grid",
}

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
ROOT = CONFIG_DIR.parent


def test_package_version() -> None:
    assert qrc_vs_qelm.__version__ == "0.1.0"


def test_configs_have_required_keys() -> None:
    for path in (CONFIG_DIR / "smoke.yaml", CONFIG_DIR / "full.yaml"):
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        assert REQUIRED_CONFIG_KEYS <= set(config)


def test_config_values() -> None:
    expected = {
        CONFIG_DIR / "smoke.yaml": {"profile": "smoke", "n_qubits": 4, "T": 400},
        CONFIG_DIR / "full.yaml": {"profile": "full", "n_qubits": 5, "T": 3000},
    }
    for path, values in expected.items():
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        for key, expected_value in values.items():
            assert config[key] == expected_value
        assert config["washout"] == 100


def test_full_config_freezes_wp7_axes() -> None:
    with (CONFIG_DIR / "full.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["primary"]["gamma"] == 0.3
    assert config["primary"]["quantize_levels"] == 33
    assert config["primary"]["observables"]["dimension"] == 45
    assert config["phase_diagram"]["taus"] == [0, 1, 2, 3, 5, 8, 10, 12, 16]
    assert config["phase_diagram"]["degrees"] == [1, 2, 3]
    assert config["phase_diagram"]["robustness_degrees"] == [1, 2, 3, 5]
    assert config["secondary_product_family"]["degree_pairs"] == [[1, 1], [1, 2], [2, 2]]
    assert config["hp_grids"]["qrc"]["cardinality"] == 6
    assert config["hp_grids"]["qelm"]["cardinality"] == 6
    assert config["shot_tiers"] == ["infinity", 100000, 10000, 1000]
    assert config["gamma_grid"] == [0.005, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    assert config["runtime_budget"]["active_shrink"]["stage1"]["applied"] is True


def test_frozen_json_manifest_matches_full_config() -> None:
    with (CONFIG_DIR / "full.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    manifest_path = ROOT / "data" / "frozen" / "frozen_v1_config.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["primary"]["n_qubits"] == config["primary"]["n_qubits"]
    assert manifest["primary"]["gamma"] == config["primary"]["gamma"]
    assert manifest["primary"]["observables"]["dimension"] == config["primary"]["observables"]["dimension"]
    assert manifest["phase_diagram"]["taus"] == config["phase_diagram"]["taus"]
    assert manifest["phase_diagram"]["degrees"] == config["phase_diagram"]["degrees"]
    assert manifest["shot_tiers"] == config["shot_tiers"]
    assert manifest["runtime_projection"]["shrink"]["applied"] is True
    stage1 = manifest["runtime_projection"]["stages"]["stage1_active_after_shrink"]
    assert stage1["input_seeds_kept"] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert stage1["projection"]["total_hours"] < manifest["runtime_projection"]["stage_budget_hours"]
