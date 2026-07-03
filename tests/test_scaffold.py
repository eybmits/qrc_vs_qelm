from pathlib import Path

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


def test_package_version() -> None:
    assert qrc_vs_qelm.__version__ == "0.1.0"


def test_config_stubs_have_required_keys() -> None:
    for path in (Path("configs/smoke.yaml"), Path("configs/full.yaml")):
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        assert REQUIRED_CONFIG_KEYS <= set(config)


def test_config_stub_values() -> None:
    expected = {
        "configs/smoke.yaml": {"profile": "smoke", "n_qubits": 4, "T": 400},
        "configs/full.yaml": {"profile": "full", "n_qubits": 5, "T": 3000},
    }
    for path, values in expected.items():
        with Path(path).open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        for key, expected_value in values.items():
            assert config[key] == expected_value
        assert config["washout"] == 100
