import dataclasses
import json

import pytest

from qrc_vs_qelm.seeds import SEED_GRIDS, SeedCell, grid_manifest


def test_seed_grids_have_frozen_expected_sizes():
    assert len(SEED_GRIDS["stage1_smoke"]) == 1
    assert len(SEED_GRIDS["stage1_full"]) == 30
    assert len(SEED_GRIDS["stage2_shots"]) == 9
    assert len(SEED_GRIDS["stage3_gamma"]) == 10
    assert len(SEED_GRIDS["pilot"]) == 15

    assert isinstance(SEED_GRIDS["stage1_full"], tuple)
    with pytest.raises(TypeError):
        SEED_GRIDS["new"] = (SeedCell(1, 2),)  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        SEED_GRIDS["stage1_smoke"][0].input_seed = 99  # type: ignore[misc]


def test_pilot_seed_design_is_disjoint_from_confirmatory_grids():
    pilot = {(cell.input_seed, cell.coupling_seed) for cell in SEED_GRIDS["pilot"]}
    pilot_input_seeds = {cell.input_seed for cell in SEED_GRIDS["pilot"]}
    pilot_coupling_seeds = {cell.coupling_seed for cell in SEED_GRIDS["pilot"]}

    assert pilot_input_seeds == {900, 901, 902, 903, 904}
    assert pilot_coupling_seeds == {987654, 123456, 555553}
    assert len(pilot) == len(pilot_input_seeds) * len(pilot_coupling_seeds)

    for name, grid in SEED_GRIDS.items():
        if name == "pilot":
            continue
        confirmatory = {(cell.input_seed, cell.coupling_seed) for cell in grid}
        confirmatory_input_seeds = {cell.input_seed for cell in grid}
        confirmatory_coupling_seeds = {cell.coupling_seed for cell in grid}
        assert pilot.isdisjoint(confirmatory), name
        assert pilot_input_seeds.isdisjoint(confirmatory_input_seeds), name
        assert pilot_coupling_seeds.isdisjoint(confirmatory_coupling_seeds), name


def test_grid_manifest_round_trips_to_json():
    manifest = grid_manifest("stage2_shots")
    encoded = json.dumps(manifest, sort_keys=True)
    decoded = json.loads(encoded)

    assert decoded["name"] == "stage2_shots"
    assert decoded["n_cells"] == 9
    assert decoded["cell_count"] == 9
    assert decoded["cells"][0] == {
        "input_seed": 0,
        "coupling_seed": 314159,
        "shot_seed": 0,
    }
