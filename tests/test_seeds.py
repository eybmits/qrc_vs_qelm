import dataclasses
import json

import pytest

from qrc_vs_qelm.seeds import SEED_GRIDS, SeedCell, grid_manifest


def test_seed_grids_have_frozen_expected_sizes():
    assert len(SEED_GRIDS["stage1_smoke"]) == 1
    assert len(SEED_GRIDS["stage1_full"]) == 30
    assert len(SEED_GRIDS["stage2_shots"]) == 9
    assert len(SEED_GRIDS["stage3_gamma"]) == 10
    assert len(SEED_GRIDS["pilot"]) == 6

    assert isinstance(SEED_GRIDS["stage1_full"], tuple)
    with pytest.raises(TypeError):
        SEED_GRIDS["new"] = (SeedCell(1, 2),)  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        SEED_GRIDS["stage1_smoke"][0].input_seed = 99  # type: ignore[misc]


def test_pilot_seed_design_is_disjoint_from_confirmatory_grids():
    pilot = {(cell.input_seed, cell.coupling_seed) for cell in SEED_GRIDS["pilot"]}
    confirmatory = {
        (cell.input_seed, cell.coupling_seed)
        for name, grid in SEED_GRIDS.items()
        if name != "pilot"
        for cell in grid
    }

    assert pilot.isdisjoint(confirmatory)
    assert {cell.input_seed for cell in SEED_GRIDS["pilot"]}.isdisjoint(
        {cell.input_seed for name, grid in SEED_GRIDS.items() if name != "pilot" for cell in grid}
    )
    assert {cell.coupling_seed for cell in SEED_GRIDS["pilot"]}.isdisjoint(
        {cell.coupling_seed for name, grid in SEED_GRIDS.items() if name != "pilot" for cell in grid}
    )


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
