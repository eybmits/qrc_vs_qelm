"""Frozen seed grids for the QRC-vs-QELM comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class SeedCell:
    """One deterministic comparison cell."""

    input_seed: int
    coupling_seed: int
    shot_seed: int = 0


_STAGE1_COUPLINGS = (314159, 271828, 161803)
_STAGE3_COUPLINGS = (314159, 271828)
_PILOT_INPUTS = (900, 901, 902, 903, 904)
_PILOT_COUPLINGS = (987654, 123456, 555553)


_SEED_GRIDS: dict[str, tuple[SeedCell, ...]] = {
    "stage1_smoke": (SeedCell(input_seed=0, coupling_seed=314159),),
    "stage1_full": tuple(
        SeedCell(input_seed=input_seed, coupling_seed=coupling_seed)
        for input_seed in range(10)
        for coupling_seed in _STAGE1_COUPLINGS
    ),
    "stage2_shots": tuple(
        SeedCell(input_seed=0, coupling_seed=coupling_seed, shot_seed=shot_seed)
        for coupling_seed in _STAGE1_COUPLINGS
        for shot_seed in (0, 1, 2)
    ),
    "stage3_gamma": tuple(
        SeedCell(input_seed=input_seed, coupling_seed=coupling_seed)
        for input_seed in range(5)
        for coupling_seed in _STAGE3_COUPLINGS
    ),
    "pilot": tuple(
        SeedCell(input_seed=input_seed, coupling_seed=coupling_seed)
        for input_seed in _PILOT_INPUTS
        for coupling_seed in _PILOT_COUPLINGS
    ),
}


SEED_GRIDS: Mapping[str, tuple[SeedCell, ...]] = MappingProxyType(_SEED_GRIDS)


def grid_manifest(name: str) -> dict:
    """Return a JSON-serializable manifest fragment for a frozen seed grid."""

    cells = SEED_GRIDS[name]
    return {
        "name": name,
        "n_cells": len(cells),
        "cell_count": len(cells),
        "cells": [asdict(cell) for cell in cells],
    }


__all__ = ["SeedCell", "SEED_GRIDS", "grid_manifest"]
