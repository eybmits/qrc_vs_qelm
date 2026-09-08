"""Read released bridge results without requiring unpublished development runs.

The frozen runner remains untouched and retains its strict full-audit contract.
Only two explicitly named development outputs may be absent here. All available
files, all numerical sources, and all production results must match their locks.
The returned audit distinguishes release verification from the full local audit.
"""

import json
from pathlib import Path

import pandas as pd

import run_feature_bridge as study

RESULTS = study.RESULTS
DEVELOPMENT_OUTPUTS = frozenset({
    "results_feature_density_smoke_a/manifest.json",
    "results_feature_density_smoke_a/metrics.csv",
})


def verify_released_material(root=None):
    root = study.ROOT if root is None else Path(root)
    freeze = json.loads((root / "configs/feature_bridge_freeze.json").read_text())
    if freeze["status"] != "FROZEN":
        raise ValueError("bridge is not frozen")
    expected_paths = set(study.parent.material()) | DEVELOPMENT_OUTPUTS | {
        "configs/feature_bridge.json", "configs/feature_density_freeze.json",
        "pra_extension/FEATURE_BRIDGE_PROTOCOL.md",
        "pra_extension/code/run_feature_bridge.py",
        "pra_extension/tests/test_feature_bridge.py",
        "results_feature_density_v1/manifest.json",
        "results_feature_density_v1/metrics.csv",
    }
    if set(freeze["material"]) != expected_paths:
        raise ValueError("bridge material inventory changed")
    missing_development = []
    for relative, expected in freeze["material"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("material path escapes repository")
        if not path.exists() and relative in DEVELOPMENT_OUTPUTS:
            missing_development.append(relative)
            continue
        if not path.is_file():
            raise ValueError(f"missing required release material: {relative}")
        if study.base.config_file_sha256(path) != expected:
            raise ValueError(f"release material changed: {relative}")
    return freeze, {
        "release_material_verified": True,
        "full_development_audit": not missing_development,
        "unavailable_development_outputs": sorted(missing_development),
    }


def load_completed(directory=RESULTS):
    # Also retain the parent's dynamic source-set check, not just listed hashes.
    study.parent.verify()
    freeze, audit = verify_released_material()
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest["status"] != "COMPLETED" or manifest["smoke"]
            or manifest["material"] != freeze["material"]
            or manifest["freeze"] != freeze):
        raise ValueError("invalid bridge provenance")
    if manifest["metrics_sha256"] != study.base.config_file_sha256(directory / "metrics.csv"):
        raise ValueError("bridge CSV changed")
    frame = pd.read_csv(directory / "metrics.csv", float_precision="round_trip")
    study.coverage(frame)
    study.validate_anchors(frame, manifest["cells"])
    return frame, {**manifest, "release_audit": audit}
