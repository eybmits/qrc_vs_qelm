"""Portable inputs for the archived, hash-frozen bridge smoke regression.

Keep the original test and numerical runner byte-identical to their freeze.
Inject only development I/O, never numerical functions, into that one test.
Production reads and strict frozen audits are not patched.
"""

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))


@pytest.fixture(autouse=True)
def portable_bridge_development_inputs(request, tmp_path, monkeypatch):
    if (request.node.path.name != "test_feature_bridge.py"
            or request.node.name != "test_development_reproduces_parent_and_is_deterministic"):
        return
    import feature_bridge_release as released
    import run_feature_bridge as study

    released.verify_released_material()
    directory = tmp_path / "generated_parent"
    study.parent.run(directory, smoke=True)
    manifest = json.loads((directory / "manifest.json").read_text())
    frame = pd.read_csv(directory / "metrics.csv", float_precision="round_trip")
    study.parent.coverage(frame, True)
    original = study.parent_artifacts

    def parent_artifacts(smoke=False):
        return (frame.copy(), manifest) if smoke else original(False)

    def development_material():
        freeze, _ = released.verify_released_material()
        # This is a fresh development run, not a claim to reproduce historical
        # manifest bytes (which include wall-clock durations).
        material = {k: v for k, v in freeze["material"].items()
                    if k not in released.DEVELOPMENT_OUTPUTS}
        material.update({f"generated_parent/{name}": study.base.config_file_sha256(directory / name)
                         for name in ("manifest.json", "metrics.csv")})
        return material

    monkeypatch.setattr(study, "parent_artifacts", parent_artifacts)
    monkeypatch.setattr(study, "material", development_material)
