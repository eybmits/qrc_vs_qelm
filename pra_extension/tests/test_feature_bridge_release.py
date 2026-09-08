"""Release I/O is portable but still fails closed on source/result changes."""

import json
import shutil

import pandas as pd
import pytest

import feature_bridge_release as released


@pytest.fixture
def release_tree(tmp_path):
    root = released.study.ROOT
    freeze_path = root / "configs/feature_bridge_freeze.json"
    freeze = json.loads(freeze_path.read_text())
    for relative in ["configs/feature_bridge_freeze.json", *freeze["material"]]:
        if relative in released.DEVELOPMENT_OUTPUTS:
            continue
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, destination)
    return tmp_path


def test_absent_development_outputs_are_reported_not_claimed_verified(release_tree):
    _, audit = released.verify_released_material(release_tree)
    assert audit["release_material_verified"]
    assert not audit["full_development_audit"]
    assert set(audit["unavailable_development_outputs"]) == released.DEVELOPMENT_OUTPUTS


@pytest.mark.parametrize("relative", [
    "configs/feature_bridge.json",
    "pra_extension/code/run_feature_bridge.py",
    "results_feature_density_v1/metrics.csv",
])
def test_required_release_material_cannot_be_missing(release_tree, relative):
    (release_tree / relative).unlink()
    with pytest.raises(ValueError, match="missing required"):
        released.verify_released_material(release_tree)


@pytest.mark.parametrize("relative", [
    "pra_extension/code/run_feature_bridge.py",
    "results_feature_density_v1/metrics.csv",
    "results_feature_density_smoke_a/metrics.csv",
])
def test_even_optional_development_files_are_checked_when_present(release_tree, relative):
    path = release_tree / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("corrupt")
    with pytest.raises(ValueError, match="material changed"):
        released.verify_released_material(release_tree)


def test_freeze_cannot_silently_omit_a_source(release_tree):
    path = release_tree / "configs/feature_bridge_freeze.json"
    freeze = json.loads(path.read_text())
    del freeze["material"]["pra_extension/code/run_feature_bridge.py"]
    path.write_text(json.dumps(freeze))
    with pytest.raises(ValueError, match="inventory changed"):
        released.verify_released_material(release_tree)


def test_release_reader_preserves_completed_rows():
    frame, manifest = released.load_completed()
    expected = pd.read_csv(released.RESULTS / "metrics.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(frame, expected)
    assert len(frame) == 144
    assert manifest["release_audit"]["release_material_verified"]


def test_release_reader_rejects_modified_results(tmp_path):
    for name in ("manifest.json", "metrics.csv"):
        shutil.copy2(released.RESULTS / name, tmp_path / name)
    (tmp_path / "metrics.csv").write_text("corrupt")
    with pytest.raises(ValueError, match="CSV changed"):
        released.load_completed(tmp_path)


def test_release_reader_rejects_modified_freeze_reference(tmp_path):
    for name in ("manifest.json", "metrics.csv"):
        shutil.copy2(released.RESULTS / name, tmp_path / name)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["freeze"]["status"] = "UNFROZEN"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="provenance"):
        released.load_completed(tmp_path)
