"""Bind full-suite JUnit evidence to the exact canonical study material."""

from pathlib import Path

import pytest

from qrc_vs_qelm.provenance import material_file_manifest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "canonical_emulation.yaml"
RUNNER = ROOT / "scripts" / "run_canonical_emulation.py"


@pytest.fixture(scope="session", autouse=True)
def record_canonical_material_identity(record_testsuite_property):
    manifest = material_file_manifest(
        repo_root=ROOT,
        config_path=CONFIG,
        runner_path=RUNNER,
    )
    record_testsuite_property(
        "canonical_material_identity_sha256",
        manifest["identity_sha256"],
    )
    record_testsuite_property(
        "canonical_config_sha256",
        manifest["files"]["configuration"]["sha256"],
    )
