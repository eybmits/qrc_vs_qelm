from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pra_extension/code"))
import render_resource_variants as variants  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return variants.load_source()


def test_all_existing_observations_retained(data):
    assert len(data) == 912
    for panel, points in (("a", 14), ("b", 6), ("c", 4)):
        summaries = variants.summaries(data, panel)
        assert len(summaries) == (4 if panel == "c" else 3)
        assert all(len(s) == points for s in summaries.values())
        for s in summaries.values():
            assert (s.q10 <= s["median"]).all()
            assert (s["median"] <= s.q90).all()


@pytest.mark.parametrize("panel", list("abc"))
@pytest.mark.parametrize("version", [1, 2, 3])
def test_variant_scales_and_data(data, panel, version):
    variants.setup_style()
    fig = plt.figure(figsize=(5.25, 5.5))
    axes = variants.draw_variant(fig, data, panel, version)
    fig.canvas.draw()
    assert len(axes) == (3 if version == 3 else 2 if (panel, version) == ("b", 2) else 1)
    assert axes[0].get_yscale() == ("log" if panel == "a" else "linear")
    assert axes[0].get_xscale() == ("log" if panel == "c" else "linear")
    if version != 3:
        summaries = variants.summaries(data, panel)
        model_lines = [line for line in axes[0].lines if line.get_marker() in ("o", "D", "s", "*")]
        assert len(model_lines) == len(summaries)
        for line, s in zip(model_lines, [summaries[k] for k in
                                       (variants.SHOT_STYLES if panel == "c" else (2, 3, 4))], strict=True):
            np.testing.assert_array_equal(line.get_ydata(), s["median"].to_numpy())
    for ax in axes[:3 if version == 3 else 1]:
        lo, hi = ax.get_ylim()
        for s in variants.summaries(data, panel).values():
            assert s["median"].between(lo, hi).all()
    plt.close(fig)
