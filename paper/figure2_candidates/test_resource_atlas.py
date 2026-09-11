"""Presentation and aggregation checks for the isolated Figure 2 preview."""
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("resource_atlas", Path(__file__).with_name("render_resource_atlas.py"))
atlas = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(atlas)


@pytest.fixture(scope="module")
def data():
    return atlas.load_data()


def test_history_criterion_and_feature_coverage(data):
    history, features, shots, _ = data
    passed, f, *_ = atlas.summaries(history, features, shots)
    assert passed.shape == (11, 6)
    assert passed.to_numpy().sum() == 19
    for rate, window in ((1.25, 6), (1.5, 5), (2, 4), (2.5, 3), (3, 3), (4, 2)):
        assert passed.loc[rate][lambda x: x].index.min() == window
    assert not passed.loc[0:1].to_numpy().any()
    assert len(f) == 18
    assert f.high.max() < .36
    for (n, b), summary in f.iterrows():
        raw = features[(features.teacher_n_qubits == n) & (features.feature_budget == b)].total_nrmse
        np.testing.assert_allclose(summary, [raw.median(), raw.quantile(.1), raw.quantile(.9)])


def test_selection_frequencies_match_all_raw_choices(data):
    history, features, shots, selected = data
    _, _, med, low, high, share = atlas.summaries(history, features, shots)
    raw_shares = selected.groupby(["total_shots", "feature_budget"]).size().unstack(fill_value=0) / 192
    np.testing.assert_allclose(share.to_numpy(), raw_shares.to_numpy(), atol=1e-12)
    np.testing.assert_allclose(share.sum(axis=1), 1)
    assert (low <= med).all().all() and (med <= high).all().all()
    assert share.loc[300].idxmax() == "selected_fraction_B36"
    assert share.loc[19200].idxmax() == "selected_fraction_B72"


def test_fail_closed_on_incomplete_or_changed_selection(data):
    h, f, s, selected = data
    with pytest.raises(ValueError):
        atlas.summaries(h.iloc[:-1], f, s)
    with pytest.raises(ValueError):
        atlas.validate_selection(s, selected.iloc[:-1])
    changed = selected.copy()
    changed.loc[0, "test_noisy_feature_nrmse"] = 100
    # Change all repetition errors for one root/shot so the median must differ.
    first = changed.iloc[0]
    changed.loc[(changed.root_id == first.root_id) & (changed.total_shots == first.total_shots),
                "test_noisy_feature_nrmse"] = 100
    with pytest.raises(AssertionError):
        atlas.validate_selection(s, changed)


def test_paper_size_text_and_axes_fit(data):
    with plt.rc_context(atlas.style()):
        fig = atlas.build_figure(*data[:3])
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for text in fig.findobj(Text):
            if not text.get_visible() or not text.get_text():
                continue
            bbox = text.get_window_extent(renderer)
            assert bbox.x0 >= -1 and bbox.y0 >= -1, text.get_text()
            assert bbox.x1 <= fig.bbox.x1 + 1 and bbox.y1 <= fig.bbox.y1 + 1, text.get_text()
            assert text.get_fontsize() * 6.221 / atlas.SIZE[0] >= 7, text.get_text()
        # The figure has a map, three aligned feature axes, a table and selection strip.
        assert len(fig.axes) == 6
        bounds = [ax.get_position() for ax in fig.axes]
        for i, box in enumerate(bounds):
            for other in bounds[i + 1:]:
                assert not box.overlaps(other)
        plt.close(fig)


def test_reproducible_preview_does_not_change_manuscript(tmp_path):
    pdf = atlas.ROOT / "paper/icaart2027/submission.pdf"
    before = atlas.digest(pdf)
    one = atlas.render(tmp_path / "one")
    two = atlas.render(tmp_path / "two")
    assert one["artifacts_sha256"] == two["artifacts_sha256"]
    assert one["status"] == "preview_only_not_integrated"
    assert atlas.digest(pdf) == before


def test_published_atlas_keeps_all_sources_and_uncertainty():
    import json
    directory = atlas.ROOT / "paper/figures/resource_atlas"
    ledger = json.loads((directory / "fig03_resource_diagnosis_source.json").read_text())
    assert ledger["status"] == "author_selected_publication_asset"
    assert "percentiles drawn below each median" in ledger["shot_uncertainty"]
    for name, expected in ledger["artifacts_sha256"].items():
        assert atlas.digest(directory / name) == expected
    for name, expected in ledger["sources_sha256"].items():
        assert atlas.digest(atlas.ROOT / name) == expected
    assert len(ledger["history_map"]) == 11
    assert len(ledger["feature_summary"]) == 18
