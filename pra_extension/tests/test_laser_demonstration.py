"""Leakage, causality and physical checks for the separate laser demonstration."""

from dataclasses import replace
import importlib.util
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[2] / "pra_extension/code/laser_demonstration.py"
SPEC = importlib.util.spec_from_file_location("laser_demonstration", PATH)
demo = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = demo
SPEC.loader.exec_module(demo)


def test_input_scaling_does_not_fit_on_validation_or_test():
    cfg = replace(demo.Config(), block_lengths=(10, 10, 10))
    raw = np.tile(np.arange(10.0), 3)
    first, metadata = demo.input_blocks(raw, cfg)
    altered = raw.copy()
    altered[10:] *= 100
    second, modified = demo.input_blocks(altered, cfg)
    np.testing.assert_array_equal(first[0], second[0])
    assert metadata["training_max"] == modified["training_max"] == 9
    assert modified["blocks"][1]["clipped"] == 9


def test_pair_windows_equal_at_fourth_common_input():
    cfg = replace(demo.Config(), prefix=8, continuation=12)
    first, second = demo.pair_histories(np.arange(28) / 28, cfg)
    a = demo.delay_embed(first, cfg.window)[cfg.prefix:]
    b = demo.delay_embed(second, cfg.window)[cfg.prefix:]
    assert not np.array_equal(a[2], b[2])
    np.testing.assert_array_equal(a[3:], b[3:])


def test_half_gap_is_a_pointwise_lower_bound_not_a_fit_metric():
    a, b = np.arange(8.0), -np.arange(8.0)
    pred = np.zeros(8)
    result = demo.pair_check(a, b, pred, pred, 4)
    assert result["predictions_identical"]
    assert result["pointwise_bound_holds"]
    assert result["half_gap_at_expiry"] == 3
    assert result["pair_bound_min_slack"] == 0


def test_readout_transforms_use_only_training_data():
    rng = np.random.default_rng(29)
    xt, xv = rng.normal(size=(100, 4)), rng.normal(size=(30, 4))
    yt, yv = xt[:, 0], xv[:, 0]
    one = demo.fit_readout(xt, yt, xv, yv)
    two = demo.fit_readout(xt, yt, xv + 1e5, yv)
    np.testing.assert_allclose(one.mean, xt.mean(axis=0))
    np.testing.assert_array_equal(one.mean, two.mean)
    np.testing.assert_array_equal(one.scale, two.scale)
    assert np.mean((one.predict(xv) - yv) ** 2) < 1e-10


@pytest.mark.parametrize("rate", [4.0, 0.2])
def test_matched_channels_physical_and_equal_window_predictions(rate):
    cfg = replace(demo.Config(), n_qubits=2, window=2, prefix=4, continuation=6)
    qrc, qelm, obs, _ = demo.architecture(917000, rate, cfg)
    first, second = demo.pair_histories(np.arange(14.0) / 14, cfg)
    a = qelm.features(demo.delay_embed(first, 2), obs, encoding_scheme="one_to_one")
    b = qelm.features(demo.delay_embed(second, 2), obs, encoding_scheme="one_to_one")
    np.testing.assert_array_equal(a[cfg.prefix + 1:], b[cfg.prefix + 1:])
    assert demo.physical_check(qrc, first)["passed"]
    assert qrc.run(first, obs, n_virtual=cfg.snapshots).shape == a.shape


def test_source_tampering_rejected(tmp_path):
    wrong = tmp_path / "laser.dat"
    wrong.write_text("1,2,3")
    with pytest.raises(ValueError, match="hash mismatch"):
        demo.load_laser(wrong)


def test_renderer_rejects_selective_seed_coverage(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(PATH.parent))
    renderer = importlib.import_module("render_laser_demonstration")
    with pytest.raises(ValueError, match="Incomplete coverage"):
        renderer.validate_rows(tmp_path, {"results": []})


def test_architecture_shares_couplings_across_both_memory_regimes():
    cfg = replace(demo.Config(), n_qubits=2, window=2)
    fast, _, _, fast_j = demo.architecture(917000, 4.0, cfg)
    slow, _, _, slow_j = demo.architecture(917000, 0.2, cfg)
    np.testing.assert_array_equal(fast_j, slow_j)
    np.testing.assert_array_equal(fast.drive_super, slow.drive_super)


def test_compact_layout_retains_all_data_and_aligned_time_axes(monkeypatch):
    monkeypatch.syspath_prepend(str(PATH.parent))
    renderer = importlib.import_module("render_laser_demonstration")
    x = np.linspace(0, 1, 128)
    rows = {4.0: (x, x + 0.01), 0.2: (x**2, x**2 + 0.2)}
    fig, axes = renderer.compact_figure(255 * x, rows, {4.0: 0.03, 0.2: 0.3})
    try:
        np.testing.assert_array_equal(axes[0].lines[0].get_ydata(), 255 * x)
        for ax, rate in zip(axes[1:], (4.0, 0.2)):
            np.testing.assert_array_equal(ax.lines[0].get_ydata(), rows[rate][0])
            np.testing.assert_array_equal(ax.lines[1].get_ydata(), rows[rate][1])
            np.testing.assert_array_equal(ax.lines[1].get_xdata(), np.arange(128))
        assert len({tuple(ax.get_xlim()) for ax in axes}) == 1
        assert len({ax.get_position().x0 for ax in axes}) == 1
        assert len({ax.get_position().width for ax in axes}) == 1
        assert len(fig.legends) == 1
        assert [t.get_text() for t in fig.legends[0].texts] == ["QRC", "QELM"]
        assert fig.get_figwidth() == pytest.approx((6.221 - 0.8 / 2.54) / 2)
    finally:
        renderer.plt.close(fig)
