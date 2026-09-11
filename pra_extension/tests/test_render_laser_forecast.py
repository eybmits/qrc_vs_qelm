"""Portable figure binding tests using the small exported source tables."""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'pra_extension/code'))
import render_laser_forecast as render
from laser_demonstration import digest

ART = ROOT/'paper/figures/laser_forecast_candidate'


@pytest.fixture
def inputs():
    rows = pd.read_csv(ART/'laser_forecast_all_rows.csv')
    traces = {rate: {col: frame[col].to_numpy() for col in frame.columns}
              for rate in (4.,.2)
              for frame in [pd.read_csv(ART/f'laser_forecast_display_seed_gamma{rate:g}.csv')]}
    return rows, traces


def test_all_seeds_horizons_and_interventions_are_retained(inputs):
    rows, traces = inputs
    assert len(rows)==720
    assert not rows.duplicated(['seed','gamma','horizon','model','window']).any()
    assert sorted(rows.seed.unique())==list(range(918000,918012))
    assert rows.groupby(['gamma','horizon']).size().to_list()==[180]*4
    for trace in traces.values():
        np.testing.assert_array_equal(trace['test_indices'],np.arange(128,2044))
        assert all(len(v)==1916 for v in trace.values())
    np.testing.assert_array_equal(traces[4.]['target_h1'],traces[.2]['target_h1'])
    np.testing.assert_array_equal(traces[4.]['target_h4'],traces[.2]['target_h4'])


@pytest.mark.parametrize('horizon',[1,4])
@pytest.mark.parametrize('frozen',[False,True])
def test_plot_marks_are_bound_to_measured_targets_and_complete_source(inputs,horizon,frozen):
    rows, traces = inputs
    with plt.rc_context(render.style()):
        fig, axes = render.build_figure(rows,traces,horizon,frozen=frozen)
        for ax,rate in zip(axes['time'],(4.,.2)):
            lines = {line.get_gid(): line for line in ax.lines}
            for gid,key in [('measured_future','target'),('qrc_forecast','qrc'),('qelm_forecast','qelm')]:
                np.testing.assert_array_equal(lines[gid].get_ydata(),traces[rate][f'{key}_h{horizon}'][:128])
                np.testing.assert_array_equal(lines[gid].get_xdata(),np.arange(128))
        assert axes['time'][0].get_ylim()==axes['time'][1].get_ylim()
        memory = {line.get_gid():line for line in axes['memory'].lines}
        for rate in (4.,.2):
            subset = rows[(rows.gamma==rate)&(rows.horizon==horizon)]
            selected = subset[subset.model==('replay_frozen' if frozen else 'replay_refit')]
            expected = selected.groupby('window').nrmse.median().sort_index()
            np.testing.assert_array_equal(memory[f'restricted_gamma{rate:g}'].get_xdata(),expected.index)
            np.testing.assert_allclose(memory[f'restricted_gamma{rate:g}'].get_ydata(),expected)
            np.testing.assert_allclose(memory[f'full_gamma{rate:g}'].get_ydata(),
                                       subset[subset.model=='qrc'].nrmse.median())
        assert axes['memory'].get_yscale()==('log' if frozen else 'linear')
        texts=[t.get_text().lower() for t in fig.findobj(match=plt.Text)]
        assert not any('floor' in text or 'unavoidable' in text for text in texts)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        # The accuracy legend must not cover any seed marker.
        legend = axes['accuracy'].get_legend().get_window_extent(renderer)
        for collection in axes['accuracy'].collections:
            if isinstance(collection, render.matplotlib.collections.PathCollection):
                for xy in axes['accuracy'].transData.transform(collection.get_offsets()):
                    assert not legend.padded(3).contains(*xy)
        for text in fig.findobj(match=plt.Text):
            if text.get_visible() and text.get_text() and text.get_clip_on() is False:
                box = text.get_window_extent(renderer)
                # Matplotlib may generate invisible out-of-range tick positions;
                # axis/legend titles and labels are explicitly checked instead.
                if text in fig.texts or any(text is a.xaxis.label or text is a.yaxis.label for a in fig.axes):
                    assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.x1+1 and box.y1<=fig.bbox.y1+1
        for ax in fig.axes:
            box=ax.yaxis.label.get_window_extent(renderer)
            assert not any(box.overlaps(other.get_window_extent(renderer))
                           for other in fig.axes if other is not ax)
        plt.close(fig)


def test_export_hashes_and_paired_statistics_match(inputs):
    rows, _ = inputs
    manifest = json.loads((ART/'laser_forecast_source.json').read_text())
    assert manifest['status']=='separate_forecasting_candidate_not_integrated'
    for name, expected in manifest['artifacts'].items():
        assert digest(ART/name)==expected
    for path,expected in manifest['sources'].items():
        if not path.startswith('results_'):
            assert digest(ROOT/path)==expected
    for rate in (.2,4.):
        for horizon in (1,4):
            selected = rows[(rows.gamma==rate)&(rows.horizon==horizon)]
            full = selected[selected.model=='qrc'].set_index('seed').nrmse.sort_index()
            for key,model in [('qelm_minus_qrc','qelm'),('restricted_d4_minus_qrc','replay_refit')]:
                alternative = selected[(selected.model==model)&(selected.window==4)].set_index('seed').nrmse.sort_index()
                expected = manifest['summary']['paired_contrasts'][f'gamma{rate:g}_h{horizon}'][key]
                np.testing.assert_allclose(alternative-full,expected['differences'],atol=1e-15)
                assert np.mean(alternative-full)==pytest.approx(expected['mean'],abs=1e-15)
                assert len(full)==expected['n_seeds']==12
