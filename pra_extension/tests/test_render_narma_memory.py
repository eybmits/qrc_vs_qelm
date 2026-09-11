"""Plot source coverage and annotation/curve identity, without bundled results."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import render_narma_memory as render
from verify_narma_memory import independent_target
from narma_memory import NarmaConfig,target_series


def test_verifier_recursion_matches_production_on_new_input():
    cfg=NarmaConfig()
    u=np.random.default_rng(83).uniform(size=333)
    np.testing.assert_allclose(independent_target(u,10,.2)[10:],target_series(u,cfg)[10:],rtol=1e-14,atol=1e-14)


def test_plot_uses_all_windows_and_actual_target_arrays():
    rows=[]
    for seed in range(12):
        for model,windows in [('replay',(1,4,8,12,24,48)),('quadratic',(4,12)),('qelm',(4,)),('qrc',(None,))]:
            for d in windows:
                rows.append({'seed':seed,'model':model,'window':d,'nrmse':.5+.01*seed})
    x=np.linspace(0,8,1919)
    trace={'test_target':.2+.02*np.sin(x),'training_target_sd':.02}
    for key in ('qrc','qelm','replay_d12'):
        trace[key]=trace['test_target']+.002
    for key in ('target','qrc','qelm'):
        trace[f'pair_{key}_a']=np.linspace(.2,.3,47)
        trace[f'pair_{key}_b']=trace[f'pair_{key}_a']+.01
    trace['pair_qelm_b'][3:]=trace['pair_qelm_a'][3:]
    with plt.rc_context(render.style()):
        fig,axes=render.build_figure(pd.DataFrame(rows),trace)
        fig.canvas.draw()
        for ax in axes['time']:
            actual=next(l for l in ax.lines if l.get_gid()=='true_target')
            np.testing.assert_array_equal(actual.get_ydata(),trace['test_target'][:96])
        line=next(l for l in axes['memory'].lines if l.get_gid()=='history_control')
        np.testing.assert_array_equal(line.get_xdata(),[1,4,8,12,24,48])
        pair=next(l for l in axes['pair'].lines if l.get_gid()=='pair_gap_qelm')
        assert pair.get_xdata()[0]==4 and np.all(pair.get_ydata()==0)
        renderer=fig.canvas.get_renderer()
        legend=fig.legends[0].get_window_extent(renderer)
        qrc_title=axes['time'][1]._left_title.get_window_extent(renderer)
        assert not legend.overlaps(qrc_title)
        plt.close(fig)
