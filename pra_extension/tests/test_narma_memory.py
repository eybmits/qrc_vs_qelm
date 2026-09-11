"""Independent task/alignment and prospective selection checks."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import narma_memory as study


def reference(u, order=10, scale=.2):
    y=np.zeros(len(u))
    for t in range(order-1,len(u)-1):
        y[t+1]=.3*y[t]+.05*y[t]*sum(y[j] for j in range(t-order+1,t+1))
        y[t+1]+=1.5*scale**2*u[t-order+1]*u[t]+.1
    return y


def test_direct_target_recursion_and_future_alignment():
    cfg=study.NarmaConfig()
    raw,blocks,indices,targets=study.make_data(cfg.seeds[0],cfg)
    assert [len(t) for t in indices]==[3967,895,1919]
    for u,b,t,y in zip(raw,blocks,indices,targets):
        np.testing.assert_allclose(y,reference(u)[t+1],rtol=1e-14,atol=1e-14)
        assert t[-1]+1<len(u)
        np.testing.assert_array_equal(b,np.rint(u*32)/32)
        changed=u.copy()
        changed[t[0]+1:]=1-changed[t[0]+1:]
        assert study.target_series(changed,cfg)[t[0]+1]==y[0]
        assert np.max(np.abs(y-study.target_series(b,cfg)[t+1]))>1e-6


def test_all_roots_and_splits_are_independent_and_reproducible():
    cfg=study.NarmaConfig()
    one=study.make_data(920000,cfg)[0]
    two=study.make_data(920000,cfg)[0]
    other=study.make_data(920001,cfg)[0]
    for a,b,c in zip(one,two,other):
        np.testing.assert_array_equal(a,b)
        assert not np.array_equal(a,c)
    assert not np.array_equal(one[0][:1024],one[1])


def test_candidate_selection_ignores_test_error():
    a=({'validation_mse':.1,'nrmse':10.},'A')
    b=({'validation_mse':.2,'nrmse':.001},'B')
    assert study.choose([a,b])[1]=='A'


def test_target_history_really_extends_beyond_four_inputs():
    cfg=study.NarmaConfig()
    raw=study.make_data(920000,cfg)[0][2]
    a=np.r_[raw[:128],raw[256:304]]
    b=np.r_[raw[128:256],raw[256:304]]
    np.testing.assert_array_equal(a[128:],b[128:])
    ya,yb=study.target_series(a,cfg),study.target_series(b,cfg)
    assert np.max(np.abs(ya[132:140]-yb[132:140]))>1e-4


def test_quadratic_control_counts_and_window_invariance():
    u=np.linspace(0,1,32)
    for d,features in ((4,14),(12,90)):
        x=study.quadratic_features(u,np.array([20,21]),d)
        assert x.shape==(2,features)
        np.testing.assert_array_equal(x[0,:d],u[20-np.arange(d)])
        altered=u.copy()
        altered[:20-d+1]=1-altered[:20-d+1]
        np.testing.assert_array_equal(x,study.quadratic_features(altered,np.array([20,21]),d))


def test_config_and_incomplete_cohort_rejected():
    cfg=study.NarmaConfig()
    assert cfg.seeds==tuple(range(920000,920012))
    assert cfg.rates==(.1,.3,1.,4.)
    with pytest.raises(ValueError,match='Incomplete'):
        study.summarize([],cfg)
    with pytest.raises(ValueError,match='warm-up'):
        study.make_data(920000,replace(cfg,washout=8))
