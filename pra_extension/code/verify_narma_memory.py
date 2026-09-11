"""Independent recurrence, all scores, and saved-readout reconstruction checks."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

import narma_memory as study
from laser_demonstration import digest, dump
from render_narma_memory import check_sources


def independent_target(u, order, scale):
    """Compute y[t+1] using the explicit lag convention, not core.tasks."""
    y=np.zeros(len(u),dtype=float)
    for t in range(order-1,len(u)-1):
        feedback=sum(float(y[t-j]) for j in range(order))
        y[t+1]=.3*y[t]+.05*y[t]*feedback+1.5*scale**2*u[t-order+1]*u[t]+.1
    return y


def parameter_prediction(saved,key,features):
    standardized=(features-saved[f'{key}_mean'])/saved[f'{key}_scale']
    return ((standardized@saved[f'{key}_weights']+saved[f'{key}_intercept'])
            *saved[f'{key}_target_sd']+saved[f'{key}_target_mean'])


def verify(source,output):
    summary,_,_=check_sources(source)
    cfg=study.NarmaConfig()
    score_checks=0
    reconstructed=0
    largest_prediction_difference=0.
    for cell in summary['results']:
        with np.load(source/cell['array_file']) as saved:
            rng=np.random.default_rng(np.random.SeedSequence([cell['seed'],7301]))
            for label,n in zip(('train','validation','test'),cfg.block_lengths):
                np.testing.assert_array_equal(saved[f'{label}_raw_input'],rng.uniform(size=n))
            u=saved['test_raw_input']
            t=np.arange(cfg.washout,len(u)-1)
            target=independent_target(u,cfg.order,cfg.input_scale)[t+1]
            np.testing.assert_allclose(saved['test_target'],target,rtol=1e-14,atol=1e-14)
            ut=saved['train_raw_input']
            yt=independent_target(ut,cfg.order,cfg.input_scale)[cfg.washout+1:]
            sd=float(np.std(yt))
            np.testing.assert_allclose(sd,saved['training_target_sd'],rtol=1e-14)
            for row in cell['rows']+cell['candidate_rows']:
                candidate=any(row is r for r in cell['candidate_rows'])
                if candidate:
                    key=f"candidate_{row['model']}_gamma{row['gamma']:g}"
                elif row['model'] in ('replay','quadratic'):
                    key=f"{row['model']}_d{row['window']}"
                else:
                    key=row['model']
                rmse=float(np.linalg.norm(saved[key]-target)/np.sqrt(len(target)))
                np.testing.assert_allclose([rmse,rmse/sd],[row['rmse'],row['nrmse']],rtol=1e-12,atol=1e-14)
                score_checks+=1
            for name in ('qrc','qelm'):
                np.testing.assert_array_equal(saved[name],saved[f"candidate_{name}_gamma{cell['selection'][name]:g}"])
            for label in ('a','b'):
                raw=saved[f'pair_raw_{label}']
                expected=independent_target(raw,cfg.order,cfg.input_scale)[cfg.prefix+1:]
                np.testing.assert_allclose(saved[f'pair_target_{label}'],expected,rtol=1e-14,atol=1e-14)
            np.testing.assert_allclose(saved['pair_qelm_a'][3:],saved['pair_qelm_b'][3:],rtol=0,atol=1e-11)
            # Reconstruct every fitted selected-model readout at 12 predeclared
            # evenly spread test rows. Full scores above use all 1919 rows.
            block=np.rint(u*32)/32
            picks=np.linspace(0,len(t)-1,12,dtype=int)
            qrc,_,obs,_=study.architecture(cell['seed'],cell['selection']['qrc'],cfg)
            _,qelm,_,_=study.architecture(cell['seed'],cell['selection']['qelm'],cfg)
            features={
                'qrc':qrc.run(block,obs,n_virtual=cfg.snapshots)[t[picks]],
                'qelm':qelm.features(study.delay_embed(block,4),obs,encoding_scheme='one_to_one')[t[picks]],
            }
            for d in cfg.replay_windows:
                features[f'replay_d{d}']=study.replay_features(qrc,block,t[picks],d,obs,cfg.snapshots)
            for d in cfg.quadratic_windows:
                features[f'quadratic_d{d}']=study.quadratic_features(block,t[picks],d)
            for key,x in features.items():
                predicted=parameter_prediction(saved,key,x)
                difference=float(np.max(np.abs(predicted-saved[key][picks])))
                largest_prediction_difference=max(largest_prediction_difference,difference)
                np.testing.assert_allclose(predicted,saved[key][picks],rtol=1e-9,atol=1e-10)
                reconstructed+=len(picks)
        print(f"Verified root {cell['seed']}",flush=True)
    report={'passed':True,'roots_checked':len(cfg.seeds),
            'full_test_scores_recomputed':score_checks,
            'predictions_reconstructed_from_saved_readouts':reconstructed,
            'max_reconstructed_prediction_difference':largest_prediction_difference,
            'independent_recursion_and_rng_checks':True,
            'same_window_predictions_identical':True,
            'summary_sha256':digest(source/'summary.json'),
            'scope':'All test scores; readout reconstruction sampled at 12 fixed rows per model/root; not hardware validation'}
    dump(output,report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    print(verify(args.source,args.output))
