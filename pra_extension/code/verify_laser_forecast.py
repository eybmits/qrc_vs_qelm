"""Independently bind every forecast score to stored predictions and real targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from laser_demonstration import ROOT, DATA_HASH, digest, dump
from render_laser_forecast import validate_source
import laser_forecast as forecast


def verify(source: Path, data: Path) -> dict:
    summary, _, _ = validate_source(source)
    if digest(data) != DATA_HASH:
        raise ValueError('Forecast verification requires the frozen laser record')
    raw = np.loadtxt(data, delimiter=',').reshape(-1)
    cfg = forecast.ForecastConfig()
    ends = np.r_[0, np.cumsum(cfg.block_lengths)]
    origins = [np.arange(cfg.washout, size-max(cfg.horizons)) for size in cfg.block_lengths]
    count = 0
    max_error = 0.
    for cell in summary['results']:
        with np.load(source/cell['array_file']) as arrays:
            np.testing.assert_array_equal(arrays['test_indices'], origins[2])
            for horizon in cfg.horizons:
                expected = raw[ends[2]+origins[2]+horizon]
                np.testing.assert_array_equal(arrays[f'target_h{horizon}'], expected)
                train = raw[origins[0]+horizon]
                sd = float(np.sqrt(np.mean((train-train.mean())**2)))
                for row in (r for r in cell['rows'] if r['horizon']==horizon):
                    model, window = row['model'], row['window']
                    key = {'replay_refit': f'replay_d{window}',
                           'replay_frozen': f'frozen_d{window}'}.get(model, model)
                    key += f'_h{horizon}'
                    error = arrays[key]-expected
                    rmse = float(np.linalg.norm(error)/np.sqrt(len(error)))
                    np.testing.assert_allclose([row['rmse'], row['nrmse']], [rmse, rmse/sd],
                                               rtol=2e-13, atol=2e-13)
                    max_error = max(max_error, abs(rmse-row['rmse']))
                    count += 1
    # Reconstruct the displayed seed's predictions from actual channel features
    # and persisted readout parameters, not from a second call to the score helper.
    blocks, indices, _, _ = forecast.forecast_blocks(raw, cfg)
    selected = np.array([0, 1, 31, 127, len(indices[2])-1])
    reconstructed = 0
    for rate in cfg.rates:
        qrc, qelm, obs, _ = forecast.architecture(cfg.seeds[0], rate, cfg)
        features = {
            'qrc': qrc.run(blocks[2], obs, n_virtual=cfg.snapshots)[indices[2][selected]],
            'qelm': qelm.features(forecast.delay_embed(blocks[2],cfg.window),obs,
                                  encoding_scheme='one_to_one')[indices[2][selected]],
            # Independent step-by-step replay avoids the batch implementation.
            'replay_d4': np.array([qrc.run(blocks[2][t-3:t+1],obs,n_virtual=cfg.snapshots)[-1]
                                   for t in indices[2][selected]]),
        }
        with np.load(source/f'seed{cfg.seeds[0]}_gamma{rate:g}.npz') as arrays:
            for h in cfg.horizons:
                for model, x in features.items():
                    key = f'{model}_h{h}'
                    predicted = (((x-arrays[key+'_mean'])/arrays[key+'_scale'])
                                 @ arrays[key+'_weights']+arrays[key+'_intercept'])
                    predicted = predicted*arrays[key+'_target_sd']+arrays[key+'_target_mean']
                    np.testing.assert_allclose(predicted,arrays[key][selected],rtol=2e-8,atol=2e-6)
                    reconstructed += len(selected)
    return {'passed': True, 'all_score_rows_recomputed': count,
            'source_target_rows_per_cell_and_horizon': len(origins[2]),
            'display_seed_predictions_reconstructed_from_channels': reconstructed,
            'maximum_rmse_recomputation_difference': max_error,
            'summary_sha256': digest(source/'summary.json'), 'data_sha256': digest(data),
            'verifier_sha256': digest(Path(__file__)),
            'scope': 'All 720 scores and every true target checked; channel/readout reconstruction '
                     'spot-checks 60 displayed-seed predictions across both rates/horizons.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'results_laser_forecast_v1')
    parser.add_argument('--data', type=Path, default=ROOT/'data/santa_fe_laser/SF_Acont.dat')
    parser.add_argument('--output', type=Path, default=ROOT/'paper/figures/laser_forecast_candidate/verification.json')
    args = parser.parse_args()
    result = verify(args.source,args.data)
    dump(args.output,result)
    print(json.dumps(result,indent=2))
