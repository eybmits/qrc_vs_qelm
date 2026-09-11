"""Fixed-size fresh-root replication, without modifying original study runners."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

import narma_memory
import laser_forecast
from laser_demonstration import ROOT, digest, dump, ALPHAS


NARMA_ROOTS=tuple(range(921000,921024))
LASER_ROOTS=tuple(range(922000,922024))


def configs():
    return {'narma':replace(narma_memory.NarmaConfig(),seeds=NARMA_ROOTS,bootstrap_seed=939991),
            'laser':replace(laser_forecast.ForecastConfig(),seeds=LASER_ROOTS,bootstrap_seed=939992)}


def sources():
    selected=[Path(__file__),ROOT/'pra_extension/TASK_MEMORY_REPLICATION_PROTOCOL.md',
              ROOT/'pra_extension/NARMA_MEMORY_PROTOCOL.md',ROOT/'pra_extension/LASER_FORECAST_PROTOCOL.md']
    selected.extend(ROOT/'pra_extension/code'/name for name in
                    ('narma_memory.py','laser_forecast.py','laser_demonstration.py'))
    selected.extend((ROOT/'src/qrc_vs_qelm').rglob('*.py'))
    return {str(p.relative_to(ROOT)):digest(p) for p in selected}


def initialize(output):
    output.mkdir(parents=True,exist_ok=False)
    spec={'status':'specified_before_fresh_results','sources':sources(),
          'configurations':{k:asdict(v) for k,v in configs().items()},'alphas':ALPHAS,
          'original_summaries':{p:digest(ROOT/p/'summary.json') for p in
                                ('results_narma_memory_v1','results_laser_forecast_v1')},
          'display':{'narma_seed':NARMA_ROOTS[0],'laser_seed':LASER_ROOTS[0],
                     'scored_rows':[0,96]},
          'scope':'fresh input/coupling roots for NARMA; fresh couplings conditional on one reused laser record'}
    dump(output/'replication_specification.json',spec)


def validate_spec(output):
    spec=json.loads((output/'replication_specification.json').read_text())
    if spec['sources']!=sources():
        raise ValueError('Replication implementation changed after specification')
    wanted=json.loads(json.dumps({k:asdict(v) for k,v in configs().items()}))
    if spec['configurations']!=wanted:
        raise ValueError('Replication configuration drift')
    return spec


def run_study(output,study):
    validate_spec(output)
    cfg=configs()[study]
    if study=='narma':
        return narma_memory.run(output/'narma',cfg)
    return laser_forecast.run(ROOT/'data/santa_fe_laser/SF_Acont.dat',output/'laser',cfg)


def validate_summary(output,study):
    spec=validate_spec(output)
    source=output/study
    summary=json.loads((source/'summary.json').read_text())
    if summary['config']!=spec['configurations'][study]:
        raise ValueError('Unexpected realized study configuration')
    for name,expected in summary['files_sha256'].items():
        if digest(source/name)!=expected:
            raise ValueError(f'Changed result source {study}/{name}')
    module=narma_memory if study=='narma' else laser_forecast
    rebuilt=module.summarize(summary['results'],configs()[study])
    if rebuilt['paired_contrasts']!=summary['paired_contrasts']:
        raise ValueError('Paired contrast mismatch')
    if not all(summary['checks'].values()):
        raise ValueError('Incomplete or nonphysical study')
    return summary


def finish(output):
    studies={k:validate_summary(output,k) for k in configs()}
    primary={
        'narma_qelm_minus_qrc':studies['narma']['paired_contrasts']['qelm_minus_qrc'],
        'narma_replay4_minus_qrc':studies['narma']['paired_contrasts']['replay_d4_minus_qrc'],
        'laser_fast_h1_qelm_minus_qrc':studies['laser']['paired_contrasts']['gamma4_h1']['qelm_minus_qrc'],
        'laser_fast_h1_replay4_minus_qrc':studies['laser']['paired_contrasts']['gamma4_h1']['restricted_d4_minus_qrc'],
    }
    gates={k:{'positive_roots':v['positive_seeds'],'total_roots':v['n_seeds'],
              'mean':v['mean'],'mean_ci95':v['mean_ci95'],
              'passed':v['positive_seeds']>=20 and v['mean_ci95'][0]>0}
           for k,v in primary.items()}
    result={'specification_sha256':digest(output/'replication_specification.json'),
            'study_summaries_sha256':{k:digest(output/k/'summary.json') for k in studies},
            'direction_checks':gates,'all_primary_directions_replicated':all(v['passed'] for v in gates.values()),
            'scope':'prospective directional replication; not new laser data or universal QRC superiority'}
    dump(output/'replication_report.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('initialize','narma','laser','finish'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='initialize':
        initialize(args.output)
    elif args.action=='finish':
        print(json.dumps(finish(args.output),indent=2))
    else:
        run_study(args.output,args.action)
