"""Source-backed Cobalt candidate: actual NARMA target, not QRC emulation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd

import narma_memory as study
from laser_demonstration import ROOT, digest, dump
from render_laser_demonstration import save
from render_laser_forecast import style
import paper_palette as color

SIZE=(9.,5.6)


def check_sources(source):
    summary=json.loads((source/'summary.json').read_text())
    cfg=study.NarmaConfig()
    if summary['config']!=json.loads(json.dumps(study.asdict(cfg))):
        raise ValueError('Unspecified NARMA configuration')
    for name,expected in summary['files_sha256'].items():
        if digest(source/name)!=expected:
            raise ValueError(f'Changed source {name}')
    spec=json.loads((source/'run_specification.json').read_text())
    for name,expected in spec['sources'].items():
        if digest(ROOT/name)!=expected:
            raise ValueError(f'Changed specified implementation {name}')
    reconstructed=study.summarize(summary['results'],cfg)
    if reconstructed['paired_contrasts']!=summary['paired_contrasts']:
        raise ValueError('Paired summary mismatch')
    expected_keys={('qrc',None),('qelm',4),('training_mean',None)}
    expected_keys.update(('replay',d) for d in cfg.replay_windows)
    expected_keys.update(('quadratic',d) for d in cfg.quadratic_windows)
    for cell in summary['results']:
        if len(cell['rows'])!=len(expected_keys) or {(r['model'],r['window']) for r in cell['rows']}!=expected_keys:
            raise ValueError('Incomplete model/window coverage')
        if len(cell['candidate_rows'])!=8:
            raise ValueError('Incomplete loss candidates')
        for model in ('qrc','qelm'):
            best=min((r for r in cell['candidate_rows'] if r['model']==model),
                     key=lambda r:r['validation_mse'])
            if cell['selection'][model]!=best['gamma']:
                raise ValueError('Selection does not match validation minimum')
    rows=pd.DataFrame([{**r,'seed':cell['seed']} for cell in summary['results'] for r in cell['rows']])
    first=next(c for c in summary['results'] if c['seed']==cfg.seeds[0])
    with np.load(source/first['array_file']) as saved:
        trace={k:saved[k].copy() for k in saved.files}
    return summary,rows,trace


def axis(fig,x,y,w,h):
    ax=fig.add_axes([x/SIZE[0],y/SIZE[1],w/SIZE[0],h/SIZE[1]])
    ax.grid(color=color.GRID,lw=.5,ls=(0,(4,4)))
    ax.set_axisbelow(True)
    ax.tick_params(length=3,width=.65,pad=3)
    return ax


def title(fig,x,y,letter,text):
    fig.text(x/SIZE[0],y/SIZE[1],letter,fontsize=11,weight='bold')
    fig.text((x+.21)/SIZE[0],y/SIZE[1],text,fontsize=10.7)


def build_figure(rows,trace):
    fig=plt.figure(figsize=SIZE)
    time_axes=[axis(fig,.64,bottom,3.68,1.15) for bottom in (3.83,2.25,.67)]
    memory=axis(fig,5.18,3.24,3.55,1.74)
    pair=axis(fig,5.18,.67,3.55,1.63)
    title(fig,.64,5.29,'a','Predicting the actual NARMA10 target')
    title(fig,5.18,5.29,'b','More accessible history')
    title(fig,5.18,2.60,'c','Same recent inputs, different targets')
    names=[('qelm','QELM · four-input window',color.CORAL),
           ('qrc','QRC · recurrent state',color.COBALT),
           ('replay_d12','Reset/replay · twelve inputs',color.VIOLET)]
    target=trace['test_target'][:96]
    visible=np.concatenate([target]+[trace[key][:96] for key,_,_ in names])
    lo,hi=float(visible.min()),float(visible.max())
    span=hi-lo
    for ax,(key,label,c) in zip(time_axes,names):
        x=np.arange(96)
        ax.plot(x,target,color=color.INK,lw=1.05,label='Target',gid='true_target')
        ax.plot(x,trace[key][:96],color=c,lw=.9,ls='--',marker='o',mfc='white',
                ms=2.6,mew=.6,markevery=3,label='Prediction',gid=key)
        ax.set_title(label,loc='left',fontsize=9.5,pad=7,color=c)
        error=np.sqrt(np.mean((trace[key]-trace['test_target'])**2))/float(trace['training_target_sd'])
        ax.set_title(f'Test: {error:.1%}',loc='right',fontsize=8.4,pad=7,color=color.MUTED)
        ax.set_ylim(lo-.07*span,hi+.10*span)
        ax.set_xlim(-1,96)
        ax.set_xticks([0,24,48,72,95])
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.set_ylabel('NARMA output',fontsize=9.6)
    for ax in time_axes[:-1]:
        ax.tick_params(labelbottom=False)
    time_axes[-1].set_xlabel('Target step in test excerpt')
    # A neutral style key applies to all three model-coloured panels.
    fig.legend(handles=[Line2D([],[],color=color.INK,lw=1,label='Target'),
                        Line2D([],[],color=color.MUTED,ls='--',marker='o',mfc='white',
                               ms=3,lw=.9,label='Prediction')],
               loc='center',bbox_to_anchor=(2.48/SIZE[0],3.73/SIZE[1]),ncol=2,
               fontsize=8.4,handlelength=1.8,borderpad=.3,columnspacing=1.1)

    grouped=rows[rows.model=='replay'].groupby('window').nrmse
    x=np.array(study.NarmaConfig().replay_windows)
    med=grouped.median().loc[x].to_numpy()
    low=grouped.quantile(.1).loc[x].to_numpy()
    high=grouped.quantile(.9).loc[x].to_numpy()
    memory.axvspan(.85,4,color=color.CORAL,alpha=.08,lw=0)
    memory.fill_between(x,low,high,color=color.VIOLET,alpha=.17,lw=0)
    memory.plot(x,med,color=color.VIOLET,marker='D',ms=3.5,lw=1.1,
                label='Reset/replay (same circuit)',gid='history_control')
    qrc=rows[rows.model=='qrc'].nrmse.median()
    memory.axhline(qrc,color=color.COBALT,lw=1,ls=(0,(4,3)),
                   label='QRC, full history',gid='full_qrc')
    qelm=rows[rows.model=='qelm'].nrmse
    loq,hiq=qelm.quantile([.1,.9])
    memory.errorbar(4,qelm.median(),yerr=[[qelm.median()-loq],[hiq-qelm.median()]],
                    color=color.CORAL,marker='s',ms=4.3,lw=.9,capsize=2,
                    label='Parallel QELM (4 inputs)',gid='parallel_qelm')
    classical=rows[rows.model=='quadratic'].groupby('window').nrmse
    # Two isolated points, not a fitted or interpolated capacity curve.
    for d in (4,12):
        vals=classical.get_group(d)
        medc=vals.median()
        loc,hic=vals.quantile([.1,.9])
        memory.errorbar(d,medc,yerr=[[medc-loc],[hic-medc]],color=color.MUTED,
                        marker='*',ms=6,mfc='white',mew=.8,lw=.7,capsize=2,
                        label='Quadratic window control' if d==4 else None,
                        gid=f'quadratic_{d}')
    memory.set_xscale('log',base=2)
    memory.set_xticks(x,labels=[str(d) for d in x])
    memory.set_xlim(.85,57)
    memory.set_ylim(0,max(.95,float(high.max())*1.48))
    memory.set_ylabel('Test error (NRMSE)')
    memory.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    memory.yaxis.set_major_locator(MaxNLocator(4))
    memory.set_xlabel('Number of recent inputs retained')
    memory.legend(loc='upper right',fontsize=7.5,borderpad=.32,labelspacing=.25,
                  handlelength=1.7)

    x=np.arange(1,len(trace['pair_target_a'])+1)
    pair.axvspan(4,x[-1]+.5,color=color.COBALT,alpha=.07,lw=0)
    pair.axvline(4,color=color.MUTED,ls=(0,(3,3)),lw=.8)
    for key,c,label in [('target',color.INK,'Actual target difference'),
                        ('qrc',color.COBALT,'QRC prediction difference'),
                        ('qelm',color.CORAL,'QELM prediction difference')]:
        gap=np.abs(trace[f'pair_{key}_a']-trace[f'pair_{key}_b'])
        start=3 if key=='qelm' else 0
        pair.plot(x[start:],gap[start:],color=c,lw=1.1,
                  ls='--' if key=='qrc' else '-',marker='o' if key=='qelm' else None,
                  markevery=4,ms=2.7,mfc='white',mew=.7,label=label,gid=f'pair_gap_{key}')
    pair.set_xlim(.5,32.5)
    pair.set_xticks([1,4,12,24,32])
    pair.set_ylim(bottom=-.001)
    pair.set_ylim(top=pair.get_ylim()[1]*1.35)
    pair.yaxis.set_major_locator(MaxNLocator(4))
    pair.set_ylabel('Output difference',fontsize=9.6)
    pair.set_xlabel('Number of identical inputs received')
    pair.legend(loc='upper right',fontsize=7.5,labelspacing=.3,borderpad=.35,handlelength=1.7)
    pair.text(.99,.51,'After 4 inputs:\nsame QELM window',ha='right',va='top',
              transform=pair.transAxes,color=color.MUTED,fontsize=8.4)
    fig.text(.64/SIZE[0],.12/SIZE[1],
             'Synthetic NARMA10 · independent test inputs · 12 runs · exact quantum expectations',
             fontsize=8.4,color=color.MUTED)
    return fig,{'time':time_axes,'memory':memory,'pair':pair}


def all_seed_figure(rows):
    fig,ax=plt.subplots(figsize=(8.,4.6))
    fig.subplots_adjust(left=.29,right=.95,top=.89,bottom=.16)
    labels=[('qelm',4,'QELM · 4 inputs',color.CORAL),
            ('replay',4,'Reset/replay · 4 inputs',color.VIOLET),
            ('qrc',None,'QRC · full history',color.COBALT),
            ('replay',12,'Reset/replay · 12 inputs',color.VIOLET),
            ('quadratic',4,'Quadratic · 4 inputs',color.MUTED),
            ('quadratic',12,'Quadratic · 12 inputs',color.MUTED)]
    for i,(name,d,label,c) in enumerate(labels):
        selected=rows[(rows.model==name)&(rows.window.isna() if d is None else rows.window==d)]
        vals=selected.sort_values('seed').nrmse.to_numpy()
        y=len(labels)-1-i
        ax.scatter(vals,y+np.linspace(-.11,.11,len(vals)),color=c,s=18,alpha=.65)
        ax.plot([np.median(vals)]*2,[y-.22,y+.22],color=c,lw=2)
    ax.set_yticks(range(len(labels)),labels=[l[2] for l in labels[::-1]])
    ax.set_xlabel('Test error (NRMSE); each dot is one independent run')
    ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.set_xlim(0,max(rows[rows.model!='training_mean'].nrmse)*1.08)
    ax.grid(axis='x',color=color.GRID,ls=(0,(4,4)),lw=.6)
    ax.set_axisbelow(True)
    ax.set_title('Every run, including the classical context checks',loc='left',fontsize=12,pad=15)
    return fig


def render(source,output):
    summary,rows,trace=check_sources(source)
    output.mkdir(parents=True,exist_ok=True)
    artifacts={}
    with plt.rc_context(style()):
        fig,_=build_figure(rows,trace)
        artifacts.update(save(fig,output,'narma_memory_candidate'))
        artifacts.update(save(all_seed_figure(rows),output,'narma_memory_all_runs'))
    for name,data in (
        ('all_selected_rows.csv',rows),
        ('all_candidate_rows.csv',pd.DataFrame([{**r,'seed':c['seed']} for c in summary['results'] for r in c['candidate_rows']])),
        ('display_test_trace.csv',pd.DataFrame({k:trace[k] for k in ('test_indices','test_target','qrc','qelm','replay_d12','quadratic_d12')})),
        ('display_pair_trace.csv',pd.DataFrame({k:trace[k] for k in trace if k.startswith('pair_') and len(trace[k])==47})),
    ):
        data.to_csv(output/name,index=False)
        artifacts[name]=digest(output/name)
    dump(output/'figure_source.json',{
        'status':'separate_candidate_not_integrated_into_paper',
        'summary_sha256':digest(source/'summary.json'),
        'renderer_sha256':digest(Path(__file__)),
        'display_seed':study.NarmaConfig().seeds[0], 'display_excerpt':[0,96],
        'artifacts':artifacts, 'size_inches':SIZE,
        'caption':('Learning the task itself. a The models predict the actual NARMA10 target from driving inputs only. '
                   'The same fixed test excerpt is shown; errors describe its full test split. '
                   'b Restricting and then expanding the same reservoir circuit\'s accessible history tests whether older inputs help. '
                   'Lines show medians over 12 independent runs; shading/bars show 10th–90th percentiles, not confidence intervals. '
                   'Gray stars are classical quadratic controls (14/90 features), not matched quantum-advantage baselines. '
                   'c Different earlier inputs followed by an identical continuation can still require different target values (first 32 shared steps shown). '
                   'After four common inputs the four-input QELM must give the same prediction. '
                   'Synthetic benchmark, exact-expectation simulation; settings selected on validation only. '
                   'Reset/replay uses sequential encoding, and longer windows cost more evolution steps.'),
    })
    return output/'narma_memory_candidate.png'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    print(render(args.source,args.output))
