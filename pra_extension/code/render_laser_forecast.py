"""Cobalt-style candidates for direct forecasting, separate from the manuscript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, MaxNLocator, MultipleLocator, LogLocator, FuncFormatter
import numpy as np
import pandas as pd

import laser_forecast as experiment
from laser_demonstration import ROOT, digest, dump
import paper_palette as colors
from render_laser_demonstration import save

SIZE=(8.0,4.6)
MODEL_COLORS={'qrc':colors.INK,'qelm':colors.CORAL}
RATE_COLORS={4.:colors.COBALT,.2:colors.VIOLET}


def validate_source(source):
    summary=json.loads((source/'summary.json').read_text())
    cfg=experiment.ForecastConfig()
    if summary['config']!=json.loads(json.dumps(experiment.asdict(cfg))):
        raise ValueError('Forecast configuration differs from the specified cohort')
    for filename, expected in summary['files_sha256'].items():
        if digest(source/filename)!=expected:
            raise ValueError(f'Changed forecasting source: {filename}')
    spec=json.loads((source/'run_specification.json').read_text())
    for name, expected in spec['sources'].items():
        if digest(ROOT/name)!=expected:
            raise ValueError(f'Changed specified implementation: {name}')
    reconstructed=experiment.summarize(summary['results'],cfg)
    if reconstructed['paired_contrasts']!=summary['paired_contrasts']:
        raise ValueError('Paired contrast mismatch')
    rows=pd.DataFrame([{**row,'seed':cell['seed'],'gamma':cell['gamma']}
                       for cell in summary['results'] for row in cell['rows']])
    expected = {(seed, rate, horizon, model, window)
                for seed in cfg.seeds for rate in cfg.rates for horizon in cfg.horizons
                for model, window in [('qrc', None), ('qelm', cfg.window), ('persistence', 1)]
                + [(model, window) for model in ('replay_refit', 'replay_frozen')
                   for window in cfg.replay_windows]}
    actual = {(r.seed, r.gamma, r.horizon, r.model,
               None if pd.isna(r.window) else int(r.window)) for r in rows.itertuples()}
    if len(rows)!=len(expected) or actual!=expected or not np.isfinite(rows[['nrmse','rmse']].to_numpy()).all():
        raise ValueError('Keep all 720 root/model/window/horizon rows')
    traces={}
    for rate in cfg.rates:
        with np.load(source/f'seed{cfg.seeds[0]}_gamma{rate:g}.npz') as stored:
            traces[rate]={key:stored[key].copy() for key in stored.files
                          if key in {'test_indices'} or key.startswith(('target_h','qrc_h','qelm_h'))
                          and stored[key].ndim==1 and len(stored[key])==1916}
    return summary, rows, traces


def style():
    return {'font.family':['Helvetica Neue','Helvetica','Arial','sans-serif'],
            'font.size':10,'axes.labelsize':10.5,'axes.labelweight':'normal',
            'axes.linewidth':.7,'text.color':colors.INK,'axes.labelcolor':colors.INK,
            'axes.edgecolor':colors.INK,'xtick.color':colors.INK,'ytick.color':colors.INK,
            'legend.fontsize':9,'legend.framealpha':1,'legend.edgecolor':colors.INK,
            'pdf.fonttype':42,'savefig.facecolor':'white'}


def axis(fig,x,y,w,h):
    ax=fig.add_axes([x/SIZE[0],y/SIZE[1],w/SIZE[0],h/SIZE[1]])
    ax.grid(color=colors.GRID,lw=.5,ls=(0,(4,4)))
    ax.set_axisbelow(True)
    ax.tick_params(length=3,width=.7,pad=3)
    return ax


def title(fig,x,y,letter,label):
    fig.text(x/SIZE[0],y/SIZE[1],letter,fontsize=11,weight='bold')
    fig.text((x+.2)/SIZE[0],y/SIZE[1],label,fontsize=10.5)


def build_figure(rows,traces,horizon,*,frozen=False):
    rows=rows[rows.horizon==horizon]
    fig=plt.figure(figsize=SIZE)
    time_axes=[axis(fig,.55,bottom,3.08,1.44) for bottom in (2.73,.55)]
    accuracy=axis(fig,4.43,2.73,3.32,1.44)
    memory=axis(fig,4.43,.55,3.32,1.44)
    title(fig,.55,4.36,'a','Next measured value' if horizon==1 else 'Four-step laser forecast')
    title(fig,4.43,4.36,'b','Forecast accuracy')
    title(fig,4.43,2.19,'c','Without refitting the readout' if frozen else 'Restricting reservoir history')
    values=np.concatenate([traces[r][f'{key}_h{horizon}'][:128]
                           for r in (4.,.2) for key in ('target','qrc','qelm')])
    lo,hi=float(values.min()),float(values.max())
    span=max(hi-lo,1)
    for ax,rate,label in zip(time_axes,(4.,.2),('Fast loss','Slow loss')):
        data=traces[rate]
        x=np.arange(128)
        ax.plot(x,data[f'target_h{horizon}'][:128],color=colors.COBALT,lw=1.15,
                label='Measured future',gid='measured_future')
        ax.plot(x,data[f'qrc_h{horizon}'][:128],color=colors.INK,lw=.7,ls='--',
                marker='.',ms=2.1,label='QRC',gid='qrc_forecast')
        ax.plot(x,data[f'qelm_h{horizon}'][:128],color=colors.CORAL,lw=.7,
                marker='o',mfc='white',ms=2.8,mew=.7,markevery=3,label='QELM',gid='qelm_forecast')
        ax.set_ylim(lo-.08*span,hi+.23*span)
        ax.set_xlim(-1.5,128.5)
        ax.set_xticks([0,32,64,96,127])
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_ylabel('Laser intensity')
        ax.text(.025,.97,label,transform=ax.transAxes,va='top',fontsize=9.4,
                bbox={'boxstyle':'round,pad=.2','facecolor':'white','edgecolor':colors.INK,'linewidth':.65})
    time_axes[0].tick_params(labelbottom=False)
    time_axes[-1].set_xlabel('Target step in test excerpt')
    handles,labels=time_axes[0].get_legend_handles_labels()
    legend=fig.legend(handles,labels,ncol=3,loc='center',
                      bbox_to_anchor=(2.08/SIZE[0],2.36/SIZE[1]),
                      handlelength=1.3,handletextpad=.35,columnspacing=.8,
                      borderpad=.4,fontsize=8.8)
    legend.get_frame().set_linewidth(.65)

    for center,rate in enumerate((4.,.2)):
        data=rows[(rows.gamma==rate)&rows.model.isin(['qrc','qelm'])]
        pivot=data.pivot(index='seed',columns='model',values='nrmse').sort_index()
        jitter=np.linspace(-.045,.045,len(pivot))
        for i,(_,row) in enumerate(pivot.iterrows()):
            accuracy.plot([center-.13+jitter[i],center+.13+jitter[i]],
                          [row.qrc,row.qelm],color=colors.GRID,lw=.7,zorder=1)
        for offset,name,marker in ((-.13,'qrc','o'),(.13,'qelm','s')):
            vals=pivot[name].to_numpy()
            median=np.median(vals)
            low,high=np.quantile(vals,[.1,.9])
            accuracy.scatter(center+offset+jitter,vals,color=MODEL_COLORS[name],
                             s=8,alpha=.4,zorder=2)
            accuracy.errorbar(center+offset,median,yerr=[[median-low],[high-median]],
                              color=MODEL_COLORS[name],marker=marker,ms=5,lw=1,
                              capsize=3,zorder=3)
    accuracy.set_xticks([0,1],labels=['Fast loss','Slow loss'])
    accuracy.set_xlim(-.5,1.5)
    accuracy.set_ylabel('Test forecast error')
    accuracy.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    accuracy.yaxis.set_major_locator(MultipleLocator(.1 if horizon==1 else .2))
    accuracy.set_ylim(bottom=0)
    accuracy.legend(handles=[Line2D([],[],color=MODEL_COLORS[k],marker=m,lw=.8,label=lab)
                             for k,m,lab in [('qrc','o','QRC'),('qelm','s','QELM')]],
                    loc='upper center',ncol=2,fontsize=8.6,borderpad=.3)
    accuracy.set_ylim(0,accuracy.get_ylim()[1]*1.45)

    model='replay_frozen' if frozen else 'replay_refit'
    for rate,marker in ((4.,'o'),(.2,'D')):
        selected=rows[(rows.gamma==rate)&(rows.model==model)]
        grouped=selected.groupby('window').nrmse
        x=np.array(sorted(selected.window.unique()),float)
        median=grouped.median().loc[x].to_numpy()
        lower=grouped.quantile(.1).loc[x].to_numpy()
        upper=grouped.quantile(.9).loc[x].to_numpy()
        memory.fill_between(x,lower,upper,color=RATE_COLORS[rate],alpha=.14,lw=0)
        memory.plot(x,median,color=RATE_COLORS[rate],marker=marker,ms=3.5,lw=1,
                    label='Fast loss' if rate==4 else 'Slow loss',gid=f'restricted_gamma{rate:g}')
        reference=float(rows[(rows.gamma==rate)&(rows.model=='qrc')].nrmse.median())
        memory.axhline(reference,color=RATE_COLORS[rate],ls=(0,(4,3)),lw=.9,
                       gid=f'full_gamma{rate:g}')
    memory.set_xscale('log',base=2)
    memory.set_xticks([1,2,4,8,16,32],labels=['1','2','4','8','16','32'])
    memory.set_xlim(.85,37)
    memory.set_ylabel('Test forecast error')
    memory.set_xlabel('Inputs retained after reset')
    memory.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    if frozen:
        memory.set_yscale('log')
        memory.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
        memory.yaxis.set_major_formatter(FuncFormatter(
            lambda y, _: f'{y/10000:g}M%' if y>=10000 else
                         f'{y/10:g}k%' if y>=10 else f'{100*y:g}%'))
        memory.set_ylim(top=memory.get_ylim()[1]*8)
    else:
        memory.yaxis.set_major_locator(MaxNLocator(4))
        memory.set_ylim(bottom=0)
        memory.set_ylim(0,memory.get_ylim()[1]*1.3)
    memory.legend(loc='upper right',fontsize=8.4,ncol=2,borderpad=.3,columnspacing=.65)
    memory.text(.98,.64,'Dashed: full history',ha='right',va='top',transform=memory.transAxes,
                color=colors.MUTED,fontsize=8.2)
    return fig,{'time':time_axes,'accuracy':accuracy,'memory':memory,'legend':legend}


def render(source,output):
    summary,rows,traces=validate_source(source)
    output.mkdir(parents=True,exist_ok=True)
    artifacts={}
    for h in (1,4):
        with plt.rc_context(style()):
            fig,_=build_figure(rows,traces,h)
            artifacts.update(save(fig,output,f'laser_forecast_h{h}'))
        with plt.rc_context(style()):
            fig,_=build_figure(rows,traces,h,frozen=True)
            artifacts.update(save(fig,output,f'laser_forecast_h{h}_frozen_readout'))
    rows.to_csv(output/'laser_forecast_all_rows.csv',index=False)
    artifacts['laser_forecast_all_rows.csv']=digest(output/'laser_forecast_all_rows.csv')
    for rate,arrays in traces.items():
        frame=pd.DataFrame(arrays)
        name=f'laser_forecast_display_seed_gamma{rate:g}.csv'
        frame.to_csv(output/name,index=False)
        artifacts[name]=digest(output/name)
    dump(output/'laser_forecast_source.json',{
        'status':'separate_forecasting_candidate_not_integrated',
        'sources':{str((source/'summary.json').relative_to(ROOT)):digest(source/'summary.json'),
                   str(Path(__file__).relative_to(ROOT)):digest(Path(__file__)),
                   str((ROOT/'pra_extension/code/paper_palette.py').relative_to(ROOT)):
                   digest(ROOT/'pra_extension/code/paper_palette.py')},
        'summary':summary,'figure_size_inches':list(SIZE),'artifacts':artifacts,
        'scope':'Targets are actual future laser values. These are not QRC-output-emulation plots.',
        'caption':'a Actual future laser intensity and independently trained QRC/QELM forecasts '
                  'for the predeclared first seed. b Paired seed results, medians and 10th-90th '
                  'percentiles across 12 couplings per loss setting. c The same QRC is reset '
                  'and given only the last d inputs; readouts are refitted for each d. '
                  'Dashed lines mark full-history QRC median errors, not information floors. '
                  'All errors are NRMSE relative to training-target SD. The frozen-readout '
                  'variants apply the original full-QRC readout without refitting.',
    })
    print(json.dumps({'output':str(output),'artifacts':artifacts},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'results_laser_forecast_v1')
    p.add_argument('--output',type=Path,default=ROOT/'paper/figures/laser_forecast_candidate')
    a=p.parse_args()
    render(a.source,a.output)
