"""Reproducible paper diagrams and archived-summary plots; submit via Slurm."""
import argparse
import json
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,FancyBboxPatch,Rectangle
import numpy as np
import pandas as pd

from direction_pipeline import code_hashes,output_transaction,sha256,verify_artifacts
from temporal_folds import project_path

SNAPSHOT='docs/paper/main_review_v1'
COLORS={'F01':'#0072B2','F02':'#009E73','F03':'#D55E00','F04':'#7B3294'}


def load_snapshot(directory):
    manifest=json.loads((directory/'source_manifest.json').read_text())
    tables={}
    for name,record in manifest['tables'].items():
        path=project_path(record['path'])
        if sha256(path)!=record['sha256']:raise ValueError(f'Changed snapshot table: {name}')
        tables[name]=pd.read_csv(path)
        if len(tables[name])!=record['rows']:raise ValueError('Changed table coverage.')
    if len(tables['temporal_dates'])!=456 or len(tables['interactions'])!=72:
        raise ValueError('Require complete date analysis.')
    return manifest,tables


def save(fig,out,name):
    for suffix in ['pdf','png']:
        fig.savefig(out/f'{name}.{suffix}',dpi=300,bbox_inches='tight')
    plt.close(fig)


def box(ax,x,y,w,h,text,color='#E8F1F8',size=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',fc=color,ec='0.3',lw=1))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size)


def arrow(ax,start,end,dashed=False):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=12,lw=1.2,
                                 color='0.25',linestyle='--' if dashed else '-'))


def workflow(out):
    fig,ax=plt.subplots(figsize=(13,7));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    box(ax,.02,.70,.24,.19,'CERRA 10-m wind speed\nNative grid, 3-hourly analyses\nFixed targets + neighbors')
    box(ax,.37,.70,.25,.19,'Dynamic sequences\nCenter: 3 channels / spatial: 11\nHistory: 8 / 24 / 56 samples')
    box(ax,.02,.29,.24,.22,'ESA WorldCover + GLO-30 DSM\nTarget-cell area aggregation\n11 class fractions + 3 descriptors','#E8F4E8')
    box(ax,.37,.29,.25,.22,'Static branch: 14 features\nFit scaling on training grids\nNo-RS: zero after scaling','#E8F4E8')
    box(ax,.74,.49,.23,.22,'Shared LSTM + static fusion\nOne target-cell scalar/sample\nRolling-origin prediction')
    box(ax,.37,.02,.25,.15,'Train / validation / test\nTraining-only preprocessing\nValidation-only early stopping','#FFF2D9',9)
    box(ax,.74,.08,.23,.19,'Evaluation at known target grids\nFuture CERRA reference\nRMSE / MAE / paired intervals','#F2E9F6',9)
    for a,b in [((.26,.80),(.37,.80)),((.26,.40),(.37,.40)),((.62,.80),(.74,.65)),
                 ((.62,.40),(.74,.55)),((.50,.17),(.50,.29)),((.85,.49),(.85,.27))]:arrow(ax,a,b)
    ax.text(.02,.96,'Reanalysis and remote sensing are separate input streams',fontsize=13,weight='bold')
    ax.text(.02,.61,'Same ROI / point-to-grid mapping; no future test values enter inputs',fontsize=10)
    ax.text(.02,.02,'Direction is a completed ablation; the selected main representation omits it.\nWind-energy applications motivate this retrospective grid-scale experiment.',fontsize=9)
    save(fig,out,'F01_workflow')


def spatial(out):
    mapping=pd.read_csv(project_path('data/rs_static/features_v1/point_grid_mapping.csv'))
    unique=mapping.drop_duplicates('grid_id')
    cases=json.loads(project_path('result/rs_hpc_v3/P0b/v1/figure_manifest.json').read_text())
    fig,(a,b)=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
    a.scatter(unique.grid_lon,unique.grid_lat,s=15,color='0.35',label='98 unique target cells')
    a.add_patch(Rectangle((8.5,49.5),3,2,fill=False,ls='--',ec='0.4'))
    for location in cases[0]['locations']:
        a.scatter(location['lon'],location['lat'],s=42,c='#D55E00')
        a.annotate(location['grid_id'],(location['lon'],location['lat']),xytext=(4,4),textcoords='offset points',fontsize=8)
    a.set(xlabel='Longitude (degrees E)',ylabel='Latitude (degrees N)',title='Fixed target ROI; case cells labeled',xlim=(8.3,11.7),ylim=(49.35,51.65))
    a.set_aspect(1/np.cos(np.deg2rad(50.5)));a.grid(alpha=.2);a.legend(fontsize=8)
    for radius,color,name in [(2,'0.7','S02: 5 x 5'),(1,'#0072B2','S01: 3 x 3')]:
        points=np.array([(dy,dx) for dy in range(-radius,radius+1) for dx in range(-radius,radius+1)])
        b.scatter(points[:,1],points[:,0],s=35,marker='s',c=color,label=name)
    far=np.array([(y,x) for y in [-10,0,10] for x in [-10,0,10] if (y,x)!=(0,0)])
    b.scatter(far[:,1],far[:,0],s=70,marker='x',c='#D55E00',label='A01: center + 8 distant cells')
    b.scatter([0],[0],s=70,marker='o',c='black',label='T01: center')
    b.set(xlabel='Array-index offset dx',ylabel='Array-index offset dy',title='Per-target input configurations',xlim=(-12,12),ylim=(-12,12));b.set_aspect('equal');b.grid(alpha=.2);b.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.15))
    fig.suptitle('100 point identifiers map to 98 cells; unpredicted locations remain blank\nNeighbor inputs may extend beyond the target ROI; index axes are not geographic east/north',fontsize=11)
    save(fig,out,'F02_roi_and_neighbors')


def architecture(out):
    fig,axes=plt.subplots(2,1,figsize=(13,7.5),layout='constrained')
    for ax in axes:ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    a,b=axes
    for x,w,text in [(.01,.21,'Dynamic sequence\n[B, L, C]'),(.30,.19,'LSTM(64)\n[B, 64]'),(.56,.17,'Dropout(0.4)'),(.82,.16,'Dense(1)\n[B, 1]')]:box(a,x,.35,w,.38,text)
    for x1,x2 in [(.22,.30),(.49,.56),(.73,.82)]:arrow(a,(x1,.54),(x2,.54))
    a.text(.01,.9,'A. Original temporal/spatial baseline: L=56; C=3 or 11; parameters=17,473 or 19,521',fontsize=11,weight='bold')
    box(b,.01,.65,.20,.22,'Dynamic [B, L, C]')
    box(b,.28,.65,.22,.22,'LSTM64 + Dropout0.4\n[B, 64]',size=9)
    box(b,.01,.20,.20,.25,'Static [B, 14]\nOff: standardized zeros','#E8F4E8',9)
    box(b,.28,.20,.22,.25,'Dense8 (ReLU)\n[B, 8]','#E8F4E8',9)
    box(b,.59,.41,.16,.25,'Concatenate\n[B, 72]',size=9)
    box(b,.81,.41,.18,.25,'Dense16 (ReLU)\nDense1 (linear)\n[B, 1]',size=9)
    for start,end in [((.21,.76),(.28,.76)),((.21,.32),(.28,.32)),((.50,.76),(.59,.58)),((.50,.32),(.59,.48)),((.75,.53),(.81,.53))]:arrow(b,start,end)
    b.text(.01,.97,'B. Fusion controls: L56 center/spatial; L8/L24 spatial only; parameters=18,713 or 20,761',fontsize=11,weight='bold')
    b.text(.01,.02,'B = batch; C = per-time dynamic features. Shared weights across target cells; scalar output, not a 2-D field decoder.',fontsize=9)
    save(fig,out,'F03_architecture')


def protocol(out,manifest):
    config=json.loads(project_path(manifest['sources']['folds']['path']).read_text())
    fig,ax=plt.subplots(figsize=(12,4.5),layout='constrained')
    for y,fold in enumerate(config['folds']):
        dates=[pd.Timestamp(fold[key]) for key in ['train_start','validation_start','test_start','test_end_exclusive']]
        for start,end,color in zip(dates,dates[1:],['#0072B2','#E69F00','#009E73']):
            ax.barh(y,(end-start).days,left=start,height=.5,color=color)
        ax.text(dates[-1]+pd.Timedelta(days=2),y,fold['test_start'][:10],va='center',fontsize=9)
    ax.set_yticks(range(4),['F01 spring','F02 summer','F03 autumn','F04 original winter']);ax.invert_yaxis()
    import matplotlib.dates as mdates
    ax.xaxis_date();ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'));ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.set(xlim=(pd.Timestamp('2024-01-01'),pd.Timestamp('2025-01-23')),xlabel='2024 calendar time',title='60-day training / 28-day validation / 28-day test')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c,label=n) for c,n in zip(['#0072B2','#E69F00','#009E73'],['Training: fit preprocessing','Validation: early stopping','Test: evaluation only'])],loc='upper left',bbox_to_anchor=(0,-.16),ncol=3,fontsize=9)
    fig.suptitle('RS and temporal controls use F01–F03 only; F04 has already been evaluated\nContext can cross boundaries; training/validation labels cannot cross the next boundary',fontsize=10)
    save(fig,out,'F04_temporal_protocol')


def forest(ax,frame,labels,value='gain_pct',low='date_p025_pct',high='date_p975_pct',xlabel='RMSE improvement (%)'):
    for y,((_,r),text) in enumerate(zip(frame.iterrows(),labels)):
        color=COLORS[r.fold_id]
        ax.plot([r[low],r[high]],[y,y],color=color,lw=2)
        ax.scatter(r[value],y,c=color,s=28,zorder=3)
    ax.axvline(0,c='0.3',ls='--',lw=.8);ax.set_yticks(range(len(frame)),labels);ax.invert_yaxis();ax.set_xlabel(xlabel);ax.grid(axis='x',alpha=.15)


def result_plots(out,tables):
    seasonal=tables['seasonal'];p1=tables['p1']
    a=seasonal[seasonal.experiment.eq('A01')&seasonal.lead_hours.eq(3)].sort_values('fold_id')
    a=a.rename(columns={'RMSE_vs_T01_gain_pct':'gain_pct','RMSE_vs_T01_date_p025_pct':'date_p025_pct','RMSE_vs_T01_date_p975_pct':'date_p975_pct'})
    b=p1[p1.experiment.eq('A01')&p1.lead_hours.eq(3)].sort_values(['fold_id','model_family'])
    b=b.rename(columns={'RMSE_same_family_center_gain_pct':'gain_pct','RMSE_same_family_center_date_p025_pct':'date_p025_pct','RMSE_same_family_center_date_p975_pct':'date_p975_pct'})
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    forest(axes[0],a,list(a.fold_id));forest(axes[1],b,[f'{r.fold_id}: {r.model_family}' for r in b.itertuples()])
    axes[0].set_title('Original LSTM: A01 vs T01');axes[1].set_title('Simpler baselines: spatial vs same-model center')
    fig.suptitle('Spatial dynamic-input gains at 3 h; original target-date intervals')
    save(fig,out,'F05_spatial_dynamic_gain')
    static=tables['p3b_dates'];static=static[static.seed.eq('fixed_seed_mean')&static.reference.eq('RS_S0')]
    for metric in ['RMSE','MAE']:
        q=static[static.metric.eq(metric)].sort_values(['fold_id','model'])
        fig,ax=plt.subplots(figsize=(9,7),layout='constrained')
        forest(ax,q,[f'{r.fold_id}: {r.model}' for r in q.itertuples()],xlabel=f'{metric} improvement vs no-static (%)')
        ax.set_title(f'Static representations, L=56, lead=3 h ({metric})\nCoordinates / land cover / DSM / full RS; target-date intervals')
        save(fig,out,f'F06_static_controls_{metric}')
    data=tables['temporal_dates'];interaction=tables['interactions']
    for metric in ['RMSE','MAE']:
        for stage,prefix in [('history','F07_history'),('horizon','F08_horizon')]:
            q=data[data.stage.eq(stage)&data.seed.eq('fixed_seed_mean')&data.metric.eq(metric)&data.model.str.endswith('RS_S1')&data.reference.str.endswith('RS_S0')]
            r=interaction[interaction.stage.eq(stage)&interaction.seed.eq('fixed_seed_mean')&interaction.metric.eq(metric)]
            fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
            forest(axes[0],q,[f'{v.fold_id}: {v.model.split("_RS")[0]}' for v in q.itertuples()],xlabel=f'{metric} improvement with RS (%)')
            forest(axes[1],r,[f'{v.fold_id}: {v.model.split("_RS")[0]} vs {v.baseline_model.split("_RS")[0]}' for v in r.itertuples()],
                   value='RS_gain_difference_pp',low='date_p025_pp',high='date_p975_pp',xlabel='Difference in RS improvement (percentage points)')
            axes[0].set_title('Within-condition RS effect');axes[1].set_title('Direct paired difference in RS effects')
            fig.suptitle(f'{stage.capitalize()} controls: {metric}; common origin-date resampling\nFixed-three-seed mean; intervals exclude retraining uncertainty and have no multiplicity correction',fontsize=10)
            save(fig,out,f'{prefix}_{metric}')


def surface(out):
    features=pd.read_csv(project_path('data/rs_static/features_v1/static_features.csv'))
    mapping=pd.read_csv(project_path('data/rs_static/features_v1/point_grid_mapping.csv')).drop_duplicates('grid_id')
    data=mapping.merge(features,on='grid_id',validate='one_to_one')
    fig,axes=plt.subplots(1,3,figsize=(13,4.7),layout='constrained')
    for ax,column,title in zip(axes,['lc_50','dsm_mean_m','dsm_slope_mean_deg'],['Built-up area fraction','DSM mean surface elevation (m)','Mean DSM-derived slope (degrees)']):
        image=ax.scatter(data.grid_lon,data.grid_lat,c=data[column],cmap='viridis',s=28)
        ax.set(xlabel='Longitude (degrees E)',ylabel='Latitude (degrees N)',title=title)
        ax.set_aspect(1/np.cos(np.deg2rad(50.5)));fig.colorbar(image,ax=ax,shrink=.8)
    fig.suptitle('Target-cell descriptors at the 98 evaluated grids; unpredicted locations remain blank',fontsize=11)
    save(fig,out,'S01_surface_descriptors')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=project_path,default=project_path(SNAPSHOT))
    parser.add_argument('--preflight',action='store_true',help='Only validate small snapshot tables; no plotting')
    args=parser.parse_args()
    manifest,tables=load_snapshot(args.snapshot)
    if args.preflight:
        print(f'Paper snapshot verified: {len(tables)} tables; no plots rendered. Main SHA={manifest["manuscript"]["sha256"]}')
        return
    job=os.environ.get('SLURM_JOB_ID')
    if not job:parser.error('Submit slurm/run_paper_figures.sh.')
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    provenance=project_path('result/rs_hpc_v3/P0b/v1');verify_artifacts(provenance)
    static=project_path('data/rs_static/features_v1');verify_artifacts(static)
    cfg=json.loads(project_path(manifest['sources']['p3_config']['path']).read_text())
    for name in ['folds','p3_config','case_manifest','static_ready']:
        if sha256(project_path(manifest['sources'][name]['path']))!=manifest['sources'][name]['sha256']:
            raise ValueError('A snapshot metadata source changed.')
    fingerprint=dict(snapshot_sha256=sha256(args.snapshot/'source_manifest.json'),
        p0b_completion_sha256=sha256(provenance/'completion.json'),static_completion_sha256=sha256(static/'completion.json'),
        model_contract_sha256=sha256(project_path('rs_static_contract.py')),
        code=code_hashes(['plot_paper_review.py','direction_pipeline.py']))
    final=project_path(f'result/paper_figures/review_v1_{job}')
    with output_transaction(final,fingerprint) as work:
        if work is None:return
        workflow(work);spatial(work);architecture(work);protocol(work,manifest);surface(work);result_plots(work,tables)
        for fold in ['F01','F02','F03','F04']:
            for kind in ['maps','series']:
                shutil.copyfile(provenance/f'{fold}_{kind}.png',work/f'S_case_{fold}_{kind}.png')
        captions='''# Figure use and captions

F01: Conceptual workflow with separate CERRA dynamics and remote-sensing surface streams. Direction is an ablation, not the selected main input. Test reference values are used only for evaluation.
F02: Real target coordinates and index-space neighborhoods. Target ROI does not imply continuous prediction coverage; context may extend outside the ROI. Labels identify actual fixed-case cells, not assumed point-ID ordering.
F03: Original direct-head baseline and matched static-fusion model. B is batch size; L history samples; C dynamic channels. Each sample predicts one target cell. Static-off groups retain the same architecture.
F04: Fixed 2024 temporal protocol. F04 was the original confirmation window and is already used. It is not an unseen confirmation for new RS/history models.
F05: Spatial RMSE gains at 3h. Original LSTM and simpler baselines use their archived target-date intervals. Three-day circular date blocks, 2000 draws; no multiplicity correction. Not a distribution of scores across seeds.
F06: Static representations relative to no-static spatial dynamics at L56/H3. Fixed-three-seed mean, archived target-date intervals. Both RMSE and MAE plots are provided; retain adverse autumn results.
F07: History dependence of RS effects and direct differences versus L56. Shared origin-date block resampling; actual input starts vary, but eligible origins/labels/counts are matched. Right panel units are percentage points.
F08: Lead dependence at L56. 3h metrics are recomputed on common 6h origins; future targets differ. Right panel directly tests differences in RS effects. Do not infer difference from non-overlap of separate confidence intervals.
S01: Example surface descriptors at 98 target grids. DSM describes the surface, not bare-earth terrain or measured aerodynamic roughness. WorldCover map attribution: © ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium. Cite archived product metadata and GLO-30 license.
S_case_*: Copied verified P0b artifacts, not reselected by current performance. Seed42, earliest common complete 72h origin interval per fold; locations are longitude-sorted first/middle/last target cells. The time-series x-axis is target_time. Reference is future CERRA, not station observations. Maps are points at actual evaluated grids with blank unpredicted locations. Use the copied case_manifest.json for exact origin/target times. A line plot does not establish eliminated phase lag, an extreme-peak skill metric or a causal weather-propagation mechanism.

Intervals describe dates in the observed windows across fixed fitted models, exclude retraining uncertainty, and are post hoc without multiplicity correction. Earlier P3/P3b intervals group by target date; temporal-control intervals group by forecast-origin date. These conventions are deliberately labeled, not pooled silently.

These newly generated figures still require visual inspection before submission. Choose 6–8 main figures and move remaining metric variants/maps to the supplement. Captions must retain the relevant definitions, units, baseline, lead, seed aggregation and uncertainty convention.
'''
        (work/'CAPTIONS_AND_USE.md').write_text(captions)
        shutil.copyfile(provenance/'figure_manifest.json',work/'case_manifest.json')
        (work/'figure_manifest.json').write_text(json.dumps(dict(status='generated_pending_visual_review',
            manuscript_snapshot=manifest['manuscript'],snapshot=str(args.snapshot.relative_to(project_path('.'))),
            selected_model_configuration=cfg['architecture'],outputs=sorted(p.name for p in work.iterdir()),
            scope='Summary plots and metadata schematics; no training, GRIB extraction, new bootstrap or independent confirmation',
            source_manifest_sha256=sha256(args.snapshot/'source_manifest.json')),indent=2))
    print(f'[Paper figures] Completed: {final}')


if __name__=='__main__':main()
