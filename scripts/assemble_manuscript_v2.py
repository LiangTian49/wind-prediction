"""Create the reviewed manuscript, upload assets and original backup; no model execution."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from direction_pipeline import verify_artifacts

BUNDLE=ROOT/'docs/paper/manuscript_v2'
REVIEW=ROOT/'docs/paper/main_review_v1'
FIGURES=ROOT/'result/paper_figures/review_v1_913096'


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def math_number(value,places=3):
    return f'{value:.{places}f}'


def uncertainty(row,metric):
    return rf'\({row[metric].mean():.4f}\pm{row[metric].std(ddof=1):.4f}\)'


def interval(row,prefix='gain',places=3):
    keys=('gain_pct','date_p025_pct','date_p975_pct') if prefix=='gain' else ('RS_gain_difference_pp','date_p025_pp','date_p975_pp')
    return rf'\({row[keys[0]]:.{places}f}\;[{row[keys[1]]:.{places}f},{row[keys[2]]:.{places}f}]\)'


def table(headers,rows,caption,label,alignment=None,wide=False):
    result=[r'\begin{table}[htbp]\centering\small',r'\caption{'+caption+'}',r'\label{'+label+'}']
    if wide:result.append(r'\resizebox{\textwidth}{!}{%')
    result += [r'\begin{tabular}{'+(alignment or 'l'*len(headers))+r'}\toprule',
               ' & '.join(headers)+r'\\\midrule']
    result += [' & '.join(str(v) for v in row)+r'\\' for row in rows]
    result += [r'\bottomrule\end{tabular}']
    if wide:result.append('}')
    result.append(r'\end{table}')
    return '\n'.join(result)


def main():
    if BUNDLE.exists():raise FileExistsError('Preserve manuscript_v2; use a new version for a new assembly.')
    source_manifest=json.loads((REVIEW/'source_manifest.json').read_text())
    original=ROOT/'main.tex'
    if digest(original)!=source_manifest['manuscript']['sha256']:
        raise ValueError('main.tex changed since review; inspect before replacing.')
    verify_artifacts(FIGURES)
    data={}
    for name,record in source_manifest['tables'].items():
        path=ROOT/record['path']
        if digest(path)!=record['sha256']:raise ValueError('Changed reviewed table.')
        data[name]=pd.read_csv(path)
    folds=json.loads((ROOT/'configs/seasonal_folds_2024.json').read_text())['folds']
    replacements={}
    replacements['FOLD_TABLE']=table(['Window','Training start','Validation start','Test start','Test end (exclusive)'],
        [[f['fold_id']]+[f[k][:10] for k in ['train_start','validation_start','test_start','test_end_exclusive']] for f in folds],
        'Fixed 2024 windows. Each partition spans 60/28/28 days; timestamps are UTC and intervals are left-closed/right-open.', 'tab:folds',wide=True)
    replacements['EXPERIMENT_TABLE']=table(['Stage','Lead (h)','Windows','Seeds','New fits','Reused fits'],[
        ['Original spatial','3/6/12/24','F01--F04','42--46','320','0'],
        ['Ridge / tree candidates','3/6/12/24','F01--F03','42','144','0'],
        ['Direction controls','3','F01--F03','42','12','0'],
        ['Four-group surface fusion','3','F01--F03','42--44','36','0'],
        ['Coordinate/source controls','3','F01--F03','42--44','27','18'],
        ['History controls (8/24/56)','3','F01--F03','42--44','36','18'],
        ['Lead controls (3/6)','3/6','F01--F03','42--44','18','18']],
        'Completed experimental stages. Reused references are not new training runs. Ridge/tree selection retains 48 selected fits from 144 candidates. Earlier exploratory protocols are excluded from the formal result tables.', 'tab:experiments',wide=True)
    seasonal=data['seasonal'];rows=[]
    for fold,block in seasonal[seasonal.lead_hours.eq(3)].groupby('fold_id'):
        t=block[block.experiment.eq('T01')].iloc[0];a=block[block.experiment.eq('A01')].iloc[0]
        u=lambda r,m:rf'\({r[m+"_mean"]:.4f}\pm{r[m+"_std"]:.4f}\)'
        rows.append([fold,u(t,'RMSE'),u(a,'RMSE'),u(t,'MAE'),u(a,'MAE'),
            rf'\({a.RMSE_vs_T01_gain_pct:.2f}\;[{a.RMSE_vs_T01_date_p025_pct:.2f},{a.RMSE_vs_T01_date_p975_pct:.2f}]\)'])
    replacements['SPATIAL_TABLE']=table(['Window','T01 RMSE','A01 RMSE','T01 MAE','A01 MAE','RMSE gain (\\%) [date interval]'],rows,
        'Original three-hour spatial results. Absolute errors are mean $\\pm$ sample SD over five seeds, in $\\ms$. Relative gains are within-seed paired means; intervals use target dates.', 'tab:spatial',wide=True)
    p=data['p1'];rows=[]
    for r in p[p.experiment.eq('A01')&p.lead_hours.eq(3)].sort_values(['fold_id','model_family']).itertuples():
        family='Ridge' if r.model_family=='ridge' else 'Histogram boosting'
        rows.append([r.fold_id,family,f'{r.RMSE:.4f}',f'{r.MAE:.4f}',
            rf'\({r.RMSE_same_family_center_gain_pct:.2f}\;[{r.RMSE_same_family_center_date_p025_pct:.2f},{r.RMSE_same_family_center_date_p975_pct:.2f}]\)'])
    replacements['BASELINE_TABLE']=table(['Window','Model','RMSE','MAE','Spatial RMSE gain (\\%) [date interval]'],rows,
        'Validation-selected simpler spatial baselines at three hours. Absolute errors are in $\\ms$; spatial gains use each model family\'s center reference. These are fixed fitted models, not multi-seed ensembles.', 'tab:baselines',wide=True)
    rows=[]
    for (fold,group),block in data['p3'].groupby(['fold_id','group']):
        rows.append([fold,rf'\model{{{group}}}',uncertainty(block,'RMSE'),uncertainty(block,'MAE')])
    replacements['FUSION_TABLE']=table(['Window','Condition','RMSE ($\\ms$)','MAE ($\\ms$)'],rows,
        'Four-group fusion results at 56 observations and a three-hour lead. Entries are mean $\\pm$ sample SD across three seeds. T/S identify center/spatial dynamics; 0/1 identify static-off/on inputs.', 'tab:fusion')
    temporal=data['temporal_dates'];fixed=temporal[temporal.seed.eq('fixed_seed_mean')]
    def metric_row(block):
        return [interval(block[block.metric.eq(m)].iloc[0]) for m in ['RMSE','MAE']]
    hist=fixed[fixed.stage.eq('history')&fixed.model.str.endswith('RS_S1')&fixed.reference.str.endswith('RS_S0')];rows=[]
    for (fold,model),block in hist.groupby(['fold_id','model'],sort=False):
        rows.append([fold,model.split('_')[0][1:]]+metric_row(block))
    replacements['HISTORY_TABLE']=table(['Window','Observations','RS RMSE gain (\\%) [date interval]','RS MAE gain (\\%) [date interval]'],rows,
        'Surface-feature effects within each matched historical length. Positive gains favor full surface descriptors. Intervals use forecast-origin dates across three fixed fitted seeds.', 'tab:history',wide=True)
    horizon=fixed[fixed.stage.eq('horizon')&fixed.model.str.endswith('RS_S1')&fixed.reference.str.endswith('RS_S0')];rows=[]
    for (fold,model),block in horizon.groupby(['fold_id','model'],sort=False):
        rows.append([fold,model.split('_H')[1].split('_')[0]]+metric_row(block))
    replacements['HORIZON_TABLE']=table(['Window','Lead (h)','RS RMSE gain (\\%) [date interval]','RS MAE gain (\\%) [date interval]'],rows,
        'Surface-feature effects at common three-/six-hour forecast origins with $L=56$. Targets differ by lead. Three-hour test metrics are recomputed on the common subset, so they differ slightly from the full three-hour results. Intervals use origin dates.', 'tab:horizon',wide=True)
    surface=data['p3b_dates'];surface=surface[surface.seed.eq('fixed_seed_mean')&surface.reference.ne('Persistence')];rows=[]
    for (fold,model,reference),block in surface.groupby(['fold_id','model','reference'],sort=False):
        rows.append([fold,rf'\model{{{model}}}',rf'\model{{{reference}}}']+metric_row(block))
    replacements['STATIC_TABLE']=table(['Window','Model','Reference','RMSE gain (\\%) [interval]','MAE gain (\\%) [interval]'],rows,
        'All registered static-input contrasts, $L=56$, lead three hours. Fixed-three-seed mean gains and target-date intervals are retained, including adverse autumn results.', 'tab:static-full',wide=True)
    effects=data['interactions'];effects=effects[effects.seed.eq('fixed_seed_mean')];rows=[]
    for (stage,fold,model,base),block in effects.groupby(['stage','fold_id','model','baseline_model'],sort=False):
        condition=(model.split('_')[0]+' minus '+base.split('_')[0]) if stage=='history' else '6 h minus 3 h'
        rows.append([stage.capitalize(),fold,condition]+[interval(block[block.metric.eq(m)].iloc[0],prefix='interaction') for m in ['RMSE','MAE']])
    replacements['INTERACTION_TABLE']=table(['Control','Window','Contrast','RMSE difference (pp) [interval]','MAE difference (pp) [interval]'],rows,
        'Direct differences in surface-feature gains under shared origin-date resamples. Units are percentage points, not percent error reductions. A positive difference indicates a larger relative surface-feature gain in the first setting.', 'tab:interactions',wide=True)
    cases=[]
    for fold in ['F01','F03','F04']:
        for kind,term in [('maps','map'),('series','series')]:
            caption=(f'Fixed {fold} regional snapshot, original three-hour spatial experiment, seed 42. Actual evaluated cells are shown without interpolation; blank locations are unpredicted.' if kind=='maps' else f'Fixed {fold} 72-hour origin case, three-hour lead, seed 42, at the same longitude-selected target cells. Curves use target timestamps and the CERRA reference; no case-based lag or extreme-peak claim is made.')
            cases.append('\\begin{figure}[htbp]\n\\centering\\includegraphics[width=\\textwidth]{S_case_'+fold+'_'+kind+'.png}\n\\caption{'+caption+'}\n\\label{fig:'+fold.lower()+'-'+term+'}\n\\end{figure}\n\\FloatBarrier')
    replacements['OTHER_CASE_FIGURES']='\n'.join(cases)
    text=(ROOT/'docs/paper/templates/main_v2.tex.in').read_text()
    for key,value in replacements.items():
        token='@@'+key+'@@'
        if text.count(token)!=1:raise ValueError(f'Missing/duplicate template token: {token}')
        text=text.replace(token,value)
    if re.search(r'@@\w+@@',text):raise ValueError('Unresolved manuscript token.')
    BUNDLE.mkdir(parents=True)
    assets=BUNDLE/'figures';assets.mkdir()
    for path in sorted(FIGURES.iterdir()):
        if path.suffix in ['.pdf','.png']:shutil.copyfile(path,assets/path.name)
    for name in ['CAPTIONS_AND_USE.md','case_manifest.json','figure_manifest.json']:
        shutil.copyfile(FIGURES/name,BUNDLE/name)
    shutil.copytree(REVIEW/'tables',BUNDLE/'tables')
    (BUNDLE/'main.tex').write_text(text)
    archive=ROOT/'docs/paper/manuscript_archive';archive.mkdir(exist_ok=True)
    backup=archive/f'main_before_v2_{digest(original)[:12]}.tex'
    if backup.exists() and digest(backup)!=digest(original):raise ValueError('Backup name collision.')
    if not backup.exists():shutil.copyfile(original,backup)
    (BUNDLE/'assembly_provenance.json').write_text(json.dumps(dict(
        status='assembled_pending_compile',created_at_utc=datetime.now(timezone.utc).isoformat(),
        original_main_sha256=digest(original),original_backup=str(backup.relative_to(ROOT)),
        figure_job='913096',figure_source_completion_sha256=digest(FIGURES/'completion.json'),
        result_snapshot_manifest_sha256=digest(REVIEW/'source_manifest.json'),
        template_sha256=digest(ROOT/'docs/paper/templates/main_v2.tex.in'),
        table_sources=source_manifest['tables'],main_sha256=digest(BUNDLE/'main.tex'),
        original_figures={p.name:digest(p) for p in assets.iterdir()},
        scope='Revised manuscript and figure integration; no new scientific experiment.'),indent=2)+'\n')
    # Publish root source/assets only after the complete versioned source and backup exist.
    root_assets=ROOT/'figures';root_assets.mkdir(exist_ok=True)
    for path in assets.iterdir():
        target=root_assets/path.name
        if target.exists() and digest(target)!=digest(path):raise FileExistsError(f'Existing figure differs: {target}')
        if not target.exists():shutil.copyfile(path,target)
    shutil.copyfile(BUNDLE/'main.tex',ROOT/'main.tex')
    print(f'Assembled {BUNDLE}; root main.tex updated; original retained at {backup}.')


if __name__=='__main__':main()
