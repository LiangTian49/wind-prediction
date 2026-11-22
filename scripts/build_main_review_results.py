"""Build a manuscript-facing result snapshot from small archived summaries only."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'docs/paper/main_review_v1'
SOURCES={
 'seasonal':'reports/seasonal/formal_results.csv',
 'p1':'reports/p1/unified_results.csv',
 'p1c':'runs/rs_hpc_v3/P1c/summary.csv',
 'direction':'runs/rs_hpc_v3/P1c/dynamic_selection.json',
 'p3':'runs/rs_hpc_v3/P3/summary.csv',
 'p3_dates':'result/rs_hpc_v3/P3/analysis_912912/date_intervals.csv',
 'p3b':'runs/rs_hpc_v3/P3b/v1/combined_results.csv',
 'p3b_dates':'result/rs_hpc_v3/P3b/v1/analysis_912973/date_intervals.csv',
 'history':'runs/rs_hpc_v3/P1b_history/v1/combined_results.csv',
 'horizon':'runs/rs_hpc_v3/P3c_horizon/v1/combined_results.csv',
 'temporal_dates':'result/rs_hpc_v3/temporal_controls/analysis_913084/date_intervals.csv',
 'interactions':'result/rs_hpc_v3/temporal_controls/analysis_913084/RS_gain_interactions.csv',
 'folds':'configs/seasonal_folds_2024.json',
 'roi':'result/rs_hpc_v3/P0b/v1/roi_dimensions.json',
 'p3_config':'configs/rs_hpc_v3_p3.json',
 'static_ready':'data/rs_static/features_v1/READY.json',
 'static_qa':'data/rs_static/features_v1/qa_summary.json',
 'case_manifest':'result/rs_hpc_v3/P0b/v1/figure_manifest.json',
}


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def md_table(frame,columns):
    lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join('' if pd.isna(v) else f'{v:.3f}' if isinstance(v,float) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir',type=Path,default=DEFAULT)
    args=parser.parse_args()
    out=args.output_dir.resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Output must stay inside project.')
    if (out/'CURRENT_EXPERIMENT_RESULTS.md').exists():raise FileExistsError('Choose a new snapshot directory; old review retained.')
    tables=out/'tables';tables.mkdir(parents=True,exist_ok=True)
    frames={name:pd.read_csv(ROOT/path) for name,path in SOURCES.items() if path.endswith('.csv')}
    assert len(frames['p3'])==36 and len(frames['p3b'])==45 and len(frames['history'])==54 and len(frames['horizon'])==36
    assert len(frames['temporal_dates'])==456 and len(frames['interactions'])==72
    table_records={}
    for name,frame in frames.items():
        path=tables/f'{name}.csv';frame.to_csv(path,index=False)
        table_records[name]=dict(path=str(path.relative_to(ROOT)),sha256=digest(path),rows=len(frame))
    manifest=dict(status='archived_summary_snapshot',created_at_utc=datetime.now(timezone.utc).isoformat(),
        manuscript=dict(path='main.tex',sha256=digest(ROOT/'main.tex'),lines=len((ROOT/'main.tex').read_text().splitlines())),
        sources={name:dict(path=path,sha256=digest(ROOT/path)) for name,path in SOURCES.items()},tables=table_records)
    (out/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    seasonal=frames['seasonal']
    anchor=seasonal[seasonal.lead_hours.eq(3)&seasonal.experiment.isin(['T01','A01'])]
    spatial=[]
    for fold,b in anchor.groupby('fold_id'):
        t=b[b.experiment.eq('T01')].iloc[0];a=b[b.experiment.eq('A01')].iloc[0]
        spatial.append(dict(fold=fold,T01_RMSE=t.RMSE_mean,A01_RMSE=a.RMSE_mean,
            gain_pct=a.RMSE_vs_T01_gain_pct,low=a.RMSE_vs_T01_date_p025_pct,high=a.RMSE_vs_T01_date_p975_pct,
            MAE_gain_pct=a.MAE_vs_T01_gain_pct))
    lines=['# 截至当前的实验结果：供main.tex改稿使用','',
        f'原稿快照：main.tex，共{manifest["manuscript"]["lines"]}行；SHA256={manifest["manuscript"]["sha256"]}。原稿未修改。',
        '所有数字来自已经完成的归档结果；表格副本位于tables/，来源与摘要见source_manifest.json。没有新训练、全量提取或原始预测重算。','',
        '## 1. 任务、数据及解释范围','',
        '同一ROI内已知CERRA目标格的10m风速滚动起报回溯预测。2024年3小时采样，共2928个时刻；100个point_id映射98个唯一格。ROI边界49.5–51.5°N、8.5–11.5°E，南北约222km，东西约208–217km；不是严格200×200km。目标网格集合不等于连续全域覆盖，外围输入可能超出目标ROI边界。',
        '原时间/空间研究为3/6/12/24h；遥感主研究为3h，扩展到6h。静态输入是WorldCover 2021 v200的11类比例与GLO-30 DSM的均值/SD/坡度。CERRA风速/风向与遥感地表数据是两路来源。没有风功率、轮毂高度、独立站点、未来风向、未见地点或业务实时验证。','',
        '## 2. 完成清单与预算','',
        '| 实验 | 完成量 | 结果身份 |','|---|---|---|',
        '| 旧V1/V2探索 | 84＋80次 | 仅历史追溯，不能把旧指标直接当正式主结果 |',
        '| 固定四窗口 | 320次，含冬季80次 | 原协议结果；F04已使用，不能再作新增RS方法的未见确认 |',
        '| P1 | 48个选定配置、144次候选拟合 | Ridge与histogram gradient boosting；不是LightGBM |',
        '| P1c方向 | 12次、42日期区间 | 按验证选择关闭方向，测试探索结果仍保留 |',
        '| P0b区域展示 | 368份归档、98格、4组固定72h案例 | 复用预测，非新增训练 |',
        '| P2 | 98格14维、100点映射，全部QA通过 | 遥感数据准备，不是性能改善证据 |',
        '| P3 | 36次，4组×3折×3seed | 192日期区间 |',
        '| P3b | 27次新增＋18份复用 | 288日期区间，来源/位置消融 |',
        '| 历史长度 | 36次新增＋18份56步复用 | 所有分区共同起报/标签/样本 |',
        '| 6h | 18次新增＋18份3h复用 | 共同起报，不同未来目标；训练标签边界数量不同 |',
        '| 联合日期分析913084 | 456配对区间、72增量差区间、9576逐日贡献 | 54新增模型＋18复用模型；90评价行不是90新训练 |','',
        '## 3. 空间动态信息：正式3h结果','',
        md_table(pd.DataFrame(spatial),['折','T01 RMSE','A01 RMSE','RMSE改善%','日期下界%','日期上界%','MAE改善%']),
        'RMSE/MAE单位m/s。改善先逐seed配对计算再平均，不能要求与四舍五入后的均值之比完全相同。春夏冬相对中心的区间为正，秋季跨零；冬季相对Persistence所有空间组/提前量区间均跨零。不能把相对中心改善写成普遍优于Persistence。','',
        '全部组/提前量/窗口的绝对误差、seed SD与两类参照区间见tables/seasonal.csv（保留T01/S01/S02/A01及Persistence），不能只报告最优窗口。','',
        '## 4. 强基线：3h空间增量','']
    p1=frames['p1'];subset=p1[p1.lead_hours.eq(3)&p1.experiment.eq('A01')]
    cols=['fold_id','model_family','RMSE','MAE','RMSE_same_family_center_gain_pct','RMSE_same_family_center_date_p025_pct','RMSE_same_family_center_date_p975_pct']
    lines += [md_table(subset[cols],['折','模型','RMSE','MAE','空间RMSE改善%','日期下界%','日期上界%']),
        'Ridge与树模型在春夏重现空间收益，秋季区间跨零。这支持动态空间信息价值并非LSTM独有，不证明遥感增量跨模型成立：同信息RS树模型尚未执行。完整48配置见tables/p1.csv。','',
        '## 5. 风向、地表准备和模型定义','',
        'P1c三折空间方向验证收益−2.468%、−0.873%、−1.761%，等权−1.7005%，按原规则direction_enabled=false。测试空间方向RMSE收益2.28%/0.16%/0.66%为探索结果，不能反过来改选。P1c的动态5/29维是方向关闭/开启的同形状对照；P3采用独立冻结的无方向3/11维。',
        'P3中心动态每步3维，A01空间11维；14维静态→Dense8，动态LSTM64→Dropout0.4，与静态拼接72→Dense16→Dense1。中心/空间分别18713/20761参数；旧LSTM直接Dense1中心/空间为17473/19521参数。无RS在缩放后14维置零，不能将旧单头模型冒充融合无RS参照。','',
        '## 6. 遥感默认56步结果与来源/位置对照','',
        'P3四组绝对误差及验证值见tables/p3.csv；P3b包含五个空间静态条件的绝对结果见tables/p3b.csv。下表是默认56步、固定三seed平均配对收益：','']
    r=frames['p3b_dates'];r=r[r.seed.eq('fixed_seed_mean')&r.reference.ne('Persistence')]
    cols=['fold_id','model','reference','metric','gain_pct','date_p025_pct','date_p975_pct']
    lines += [md_table(r[cols],['折','模型','参照','指标','改善%','日期下界%','日期上界%']),
        '夏季完整RS的RMSE/MAE相对无RS区间为正；相对坐标亦为正，但不是给定坐标后的条件增量。仅DSM夏季两个指标区间为正，LC仅MAE区间为正；完整RS相对任一单源的夏季区间均跨零，不能宣称融合互补。春季完整RS相对无RS仍跨零；秋季各静态组MAE相对无RS均负。','',
        '## 7. 历史长度与提前量：最新日期分析','',
        '来源913084。历史8/24步通过截取原56步窗口保持训练42400/验证22300/测试22300样本一致，参数均20761。6h为42300/22200/22200，3h参照在相同起报子集重算，目标时刻必然不同。表内H3/H6为小时，L8/L24/L56为观测数，首末观测时距21/69/165h。','']
    r=frames['temporal_dates'];r=r[r.seed.eq('fixed_seed_mean')&r.model.str.endswith('RS_S1')&r.reference.str.endswith('RS_S0')]
    cols=['stage','fold_id','model','metric','gain_pct','date_p025_pct','date_p975_pct']
    lines += [md_table(r[cols],['阶段','折','条件','指标','RS改善%','日期下界%','日期上界%']),'',
        '直接比较RS增量之差，单位为百分点，而非百分比误差变化：','']
    r=frames['interactions'];r=r[r.seed.eq('fixed_seed_mean')]
    cols=['stage','fold_id','model','baseline_model','metric','RS_gain_difference_pp','date_p025_pp','date_p975_pp']
    lines += [md_table(r[cols],['阶段','折','条件','基准','指标','增量差pp','日期下界pp','日期上界pp']),
        '春季8步RS收益+0.757% [0.271,1.187]，比56步RS增量高0.531个百分点 [0.341,0.713]；有RS的8步模型相对有RS的56步模型绝对RMSE改善0.999% [0.339,1.834]。',
        '夏季三长度的RS两个指标均有利，但短长历史的增量差区间跨零。秋季8步增量比56步大主要伴随无RS参照变弱，8步RS本身仍跨零。6h春夏RS区间跨零、秋季为负；6h减3h的增量差三个窗口均跨零。','',
        '## 8. 不确定性、案例及图件使用','',
        '种子SD描述训练随机性；三日循环区块2000次的95%日期区间描述固定模型在当前窗口内的日期波动，不含重训不确定性，无多重比较校正。P3/P3b按target日期，联合分析按forecast_origin日期；数值相近不代表同一估计，绘图不可静默混合。',
        '默认56步夏季RS有22/28天平方误差下降，删除前三正贡献日后仍约0.61% RMSE改善；这是跨seed平均损失后的事后敏感性分析，估计量不同于逐seed百分比均值。',
        'P0b固定seed42、每折最早共同完整72h起报段，位置按经度排序首/中/末真实目标格。F02案例为7月1–4日，不是原稿7月16日事件；不把point_id 0/45/99直接命名西/中/东。所有曲线以真实target_time对齐；不拼接点位或平均预测作为原模型。','',
        '## 9. 可以写入论文与尚未完成','',
        '核心结论：空间风场历史有重复出现的预测价值；遥感地表特征具有历史配置与窗口依赖的条件性增量，无法概括为全年普遍改善。区域应用指在已知、选定格点上逐目标滚动预测，不是联合输出整幅连续二维风场。',
        '尚未完成：独立新年份确认、空间留出/第二ROI、遥感同信息树模型、16维坐标＋RS条件增量、同源粗化分辨率对照、完整距离曲线、HPC性能基准、功率/轮毂高度/独立站点或极端风峰验证。',
        '没有相位误差指标或事件追踪来证明消除phase lag；没有通用5%阈值可据以证明物理预测能力。旧13.04%/10.93%及MAPE/R²表未绑定当前正式协议，不能沿用。','']
    (out/'CURRENT_EXPERIMENT_RESULTS.md').write_text('\n'.join(lines))
    print(f'Built result snapshot and {len(frames)} small tables: {out}')


if __name__=='__main__':main()
