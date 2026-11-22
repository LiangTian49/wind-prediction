# P3b：坐标与遥感来源有限对照

最新验收：P3b日期分析912973已验收：COMPLETED/0:0、54秒，288区间/6048逐日贡献/45份绝对结果齐全。夏季完整RS优于简单坐标，DSM单源相对无RS的RMSE/MAE区间为正；完整RS相对单源夏季区间均跨零，不能声称融合互补。秋季各静态组MAE相对无RS均退步。结果和论文补充已记录；当前无待重提P3/P3b任务，新年份方案尚未冻结。

最新验收：P3b新增27/27训练已验收（912913/912946），18份P3参照复用，combined_results45行、paired_results63行。夏季LC/DSM相对无RS RMSE均改善约0.68%，坐标约0.004%；完整RS相对坐标改善0.896%。当前仅开发均值，新增对照日期区间尚未执行。已准备slurm/run_p3b_analysis.sh，下一步用户手动提交，新增训练0次。

最新验收：P3b先导912913已3/3验收，全部COMPLETED/0:0，耗时3分48秒至4分28秒；产物、样本签名、静态结构及18份参照检查通过。可手动提交sbatch --array=3-26%4 slurm/run_p3b.sh补齐24次；剩余任务当前未核实已提交或完成。

2026-11-17。代码、配置、静态表示、旧结果引用检查及轻量测试已准备。真实P3b训练尚未运行。本轮只新增27次拟合，不重复原320次或原P3参照。

## 1. 实际入口和冻结来源

- 训练：train_p3b.py；静态表示与来源检查：p3b_contract.py。
- 配置：configs/rs_hpc_v3_p3b_v1.json。
- 旧结果清单：configs/rs_hpc_v3_p3b_v1_references.json，包含18份原RS_S0/RS_S1的run_id和completion SHA256。
- 汇总：summarize_p3b.py。
- Slurm：slurm/run_p3b.sh。
- 科学方案：[NEXT_RS_CONTROLS_PROTOCOL](../experiments/NEXT_RS_CONTROLS_PROTOCOL_2026-11-17.md)。

动态输入仍为A01的11维V/时间编码、56步历史、未来3h、F01–F03、seed42/43/44。LSTM64＋Dropout0.4＋静态Dense8＋拼接＋Dense16＋Dense1；静态14槽，参数20761。Adam lr0.001、batch64、最多100轮、验证早停patience10并恢复最佳权重。目标仍为已知目标格的10m CERRA风速。

| 组 | 静态14槽 | 新增拟合 |
|---|---|---:|
| S_COORD | 槽0标准化grid_lat、槽1标准化grid_lon，其他12槽为零 | 9 |
| S_LC | 原11个标准化土地覆盖比例，DSM三槽关闭 | 9 |
| S_DSM | 原土地覆盖11槽关闭，保留三个标准化DSM量 | 9 |

LC/DSM都先沿用原14维scaler，再缩放后置零；坐标scaler同样仅fit唯一训练网格。两者记录独立保存。不因样本重复加权静态scaler。所有点位到格网连接保持窗口顺序；原始100个point_id训练，按98格去重评价。

## 2. 首先手动提交3项先导

从项目根目录执行：

```bash
sbatch --array=0-2%3 slurm/run_p3b.sh
```

| 索引 | fold | seed | 组 |
|---:|---|---:|---|
| 0 | F01 | 42 | S_COORD |
| 1 | F01 | 42 | S_LC |
| 2 | F01 | 42 | S_DSM |

每项1个CPU节点、1任务、8工作线程、16GB、4小时，最多3项并行，不申请GPU。模块使用目前可工作的Stages/2025 GCCcore/.13.3.0 Python/3.12.3并激活项目wind-env。实际AllocCPUS可能48，成本看sacct，不将8工作线程当作全部分配核数。

日志绝对位置：/p/project1/sdlrs/tian1/wind_prediction/logs/p3b_ARRAYID_TASKID.out/.err。
结果：runs/rs_hpc_v3/P3b/v1/INDEX_FOLD_GROUP_seedSEED/。

旧结果保留。既有新阶段run的指纹一致时验证后跳过；不一致时拒绝覆盖。中断产生的隐藏部分目录保留。

## 3. 先导验收

激活wind-env后：

```bash
python summarize_p3b.py --expected pilot --require_complete
```

要求3/3新任务完成，同时18份旧参照元数据有效。先导combined_results包含3项新拟合和同折同seed的2份旧参照，即5行；18份参照清单不等于18次新增训练。sacct应显示三个array任务COMPLETED/0:0。

每项要求：

- train/validation/test数量42400/22300/22300；98个唯一目标网格；动态56×11、静态14、参数20761；RMSE/MAE有限。
- model.keras、preprocessors.pkl（动态/遥感/坐标三个scaler及mask规则）、config.json、training_log.csv、predictions.csv、result.json、reference_sources.json及completion.json齐全。
- 训练前日志出现Pre-fit reference, scaler and all-partition sample checks passed。全部分区样本签名、动态/遥感scaler统计和测试键/标签/Persistence与匹配P3一致；失败时不进入fit。
- 早停只使用验证集，Persistence逐点逐窗口，全部对照共享样本。先导验收不要求收益为正。

汇总输出new_results.csv、combined_results.csv、paired_results.csv、paired_seed_summary.csv。分别标明新拟合和P3_reused来源，报告全部7种预定比较。seed标准差不充当日期区间。

## 4. 先导验收后再补24项

```bash
sbatch --array=3-26%4 slurm/run_p3b.sh
```

最多4项并行。完成后：

```bash
python summarize_p3b.py --expected full --require_complete
```

预期27/27新任务、18份复用参照，combined_results共45行，paired_results共63行（3折×3种子×7比较）。不能把旧18份结果计为本轮训练成本。

日期分析将在完整结果验收后准备；当前未提供P3b日期分析脚本，也未运行该阶段。新的年份/空间确认和额外16维坐标＋RS对照不在本轮提交范围。

## 5. 轻量检查和计时说明

可在wind-env运行以下命令，不会训练或重建全年窗口：

```bash
python train_p3b.py --preflight
python train_p3b.py --list
python -m unittest discover -s tests -p 'test_p3*.py'
bash -n slurm/run_p3b.sh
```

已运行11项轻量回归检查并通过，日志logs/p3b_regression_tests.out；Python编译、Slurm语法及18份元数据预检通过。没有真实训练、全量提取或性能基准。

preflight只读配置、已冻结元数据与小型静态表；完整参照产物、窗口和模型检查留在Slurm训练前。真实拟合不允许在登录节点执行。

result.json的preparation_seconds覆盖窗口与静态表示准备，fit_seconds仅model.fit，prediction_seconds仅测试集predict；elapsed_seconds覆盖execute入口至结果保存前的流程，不包括最后原子产物哈希/发布。sacct记录整个Slurm作业耗时与实际分配资源。当前不是P6固定工作量性能基准，不能拿不同早停轮数的耗时直接宣称加速。

坐标与遥感的比较仅检验本网络下的信息表示，不能单独证明“给定坐标后遥感仍有条件增量”，也不能证明物理机制或空间迁移。所有三个开发窗口的正负结果均保留。

## 全部训练完成后的日期分析

已准备analyze_p3b_dates.py及slurm/run_p3b_analysis.sh；提交sbatch slurm/run_p3b_analysis.sh。1线程、4GB、30分钟。日志logs/p3b_dates_JOBID.out/.err，输出result/rs_hpc_v3/P3b/v1/analysis_JOBID/。预期288条区间、6048条逐日贡献、45行absolute_results.csv、method.json和completion.json。7个静态比较＋5个Persistence比较，逐seed与固定三seed均值分别报告。日期分析尚未实际运行。新增2项源组合/路径校验测试通过，不涉及真实训练。
