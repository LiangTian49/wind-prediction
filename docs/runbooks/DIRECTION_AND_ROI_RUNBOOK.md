# 风向提取、区域制图与P1c提交说明

截至2026-11-14，P0b作业912777、风向提取912778及P1c先导912779/补齐912802均已完成并验收，共12次P1c拟合。日期分析912824的42条区间也已验收。当前下一步为sbatch slurm/run_p2_download.sh，见[P2下载说明](P2_DOWNLOAD_RUNBOOK.md)。下方命令保留为历史执行记录。

三折空间方向验证收益平均-1.70%，按预先登记规则关闭方向。测试收益及后续日期区间保留用于探索性报告。后续训练命令作为历史流程记录，当前12次均已完成。

## 已完成的两项准备作业（历史命令）

```bash
sbatch slurm/run_extract_direction.sh
sbatch slurm/run_p0b.sh
```

| 作业 | 申请资源 | 时限 | 输出 | 日志 |
|---|---|---|---|---|
| 风向提取 | dp-cn，1节点、1任务、1线程、8 GB | 4h | data/CERRA/direction_rs_hpc_v3/ | logs/direction_JOBID.out/.err |
| P0b归档分析与制图 | dp-cn，1节点、1任务、1线程、8 GB | 2h | result/rs_hpc_v3/P0b/v1/ | logs/p0b_JOBID.out/.err |

dp-cn既有作业记录显示实际分配48核节点，资源成本以本次sacct的AllocCPUS和Elapsed为准。表内线程数描述脚本使用的并行线程。

风向任务扫描既有GRIB索引并抽取原中心及A01外围格。速度与方向须为10m瞬时分析、零步长、相同网格及完整2024三小时时间轴。V由原归档CSV桥接，原始GRIB与归档V在1e-6 m/s容差内核对，再以归档V构造u/v。保存原始方向用于追溯与后续分组。数据的completion.json含产物摘要，metadata.json含原GRIB SHA256、映射来源、单位、公式和核对结果。

端点归一化使用各GRIB消息的packingError。仅在[-packingError, 360+packingError]内的端点越界按360度环绕，有限性与范围异常继续严格报错。direction_raw_10系列列保留原始解码值，direction_10系列列供转换与分析使用，direction_quality.csv逐时记录误差界、原始范围和修改的唯一输入格数。若再次失败，独立临时目录会保存failure_diagnostic.json。修复后的6项针对性测试通过，全量结果待重跑。

风向验收要求作业COMPLETED/0:0，metadata.json的status为complete，2928个时刻、100个point_id、98个唯一中心、292800行；A01.csv有九点的V/u/v和原始方向。日志最后显示[Direction] Completed。completion.json须通过verify_direction_data验证，P1c启动时会自动验证。

P0b读取320个旧LSTM和48个P1预测，保存逐格RMSE、MAE、Bias及相对同模型T01和Persistence的收益。相同种子及时间内去重中心，保留重复预测最大差异检查。地图采用真实98个目标格的点图。每折使用seed42和首个共同完整72小时段；曲线位置按经度排序取首/中/末格。模型包括LSTM T01/A01、Persistence以及开发折验证选定的P1强中心。F04只用已有LSTM。固定色阶可直接比较同一时刻不同模型。

P0b验收要求summary.json为completed、archived_runs=368、grid_metric_rows=36064、cases=4。查看grid_metrics.csv、grid_skill.csv、grid_coverage.csv、pooled_metrics.csv、duplicate_grid_audit.csv、figure_manifest.json、roi_dimensions.json及architectures.json。每折地图和曲线保存PNG/SVG，另有LSTM架构SVG。逐格RMSE算术均值和pooled RMSE使用独立字段。图件需要人工检查标注与可读性。

## 已完成的P1c先导（历史命令）

```bash
sbatch --array=0-3%4 slurm/run_p1c.sh
```

| 索引 | 折 | 组 | 每步通道数 | u/v开关 | 参数量 |
|---:|---|---|---:|---|---:|
| 0 | F01 | D_T0 | 5 | 缩放后置零 | 17985 |
| 1 | F01 | D_T1 | 5 | 开启 | 17985 |
| 2 | F01 | D_S0 | 29 | 缩放后置零 | 24129 |
| 3 | F01 | D_S1 | 29 | 开启 | 24129 |

每项dp-cn，1节点、1任务、8工作线程、16 GB、4小时，最多4项并行。实际分配成本仍查sacct。3h直接预测，56步，seed42，LSTM64、Dropout0.4、Adam0.001、batch64、100轮上限、验证早停patience10、恢复最佳权重。四组V和时间处理相同。每折仅用训练段拟合缩放，关闭组在缩放之后将u/v置零。

配置为configs/rs_hpc_v3_p1c.json。输出为runs/rs_hpc_v3/P1c/INDEX_FOLD_GROUP_seed42/，日志为logs/p1c_ARRAYID_TASKID.out/.err。每项保存完整配置、参数量、模型、缩放器与关闭规则、折清单、训练日志、预测、验证及测试指标、各阶段耗时和摘要。

先导验收要求4项均COMPLETED/0:0，四份result.json，训练/验证/测试数量均为42400/22300/22300，98个唯一网格，通道和参数量与表一致。每项自动与相同折、同种子旧T01/A01核对起报、输入开始、标签和逐点Persistence。

激活wind-env后可以做小型产物汇总。

```bash
module --force purge
module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3
source wind-env/bin/activate
python summarize_p1c.py
```

先导应显示4/12。验收通过后再补其余8次。

```bash
sbatch --array=4-11%4 slurm/run_p1c.sh
```

随后执行python summarize_p1c.py --require_complete，预期12/12。三折验证集按配置中的等权RMSE增量规则生成dynamic_selection.json，测试得分不参与选择。该文件记录开发选择，P3前还需审阅并冻结动态输入与比较组。日期区间与新时间确认仍是后续分析步骤。

## 重试与版本

单项失败可以使用sbatch --array=失败索引 slurm/run_p1c.sh重试。各配置独立锁定，成功产物校验通过则跳过，失败尝试保存在独立隐藏目录供检查。提取和P0b同样采用原子目录发布，运行中断后可以原命令重试。若代码、来源或配置摘要发生变化，程序报告版本差异，需新建版本路径并明确记录，避免混合实验。

## 已完成的轻量验证

16项测试通过，覆盖风向符号与速度模长、未来输入扰动、训练段拟合、逐点Persistence、缩放后方向置零、首个共同72小时案例、任务枚举及失败重试/已完成跳过。另以合成零输入验证5/29维模型的标量输出和参数量。Python编译与三个Slurm脚本语法检查通过。真实GRIB提取、制图输出及真实训练尚待执行。
