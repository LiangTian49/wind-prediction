# 历史状态页归档

整理为每日看板前的版本，当前状态见[CURRENT_STATUS](../../CURRENT_STATUS.md)。

# 当前执行状态

目录导航与GitHub操作见[README](../../README.md)。实验文档位于docs/experiments/，操作说明位于docs/runbooks/，作业从项目根目录用sbatch slurm/脚本名.sh提交。小型可共享结果位于reports/。

截至2026-11-14，旧V1 84次、夏季V2 80次、固定四折320次合计484次。冬季80次包含在四折320次内。扩展版本rs_hpc_v3已完成P1的48个配置（144次候选拟合）、P1c的12个开发配置及其日期分析，时间协议继续标为v2_fixed。

| 工作 | 状态 | 产物 |
|---|---|---|
| P0 归档、GRIB、结果定义、统一表、冬季早停 | 已核验 | result/rs_hpc_v3/P0 |
| P0 逐行源数据复核、冬季分布和误差分层 | 已通过，912699 | result/rs_hpc_v3/P0/audit_912699 |
| P1 模型拟合与结果验收 | 48/48通过，144次候选拟合，912700/912704 | result/rs_hpc_v3/P1/P1_REVIEW.md |
| P1 日期配对区间 | 已完成并验收，912751，288条 | result/rs_hpc_v3/P1/analysis_912751/P1_DATE_REVIEW.md |
| P0b区域输出、模型架构与案例图 | 912777已完成并验收 | result/rs_hpc_v3/P0b/VERIFICATION.md |
| 风向提取 | 重跑912778已完成并验收，5个格-时刻端点修正 | data/CERRA/direction_rs_hpc_v3 |
| P1c风向四组消融 | 12/12已验收，912779/912802 | [P1c开发结果](../../reports/p1c/DEVELOPMENT_REVIEW.md) |
| P1c日期区间 | 912824已完成并验收，42条 | result/rs_hpc_v3/P1c/DATE_REVIEW.md |
| P2遥感瓦片 | 16个URL已核对，约0.79GB，下载待提交 | slurm/run_p2_download.sh |
| P1b、P3–P8 | 计划 | WIND_EXPERIMENT_TASKS_FOR_CODEX.md |

P1 以 sklearn HistGradientBoostingRegressor 作为树模型，当前环境缺少 LightGBM。Ridge 四个 alpha，树模型两个候选。所有候选仅用训练数据拟合，按固定验证段 RMSE 选取。树模型关闭内部早停。动态 MinMaxScaler 沿用旧管线且只拟合训练段。每个任务保存选中的候选，无额外重拟合。以五个旧 T01/A01 种子分别配对，并与逐窗口 Persistence 比较。P1 全部使用开发折。P1b 历史长度实验与 P4 距离实验按任务书安排在后续阶段。

## 提交和验收

P0、P1拟合及日期区间均已验收。下方提交命令保留为历史记录。
用户已采纳合作者反馈后的新顺序，执行版本见WIND_FEEDBACK_PLAN_UPDATE_2026-11-11.md第11节。
P0b、风向提取、P1c三折12次和日期分析均已通过。今晚可提交sbatch slurm/run_p2_download.sh，1工作线程、2GB、4小时，下载4个WorldCover与12个GLO-30瓦片，共792406989字节。日志logs/p2_download_JOBID.out/.err，缓存data/rs_static/raw_v1，摘要result/rs_hpc_v3/P2/download_JOBID.json。真实下载尚待执行，详见[P2操作说明](../runbooks/P2_DOWNLOAD_RUNBOOK.md)。
GRIB编码端点修正共5个唯一输入格-时刻，最大原始方向360.000824度，均在对应消息packingError内。旧V最大差4.7681e-7 m/s。原始方向及质量统计保留。全量提取验收通过。
三折空间方向验证RMSE收益为-2.47%/-0.87%/-1.76%，等权平均-1.70%，按已登记规则选择关闭方向。测试收益完整保留，具体P3动态通道和结构在配置实现时冻结，P2栅格仍缺。
冻结动态表示后进入P3四组遥感对照，随后实施P6及P7/P8。

```bash
sbatch slurm/run_p1_analysis.sh
```

1项CPU作业，1核、4 GB、30分钟时限。日志为 logs/p1_dates_JOBID.out/.err。
结果为 result/rs_hpc_v3/P1/analysis_JOBID/，验收 method.json 的 completed 状态、
48个配置及288条区间结果。3日循环日期区块，2000次，折内同一提前量的四个模型和两个指标共用日期索引。
作业912751已完成并验收。这是当前开发窗口的事后描述分析。

P1三窗口的3h空间RMSE改善分别为Ridge 5.03%/5.33%/0.90%，树模型3.42%/4.05%/0.87%。
F03 Ridge的MAE退步0.47%。全部48个开发配置的RMSE和MAE点估计均优于Persistence。
完整表和限制见 result/rs_hpc_v3/P1/P1_REVIEW.md。日期区间见analysis_912751/P1_DATE_REVIEW.md。
两种基线的3h空间增量在春夏窗口的RMSE区间为正，在秋季跨零。
秋季树模型A01相对验证选定强中心的收益为-0.13%，区间[-3.41%, 2.97%]。

```bash
sbatch slurm/run_p0_audit.sh
```

P0 一项 CPU 作业，1 核、4 GB、45 分钟时限。日志 logs/p0_JOBID.out 和 .err。验收要求 COMPLETED/0:0，result/rs_hpc_v3/P0/audit_JOBID/summary.json 标为 passed，320 行预测复核齐全，冬季分布和分层误差 CSV 可读。

P0 作业 912699 与 P1 先导 912700 均已验收。

```bash
sbatch --array=0-3%4 slurm/run_p1_baselines.sh
```

4 个数组任务覆盖 F01、3h、两种输入、两类模型，共 12 次候选拟合。每项 1 节点上的 1 个任务、4 CPU 核、12 GB、4 小时时限，最多并发 4。先导验收看四项 COMPLETED/0:0、每项 result.json 和三份归档产物，3h 点位样本为训练 42400、验证 22300、测试 22300，98 个唯一网格，10 条旧模型配对记录。日志 logs/p1_ARRAYID_TASKID.out 和 .err。新结果写入 runs/rs_hpc_v3/P1。提交同一索引时，匹配指纹及产物校验通过的任务跳过，未发布的部分尝试留在独立目录。

剩余44项已由数组912704完成，共132次候选拟合，均通过验收。

```bash
sbatch --array=4-47%4 slurm/run_p1_baselines.sh
```

最终验收为 48 个选定配置、144 次候选拟合（其中 96 次为落选候选）。每折每提前量以验证 RMSE 选中心基线。可以在登录节点汇总现有结果。

```bash
module --force purge
module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3
source wind-env/bin/activate
python summarize_p1.py --require_complete
```

## HPC 计时口径

P1 的进程单调时钟从作业分配内启动后记录。分别保存窗口与数据准备、验证选模总时间、每个候选 fit 时间、评估与输出时间及进程总耗时。总耗时还包含指纹核对、锁等待和就绪检查。这些字段用于定位开销。P6 的固定任务集合性能实验仍待实施，P1 的不同配置耗时不能用于估计并行加速。

性能阶段需控制 worker 核数、内存、节点型号、线程和固定任务集合，记录缓存状态。用分配就绪后的批次完成时间计算吞吐、加速比与效率；提交至开始的等待单列。资源成本以实际 AllocCPUS × 分配 Elapsed 计算，TotalCPU 单列。保留作业与步骤两层 sacct 输出；MaxRSS 按步骤和任务的统计口径解释。性能重复单独存放。 [Slurm sacct 字段说明](https://slurm.schedmd.com/sacct.html)

## 与前一版方案的对应

- 冬季 6h 的已有结果诊断归入 P0，作业 912699 已导出分布、分层误差和逐行核验。P1 的新基线只在 F01–F03 开发，统一比较 T01/A01。旧方案提到的树模型近邻 S01 留作可选扩展。P1b 首批历史长度对照改为开发窗口的 3h，冬季 6h 的历史长度试验仍是后续回顾性诊断。
- 等点数距离对照归入 P4。新增三个偏移先用三个种子，共 27 次。若形成五种子正式表，再补 18 次，总数仍为 45。所有距离采用同一批种子。
- 新年份确认归入 P7，仍需事先冻结。新年份内按时间重新拟合用于检验方法能否复现，直接使用冻结权重用于检验跨年迁移。P2/P3 遥感和 P6 HPC 是新增的研究工作。
