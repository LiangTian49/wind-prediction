# Wind Prediction

使用CERRA历史风场研究ROI内选定网格的短期风速预测。已完成空间输入、强基线、风向、遥感地表特征、来源/坐标、历史长度与提前量开发对照；独立年份确认入口已实现，目前处于数据下载恢复阶段，尚未完成新年确认或HPC性能评估。实时状态以CURRENT_STATUS.md为准。

[每日看板与待决事项](CURRENT_STATUS.md) · [实验总记录](docs/experiments/EXPERIMENT_RECORD_AND_PLAN.md) · [每日进展](docs/experiments/PROGRESS_LOG.md) · [结果表与图件](reports/README.md)

## 项目结构

```text
wind_prediction/
├── CURRENT_STATUS.md        当前阶段、作业与验收入口
├── configs/                 日期折、冻结配置和实验清单
├── slurm/                   全部Slurm提交脚本
├── tests/                   合成数据和协议测试
├── docs/
│   ├── experiments/         实验记录、进展、任务书与反馈
│   ├── runbooks/            数据准备及作业操作说明
│   ├── paper/               论文初稿
│   ├── archive/             早期计划、笔记与图件
│   └── maintenance/         目录迁移记录和GitHub更新流程
├── reports/                 可提交GitHub的小型结果快照
├── scripts/                 维护工具，例如结果快照导出
├── environment/             当前wind-env依赖快照
├── legacy/                  旧版提取、检查及CERRA训练脚本
├── *.py                     当前实验Python模块
├── data/                    集群本地数据
├── result/                  完整分析结果与图件
├── runs/                    训练、模型、预处理器及预测
├── logs/                    Slurm输出和检查日志
├── model/                   早期模型检查点
└── wind-env/                集群本地Python环境
```

当前Python模块保留原路径和内容，归档产物中的代码摘要可以继续核验。根目录的run_ablation.sh、run_confirmation.sh是指向slurm/的兼容链接，供历史冻结清单及子脚本调用。ERA5历史主入口model_era5.py保留在根目录。

## 代码入口

| 任务 | Python模块 |
|---|---|
| 既有LSTM及时间协议 | train.py、temporal_folds.py |
| 空间数据提取与核验 | extract_cerra_spatial.py、validate_annual_data.py |
| 强基线与分析 | train_p1_baseline.py、summarize_p1.py、analyze_p1_dates.py |
| 风向提取与消融 | extract_cerra_direction.py、direction_pipeline.py、train_p1c.py |
| 风向汇总与日期区间 | summarize_p1c.py、analyze_p1c_dates.py |
| 遥感栅格QA与静态特征 | rs_gis.py、prepare_rs_static.py |
| 空间展示 | plot_spatial_setup.py、plot_p0b.py |
| 归档评估 | analyze_experiments.py、review_seasonal_results.py |

## 在JSC使用

从项目根目录操作，环境沿用已验证的模块组合。

```bash
source environment/activate_wind_env.sh
```

训练、全量数据提取、归档预测处理及性能实验通过用户手动提交Slurm。提交前查看CURRENT_STATUS.md，确认相应阶段尚未完成。脚本的规范路径现在带slurm/前缀，例如：

```bash
sbatch slurm/run_p1c_analysis.sh
```

新训练继续写入版本化runs/子目录，日志位于logs/。数据、模型、环境和完整运行产物由.gitignore排除。当前环境快照为environment/requirements-wind-env.txt，原requirements.txt保留历史依赖记录。

## 轻量检查

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1   python -m unittest discover -s tests -t .
```

这套检查使用合成样本和小型协议夹具。真实数据与模型作业按各阶段runbook验收。

## 更新GitHub

先将已完成的小型表格、报告和选定图件导出到reports/。

```bash
python scripts/export_reports.py
git status --short
# 具体暂存范围见下方完整流程；论文/合作者反馈可先保留本地。
git add --dry-run -A
git diff --cached --stat
# 选择并审阅暂存内容后，再commit和push。
```

reports/manifest.json记录每份快照的原始路径和SHA256。Git操作由用户执行，发布前可检查暂存内容。

[完整GitHub更新流程](docs/maintenance/GITHUB_WORKFLOW.md) · [实验操作说明](docs/runbooks/DIRECTION_AND_ROI_RUNBOOK.md)
