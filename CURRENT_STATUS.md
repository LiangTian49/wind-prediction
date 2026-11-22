# 每日实验看板

最近验收：2026-11-18，制图913096和论文v2已验收；历史36次、6h18次及联合日期分析913084均已验收。

## 今天需要知道

- 2026-11-21：下载未齐，仅2025年1/2/5月（3/24）成功；5超时、16被CDS队列限制拒绝。已修复串行/远端恢复/20h，先手动 `sbatch slurm/run_cerra_download_recover.sh`。不要重复%2或提交确认训练；见下载runbook最新记录。

- 2026-11-20下载先导913327_0已验收：2025年1月下载成功，COMPLETED/0:0、2分28秒，两变量各248时刻且an/step0/原网格header通过。下一步手动 `sbatch --array=1-23%2 slurm/run_cerra_download.sh`；目前仅1/24月完成，先不提交年度确认训练。

- CDS安装913326已验收：COMPLETED/0:0，已有库版本保持不变、cdsapi导入成功。下一步设置CDSAPI_KEY后手动 `sbatch --array=0 slurm/run_cerra_download.sh` 下载2025年1月先导；此前913325模块失败已修复，认证与实际下载仍待验证。

- 历史故障913325：模块查找失败、未开始安装；后续913326已成功验证修复。

- [CERRA 2023/2025月度下载](docs/runbooks/CERRA_CONFIRMATION_DOWNLOAD.md)入口已准备（24任务，先2025年1月）。wind-env缺cdsapi，先手动提交run_cerra_download_setup.sh；通过后设置CDSAPI_KEY并提交下载先导。5项合成检查通过，实际安装/下载均未执行，确认模型仍只用风速。

- 2026-11-19：[独立年份确认v1入口](docs/runbooks/INDEPENDENT_YEAR_CONFIRMATION_V1.md)已实现，7项合成检查及旧P3b预检通过。计划45次（先5、后40），训练需新GRIB→header→提取QA→冻结；目前本地仅2024源，新年真实工作均未执行，不直接提交训练。

- 已完成[20项核心文献评估](docs/paper/literature_review_2026-11-18/LITERATURE_REVIEW.md)：路线合理，创新应聚焦受控信息增量与条件性发现。独立年份和同信息RS树模型为建议优先项，尚未改动协议/配置或新增提交入口；v2稿保持不变。

- 联合日期分析913084成功验收：47秒、0:0，456配对区间、72增量差区间、9576逐日贡献及54/36行绝对结果齐全。
- 春季8步RS RMSE改善0.757% [0.271,1.187]；其RS增量比56步大0.531个百分点 [0.341,0.713]。夏季三个长度的RS收益均有利，但短长历史之间的增量差跨零。
- 秋季8步的相对RS增量比56步大，但主要伴随无RS参照变弱；8步RS自身收益仍跨零，不能说秋季问题解决。
- 6h秋季RS RMSE/MAE均退步；3h到6h的RS增量差三个窗口均跨零，无统一提前量规律。
- 本次按起报日期抽样，旧P3/P3b按目标日期，口径区别已明确记录，旧结果保留。

## 当前执行顺序

| 阶段 | 状态 | 当前操作 |
|---|---|---|
| 历史长度36项＋6h18项 | 全部验收 | 不重提 |
| 联合日期分析 | 913084全部验收 | 不重提 |
| 论文更新 | v2正文/图片/上传包已验收 | 作者审阅并补齐投稿信息 |
| 后续确认 | 年份/数据与新方案尚未冻结 | 先定主要问题，再核实未见时段；当前无新训练命令 |

## 实验结果摘要

| 实验 | 完成情况 | 当前结论 | 详细结果 |
|---|---|---|---|
| 固定四窗口LSTM | 320/320，含冬季80次 | A01的3h平均RMSE收益四窗口为2.68%–4.54%；秋季日期区间跨零，冬季相对Persistence优势不稳定 | [统一表](reports/seasonal/formal_results.csv)、[日期审查](reports/seasonal/DATE_REVIEW.md) |
| P1强基线 | 48个选定配置，144次候选拟合 | 3h空间收益在Ridge和树模型的春夏窗口为正，秋季空间增量区间跨零 | [统一表](reports/p1/unified_results.csv)、[审查](reports/p1/DATE_REVIEW.md) |
| P1c风向 | 12/12，三折、seed42，日期分析完成 | 空间方向测试RMSE收益为2.28%/0.16%/0.66%，春季日期区间为正、夏秋跨零；验证选择关闭方向 | [开发结果](reports/p1c/DEVELOPMENT_REVIEW.md)、[日期区间](reports/p1c/dates/date_intervals.csv) |
| P0b区域展示 | 已验收 | 98个唯一目标格、368份归档的逐格指标和固定案例齐全 | [地图](reports/roi/F01_maps.png)、[时序图](reports/roi/F01_series.png) |
| P2遥感 | 912843已验收，98格14维 | 全部qa_pass，最低有效覆盖率接近100% | [静态QA验收](reports/p2/STATIC_REVIEW.md) |
| P3/P6/P7/P8 | P3开发36/36已验收；其他待执行 | 遥感受控对照、HPC性能和新时间/空间验证继续按阶段推进 | [采纳后的计划](docs/experiments/WIND_FEEDBACK_PLAN_UPDATE_2026-11-11.md#11-用户采纳后的执行记录2026-11-11) |

旧V1/V2分别完成84/80次，与固定四窗口合计484次。P1的144次候选拟合包含48个选定模型。P1c为单种子开发结果，日期区间为窗口内事后描述，未做多重比较校正。

## 下一步与验收

论文v2已生成并通过验收：[main.tex](main.tex)、[PDF预览](docs/paper/manuscript_v2/main.pdf)、[完整上传包](docs/paper/manuscript_v2/upload_bundle.zip)、[上传说明](docs/paper/manuscript_v2/README_UPLOAD.md)。制图913096已完成，20幅不同图件全部插入正文/附录，另保留PNG预览；11张表，24页，两遍pdfLaTeX成功，无未解析引用/重复标签/溢出警告，抽查5页通过。

原稿备份在docs/paper/manuscript_archive/。原始制图输出和实验结果均保留，修正的工作流/架构图仅存入论文副本。结果已覆盖空间动态、遥感与坐标/来源控制、历史长度、提前量及日期区块分析，负结果保留。当前无需重复提交制图或训练。

下一步：作者审阅论文，补齐作者与通讯信息、贡献/基金/利益冲突声明、代码公开地址，再适配目标期刊模板。新年份、空间确认、同信息RS树模型及P6仍未执行；本次改稿没有把它们标为已完成。

## 深入查看

- [结果总目录](reports/README.md)包含GitHub可共享的小型表格与图件。
- [实验总记录](docs/experiments/EXPERIMENT_RECORD_AND_PLAN.md)说明方法、证据、限制和长期计划。
- [每日进展](docs/experiments/PROGRESS_LOG.md)按日期记录做了什么、作业编号及结果位置。
- [项目README](README.md)说明目录、环境、Slurm使用和GitHub更新。

维护本页时更新检查时间、最新作业、今天要知道的变化、待决事项和下一步。阶段结果更新后运行python scripts/export_reports.py同步可共享快照。
