# 可共享结果快照

下载进度：[月度状态](downloads/month_status.csv)、[下载实现/验收状态](downloads/status.json)；独立年份确认：[实现状态](confirmation/implementation_status.json)。这些记录实现与验收进度，不代表下载或新年实验已全部完成。

| 内容 | 入口 |
|---|---|
| 四窗口正式结果 | [formal_results.csv](seasonal/formal_results.csv)、[审查报告](seasonal/REVIEW.md) |
| 旧方案日期区间 | [DATE_REVIEW](seasonal/DATE_REVIEW.md)、[区间表](seasonal/date_intervals.csv) |
| P1强基线 | [结果表](p1/baseline_results.csv)、[审查报告](p1/REVIEW.md) |
| P1日期区间 | [统一表](p1/unified_results.csv)、[DATE_REVIEW](p1/DATE_REVIEW.md) |
| P1c风向开发 | [结果表](p1c/development_results.csv)、[审查报告](p1c/DEVELOPMENT_REVIEW.md) |
| P1c已有日期产物 | [日期区间](p1c/dates/date_intervals.csv)、[分析报告](p1c/dates/REVIEW.md) |
| P2数据可用性 | [DATA_AVAILABILITY](p2/DATA_AVAILABILITY.md) |
| P2静态特征验收 | [STATIC_REVIEW](p2/STATIC_REVIEW.md)、[质量汇总](p2/qa_summary.json)、[14维表](p2/static_features.csv)、[来源与定义](p2/feature_metadata.json) |
| 历史/提前量日期分析 | [结论](temporal/DATE_REVIEW.md)、[配对区间](temporal/date_intervals.csv)、[增量差](temporal/RS_gain_interactions.csv) |
| 6h提前量对照 | [开发审查](horizon/DEVELOPMENT_REVIEW.md)、[结果](horizon/combined_results.csv)、[配对](horizon/paired_results.csv) |
| 历史长度对照 | [开发审查](history/DEVELOPMENT_REVIEW.md)、[结果](history/combined_results.csv)、[配对](history/paired_results.csv) |
| P3b日期审查 | [结论](p3b/DATE_REVIEW.md)、[区间](p3b/dates/date_intervals.csv)、[日期集中度](p3b/date_concentration.csv) |
| P3b坐标/来源对照 | [开发审查](p3b/DEVELOPMENT_REVIEW.md)、[绝对结果](p3b/combined_results.csv)、[配对结果](p3b/paired_results.csv) |
| P3日期分析 | [审查](p3/DATE_REVIEW.md)、[日期区间](p3/dates/date_intervals.csv)、[集中度](p3/date_concentration.csv) |
| P3先导 | [验收](p3/PILOT_REVIEW.md)、[结果](p3/summary.csv)、[配对](p3/paired_results.csv) |
| 区域图件 | [地图](roi/F01_maps.png)、[时序图](roi/F01_series.png)、[架构](roi/lstm_architecture.svg) |

这些文件从完整分析目录复制，包含预定窗口的正负结果。来源、字节数和SHA256记录在manifest.json。更新时从项目根目录运行python scripts/export_reports.py。导出工具只复制已存在的小型产物，每份最大2 MiB，完整模型和逐样本预测保存在集群。

P1c日期产物在存在有效completion.json时纳入快照，具体解释仍以实验阶段验收为准。
