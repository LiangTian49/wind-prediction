# P3遥感四组受控先导

2026-11-15更新：P3先导912852已4/4验收。遥感中心/空间RMSE增益为−0.0047%/+0.0531%，MAE均略退步。先导用于流程验收，尚无稳定遥感增益结论。保持冻结配置，下一步手动提交sbatch --array=4-35%4 slurm/run_p3.sh，剩余32次待执行。以下早期状态保留供追溯。

## 冻结配置与预算

P2作业912843已验收。数据为data/rs_static/features_v1，98格14维及100个point_id映射齐全。P3配置为configs/rs_hpc_v3_p3.json，固定P2 READY/产物摘要、P1c验证选择摘要和动态CSV摘要。

方向选择沿用三折验证规则的关闭结果。P3动态表示为原风速与小时sin/cos，中心3维、A01空间11维。静态向量仅描述目标中心网格。四组共同采用LSTM64、Dropout0.4、静态Dense8(relu)、拼接、Dense16(relu)、Dense1。无RS使用缩放后固定14维零向量。参数数目按动态条件一致，输入关闭约束影响有效自由度。

| 索引 | seed | 折 | 组 | 动态维数 | 静态信息 | 参数量 |
|---:|---:|---|---|---:|---|---:|
| 0 | 42 | F01 | RS_T0 | 3 | 关闭，14维零向量 | 18713 |
| 1 | 42 | F01 | RS_T1 | 3 | 开启，14维地表特征 | 18713 |
| 2 | 42 | F01 | RS_S0 | 11 | 关闭，14维零向量 | 20761 |
| 3 | 42 | F01 | RS_S1 | 11 | 开启，14维地表特征 | 20761 |

原T01/A01作为架构桥接，四个融合模型都新训练。3h直接预测，56步。Adam lr0.001、batch64、100轮上限、验证早停patience10、恢复最佳验证权重，MSE损失。

枚举顺序为seed、fold、group。预先登记seed42–46共60个位置，初轮只使用42/43/44的索引0–35，共36次。先导4次包含其中，先导通过后再补32次。五种子扩展的索引36–59共24次保留为可选后续，配置无需因补种子而变化。

## 先提交4次

从项目根目录执行。

```bash
sbatch --array=0-3%4 slurm/run_p3.sh
```

每项1节点、1任务、8工作线程、16GB、4小时，最多4项并行。dp-cn的实际分配核数和成本查sacct。日志为logs/p3_ARRAYID_TASKID.out/.err，结果为runs/rs_hpc_v3/P3/INDEX_FOLD_GROUP_seedSEED/。

当前没有提交或执行P3真实训练。4项轻量检查通过，覆盖原窗口管线等价、未来输入隔离、逐点Persistence、训练网格缩放、重复中心连接、无RS输入关闭、两种动态维数的同结构及保存加载。模型检查使用合成前向和评估，不做真实训练。

## 数据与评估控制

动态分割与滑窗复用train.py原函数。训练标签早于验证边界，验证标签早于测试边界，模型只用验证损失早停。动态MinMaxScaler只拟合训练段，静态StandardScaler只拟合唯一训练网格。保留100个point_id训练权重，测试按98格去重后计算pooled RMSE/MAE。

静态连接严格使用point_id到grid_id映射，窗口顺序由同一滑窗函数输出。三个分区都保存包含点位、起报、输入开始、目标、标签和Persistence的摘要，以核对成对实验的样本。

每项保存与同折同种子的原T01/A01，以及按验证RMSE选定的P1强中心的比较。P1强中心保持其固定拟合seed42，source_seed写入记录。主RS比较仍为RS_S1对RS_S0，补充RS_T1对RS_T0与两个空间比较。

## 先导验收

激活wind-env后运行。

```bash
python summarize_p3.py --expected pilot --require_complete
```

预期4/4，四项均COMPLETED/0:0。每项训练/验证/测试42400/22300/22300，98个唯一格、14静态维、正确动态维数与参数量。每项含model.keras、preprocessors.pkl、config.json、training_log.csv、predictions.csv、bridge_comparisons.csv及completion.json。实际参数、样本、标签及预处理来源摘要应可追溯。

汇总输出summary.csv、paired_results.csv、interaction_results.csv。主增益逐折逐种子报告，interaction记录空间与中心条件下RS误差下降量之差，正值表示空间条件下RS增量更大。单种子先导用于流程验收，后续保留全部预定组和窗口。

## 通过后再提交剩余初轮

```bash
sbatch --array=4-35%4 slurm/run_p3.sh
```

这32次补齐三种子三窗口的36次。结束后运行python summarize_p3.py --expected initial --require_complete。可选五种子补充36–59在后续评估阶段决定。

失败索引可单独重新提交。已完成结果摘要匹配则跳过，中断的部分目录保留。配置、来源或代码变化时使用新实验版本，保持原结果可追溯。完成后再准备日期区间、同信息树模型补充及新时间确认。

## 开发完成后的日期分析

2026-11-15验收36/36后，提交sbatch slurm/run_p3_analysis.sh。单项1线程、4GB、30分钟。结果在result/rs_hpc_v3/P3/analysis_JOBID/，预期192条区间、每日贡献、absolute_results.csv、method.json及completion.json。日期分析尚未实际运行。种子标准差与日期区间分开解释。
