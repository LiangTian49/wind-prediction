# 全年空间数据与固定日期接口：准备及检查

本阶段准备数据和接口，不提交训练。现有夏季数据、V1/V2实验及默认行为保留。
固定日期变体使用 `--protocol v2_fixed`，沿用V2的因果窗口规则。

## 1. 当前应提交的作业

```bash
sbatch slurm/run_prepare_year.sh
```

该CPU作业申请1节点、4CPU、4小时，激活项目wind-env，执行：

1. 从现有GRIB提取2024年1–12月，同一100个中心点、5×5邻域及八个远处网格。
2. 检查全年的每个3小时时刻，包括闰日；共2928个时刻。
3. 重新生成的中心坐标、y/x必须与原summer centers.csv一致。
4. 原夏季CSV的全部73600条中心记录必须存在且数值一致；不能只匹配少量交集。
5. 检查所有点、所有特征表的日期键、缺失/非有限值、负风速及中心标签一致性。
6. 检查空间数组形状、时间/点位/相对偏移顺序和中心值。
7. 检查四个固定日期折的完整时间覆盖、理论样本数和标签边界；不做模型训练或冬季预测评价。

默认输出：

```text
data/CERRA/spatial_2024_full/
    center.csv                 # T01输入
    S01.csv / S02.csv / A01.csv
    centers.csv / grid_mapping.csv
    neighborhoods.npz         # (100, 2928, 5, 5)
    metadata.json
    READY.json                # 仅在全部就绪检查通过后生成

result/data_validation/annual_JOBID/
    readiness.json
    fold_coverage.csv
    fold_manifest.json

logs/year_data_JOBID.out
logs/year_data_JOBID.err
```

每个CSV应有292800行。100个point_id对应98个唯一中心网格。
只有日志出现 `[Readiness] PASS` 且READY.json状态为passed，数据才算准备完成。
READY.json记录日期清单SHA256及源文件大小/修改时间；后续文件或清单变更会使检查失效。

提取不会覆盖已有输出目录。如果提取已成功、只需重做就绪检查：

```bash
sbatch --export=ALL,VALIDATE_ONLY=1 slurm/run_prepare_year.sh
```

如需另一个输出位置，使用 `--export=ALL,ANNUAL_DIR=./data/CERRA/另一个目录`；
后续训练应显式设置对应的SPATIAL_DIR。所有路径须在项目内。

## 2. 固定日期清单

执行清单为 [configs/seasonal_folds_2024.json](../../configs/seasonal_folds_2024.json)。
它与原proposed_folds.json的日期一致，增加协议、训练参数和数据目录定义。
清单内容应在运行前固定；修改后需重新做就绪检查，不能将不同版本直接合并分析。

| fold_id | 训练起点 | 验证起报边界 | 测试起报边界 | 测试标签截止（不含） | 角色 |
|---|---|---|---|---|---|
| F01 | 2024-01-04 | 2024-03-04 | 2024-04-01 | 2024-04-29 | 开发 |
| F02 | 2024-04-04 | 2024-06-03 | 2024-07-01 | 2024-07-29 | 开发 |
| F03 | 2024-07-05 | 2024-09-03 | 2024-10-01 | 2024-10-29 | 开发 |
| F04 | 2024-09-04 | 2024-11-03 | 2024-12-01 | 2024-12-29 | 锁定确认 |

均为UTC 00:00，左闭右开。各折60天训练、28天验证、28天测试。
每折单独fit scaler，仅使用train_start至validation_start之前的行。

### 窗口与标签规则

- 训练：上下文不早于train_start，target<validation_start。
- 验证：origin>=validation_start，target<test_start，可使用之前的训练历史。
- 测试：origin>=test_start，target<test_end_exclusive，可使用之前已观测历史。
- 所有输入时刻<=origin；不借用train_start以前的数据，不跨point_id。
- 每个点在整折范围内都必须具有完整的3小时序列，缺失/重复会报错，不静默丢点或补值。

对于100个point_id、56步窗口，各折的预期样本数相同：

| 提前量 | 训练 | 验证 | 测试 |
|---|---:|---:|---:|
| 3小时 | 42400 | 22300 | 22300 |
| 6小时 | 42300 | 22200 | 22200 |
| 12小时 | 42100 | 22000 | 22000 |
| 24小时 | 41700 | 21600 | 21600 |

这些是窗口数，不是独立天气事件数。冬季数据覆盖检查不会生成预测或误差分数。

## 3. 训练接口（数据通过后使用）

train.py增加：

```text
--protocol v2_fixed
--fold_manifest configs/seasonal_folds_2024.json
--fold_id F01
```

也支持自定义明确日期，不能与manifest同时传入：

```text
--protocol v2_fixed --fold_id CUSTOM01
--train_start "2024-01-04 00:00:00"
--validation_start "2024-03-04 00:00:00"
--test_start "2024-04-01 00:00:00"
--test_end "2024-04-29 00:00:00"
```

上述是参数说明，真实训练仍必须放在Slurm批处理脚本中。
config.json保存fold_spec、清单路径/哈希、实际起报/目标范围，并逐字节保存fold_manifest.json快照；results.csv与predictions.csv保存fold_id。
固定日期方案不再将70/10比例写成实际划分比例，相关字段为空。
模型、特征及原超参数不变；原 `--protocol v1/v2` 及其旧模型重放仍兼容。
F04清单角色为locked_confirmation，默认训练会拒绝；方法冻结后须显式设置
`--allow_confirmation`（批处理环境为ALLOW_CONFIRMATION=1）才能启动。

## 4. 第一批48次训练已经具备提交入口，但当前先不提交

数据就绪检查通过后，下一阶段可运行：

```bash
sbatch slurm/run_folds.sh
```

数组0–11、最多并行4任务：

- 0–3：F01的T01/S01/S02/A01；
- 4–7：F02的T01/S01/S02/A01；
- 8–11：F03的T01/S01/S02/A01。

每任务仅seed=42，运行四提前量，共48次。不会触及F04。
输出位于runs/seasonal/check_seed42/FOLD_ID/EXPERIMENT_ARRAYID_TASKID/。
日志位于logs/folds_ARRAYID_TASKID.out/.err。
任务会先核对READY.json，缺失或数据/清单发生变化时停止。

## 5. 按折分析接口

第一批结束后，可用：

```bash
sbatch slurm/run_analysis.sh \
  --protocol v2_fixed \
  --runs_root ./runs/seasonal/check_seed42 \
  --centers ./data/CERRA/spatial_2024_full/centers.csv \
  --fold_manifest ./configs/seasonal_folds_2024.json
```

所有表格增加fold_id；配对按fold_id、seed、提前量进行，不跨折匹配。
相同实验/提前量/种子在不同fold合法，同一fold重复则拒绝。
每折单独出图，coverage.json标出尚未补齐的种子，F04没有结果时标为reserved_confirmation。
不同清单版本不混合统计。自定义日期且未使用manifest的运行可从保存的fold_spec重建。

本阶段尚未自动提交任何作业。先由用户手动提交全年准备作业，检查通过后再进入48次训练。
