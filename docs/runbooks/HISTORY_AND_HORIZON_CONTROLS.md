# 先历史长度，再预测提前量：执行与验收

最新验收：联合日期分析913084已验收：47秒、COMPLETED/0:0，456配对区间/72增量差区间/9576逐日贡献及所有产物哈希通过。春季8步RS收益及相对56步增量差均为正；秋季8步相对收益增加部分源于无RS参照变弱，本身收益仍不明确。6h秋季RS两个指标退步，3h到6h增量差三窗口均跨零。历史与提前量有限开发已完成，无需补训练，论文补充已生成，新年份确认尚未冻结/执行。

最新验收：6h全部18/18已验收（913066/913068），共同起报的3h/6h RS RMSE增量春0.218%/0.721%、夏0.911%/0.503%、秋−0.845%/−1.589%。种子波动较大，日期区间尚未计算。已准备run_temporal_analysis.sh，同时分析历史与提前量，以及RS增量之差；3项合成检查通过，真实分析尚未执行。

最新验收：6h先导913066已2/2验收，全部COMPLETED/0:0，耗时约4分钟。42300/22200/22200样本覆盖、参数量、产物摘要和共同起报的3h参照重算检查通过，combined_results4行、paired_results2行。可补交索引2–17共16项；当前剩余任务未核实已提交或完成。先导为单折单seed，不用于改组选长度。

最新验收：2026-11-18历史对照已36/36验收（912974/912978），54行合并结果、63行配对结果齐全；6h阶段门槛检查通过。春季短历史RS增量较大，夏季非此趋势，秋季仍不稳定；尚无日期区间，不据此改选第二阶段长度。现在可按原方案提交固定56步6h先导0–1，两项；6h真实训练尚未执行。

2026-11-18验收历史先导912974：4/4完成，全部COMPLETED/0:0，耗时1分44秒至2分38秒。样本锚点、scaler、输入结构、产物哈希与旧参照检查通过，combined_results6行、paired_results7行。可以手动补交索引4–35共32次；6h仍需等待历史36次完成验收。

依据用户决定实施。当前为2024已使用窗口上的开发对照，新年份确认延后而非取消。旧P3/P3b及所有结果保留；没有修改冻结的数据加载、静态接口或已有训练入口。

## 1. 预算及不根据测试结果改配置的原则

| 顺序 | 对照 | 固定条件 | 新拟合 | 旧结果复用 |
|---|---|---|---:|---:|
| 第一阶段 | 历史长度8/24/56步 | A01、3h、F01–F03、seed42–44、无/有完整RS | 36（8/24步） | 原56步18份 |
| 第二阶段 | 3h/6h提前量 | A01、56步、F01–F03、seed42–44、无/有完整RS | 18（6h） | 原3h18份，但测试指标在共同起报子集上重算 |

新增合计54次，不同时启动两个阶段，不扩到全部模型、所有来源或24h任务。第二阶段固定56步，避免在当前测试结果上挑一个“最佳历史长度”后把后续结果误称确认。若用户未来决定更换这一设置，必须新版本登记预算和参照。

8/24/56指观测数量，首末观测时间跨度分别21/69/165小时；最新一条均在forecast_origin。研究问题是历史信息量变化是否改变RS的增量，不预设短历史会让RS有效。

## 2. 第一阶段如何严格匹配样本

先调用原build_partitions按56步、3h生成所有train/validation/test窗口。8/24步仅取每个已有窗口的最后8/24条输入，不增加更早起报，也不多收训练样本。动态scaler仍fit同一训练段，静态scaler fit同一组唯一训练网格。

因此三个长度均有训练42400、验证22300、测试22300个原始point窗口，标签、起报、目标、Persistence、点位权重完全一致。训练前核对原56步三个分区的完整样本签名及scaler；真实输入input_start按8/24步重新记录，另存anchor_input_start描述共同资格锚点，不能把短历史实际输入误写成56步。

无RS静态14维在缩放后置零；有RS保留原14维。动态每步11维、LSTM64/Dropout0.4/静态Dense8/拼接/Dense16/Dense1，三个长度参数量都是20761。所有组沿用Adam lr0.001、batch64、epochs100、验证loss早停patience10/恢复最佳权重。目标仍为CERRA网格10m风速。

## 3. 现在先提交历史长度4项先导

```bash
sbatch --array=0-3%4 slurm/run_history_controls.sh
```

| 索引 | fold | seed | input_len | 静态信息 |
|---:|---|---:|---:|---|
| 0 | F01 | 42 | 8 | 无RS |
| 1 | F01 | 42 | 8 | 完整RS |
| 2 | F01 | 42 | 24 | 无RS |
| 3 | F01 | 42 | 24 | 完整RS |

每项CPU1节点、1任务、8工作线程、16GB、4小时，最多4项并行。模块Stages/2025 GCCcore/.13.3.0 Python/3.12.3与项目wind-env保持已验证环境。实际AllocCPUS可能48，成本按实际分配核算。

日志logs/history_controls_ARRAYID_TASKID.out/.err（SBATCH设置为项目内绝对路径）。结果runs/rs_hpc_v3/P1b_history/v1/INDEX_FOLD_LLEN_GROUP_seedSEED/。

完成后激活wind-env：

```bash
python summarize_temporal_controls.py --stage history --expected pilot --require_complete
```

预期4/4，新拟合4行＋同折同seed旧参照2行＝combined_results6行。确认sacct全部COMPLETED/0:0；每项参数20761、动态shape为8×11或24×11、静态14、98唯一格、42400/22300/22300，model.keras/preprocessors.pkl/预测/训练日志/配置/result/参照清单/completion齐全。训练前必须通过anchor样本签名、scaler与逐点Persistence匹配。不以正收益作为先导验收条件。

先导通过后再提交剩余32项：

```bash
sbatch --array=4-35%4 slurm/run_history_controls.sh
```

全部结束：

```bash
python summarize_temporal_controls.py --stage history --expected full --require_complete
```

预期36/36新增＋18份复用＝combined_results54行，paired_results63行。其中27行是在每个历史长度上的RS增量，36行是8/24步分别相对本组56步的误差变化。不要将旧18份当作新训练成本。

## 4. 历史阶段验收后再开始6h

6h执行入口会验证历史36项全部完成、产物及科学版本一致；若尚未完成则在训练前拒绝运行。这个门槛用于执行顺序，不是要求历史实验出现正收益。

先提交2项先导，F01/seed42的无/有RS：

```bash
sbatch --array=0-1%2 slurm/run_horizon_controls.sh
```

验收：

```bash
python summarize_temporal_controls.py --stage horizon --expected pilot --require_complete
```

预期2/2，combined_results4行。通过后再补16项：

```bash
sbatch --array=2-17%4 slurm/run_horizon_controls.sh
```

全量验收：

```bash
python summarize_temporal_controls.py --stage horizon --expected full --require_complete
```

预期18/18新增、combined_results36行、paired_results18行（两提前量×三折×三seed的RS增量）。日志logs/horizon_controls_ARRAYID_TASKID.out/.err，结果runs/rs_hpc_v3/P3c_horizon/v1/。

### 6h的样本与解释

沿用原56步/3h锚点，同点下将标签与target_time向后移一条3h采样，删除每个分区最后一个不再满足边界的起报。合成测试已验证这种构造与原56步/horizon=2滑窗完全相等，没有改动核心数据加载。

6h训练/验证/测试分别42300/22200/22200（比3h各少100），标签严格早于下一边界。这一训练数量差异由提前量与标签隔离共同决定，不能声称两个提前量训练样本也完全相同。

评价时从原3h预测中取与6h相同的point_id/forecast_origin子集，重新按98格去重计算3h RMSE/MAE/Persistence指标，不修改旧预测或旧result。两个任务起报相同，但3h和6h目标时刻/标签必然不同；不能强行匹配目标后混称同一起报。相同RS开关下模型误差跨提前量变化主要用于描述任务难度，主要问题是各提前量内部RS_S1相对RS_S0的配对增量。

保存matched_3h_references，包括相同起报下的原3h指标、样本数和起报/Persistence摘要。汇总表的validation_RMSE/MAE为各模型原生验证集合，3h验证未重匹配；它们只描述原生早停，不用来声称相同验证集合下的提前量比较。train_raw_samples/validation_raw_samples/test_raw_samples单独记录覆盖差异。

## 5. 新文件与版本边界

- 通用入口train_temporal_controls.py、窗口/参照契约temporal_controls.py、汇总summarize_temporal_controls.py。
- configs/rs_hpc_v3_history_v1.json、configs/rs_hpc_v3_horizon_v1.json。
- slurm/run_history_controls.sh、slurm/run_horizon_controls.sh。
- 旧18份参照仍使用configs/rs_hpc_v3_p3b_v1_references.json；新配置/代码指纹与旧指纹分开追溯。

保存每次模型、scaler、输入列序、静态开关、真实输入长度、锚点签名、预测、最佳验证轮次和分段耗时。已有run指纹一致则验证后跳过，不一致拒绝覆盖，保留中断尝试。真实训练/性能/全量处理必须手动sbatch。

已完成7项轻量合成/模型前向与保存加载检查：所有长度的全分区样本匹配、实际输入开始、逐点Persistence、未来输入隔离、6h与原生滑窗等价、共同起报/不同标签匹配与错误拒绝、任务清单、6h顺序门槛、短历史模型参数量和保存加载。日志logs/temporal_controls_tests.out。没有真实拟合或性能测试。

日期区块分析待训练验收后另行准备，当前seed标准差不替代日期区间。两阶段均属于已使用2024窗口上的开发分析，不把其最佳组合当作独立确认。新年份草案保留但执行优先级延后；不能删除秋季、差的seed或不利历史长度。

## 6. 两阶段训练完成后的联合日期分析

历史36与6h18项均已验收。提交sbatch slurm/run_temporal_analysis.sh；单项1工作线程、8GB、30分钟，日志logs/temporal_dates_JOBID.out/.err。输出result/rs_hpc_v3/temporal_controls/analysis_JOBID/，预期date_intervals.csv 456行、RS_gain_interactions.csv 72行、daily_loss_contributions.csv 9576行、history_absolute_results.csv 54行、horizon_absolute_results.csv 36行，以及method.json/completion.json。

本版本按forecast_origin日期使用三日循环区块、2000次，在同折内所有模型和seed共享抽样。旧P3/P3b分析按target日期，本次改变用于共同起报下比较不同lead，旧结果保留。报告固定seed各自和均值，不平均预测或重采样seed，日期区间不含训练随机性且没有多重校正。

RS_gain_interactions保存两个RS百分比收益之差，单位是百分点：8/24步相对56步、6h相对3h。通过共享日期抽样直接相减求区间，不能由两个独立区间是否相交替代。本次分析已通过3项合成测试、Python编译及Slurm语法，真实分析尚未执行。
