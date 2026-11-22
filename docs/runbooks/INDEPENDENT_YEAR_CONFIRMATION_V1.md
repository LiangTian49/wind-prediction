# 独立年份确认 v1：数据、冻结、45次拟合与分析

实现日期：2026-11-19。状态：入口与合成检查已准备；本地仅发现旧2024 GRIB，真实新年份header审计、提取、冻结、训练、确认分析均未执行。本文替代旧runbook中“尚未实现入口”的操作说明，历史记录保留。

## 1. 这轮要确认什么

主要问题：在尚未用于模型误差分析的另一个年份，按相同规则重新训练后，完整遥感地表描述相对无RS的收益能否复现。

固定为56个三小时观测、3h直接预测、A01共11动态通道、98个唯一目标格/100个原point_id、同一14维静态表和相同融合架构。方向关闭，不新增气压/温度，不根据新年结果选L8或6h。

五组为RS_S0（无RS）、RS_S1（完整RS）、S_COORD（坐标）、S_LC（仅土地覆盖）、S_DSM（仅DSM）。三窗口×五组×三种子=45次新拟合。Persistence不需要拟合。这里只确认新年份重新拟合，未实现冻结2024权重迁移，不能混称跨年迁移性能。

默认优先2025；仅按可用性采用2023备用。若采用2023，应称未见回溯年份复现，而非向未来年份迁移。在新年份上看到负收益后，不得切换年份、长度、窗口或组别并仍声称原确认成功。

## 2. 新增文件

- `confirmation_protocol.py`：固定日期、任务表、产品、冻结与样本/静态/区块统计契约。
- `prepare_confirmation_year.py`：init、Slurm-only extract/数值QA、freeze。
- `train_confirmation_year.py`：复用既有动态加载、时间序列、预处理和融合定义，独立输出新模型。
- `analyze_confirmation_year.py`：完成量、身份、三分区样本签名及缩放器一致性验收，完整确认分析。
- `slurm/run_confirmation_extract.sh`、`run_confirmation_year.sh`、`run_confirmation_analysis.sh`。
- `tests/test_confirmation_year.py`：小型合成检查，无模型拟合、真实GRIB扫描或性能测试。

原始训练、提取器、P3/P3b、2024配置、旧P7草案和结果均未改。新入口不借用旧模型作为新年对照。

## 3. 先准备新年份原始数据

下载到项目内，例如 `data/CERRA/raw_2025/`，可按月多个GRIB。保留请求、回执与产品说明。输入要求：CERRA high-resolution、10m wind speed、analysis、step=0、instant、m/s，三小时00/03/06/09/12/15/18/21 UTC，全年。

不要将gust、ensemble、forecast或不同初始化的同一valid_time混入。维持完整原生Lambert网格与扫描方向；裁剪/重网格版本会被拒绝，因为旧整数格索引不再等价。文件不应重叠；若重复下载，明确选择唯一来源，不能静默去重。

2025和2023各应有2920个三小时时刻；2024的2928不能直接套用。训练与验证也需要数据，不能只下载4/7/10月。现已新增[CDS月度下载与依赖准备入口](CERRA_CONFIRMATION_DOWNLOAD.md)，尚未实际执行，全年可用性需按实际下载/header结果判断。

## 4. 初始化新版本和header审计

先在项目根目录启用工作环境（解决libpython共享库路径）：

```bash
module --force purge
module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3
source wind-env/bin/activate
```

**以下命令要求新GRIB已经存在**；当前目录尚无这些文件。以实际选定的2025文件为例：

```bash
python prepare_confirmation_year.py init \
  --year 2025 --version confirmation_v1 \
  --year-basis '2025 selected by complete compatible data availability, before error analysis' \
  --grib data/CERRA/raw_2025/*.grib

sbatch slurm/run_newyear_audit.sh --grib data/CERRA/raw_2025/*.grib
```

init不读风场数组，只绑定小型参考文件/配置并建立 `configs/confirmation_v1/request.json` 与 `folds.json`，此时仍未冻结。已存在的版本不覆盖。如果实际文件列表改变，先另建版本并重新审计。

header审计沿用已有脚本，结果为 `result/newyear_audit/headers_JOBID/inventory.json` 与completion。要求选定年的headers_complete=true、无重复/缺测/偏离三小时整点，且incompatible_wind_messages=0。显式文件列表必须与init完全相同。

若2025确实不可用，首次初始化可选择`--year 2023`和实际raw_2023文件；其余流程相同，data/runs/result路径由年份生成。不自动替你声明年份未见。

## 5. 独立年份提取与数值QA

用真实header作业号替换下列 `HEADER_JOBID`：

```bash
sbatch slurm/run_confirmation_extract.sh \
  --request configs/confirmation_v1/request.json \
  --audit result/newyear_audit/headers_HEADER_JOBID
```

此作业同时完成提取和QA：

- 检查审计文件列表/文件stat，计算原始GRIB SHA256；按消息流解码，不加载全年完整风场立方体。
- 逐条核对产品、网格、投影、扫描；复用既有centers和grid_mapping，核对全部映射单元真实坐标与边界。
- 只写本轮需要的A01，不重复生成S01/S02/大邻域张量。列顺序与既有A01一致；100点、98唯一中心、全年2920时刻、292000行。
- 检查NaN/Inf/负值、GRIB缺失值和明显不合理风值（>200m/s会报错，不截断或删除），重复/缺测会硬失败。
- 首/中/末三个时刻用ecCodes逐点访问API与行优先解码采样交叉核对，再检查CSV往返数值及键。
- 不把2024风速数值作为跨年相等条件，不读取模型分数。

输出默认 `data/CERRA/confirmation_v1_2025/`：A01.csv、centers.csv、grid_mapping.csv、READY.json、fold_coverage.csv、completion.json。数据与文件指纹独立版本化。失败的临时目录保留，不伪装成READY；同版本重试保留历史尝试。

验收看READY的status=passed、year、2920时刻、100点/98格、三个独立点核对，fold_coverage应均为42400训练、22300验证、22300测试。原始值表不代表模型测试结果，允许此阶段检查质量。

## 6. 在新年测试误差揭示前冻结

先确认选定年份确实未被你或其他分析用于模型误差检查/调参，再运行：

```bash
python prepare_confirmation_year.py freeze \
  --request configs/confirmation_v1/request.json --attest-unseen

python train_confirmation_year.py --request configs/confirmation_v1/request.json --preflight
python train_confirmation_year.py --request configs/confirmation_v1/request.json --list
```

`--attest-unseen`是操作者对科学事实的声明，程序无法判断项目外是否分析过该年。若不成立，不应传此参数来把已见数据包装成独立确认。

freeze绑定request/fold/原始源与提取结果、静态数据、代码、Slurm脚本、软件版本及UTC时间，生成独立 `configs/confirmation_v1/FROZEN.json`，不覆盖旧P7草案。其静态解释为回溯地表描述：WorldCover 2021 v200于2022发布，2023/2025可用同一版本；不声称地表没有变化或已验证业务时效。旧地表数据的metadata仍保留2024开发来源身份。

训练拒绝缺少FROZEN、request/代码/环境/文件变动。预检仅检查小型metadata、静态表与文件stat，真实训练作业再完整核对提取产物哈希。冻后不得修改科学代码；如修复问题必须登记新版本与测试数据是否已揭示，不能直接重写FROZEN。

## 7. 分批拟合：先5次，再40次

冻结预检通过之后：

```bash
sbatch --array=0-4%3 slurm/run_confirmation_year.sh
```

任务0–4是C01、seed42的全部五组，分别为RS_S0、RS_S1、S_COORD、S_LC、S_DSM。先导用于确认运行、数据连接、边界和产物完整性；不能根据新年收益决定是否保留后续组/窗口。

五项全部结束后提交验收：

```bash
sbatch slurm/run_confirmation_analysis.sh --expected pilot --summarize-only
```

要求5/5，参数20761、动态11维/静态14维、样本计数及签名一致、静态/动态/坐标scaler一致、98格齐全，无NaN，模型和预处理文件可校验。会存下绝对结果供追溯，但不生成先导汇总确认结论。

运行完整性验收通过后：

```bash
sbatch --array=5-44%4 slurm/run_confirmation_year.sh
```

共40次剩余拟合。索引5–9为C02、seed42；10–14为C03、seed42；15–29为seed43；30–44为seed44。每次保存model.keras、preprocessors.pkl、training_log.csv、config/result、预测、冻结副本及completion。

只可按原任务号重试基础设施失败，不删差的种子、窗口和日期；同版本完整产物会校验后跳过，不覆盖重训。

## 8. 全部45次结束后做一次正式确认分析

```bash
sbatch slurm/run_confirmation_analysis.sh --expected full
```

分析先完整核对45个模型的身份、文件哈希、样本签名、缩放器和误差重建，再计算：

- 每窗口、组、seed的RMSE/MAE及种子均值/样本SD。
- 主比较：RS_S1相对RS_S0的RMSE，先逐seed算收益，再窗口内三seed平均，最后三个窗口等权平均；不把全部预测混在一起算一个比值，也不只挑夏季。
- 关键次要比较RS_S1相对S_COORD；保留原七类组间对比及各组相对Persistence。
- 起报UTC日期、三日循环区块、2000次；同窗口模型/seed/指标共用抽样，窗口间独立抽样后等权合成。不同于早期P3/P3b目标日期口径，已预先写入本协议。
- 312行日期区间（含24行跨窗口等权结果）与6048行逐日损失贡献；固定seed均值、无预测集成、不重采样seed。次要区间未作多重比较校正。

输出 `result/confirmation_v1_2025/full_JOBID/`，主要文件为 `CONFIRMATION_REVIEW.json`、`absolute_results.csv`、`absolute_seed_summary.csv`、`date_intervals.csv`、`paired_seed_summary.csv`、`daily_loss_contributions.csv`、scope/completion。

主收益区间全正是支持这项预声明确认的证据；区间跨零或负值也必须作为本次结果报告。程序不会把负结果标成作业失败，也不会据此换长度/年份。区间仍只覆盖所选新年窗口和固定已拟合模型，不等于独立观测、全气候、业务或新地点验证。

## 9. 资源与日志

| 阶段 | 作业/拟合量 | 每项申请 | 日志 |
|---|---|---|---|
| header审计 | 1项 | dp-cn，1节点/1核，4GB，1h | logs/newyear_audit_JOBID.out/.err |
| 提取+数值QA | 1项 | dp-cn，1节点/1核，8GB，4h | logs/confirmation_extract_JOBID.out/.err |
| 先导 | 5次拟合，最多3项并行 | 每项dp-cn，1节点/8核，16GB，4h | logs/confirmation_fit_ARRAYID_TASK.out/.err |
| 先导验收 | 1项 | dp-cn，1节点/1核，8GB，1h | logs/confirmation_analysis_JOBID.out/.err |
| 剩余拟合 | 40次，最多4项并行 | 同先导；最多32核/64GB并行申请 | 同上fit日志 |
| 全量确认分析 | 1项 | dp-cn，1节点/1核，8GB，1h | 同上analysis日志 |

时间为申请上限，不是实测耗时。v1固定CPU与wind-env，沿用当前已验证的Stages/2025+Python3.12.3模块组合，避免libpython缺失；没有安装新库。日志SBATCH路径均为项目logs绝对路径，logs目录需提交前存在。

timing记录区分校验、数据准备、fit（含每轮验证与callback）、完整test predict与总进程内wall；不含排队/module启动。单次model.predict包含框架开销，不能称为预热后的内核延迟，不能用这轮结果宣称CPU/GPU加速比。用sacct另外保留作业Elapsed、AllocCPUS、MaxRSS与退出状态。

## 10. 现在实际应该做的事

当前尚无新GRIB，不要直接提交45次训练，也不必重审2024来冒充新年核实。先把所选年份完整匹配产品放进项目，再按init→header→extract/QA→freeze/preflight→5项→验收→40项→正式分析顺序推进。

本次只实现并做轻量合成检查。下载、真实GRIB审计/提取、新年冻结、所有拟合和正式确认分析都仍待执行；是否新年未见由操作者如实声明。
