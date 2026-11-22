# CERRA 2023/2025：按月下载并衔接独立年份确认

2026-11-21最新验收：24项任务结束不等于下载完成。仅2025-01/02/05成功（3/24）；数组913328中2项成功、5项TIMEOUT、16项因CDS最大排队请求限制失败。已有文件保留。现已修复为全项目串行锁、保存/恢复远端request_id、20小时上限（sinfo显示dp-cn最大20h）。旧超时任务在CDS可能仍排队或已完成，需要先只读清点恢复：

```bash
sbatch slurm/run_cerra_download_recover.sh
```

需要提交shell仍设置CDSAPI_KEY。该作业1核/4GB/1h，不提交新请求、不取消远端任务、不解码风场；输出logs/cerra_recovery_JOBID.out/.err及result/cerra_download/recovery_JOBID/recovery.json。精确匹配历史请求后为缺失月份保存.remote.json；未知/重复活动请求会报错并要求查看清单。

恢复清点通过后，先验证一个缺失月份的恢复/下载：

```bash
sbatch --array=2%1 slurm/run_cerra_download.sh
```

任务2成功后补交其余20项：

```bash
sbatch --array=3,5-23%1 slurm/run_cerra_download.sh
```

不再使用%2。原远端请求若已过期或状态异常，保留ID并停止供检查，不盲目重新请求；恢复接口已做mock测试，真实服务调用仍待用户sbatch验证。2025全年完整前不初始化确认/年度提取。

2026-11-20最新验收：下载先导913327_0成功，2025-01两变量各248时刻，header通过，2分28秒；可补交1–23%2。当前只有1/24月完成，年度完整性和数值QA仍待执行。

最新验收：安装913326成功，COMPLETED/0:0，原有包版本保持不变，cdsapi导入通过。无需重复安装，直接按第2–3步设置token和提交下载先导。真实认证/下载仍未执行。

2026-11-19。已实现下载及依赖安装入口；未实际安装依赖、读取密钥、发送CDS请求或下载数据。

## 请求修正

使用用户提供的数据集、年份、风速/风向、三小时时刻和GRIB格式。为匹配旧实验及确认协议，明确加入：

```python
"level_type": "surface_or_atmosphere",
"product_type": "analysis",
```

移除`leadtime_hour=["3"]`。三小时时间采样由time列表指定；leadtime是预报产品维度，不是取样频率。本项目要求实际下载的GRIB为an、stepRange=0。`data_type=reanalysis`本身不足以明确选择analysis，因为数据目录同时提供analysis和forecast。

依据：[官方API过程字段说明](https://cds.climate.copernicus.eu/api/retrieve/v1/processes/reanalysis-cerra-single-levels)、[官方CERRA用户指南](https://confluence.ecmwf.int/spaces/CKB/pages/272323672/Copernicus%2BEuropean%2BRegional%2BReAnalysis%2BCERRA%2Bproduct%2Buser%2Bguide)。当前字段列表含2023和2025，但不代表每个月都已成功取到数据；须以实际请求和header验收为准。

每月自动生成合法日历天数；完整原生网格，不加area/grid裁剪或重采样。两变量均保留下载；现有确认训练依然只取风速，不能把下载风向等同于模型已引入风向。

## 1. 先补齐wind-env缺少的下载依赖

安装作业913325验收：FAILED/1:0，22秒，失败在Stages/2025模块查找，安装程序未启动。已统一使用environment/activate_wind_env.sh显式恢复stage搜索路径；移除登录环境中的该路径后，本地恢复测试通过。请重新提交安装；尚未在计算节点验证修复。

轻量检查发现wind-env没有cdsapi、multiurl及ecmwf客户端。本次没有直接改变环境，提供独立安装作业：

```bash
sbatch slurm/run_cerra_download_setup.sh
```

1核、2GB、15分钟，日志`logs/cerra_cds_setup_JOBID.out/.err`。安装到既有wind-env；先将所有已安装包约束为原版本，再安装`cdsapi>=0.7.7,<0.8`和缺少的依赖，不升级训练库。若依赖冲突则失败，不自动放宽约束。结果保存在`environment/cds_setup_JOBID/`，completion应为passed、existing_versions_unchanged=true。

安装、下载均需计算节点具备相应出网能力；若无网络会失败，不能把脚本准备视为连通性测试成功。此安装作业只提交一次，不要与下载任务或环境冻结同时执行。

官方安装与许可说明：[CDS API setup](https://cds.climate.copernicus.eu/how-to-api)。用户需要在CDS网页账户中接受数据集使用条款。

## 2. 在提交终端设置API token

本项目不读取目录外的配置或密钥文件。使用当前提交shell的环境变量，隐藏输入：

```bash
read -rsp 'CDS personal access token: ' CDSAPI_KEY
export CDSAPI_KEY
echo
```

不要把token写进脚本、聊天、Git或命令参数。Slurm默认继承提交环境；若站点/个人配置禁用了环境继承，需按站点规则恢复该变量。新登录终端需要重新设置。代码显式传入url/key，并把CDSAPI_RC指向项目内路径，不读取home配置。

## 3. 先下载一个月

```bash
sbatch --array=0 slurm/run_cerra_download.sh
```

索引0是2025年1月，1核、4GB、20小时。包含CDS远端排队、处理和下载等待；这不是对服务耗时的保证。

输出：

- `data/CERRA/raw_2025/cerra_2025_01.grib`
- `data/CERRA/raw_2025/cerra_2025_01.download.json`
- 请求开始时保存的attempt JSON；不包含密钥或签名下载地址。
- `logs/cerra_download_ARRAYID_0.out/.err`

下载后只读GRIB header，不解码风场数值。要求207风速与260260风向各248条，完整01月三小时时间轴，参数、高度、units、analysis/step0和全网格投影/扫描匹配旧参考。通过后才发布正式`.grib`和completed receipt；保存请求、SHA256、字节数及客户端版本。

失败的`.grib.part`保留。重跑同一任务时，旧partial以preserved唯一后缀保留再重新请求；已完成文件先核对request/SHA/header后跳过。下载中的HTTP重试由客户端处理，但Slurm超时后的跨作业对已保存request_id优先复用；旧任务需先恢复编号，不保证从同一字节继续。正式GRIB若缺receipt会停止，不能覆盖原文件冒充重新下载。

## 4. 先导通过后下载剩余月份

任务映射固定：0–11为2025年1–12月；12–23为2023年1–12月。

如果要按用户原请求下载两年，先导完成后：

```bash
sbatch --array=1-23%1 slurm/run_cerra_download.sh
```

共23个剩余月度请求，最多1项并行，总计24个月度任务，不是24次训练。若只希望先完成优先确认年，可先用`--array=1-11%1`，2023另用`--array=12-23%1`。不得通过比较两个年份的模型分数再选择确认年。

两年各变量应有2920时刻；磁盘成本取决于CDS实际打包，保留完整原生网格会显著大于研究ROI提取结果，不能按小区域CSV大小估计原始下载量。

## 5. 衔接确认脚本

2025全部12个月下载成功后，仍需独立年度审计与数值QA：

```bash
module --force purge
module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3
source wind-env/bin/activate

python prepare_confirmation_year.py init \
  --year 2025 --version confirmation_v1 \
  --year-basis 'Complete compatible 2025 monthly downloads before error analysis' \
  --grib data/CERRA/raw_2025/*.grib

sbatch slurm/run_newyear_audit.sh --grib data/CERRA/raw_2025/*.grib
```

只传选定年份的12个正式GRIB，不混入2023/2024、partial或重复副本。方向消息会保留在原始文件里，年度风速审计/提取按原逻辑忽略它们。随后按[独立年份确认v1](INDEPENDENT_YEAR_CONFIRMATION_V1.md)执行提取→冻结→5先导→40剩余→分析。依赖安装应在冻结之前完成。

## 当前验收状态

9项合成/mock检查及Python/Shell语法通过。依赖安装、认证和三个2025月份下载/header检查已验证；远端恢复修复、其余月份、全年QA及数值提取尚未执行。
