# GIS空间QA和14维地表特征

## 当前状态

2026-11-14已在项目wind-env补齐rasterio 1.4.3、pyproj 3.7.2和shapely 2.1.2及其依赖。安装使用二进制wheel，现有NumPy、pandas、scikit-learn、TensorFlow和Keras版本由environment/constraints-wind-core.txt锁定并复核。pip check通过，当前完整快照为environment/requirements-wind-env.txt。

16个真实TIFF的轻量头信息检查通过。WorldCover为EPSG:4326、36000×36000、NoData=0。DSM为EPSG:4326，49度瓦片3600列，50–51度瓦片2400列，均未声明NoData。DSM的0米值按有效观测保留，读取掩膜和非有限值单独处理。98个原中心网格的投影重建最大误差约4.38e-9米。检查记录为result/rs_hpc_v3/P2/gis_header_preflight.json。

全量作业912843已完成并验收：98格14维、100点映射、全部qa_pass，最低有效覆盖率近100%。READY.json及所有产物哈希核验通过。下一步见[P3先导](P3_RUNBOOK.md)；以下保留复现方法。

## 提交

从项目根目录执行。

```bash
sbatch slurm/run_prepare_rs_static.sh
```

这是1项CPU作业，申请1工作线程、8GB内存、4小时时限。实际分配核数和成本以sacct为准。日志为logs/p2_static_JOBID.out/.err。没有模型拟合。输出版本为data/rs_static/features_v1/。

## 14维特征

| 列 | 内容 | 单位 |
|---|---|---|
| lc_10 | 树木覆盖比例 | 0–1 |
| lc_20 | 灌丛比例 | 0–1 |
| lc_30 | 草地比例 | 0–1 |
| lc_40 | 农田比例 | 0–1 |
| lc_50 | 建成区比例 | 0–1 |
| lc_60 | 裸地/稀疏植被比例 | 0–1 |
| lc_70 | 雪/冰比例 | 0–1 |
| lc_80 | 永久水体比例 | 0–1 |
| lc_90 | 草本湿地比例 | 0–1 |
| lc_95 | 红树林比例 | 0–1 |
| lc_100 | 苔藓/地衣比例 | 0–1 |
| dsm_mean_m | 面积加权平均地表高程 | m |
| dsm_sd_m | 面积加权高程总体标准差 | m |
| dsm_slope_mean_deg | 面积加权平均DSM坡度 | degree |

WorldCover代表2021年的静态地表背景，用于2024年预测研究。DSM包含植被和建筑，产品垂直基准为EGM2008；源TIFF标签与单位缺省情况另存，当前流程不做垂直基准转换。 [Copernicus DEM产品说明](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM)

## 几何、缺测与统计

用已审计GRIB的球面Lambert投影、6371229米半径、第一格心和扫描方向，重建原生网格坐标。每个中心的代表区域为以原生格心为中心的5500×5500米方格，检查其坐标与原映射一致。

土地覆盖直接读取原分类栅格。将像元四角投影到CERRA米制坐标，与目标方格相交后累计各类面积。像元边界采用投影后四边形近似，面积单位明确为原生投影平面m²。类别编码不做双线性插值。11类比例以有效已知类别面积为分母，NoData、未知类别和未覆盖面积继续保存在quality.csv。

DSM均值和标准差使用原生像元的面积权重。坡度另将DSM双线性对齐到统一30米CERRA米制网格，保留目标方格外的邻域像元，以Horn 3×3计算。完整九邻域有效时才产生坡度，缺测保持NaN。

默认工程QA门槛由本次配置固定为有效面积覆盖率至少99%，三项lc/dsm/slope均须满足。几何覆盖总量与目标格面积的相对偏差不超过0.1%，未知类别面积比例不超过1e-8。门槛记录在configs/rs_hpc_v3_p2_static.json，未参考任何预测得分。所有98格都保留，任何格未通过时输出失败报告，READY不会生成。

静态插补、缩放、恒定列处理留在后续训练流程，仅拟合训练网格集合。提取阶段不使用风速标签，也不根据空间全表筛除列。

## 验收

作业应COMPLETED/0:0，日志最后为[P2 static] READY。检查以下文件。

| 文件 | 验收重点 |
|---|---|
| READY.json | status=passed，98格，特征顺序与STATIC_FEATURES一致 |
| static_features.csv | 98行、唯一grid_id和14个有限数值列 |
| point_grid_mapping.csv | 100个原point_id，映射至98个唯一grid_id |
| quality.csv | 每格覆盖、NoData、未知类别及投影误差，qa_pass均为True |
| qa_summary.json | status=passed，所有目标格通过 |
| grid_footprints.geojson | 98个真实目标格边界及坐标 |
| raster_metadata.json | 16个源文件的CRS、分辨率、NoData、单位和标签 |
| feature_metadata.json | 版本、参数、统计方法、单位、来源与静态时间假设 |
| provenance.json / completion.json | 代码、配置、数据及产物摘要 |
| qa_格点ID.png/.svg | 真实边界、土地覆盖、DSM和坡度的人工抽查图 |

QA示例固定为DSM标准差最低/最高和建成区比例最高的三个不同目标格，不使用模型误差选择。图中土地覆盖采用最近邻显示，数值统计仍为原生面积比例。

## 失败、重试和后续

中断时保留独立临时目录，可重新提交同一命令。已完成且指纹、产物校验一致时跳过。若QA失败，完整quality.csv及qa_report.md仍发布，脚本返回失败退出码，所有目标格保留。应检查缺瓦片、掩膜、投影或源数据问题；更改配置或代码后使用新的输出版本，例如--output_dir data/rs_static/features_v2。

本轮轻量测试包含平面坡度、缺测邻域、面积交集、有效0米、未知类别、相邻无NoData瓦片的拼接、球面CERRA投影及合成QA图，11项检查通过。真实98格特征尚待上述作业验收。P3的四组先导在此后实现和提交。
