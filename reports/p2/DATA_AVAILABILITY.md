# P2 遥感地表数据可用性

当前更新（2026-11-14）：下载作业912842已完成，16个原始瓦片及回执已齐全，总计792406989字节，缓存data/rs_static/raw_v1。当前仍缺GIS依赖配置、实际栅格空间QA及98格14维特征表。下方保留初始盘点和数据设计。

已存在年度 CERRA 中心表、外围映射和已审计 Lambert 网格参数。100 个 point_id 对应 98 个唯一网格。中心纬度约 49.487–51.462°N，经度约 8.538–11.436°E。本轮扫描未找到 WorldCover、GLO-30 原始栅格或已验收的静态特征表。wind-env 中未检测到 rasterio 或 pyproj。下载、GIS 依赖配置、全量统计与空间 QA 等待后续阶段。

| 产品 | 预计需要的候选瓦片 | 下载与记录 |
|---|---|---|
| ESA WorldCover 2021 v200 Map | N48E006、N48E009、N51E006、N51E009 | 官方 S3 前缀 esa-worldcover/v200/2021/map，按瓦片下载。EPSG:4326，3° 瓦片，约 10 m，CC BY 4.0。记录 DOI 10.5281/zenodo.7254221。 |
| Copernicus DEM GLO-30 DSM | 纬度下边界 N49/N50/N51 × 经度左边界 E008/E009/E010/E011，共 12 个候选 1° 单元 | CDSE Browser 或 API，按相同产品版本查询并记录确切资产 ID、版本、垂直基准、许可及质量层。引用 DOI 10.5270/ESA-c5d3d65。 |

候选清单根据中心范围加像元和坡度计算余量确定，下载前用实际投影像元边界与官方瓦片索引求交，保留坡度所需的边缘像元。逐文件记录来源 URL、下载日期、产品/瓦片版本、SHA256、水平 CRS、垂直单位/基准、NoData 和掩膜。具体头信息等待真实瓦片核验。WorldCover 代表 2021 年地表，对 2024 年是静态近似。GLO-30 是包含植被与建筑的 DSM，获取时段以供应商产品记录为准。

提取用 GRIB 的 Lambert 参数和扫描方向重建以格心为中心的 5500 m 方格。先复算格心经纬度并与 centers.csv 核对，再把像元边界与栅格相交。土地覆盖按有效面积分配到 11 类，报告分母和覆盖率。高程均值、标准差与平均坡度采用有效区域，坡度在米制空间计算并记录分析尺度。NoData 与水面类别分别处理。初版只统计目标格，质检记录含类别比例和、有效覆盖率及若干真实边界抽查图。

统一 grid_id 为 center_y:center_x。静态表每个 grid_id 一行，动态窗口按 point_id 映射再连接，禁止按行序拼接。rs_static_contract.py 已验证重复中心得到相同向量，缩放拟合使用去重后的训练格集合。首版缺值拒绝进入模型，覆盖率门槛及异常处理在 P2 数据 QA 后冻结。四个配对组共用融合结构，同动态组只改变 14 维静态输入，关闭时填固定零向量。真实静态特征与 P3 训练仍待执行。

产品参考：[ESA WorldCover 数据下载](https://esa-worldcover.org/en/data-access)；[CDSE GLO-30 产品与许可](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM)。

## 2026-11-14准备进展

已核对4个WorldCover及12个GLO-30公开URL，HEAD均成功，总计792406989字节。此次DSM采用公开AWS COG镜像的2021发布版，精确对象版本、日期和长度见configs/rs_hpc_v3_p2_download.json。下载脚本slurm/run_p2_download.sh已准备，真实下载及GeoTIFF空间QA仍待执行。操作说明见docs/runbooks/P2_DOWNLOAD_RUNBOOK.md。 [GLO-30公开镜像](https://registry.opendata.aws/copernicus-dem/)

## GIS准备进展

wind-env已补齐rasterio/pyproj/shapely，核心机器学习包版本复核一致。16个TIFF真实头信息与98格投影坐标检查通过，完整像元覆盖统计和14维特征仍待运行。可提交sbatch slurm/run_prepare_rs_static.sh，配置为configs/rs_hpc_v3_p2_static.json，输出data/rs_static/features_v1。默认有效覆盖门槛99%，任何格失败时保留全表和诊断，READY不生成。详细说明见docs/runbooks/P2_STATIC_RUNBOOK.md。
