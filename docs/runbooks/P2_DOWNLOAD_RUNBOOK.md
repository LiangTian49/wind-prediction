# P2研究区遥感瓦片下载

作业912842已于2026-11-14成功完成，51秒、0:0。16个文件共792406989字节（约0.79GB），文件大小、回执和固定版本清单已核对。下载程序记录了逐文件SHA256。当前等待GIS空间QA及特征提取，下方提交命令保留为重试/复核入口。

```bash
sbatch slurm/run_p2_download.sh
```

1个dp-cn CPU任务、1工作线程、2GB内存、4小时时限。实际资源成本以sacct的AllocCPUS和Elapsed为准。日志为logs/p2_download_JOBID.out/.err。结果写入data/rs_static/raw_v1/及result/rs_hpc_v3/P2/download_JOBID.json。

## 产品与版本

- ESA WorldCover 2021 v200，4个3度瓦片，覆盖原ROI矩形及目标网格边缘。公开S3地址来自[产品数据页](https://esa-worldcover.org/en/data-access)。
- Copernicus GLO-30 Public，12个1度瓦片。此次采用公开COG镜像，AWS目录说明其来源为2021发布版，精确对象ETag与Last-Modified固定在配置中。[镜像目录与发布说明](https://registry.opendata.aws/copernicus-dem/)

完整清单为configs/rs_hpc_v3_p2_download.json，记录URL、长度、ETag、Last-Modified、产品版本及许可来源。程序逐项校验远程版本，保留本地SHA256。ETag作为远程对象标识，SHA256作为下载后本地摘要。

## 下载验收

要求作业COMPLETED/0:0，download_complete.json的status为downloaded_waiting_geospatial_QA、assets=16、bytes=792406989。每个TIFF有对应receipt.json，内容长度、TIFF标识与SHA256通过，日志显示16/16。已有完整文件经摘要校验后跳过，.part临时文件支持Range续传，服务器忽略Range时重新写临时文件。版本变化或缓存损坏会报错并保留文件。

下载成功后还要核对真实文件CRS、NoData、垂直单位、区域覆盖与质量层，安装/配置wind-env所需GIS依赖，再做98个目标格的14维统计和QA。本任务的完成状态只表示瓦片下载通过，P3训练等待静态特征验收。

## 重试

瞬时网络错误会重试三次。失败后可原命令重新提交，保留已经核验的瓦片和未完成的.part文件。若远程ETag发生变化，应检查产品版本并另建数据版本。计算节点网络可达性和真实下载尚待本次作业验证。
