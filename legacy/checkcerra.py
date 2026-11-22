import xarray as xr

# 1. 读取 GRIB 文件
# engine='cfgrib' 是解析 GRIB 格式的关键
grib_file = '/p/project1/geofm4eo/tian1/wind_prediction/data/CERRA/dacdd4252851172d1e8fbd4b4def733a.grib'  # 替换成您的文件名
ds = xr.open_dataset(grib_file, engine='cfgrib')

# 2. 打印数据集信息 (非常重要，看看里面有哪些变量)
print(ds)