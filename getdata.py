"""
=============================================================================
CERRA Data Extraction Script for Micro-scale Wind Prediction
-----------------------------------------------------------------------------
Description: 
This script extracts wind speed data (si10) for 100 specific geographical 
points from a high-resolution CERRA GRIB file. It utilizes "Lazy Loading" 
and an efficient 2D nearest-neighbor search to handle massive GRIB files 
(e.g., 27GB+) without memory overflow. 

It also includes a time-filtering feature to extract data only for 
specific months (e.g., Summer months) to reduce the final dataset size.
=============================================================================
"""

import xarray as xr
import pandas as pd
import numpy as np
import time

# ==========================================
# 1. Configuration Section
# ==========================================
GRIB_FILE = '/p/project1/geofm4eo/tian1/wind_prediction/data/CERRA/dacdd4252851172d1e8fbd4b4def733a.grib'   
OUTPUT_CSV = '/p/project1/geofm4eo/tian1/wind_prediction/data/CERRA/cerra_100_points_200km_summer_roi_2024.csv'

# Target months for extraction (e.g., [7, 8, 9] for July, August, September)
# To extract the whole year, you can use [1,2,3,4,5,6,7,8,9,10,11,12]
TARGET_MONTHS = [6, 7, 8]

# Generate 100 test points using your original ~31km ROI
# Latitude: 50.5 to 50.78
# Longitude: 9.0 to 9.31
np.random.seed(42)
num_points = 100
# 坐标范围：31km × 31km
#lats = np.random.uniform(50.5, 50.78, num_points)
#lons = np.random.uniform(9.0, 9.31, num_points)
# 提取脚本中修改坐标范围：200km × 200km
lats = np.random.uniform(49.5, 51.5, num_points)
lons = np.random.uniform(8.5, 11.5, num_points)
points = [{'point_id': i, 'lat': lats[i], 'lon': lons[i]} for i in range(num_points)]

# ==========================================
# 2. Efficient 2D Coordinate Search Function
# ==========================================
def get_nearest_indices(target_lat, target_lon, lat_grid, lon_grid):
    """
    Calculates the distance between the target coordinates and all points in the grid,
    returning the (y, x) indices of the nearest point.
    This is the standard approach for handling CERRA's 2D projection coordinates.
    """
    # Calculate squared distance (Euclidean distance is accurate enough and very fast for small areas)
    dist_sq = (lat_grid - target_lat)**2 + (lon_grid - target_lon)**2
    # Find the flattened index of the minimum distance
    min_idx = np.argmin(dist_sq)
    # Convert flattened index back to 2D (y, x) indices
    y_idx, x_idx = np.unravel_index(min_idx, dist_sq.shape)
    return y_idx, x_idx

# ==========================================
# 3. Lazy Loading and Extraction
# ==========================================
print("Reading GRIB file structure (Lazy Loading)...")
start_time = time.time()
ds = xr.open_dataset(GRIB_FILE, engine='cfgrib')

# Filter dataset by specific months if TARGET_MONTHS is defined
if TARGET_MONTHS:
    print(f"Filtering data for months: {TARGET_MONTHS}...")
    ds = ds.sel(time=ds['time'].dt.month.isin(TARGET_MONTHS))

# Calculate expected time steps after filtering
total_time_steps = len(ds.time)
print(f"Total time steps after filtering: {total_time_steps}")

print("Loading latitude and longitude grids into memory (this takes a few seconds)...")
# Only latitude and longitude arrays are loaded into RAM; the massive weather data remains on disk
lat_grid = ds['latitude'].values
lon_grid = ds['longitude'].values

print(f"Grid loaded successfully. Starting extraction for {num_points} points...")
df_list = []

for pt in points:
    # 1. Calculate the y, x indices for this point in the grid
    y_idx, x_idx = get_nearest_indices(pt['lat'], pt['lon'], lat_grid, lon_grid)
    
    # 2. Use isel (Index Selection) to pierce through the data cube and extract the time series
    # This step is highly memory-efficient as it only reads data for a single pixel
    ds_point = ds.isel(y=y_idx, x=x_idx)
    
    # 3. Convert to Pandas DataFrame
    # reset_index() flattens the output, creating a clean tabular format
    df_point = ds_point.to_dataframe().reset_index()
    
    # 4. Keep only the necessary columns: time, si10
    df_point = df_point[['time', 'si10']].copy()
    
    # 5. Rename to match the LSTM script's expected names
    df_point = df_point.rename(columns={
        'time': 'datetime',
        'si10': 'ws_10'  # Forced naming to ws_100 so you don't need to change TARGET_COL in LSTM script
    })
    
    # 6. Tag with point_id
    df_point['point_id'] = pt['point_id']
    
    df_list.append(df_point)
    
    if pt['point_id'] % 10 == 0 or pt['point_id'] == num_points - 1:
        print(f"  Extraction progress: {pt['point_id'] + 1}/{num_points} points completed")

# ==========================================
# 4. Merge and Save
# ==========================================
print("Merging panel data...")
final_df = pd.concat(df_list, ignore_index=True)

# Ensure the order strictly matches the LSTM requirements: first by point_id, then by datetime
final_df = final_df.sort_values(by=['point_id', 'datetime']).reset_index(drop=True)

final_df.to_csv(OUTPUT_CSV, index=False)
end_time = time.time()

# Calculate expectations vs reality
expected_rows = num_points * total_time_steps
actual_rows = len(final_df)

print(f"🎉 Success! Extraction completed in {end_time - start_time:.2f} seconds.")
print(f"Total data rows: {actual_rows} (Expected: {num_points} points x {total_time_steps} steps = {expected_rows} rows)")
print(f"CSV saved to: {OUTPUT_CSV}")