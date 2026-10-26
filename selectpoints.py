import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==========================================
# 1. 核心配置区
# ==========================================
# 指向你已经提取好的 200km 范围夏季面板数据
DATA_FILE_PATH = './data/CERRA/cerra_100_points_200km_summer_roi_2024.csv'
OUTPUT_FIGURE = 'use_case_spatial_gradientnew.png'

# 挑选 3 个具有代表性的点位 ID
# 在你的 100 个随机点中，挑选 ID 间距较大的点以保证地理距离
TARGET_POINTS = [0, 45, 99] 
POINT_LABELS = ['Point A (Western Edge)', 'Point B (Center)', 'Point C (Eastern Edge)']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c'] # 蓝, 橙, 绿经典配色

# 设定观察的时间窗口 (截取一段有明显风速波动的 72 小时冷锋过境期)
# 你可以根据生成的整体趋势，修改具体的时间段
START_TIME = '2024-07-15 00:00:00'
END_TIME = '2024-07-18 00:00:00'

# ==========================================
# 2. 数据过滤与绘图逻辑
# ==========================================
def plot_spatial_use_cases():
    print(f"Loading data from {DATA_FILE_PATH}...")
    try:
        df = pd.read_csv(DATA_FILE_PATH)
    except FileNotFoundError:
        print("Data file not found. Check the path.")
        return

    # 确保时间列为 datetime 格式
    df['datetime'] = pd.to_datetime(df['datetime'])

    # 截取特定时间窗口
    mask = (df['datetime'] >= START_TIME) & (df['datetime'] <= END_TIME)
    df_window = df.loc[mask].copy()

    if df_window.empty:
        print("The specified time window contains no data. Adjust START_TIME and END_TIME.")
        return

    plt.figure(figsize=(14, 6), dpi=300)

    # 循环绘制 3 个点的时间序列
    for idx, pt_id in enumerate(TARGET_POINTS):
        df_pt = df_window[df_window['point_id'] == pt_id]
        
        if df_pt.empty:
            print(f"Warning: No data for point_id {pt_id}")
            continue

        plt.plot(
            df_pt['datetime'], 
            df_pt['ws_10'], 
            label=f"{POINT_LABELS[idx]} (ID: {pt_id})", 
            color=COLORS[idx],
            linewidth=2,
            alpha=0.85
        )

    # ==========================================
    # 3. 论文级图表格式美化
    # ==========================================
    plt.title('Spatiotemporal Wind Speed Variations Across 200km ROI (CERRA 5.5km)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Time (UTC)', fontsize=12, labelpad=10)
    plt.ylabel('10m Wind Speed (m/s)', fontsize=12, labelpad=10)
    
    # 优化时间轴显示格式
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # 添加网格线与图例
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', frameon=True, fontsize=11, shadow=True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE, bbox_inches='tight')
    print(f"Success! High-resolution figure saved to: {OUTPUT_FIGURE}")
    plt.close()

if __name__ == "__main__":
    plot_spatial_use_cases()