import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 全局配置区 (Global Configuration)
# ==========================================

# --- 数据路径 ---
DATA_FILE_PATH = './data/CERRA/cerra_100_points_summer_roi_2024.csv'   # 请确保这里是存放您三个月CSV的文件夹路径

# --- 检查对象选择 ---
# 可选: 'ws_100' (100米风速) 或 'ws_10' (10米风速)
CHECK_TARGET = 'ws_10' 

# --- 分割比例 ---
TRAIN_SPLIT_RATIO = 0.8     # 模拟训练时的切分比例

# ==========================================
# 2. 数据加载与预计算
# ==========================================
def load_data(file_path):
    print(f"--- [Step 1] 正在加载数据 (目标: {CHECK_TARGET}) ---")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}，请检查路径。")
        return None

    # 直接读取这一个文件
    df = pd.read_csv(file_path)

    # 确保按时间排序
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(by=['point_id', 'datetime']).reset_index(drop=True)
    
    # 因为 CERRA 提取的 CSV 中直接就叫 ws_10
    if CHECK_TARGET == 'ws_10':
        if 'ws_10' in df.columns:
            df['wind_speed'] = df['ws_10']
        elif 'ws_100' in df.columns: 
            df['wind_speed'] = df['ws_100']
        else:
            raise ValueError(f"数据中找不到列名为 'ws_10' 的数据，现有列名: {df.columns}")
    else:
        raise ValueError("当前 CERRA 数据仅支持 CHECK_TARGET = 'ws_10'")

    print(f"数据加载完毕，总行数: {len(df)}")
    return df
# ==========================================
# 3. 核心检查逻辑
# ==========================================

def check_dead_and_extreme_values(df):
    print(f"\n--- [Step 2] 检查死值与异常极值 ---")
    ws = df['wind_speed']
    
    # 1. 空值检查
    n_nan = ws.isnull().sum()
    print(f"1. 空值 (NaN) 数量: {n_nan} \t [{'危险' if n_nan > 0 else '通过'}]")
    
    # 2. 零值检查 (完全静风)
    n_zeros = (ws == 0).sum()
    zero_ratio = n_zeros / len(ws) * 100
    print(f"2. 零值 (0.0) 数量: {n_zeros} ({zero_ratio:.2f}%)")
    if zero_ratio > 10:
        print("   [警告] 零值过多！可能存在传感器故障或数据缺失被填0的情况。")
    
    # 3. 极值统计
    max_val = ws.max()
    min_val = ws.min()
    mean_val = ws.mean()
    q99 = ws.quantile(0.99)
    q999 = ws.quantile(0.999)
    
    print(f"3. 统计概览:")
    print(f"   - 最小值: {min_val:.2f} m/s")
    print(f"   - 平均值: {mean_val:.2f} m/s")
    print(f"   - 99%分位: {q99:.2f} m/s (99%的数据小于此值)")
    print(f"   - 最大值: {max_val:.2f} m/s")
    
    # 4. 极值判定
    # 如果最大值是 99.9% 分位数的 1.5 倍以上，通常认为是离群点
    if max_val > q999 * 1.5:
        print(f"   [警告] 发现极端大值 ({max_val:.2f} m/s)！远超 99.9% 分位数 ({q999:.2f})。")
        print("   这可能会挤压归一化区间，建议在预处理时使用 clip() 截断。")
    else:
        print("   [通过] 极值在合理范围内。")

def check_distribution_shift(df):
    print(f"\n--- [Step 3] 检查分布一致性 (训练集 vs 测试集) ---")
    
    # 按时间切分
    unique_times = df['datetime'].sort_values().unique()
    split_idx = int(len(unique_times) * TRAIN_SPLIT_RATIO)
    split_time = unique_times[split_idx]
    
    train_ws = df[df['datetime'] <= split_time]['wind_speed']
    test_ws = df[df['datetime'] > split_time]['wind_speed']
    
    print(f"切分时间点: {split_time}")
    print(f"训练集均值: {train_ws.mean():.2f} m/s")
    print(f"测试集均值: {test_ws.mean():.2f} m/s")
    
    diff = abs(train_ws.mean() - test_ws.mean())
    if diff > 1.5:
        print(f"   [严重警告] 均值差异巨大 ({diff:.2f} m/s)！发生了显著的分布偏移 (Distribution Shift)。")
        print("   模型很可能无法适应测试集。建议：使用更长时间的数据，或改为随机打乱切分 (Shuffle Split)。")
    elif diff > 0.5:
        print(f"   [提示] 存在一定程度的季节性差异 ({diff:.2f} m/s)，属于正常但需留意的范围。")
    else:
        print("   [通过] 训练集和测试集均值接近。")

    # 画图
    plt.figure(figsize=(10, 5))
    sns.kdeplot(train_ws, label='Train Set (Past)', fill=True, color='blue', alpha=0.3)
    sns.kdeplot(test_ws, label='Test Set (Future)', fill=True, color='orange', alpha=0.3)
    plt.title(f'Distribution Check: {CHECK_TARGET}')
    plt.xlabel('Wind Speed (m/s)')
    plt.legend()
    # 保存图片
    save_path = f'check_dist_cerra_{CHECK_TARGET}.png'
    plt.savefig(save_path)
    print(f"   [绘图] 分布对比图已保存为 {save_path}，请务必查看！")
    plt.close()

def check_scaling_toxicity(df):
    print(f"\n--- [Step 4] 检查归一化 (Scaling) 是否有毒 ---")
    
    # 同样按时间切分
    unique_times = df['datetime'].sort_values().unique()
    split_time = unique_times[int(len(unique_times) * TRAIN_SPLIT_RATIO)]
    
    train_ws = df[df['datetime'] <= split_time]['wind_speed']
    test_ws = df[df['datetime'] > split_time]['wind_speed']
    
    train_max = train_ws.max()
    test_max = test_ws.max()
    
    print(f"训练集最大值 (Scaler上限): {train_max:.2f} m/s")
    print(f"测试集最大值: {test_max:.2f} m/s")
    
    if test_max > train_max:
        ratio = test_max / train_max
        print(f"   [警告] 测试集出现了比训练集更大的风速！(比例: {ratio:.2f}x)")
        print(f"   归一化后，这些值会变成 > 1.0 的数 (例如 {ratio:.2f})。")
        if ratio > 1.2:
            print("   [危险] 超出幅度较大 (>20%)，这可能导致模型预测失效。")
            print("   建议：1. 增大训练集覆盖范围； 2. 手动设置 Scaler 的上限 (比如设为 30m/s)。")
    else:
        print("   [通过] 测试集所有数据都在训练集的范围内 (Safe Scaling)。")

# ==========================================
# 4. 主执行入口
# ==========================================

if __name__ == "__main__":
    try:
        # 1. 加载
        df = load_data(DATA_FILE_PATH)
        if df is not None:
            # 2. 基础检查
            check_dead_and_extreme_values(df)
            # 3. 分布检查
            check_distribution_shift(df)
            # 4. 归一化检查
            check_scaling_toxicity(df)
            
            print("\n=== 体检结束 ===")
    except Exception as e:
        print(f"\n运行出错: {e}")
        print("请检查路径是否正确，或数据文件中是否有必需的列名。")