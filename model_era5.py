import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from datetime import datetime  # <--- [新增] 用于生成时间戳
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.optimizers import Adam

# ==========================================
# 1. 全局配置区 (Global Configuration)
# ==========================================

# --- 训练超参数 ---
MODEL_TYPE = 'simple'   # 可选: 'simple' 或 'stacked'
BATCH_SIZE = 64
EPOCHS = 100            # 最大轮数 (有早停机制，通常不会跑满)
LEARNING_RATE = 0.001
PATIENCE = 10           # 早停容忍度
DROPOUT_RATE = 0.4      # Dropout比率

# --- [新增] 实验 ID 生成 (Run ID) ---
# 获取当前时间，例如: 20251216_143000
# RUN_ID 将作为本次运行所有输出文件的唯一标识
CURRENT_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ID = f"{MODEL_TYPE}_{CURRENT_TIME}"

# --- 路径设置 ---
# 请将此路径修改为您存放三个月CSV文件的文件夹路径
DATA_DIR_PATH = './data/'       # 数据源路径
RESULT_DIR = './result/'        # 图片和日志保存路径
MODEL_DIR = './model/'          # 模型文件保存路径

# --- 数据处理参数 ---
WINDOW_SIZE = 24        # 输入窗口：过去24小时
FORECAST_HORIZON = 3    # 预测目标：未来第1个小时
TRAIN_SPLIT_RATIO = 0.8 # 训练集比例 (前80%训练, 后20%测试)

# --- 特征列定义 ---
# 基础特征 (原始数据中的列名)
RAW_FEATURE_COLS = [
    'u_component_of_wind_100m', 'v_component_of_wind_100m',
    'u_component_of_wind_10m', 'v_component_of_wind_10m'
]
TARGET_COL = 'ws_100'  # 我们计算出的合成风速(100m)作为预测目标

# ==========================================
# 2. 数据加载与特征工程
# ==========================================

def load_and_preprocess_data(data_dir):
    print(f"[Info] 实验 ID (Run ID): {RUN_ID}")  # <--- [新增] 打印ID
    print(f"[Info] 正在从 {data_dir} 加载数据...")
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not all_files:
        raise ValueError(f"错误: 在路径 {data_dir} 下未找到CSV文件！")
    
    # 1. 合并三个月的数据
    df_list = []
    for filename in all_files:
        df_temp = pd.read_csv(filename)
        df_list.append(df_temp)
    
    df = pd.concat(df_list, axis=0, ignore_index=True)
    
    # 2. 时间格式转换与排序
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(by=['point_id', 'datetime']).reset_index(drop=True)
        df['hour'] = df['datetime'].dt.hour
    else:
        raise ValueError("错误: 数据中缺少 'datetime' 列")

    # 3. 物理特征计算 (计算标量风速)
    # ws = sqrt(u^2 + v^2)
    print("[Info] 执行物理特征计算 (矢量转标量)...")
    df['ws_100'] = np.sqrt(df['u_component_of_wind_100m']**2 + df['v_component_of_wind_100m']**2)
    df['ws_10'] = np.sqrt(df['u_component_of_wind_10m']**2 + df['v_component_of_wind_10m']**2)

    # 4. 时间周期编码 (Sin/Cos)
    print("[Info] 执行时间周期编码...")
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    print(f"[Success] 数据加载完成，总行数: {len(df)}")
    return df

# ==========================================
# 3. 序列生成与数据集构建 (核心防泄露逻辑)
# ==========================================

def create_dataset(df, window_size, horizon, split_ratio):
    # 最终用于模型的特征列
    feature_cols = RAW_FEATURE_COLS + ['ws_100', 'ws_10', 'hour_sin', 'hour_cos']
    
    # 1. 按时间切分训练集和测试集 (Time Split)
    # 注意：必须按时间轴切，不能随机shuffle，否则会泄露未来信息
    # 这里的逻辑是：找到时间轴上的80%分位点
    unique_times = df['datetime'].sort_values().unique()
    split_time = unique_times[int(len(unique_times) * split_ratio)]
    
    train_df = df[df['datetime'] <= split_time].copy()
    test_df = df[df['datetime'] > split_time].copy()
    
    print(f"[Info] 训练集截止时间: {split_time}")
    print(f"       训练集样本数 (行): {len(train_df)}")
    print(f"       测试集样本数 (行): {len(test_df)}")

    # 2. 归一化 (仅在训练集上Fit)
    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols])
    
    # 定义滑动窗口生成器 (按 point_id 分组)
    def _create_sequences(data_df):
        X, y = [], []
        # 关键：按地点分组处理，防止跨地点滑窗
        grouped = data_df.groupby('point_id')
        
        for _, group in grouped:
            # 提取该地点的特征矩阵和目标向量
            group_feats = scaler.transform(group[feature_cols])
            group_target = group[TARGET_COL].values
            
            # 滑动窗口
            # 样本 i 的输入: [i, i+window_size)
            # 样本 i 的输出: i + window_size + (horizon-1)
            max_idx = len(group) - window_size - horizon + 1
            for i in range(max_idx):
                X.append(group_feats[i : i + window_size])
                y.append(group_target[i + window_size + horizon - 1])
                
        return np.array(X), np.array(y)

    print("[Info] 正在生成时间序列样本 (这可能需要一点时间)...")
    X_train, y_train = _create_sequences(train_df)
    X_test, y_test = _create_sequences(test_df)
    
    print(f"[Success] 数据集构建完成:")
    print(f"          X_train shape: {X_train.shape}")
    print(f"          y_train shape: {y_train.shape}")
    print(f"          X_test shape:  {X_test.shape}")
    
    return X_train, y_train, X_test, y_test, scaler

# ==========================================
# 4. 模型定义
# ==========================================

def build_model(input_shape, model_type):
    model = Sequential()
    
    if model_type == 'simple':
        print("[Model] 构建 Simple LSTM 模型...")
        model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
        model.add(Dropout(DROPOUT_RATE))
        
    elif model_type == 'stacked':
        print("[Model] 构建 Stacked LSTM 模型...")
        model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(DROPOUT_RATE))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(DROPOUT_RATE))
    
    model.add(Dense(1))
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    return model

# ==========================================
# 5. 可视化与保存辅助函数
# ==========================================

# [修改] 入参不再需要 model_name，直接使用全局 RUN_ID
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{RUN_ID} Loss') # [修改] 标题使用 RUN_ID
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    
    # MAE Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title(f'{RUN_ID} MAE') # [修改] 标题使用 RUN_ID
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    # [修改] 文件名使用 RUN_ID
    save_path = os.path.join(RESULT_DIR, f'history_{RUN_ID}.png')
    plt.savefig(save_path)
    print(f"[Output] 训练曲线已保存: {save_path}")
    plt.close()

# [修改] 入参不再需要 model_name
def plot_predictions(y_true, y_pred, sample_size=200):
    plt.figure(figsize=(14, 6))
    plt.plot(y_true[:sample_size], label='Actual Wind Speed (m/s)', color='blue', alpha=0.6)
    plt.plot(y_pred[:sample_size], label='Predicted Wind Speed (m/s)', color='red', linestyle='--', alpha=0.8)
    plt.title(f'{RUN_ID}: Prediction vs Actual (First {sample_size} hours)')
    plt.xlabel('Time Steps (Hours)')
    plt.ylabel('Wind Speed (m/s)')
    plt.legend()
    plt.grid(True)
    
    # [修改] 文件名使用 RUN_ID
    save_path = os.path.join(RESULT_DIR, f'pred_compare_{RUN_ID}.png')
    plt.savefig(save_path)
    print(f"[Output] 预测图已保存: {save_path}")
    plt.close()

def save_experiment_config():
    """保存本次实验的超参数配置，方便日后复盘"""
    # [修改] 文件名使用 RUN_ID
    config_path = os.path.join(RESULT_DIR, f'config_{RUN_ID}.txt')
    with open(config_path, 'w') as f:
        f.write(f"Run ID: {RUN_ID}\n") # [新增] 记录 Run ID
        f.write(f"Timestamp: {CURRENT_TIME}\n")
        f.write(f"Model Type: {MODEL_TYPE}\n")
        f.write(f"Window Size: {WINDOW_SIZE}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Dropout Rate: {DROPOUT_RATE}\n")
        f.write(f"Patience: {PATIENCE}\n")
        f.write(f"Data Dir: {DATA_DIR_PATH}\n")
    print(f"[Output] 实验配置已保存: {config_path}")

# ==========================================
# 6. 主程序入口
# ==========================================

if __name__ == "__main__":
    # 0. 创建文件夹 (如果不存在)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"[Init] 文件夹检查完毕: \n - {RESULT_DIR} \n - {MODEL_DIR}")
    print(f"[Init] 当前实验 ID: {RUN_ID}") # [新增] 打印当前 ID

    # 1. 准备数据
    try:
        df = load_and_preprocess_data(DATA_DIR_PATH)
        X_train, y_train, X_test, y_test, scaler = create_dataset(
            df, WINDOW_SIZE, FORECAST_HORIZON, TRAIN_SPLIT_RATIO
        )
    except Exception as e:
        print(f"数据处理出错: {e}")
        exit()

    # 2. 构建模型
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape, model_type=MODEL_TYPE)
    model.summary()

    # 3. 训练模型 (加入 Log 保存)
    print(f"\n[Training] 开始训练 {MODEL_TYPE} 模型...")
    
    # 定义回调函数
    callbacks_list = [
        # EarlyStopping: 监控 val_loss, 没改善就停, 并恢复最佳权重
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
        
        # [修改] CSVLogger: 文件名使用 RUN_ID
        CSVLogger(os.path.join(RESULT_DIR, f'training_log_{RUN_ID}.csv'))
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_list, # 使用列表
        verbose=1
    )

    # 4. 评估与保存
    print("\n[Evaluation] 在测试集上评估...")
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MSE: {loss:.4f}")
    print(f"Test MAE: {mae:.4f}")

    # [修改] 保存模型到 model 文件夹，文件名使用 RUN_ID
    model_path = os.path.join(MODEL_DIR, f'wind_speed_{RUN_ID}.h5')
    model.save(model_path)
    print(f"[Output] 模型已保存: {model_path}")
    
    # 保存参数配置
    save_experiment_config()

    # 5. 可视化 (图片自动保存到 result 文件夹)
    plot_training_history(history)
    y_pred = model.predict(X_test)
    plot_predictions(y_test, y_pred)

    print(f"\nDone. 所有任务完成。本次实验 ID: {RUN_ID}")