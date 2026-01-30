import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ==========================================
# 1. 全局配置区 (Global Configuration)
# ==========================================

# --- 路径设置 ---
# 请将此路径修改为您存放三个月CSV文件的文件夹路径
DATA_DIR_PATH = './data_folder/' 

# --- 数据处理参数 ---
WINDOW_SIZE = 24        # 输入窗口：过去24小时
FORECAST_HORIZON = 1    # 预测目标：未来第1个小时
TRAIN_SPLIT_RATIO = 0.8 # 训练集比例 (前80%训练, 后20%测试)

# --- 训练超参数 ---
MODEL_TYPE = 'stacked'  # 可选: 'simple' 或 'stacked'
BATCH_SIZE = 64
EPOCHS = 100            # 最大轮数 (有早停机制，通常不会跑满)
LEARNING_RATE = 0.001
PATIENCE = 10           # 早停容忍度

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
# 4. 模型定义 (包含普通LSTM和Stacked LSTM)
# ==========================================

def build_model(input_shape, model_type='stacked'):
    model = Sequential()
    
    if model_type == 'simple':
        print("[Model] 构建 Simple LSTM 模型...")
        # 单层 LSTM
        model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
        model.add(Dropout(0.2))
        
    elif model_type == 'stacked':
        print("[Model] 构建 Stacked LSTM 模型 (推荐)...")
        # 第一层：必须 return_sequences=True 以传递时间步给下一层
        model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(0.2)) # 防止过拟合
        
        # 第二层：汇总时间特征
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.2))
        
    else:
        raise ValueError("未知模型类型，请选择 'simple' 或 'stacked'")

    # 输出层 (回归任务，线性激活)
    model.add(Dense(1))
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    
    return model

# ==========================================
# 5. 可视化函数
# ==========================================

def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 5))
    
    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Val Loss (MSE)')
    plt.title(f'{model_name} Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    
    # MAE 曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title(f'{model_name} Mean Absolute Error')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'training_history_{model_name}.png')
    print(f"[Output] 训练曲线已保存为 training_history_{model_name}.png")
    # plt.show() # 如果在非交互环境运行，请注释此行

def plot_predictions(y_true, y_pred, model_name, sample_size=200):
    # 只画一部分数据以便观察细节
    plt.figure(figsize=(14, 6))
    
    plt.plot(y_true[:sample_size], label='Actual Wind Speed (m/s)', color='blue', alpha=0.6)
    plt.plot(y_pred[:sample_size], label='Predicted Wind Speed (m/s)', color='red', linestyle='--', alpha=0.8)
    
    plt.title(f'{model_name}: Prediction vs Actual (First {sample_size} hours of Test Set)')
    plt.xlabel('Time Steps (Hours)')
    plt.ylabel('Wind Speed (m/s)')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'prediction_comparison_{model_name}.png')
    print(f"[Output] 预测对比图已保存为 prediction_comparison_{model_name}.png")
    # plt.show()

# ==========================================
# 6. 主程序入口
# ==========================================

if __name__ == "__main__":
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

    # 3. 训练模型
    print(f"\n[Training] 开始训练 {MODEL_TYPE} 模型...")
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=PATIENCE, 
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )

    # 4. 评估与保存
    print("\n[Evaluation] 在测试集上评估...")
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MSE: {loss:.4f}")
    print(f"Test MAE: {mae:.4f}")

    # 保存模型
    model_filename = f'wind_speed_{MODEL_TYPE}_lstm.h5'
    model.save(model_filename)
    print(f"[Output] 模型已保存为 {model_filename}")

    # 5. 可视化
    plot_training_history(history, MODEL_TYPE)
    
    # 生成预测值进行对比绘图
    y_pred = model.predict(X_test)
    plot_predictions(y_test, y_pred, MODEL_TYPE)

    print("\nDone. 所有任务完成。")