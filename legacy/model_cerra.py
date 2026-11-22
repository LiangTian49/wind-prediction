import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # [新增] 引入 sklearn 的标准评估库
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

# --- 实验 ID 生成 (Run ID) ---
CURRENT_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ID = f"{MODEL_TYPE}_{CURRENT_TIME}"

# --- 路径设置 ---
DATA_DIR_PATH = './data/CERRA/cerra_100_points_200km_summer_roi_2024.csv' # 数据源路径

# 1. 设定所有实验结果的总根目录
BASE_OUT_DIR = './runs/' 
# 2. 为本次实验生成专属的文件夹路径 (例如: ./runs/simple_20260519_123000/)
RUN_DIR = os.path.join(BASE_OUT_DIR, RUN_ID)

# --- 数据处理参数 ---
WINDOW_SIZE = 56        # 输入窗口：过去24小时 (8*3=24)
FORECAST_HORIZON = 1    # 预测目标：未来第1个小时 (3小时后)
TRAIN_SPLIT_RATIO = 0.8 # 训练集比例 (前80%训练, 后20%测试)

# --- 特征列定义 ---
RAW_FEATURE_COLS = ['ws_10'] 
TARGET_COL = 'ws_10'

# ==========================================
# 2. 数据加载与特征工程
# ==========================================

def load_and_preprocess_data(file_path):
    print(f"[Info] 实验 ID (Run ID): {RUN_ID}")
    print(f"[Info] 正在从 {file_path} 加载数据...")
    
    if not os.path.exists(file_path):
        raise ValueError(f"错误: 找不到文件 {file_path}！请检查路径是否正确。")
    
    df = pd.read_csv(file_path)
    
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(by=['point_id', 'datetime']).reset_index(drop=True)
        df['hour'] = df['datetime'].dt.hour
    else:
        raise ValueError("错误: 数据中缺少 'datetime' 列")

    print("[Info] 执行时间周期编码...")
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    print(f"[Success] 数据加载完成，总行数: {len(df)}")
    return df

# ==========================================
# 3. 序列生成与数据集构建 (核心防泄露逻辑)
# ==========================================

def create_dataset(df, window_size, horizon, split_ratio):
    feature_cols = ['ws_10', 'hour_sin', 'hour_cos']
    
    unique_times = df['datetime'].sort_values().unique()
    split_time = unique_times[int(len(unique_times) * split_ratio)]
    
    train_df = df[df['datetime'] <= split_time].copy()
    test_df = df[df['datetime'] > split_time].copy()
    
    print(f"[Info] 训练集截止时间: {split_time}")
    print(f"       训练集样本数 (行): {len(train_df)}")
    print(f"       测试集样本数 (行): {len(test_df)}")

    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols])
    
    def _create_sequences(data_df):
        X, y = [], []
        grouped = data_df.groupby('point_id')
        
        for _, group in grouped:
            group_feats = scaler.transform(group[feature_cols])
            group_target = group[TARGET_COL].values
            
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

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training Loss (MSE)') 
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('Training MAE') 
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(RUN_DIR, 'history.png')
    plt.savefig(save_path)
    print(f"[Output] 训练曲线已保存: {save_path}")
    plt.close()

def plot_predictions(y_true, y_pred, sample_size=200):
    plt.figure(figsize=(14, 6))
    plt.plot(y_true[:sample_size], label='Actual Wind Speed (m/s)', color='blue', alpha=0.6)
    plt.plot(y_pred[:sample_size], label='Predicted Wind Speed (m/s)', color='red', linestyle='--', alpha=0.8)
    plt.title(f'Prediction vs Actual (First {sample_size} steps)')
    plt.xlabel('Time Steps')
    plt.ylabel('Wind Speed (m/s)')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(RUN_DIR, 'pred_compare.png')
    plt.savefig(save_path)
    print(f"[Output] 预测图已保存: {save_path}")
    plt.close()

def save_experiment_config():
    """保存本次实验的超参数配置"""
    config_path = os.path.join(RUN_DIR, 'config.txt')
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(f"Run ID: {RUN_ID}\n") 
        f.write(f"Timestamp: {CURRENT_TIME}\n")
        f.write(f"Model Type: {MODEL_TYPE}\n")
        f.write(f"Window Size: {WINDOW_SIZE}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Dropout Rate: {DROPOUT_RATE}\n")
        f.write(f"Patience: {PATIENCE}\n")
        f.write(f"Data Dir: {DATA_DIR_PATH}\n")
    print(f"[Output] 实验配置已保存: {config_path}")

# --- [新增] 安全计算 MAPE 的辅助函数 ---
def calculate_mape(y_true, y_pred):
    """
    计算 MAPE (Mean Absolute Percentage Error)。
    为了防止真实风速中包含 0.0 导致除以零的错误，对分母加入微小的常数保护。
    """
    y_true_safe = np.maximum(y_true, 1e-8)
    return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

# ==========================================
# 6. 主程序入口
# ==========================================

if __name__ == "__main__":
    os.makedirs(RUN_DIR, exist_ok=True)
    print(f"[Init] 成功创建实验结果文件夹: \n -> {RUN_DIR}")

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
    
    callbacks_list = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
        CSVLogger(os.path.join(RUN_DIR, 'training_log.csv'))
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_list,
        verbose=1
    )

    # 4. [全新修改区] 评估与顶级文献基准对比 
    print("\n" + "="*50)
    print("[Evaluation] 阶段 1: LSTM 模型在测试集上的表现")
    print("="*50)
    
    # 获取预测值，将其扁平化以适配 sklearn 的函数要求
    y_pred = model.predict(X_test, verbose=0).flatten()
    y_test_flat = y_test.flatten()

    # 计算文献要求的四大指标
    lstm_mse = mean_squared_error(y_test_flat, y_pred)
    lstm_rmse = np.sqrt(lstm_mse)
    lstm_mae = mean_absolute_error(y_test_flat, y_pred)
    lstm_mape = calculate_mape(y_test_flat, y_pred)
    lstm_r2 = r2_score(y_test_flat, y_pred)

    print(f"-> LSTM Test RMSE: {lstm_rmse:.4f}")
    print(f"-> LSTM Test MAE:  {lstm_mae:.4f}")
    print(f"-> LSTM Test MAPE: {lstm_mape:.2f}%")
    print(f"-> LSTM Test R²:   {lstm_r2:.4f}")

    print("\n" + "="*50)
    print("[Evaluation] 阶段 2: 计算绝对基准 (惯性预测 Persistence Forecast)")
    print("="*50)
    
    # 构造惯性预测：将真实值向右平移 1 个步长对齐
    y_naive_pred = y_test_flat[:-1]
    y_true_aligned = y_test_flat[1:]

    # 计算基准模型的四大指标
    naive_mse = mean_squared_error(y_true_aligned, y_naive_pred)
    naive_rmse = np.sqrt(naive_mse)
    naive_mae = mean_absolute_error(y_true_aligned, y_naive_pred)
    naive_mape = calculate_mape(y_true_aligned, y_naive_pred)
    naive_r2 = r2_score(y_true_aligned, y_naive_pred)
    
    print(f"-> Persistence Test RMSE: {naive_rmse:.4f}")
    print(f"-> Persistence Test MAE:  {naive_mae:.4f}")
    print(f"-> Persistence Test MAPE: {naive_mape:.2f}%")
    print(f"-> Persistence Test R²:   {naive_r2:.4f}")

    # 计算提升幅度 (Improvement) - 论文中最核心的卖点
    rmse_improvement = (naive_rmse - lstm_rmse) / naive_rmse * 100
    mae_improvement = (naive_mae - lstm_mae) / naive_mae * 100
    
    print("\n" + "="*50)
    print("[Conclusion] 最终评估结论 (论文级别)")
    print("="*50)
    print(f"相比于“惯性基准(Persistence)”，本模型实现了：")
    print(f"   - RMSE 提升: {rmse_improvement:.2f}%")
    print(f"   - MAE  提升: {mae_improvement:.2f}%\n")
    
    if rmse_improvement > 0 and mae_improvement > 0:
        print("🎉 结论: 模型成功击败了基准线，学到了有效的气象物理特征！")
    else:
        print("⚠️ 警告: 模型表现未全面超越基准线，可能存在预测滞后或过拟合。")

    # 保存模型权重
    model_path = os.path.join(RUN_DIR, 'model.h5')
    model.save(model_path)
    
    # 保存参数配置
    save_experiment_config()

    # [全面扩充] 将详细评估矩阵写入 metrics.txt
    metrics_path = os.path.join(RUN_DIR, 'metrics.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("=== Evaluation Metrics (Literature Standard) ===\n\n")
        f.write("[1] LSTM Model Results:\n")
        f.write(f"RMSE: {lstm_rmse:.4f}\n")
        f.write(f"MAE:  {lstm_mae:.4f}\n")
        f.write(f"MAPE: {lstm_mape:.2f}%\n")
        f.write(f"R²:   {lstm_r2:.4f}\n\n")
        
        f.write("[2] Persistence Baseline Results:\n")
        f.write(f"RMSE: {naive_rmse:.4f}\n")
        f.write(f"MAE:  {naive_mae:.4f}\n")
        f.write(f"MAPE: {naive_mape:.2f}%\n")
        f.write(f"R²:   {naive_r2:.4f}\n\n")
        
        f.write("[3] Improvement (%):\n")
        f.write(f"RMSE Improvement: {rmse_improvement:.2f}%\n")
        f.write(f"MAE Improvement:  {mae_improvement:.2f}%\n")

    print(f"[Output] 完整的评估指标矩阵已保存: {metrics_path}")

    # 5. 可视化
    plot_training_history(history)
    # 我们上面已经 predict 过了，可以直接画图
    plot_predictions(y_test_flat, y_pred)

    print(f"\n🎉 Done. 所有结果均已保存在: {RUN_DIR}")