"""CERRA point-wise wind forecasting; launch training through Slurm.

pred_len is the forecast lead in rows, not an output sequence length.
Uses chronological 70/10/20 boundaries; optional V2 keeps prior observed context.
Persistence uses the last observed target within each point's input window.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from temporal_folds import FoldSpec, expected_fold_counts, load_manifest, project_path, select_fold_panel


DEFAULT_DATA_PATH = './data/CERRA/cerra_100_points_200km_summer_roi_2024.csv'
TARGET_COL = 'ws_10'
TRAIN_SPLIT_RATIO = 0.7
VAL_SPLIT_RATIO = 0.1
PROTOCOLS = {
    'v1': 'chronological_70_10_20_independent_test',
    'v2': 'origin_70_10_20_context_embargo_v2',
    'v2_fixed': 'origin_fixed_dates_context_embargo_v2_fixed',
}


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('Value must be a positive integer.')
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_type', choices=['lstm', 'bilstm'], default='lstm')
    parser.add_argument('--input_len', type=positive_int, default=56)
    parser.add_argument('--pred_len', type=positive_int, default=1,
                        help='Forecast lead in time steps; outputs one wind speed.')
    parser.add_argument('--features', nargs='+', default=['ws_10', 'hour_sin', 'hour_cos'],
                        help='Ordered CSV/engineered column names, space or comma separated.')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_dir', type=Path, default=Path('./runs'),
                        help='Root for results.csv and individual run directories.')
    parser.add_argument('--data_path', type=Path, default=Path(DEFAULT_DATA_PATH))
    parser.add_argument('--batch_size', type=positive_int, default=64)
    parser.add_argument('--epochs', type=positive_int, default=100)
    parser.add_argument('--patience', type=positive_int, default=10)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--experiment', default='custom', help='Experiment label recorded in outputs.')
    parser.add_argument('--protocol', choices=list(PROTOCOLS), default='v1',
                        help='v2 borrows past context and separates forecast origins/labels.')
    parser.add_argument('--fold_manifest', type=project_path)
    parser.add_argument('--fold_id', help='Fixed-date fold ID; required for v2_fixed.')
    parser.add_argument('--train_start', help='UTC, inclusive; alternative to fold_manifest.')
    parser.add_argument('--validation_start', help='First validation forecast origin, UTC.')
    parser.add_argument('--test_start', help='First test forecast origin, UTC.')
    parser.add_argument('--test_end', help='Exclusive target boundary, UTC.')
    parser.add_argument('--allow_confirmation', action='store_true',
                        help='Explicitly enable a reserved confirmation fold after method freezing.')
    args = parser.parse_args(argv)
    args.features = [column.strip() for item in args.features
                     for column in item.split(',') if column.strip()]
    if not args.features or len(set(args.features)) != len(args.features):
        parser.error('--features must contain unique column names.')
    if not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('--lr must be finite and positive.')
    if not 0 <= args.dropout < 1:
        parser.error('--dropout must be in [0, 1).')
    dates = [args.train_start, args.validation_start, args.test_start, args.test_end]
    args.fold_spec = None
    args.fold_manifest_sha256 = None
    try:
        if args.protocol == 'v2_fixed':
            if not args.fold_id:
                parser.error('--protocol v2_fixed requires --fold_id.')
            if args.fold_manifest:
                if any(value is not None for value in dates):
                    parser.error('Choose fold_manifest or explicit dates, not both.')
                _, folds, args.fold_manifest_sha256 = load_manifest(args.fold_manifest)
                if args.fold_id not in folds:
                    parser.error(f'Unknown fold_id: {args.fold_id}')
                fold = folds[args.fold_id]
            else:
                if not all(value is not None for value in dates):
                    parser.error('Fixed-date training requires a manifest or all four date arguments.')
                fold = FoldSpec(args.fold_id, *dates)
            if fold.role == 'locked_confirmation' and not args.allow_confirmation:
                parser.error('This fold is reserved; use --allow_confirmation only after freezing the method.')
            args.fold_spec = fold.as_dict()
        elif args.fold_manifest or args.fold_id or any(value is not None for value in dates):
            parser.error('Fixed-date arguments require --protocol v2_fixed.')
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    return args


def load_and_preprocess_data(file_path, feature_cols):
    """Keep the source CSV, sorting and hourly encoding logic unchanged."""
    df = pd.read_csv(file_path)
    required = {'datetime', 'point_id', TARGET_COL}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    df['datetime'] = pd.to_datetime(df['datetime'], errors='raise')
    if df[['datetime', 'point_id']].isna().any().any():
        raise ValueError('datetime/point_id contains missing values.')
    df = df.sort_values(['point_id', 'datetime']).reset_index(drop=True)
    df['hour'] = df['datetime'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    missing = set(feature_cols).difference(df.columns)
    if missing:
        raise ValueError(f'Missing feature columns: {sorted(missing)}')
    if {'datetime', 'point_id'}.intersection(feature_cols):
        raise ValueError('datetime and point_id are identifiers, not input features.')
    columns = list(dict.fromkeys(feature_cols + [TARGET_COL]))
    values = df[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Features/target contain missing or non-finite values.')
    if (df[TARGET_COL] < 0).any():
        raise ValueError('ws_10 contains negative wind speeds.')
    if df.duplicated(['point_id', 'datetime']).any():
        raise ValueError('Duplicate point_id/datetime pairs would invalidate time steps.')
    intervals = df.groupby('point_id')['datetime'].diff().dropna().unique()
    if len(intervals) != 1 or intervals[0] != pd.Timedelta(hours=3):
        raise ValueError(f'Expected regular 3-hour sampling; found intervals: {intervals}')
    print(f'[Data] {len(df)} rows; features={feature_cols}; '
          'ws_10 assumed m/s (CSV carries no unit metadata).')
    return df


def split_dataframe(df, split_ratio=TRAIN_SPLIT_RATIO, val_ratio=VAL_SPLIT_RATIO, fold=None):
    """Original chronological partitions, with no context borrowed across boundaries."""
    if fold is not None:
        df = select_fold_panel(df, fold)
        train_df = df[df.datetime < pd.Timestamp(fold.validation_start)].copy()
        val_df = df[(df.datetime >= pd.Timestamp(fold.validation_start))
                    & (df.datetime < pd.Timestamp(fold.test_start))].copy()
        test_df = df[df.datetime >= pd.Timestamp(fold.test_start)].copy()
        return train_df, val_df, test_df, {
            'fold_id': fold.fold_id, 'fold_role': fold.role,
            'train_start': fold.train_start, 'validation_start': fold.validation_start,
            'test_start': fold.test_start, 'test_end_exclusive': fold.test_end_exclusive}
    unique_times = df['datetime'].sort_values().unique()
    if not (0 < split_ratio < 1 and 0 < val_ratio < 1 - split_ratio):
        raise ValueError('Train/validation ratios must be positive and sum to less than 1.')
    train_end = int(len(unique_times) * split_ratio)
    val_end = train_end + int(len(unique_times) * val_ratio)
    if not 0 < train_end < val_end < len(unique_times):
        raise ValueError('Need enough distinct timestamps for three time partitions.')
    val_start_time = unique_times[train_end]
    test_start_time = unique_times[val_end]
    train_df = df[df['datetime'] < val_start_time].copy()
    val_df = df[(df['datetime'] >= val_start_time)
                & (df['datetime'] < test_start_time)].copy()
    test_df = df[df['datetime'] >= test_start_time].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError('Time split produced an empty partition.')
    split_info = {'validation_start': str(val_start_time),
                  'test_start': str(test_start_time)}
    return train_df, val_df, test_df, split_info


def create_sequences(data_df, window_size, horizon, feature_cols, scaler, partition,
                     origin_start=None, target_before=None):
    """Preserve sample order and labels; retain identifiers for every window."""
    X, y, persistence, records = [], [], [], []
    for point_id, group in data_df.groupby('point_id'):
        group_feats = scaler.transform(group[feature_cols])
        group_target = group[TARGET_COL].values
        times = group['datetime'].to_numpy()
        max_idx = len(group) - window_size - horizon + 1
        for i in range(max_idx):
            target_index = i + window_size + horizon - 1
            origin = pd.Timestamp(times[i + window_size - 1])
            target = pd.Timestamp(times[target_index])
            if origin_start is not None and origin < pd.Timestamp(origin_start):
                continue
            if target_before is not None and target >= pd.Timestamp(target_before):
                continue
            X.append(group_feats[i:i + window_size])
            y.append(group_target[target_index])
            persistence.append(group_target[i + window_size - 1])
            records.append((point_id, times[i], times[i + window_size - 1], times[target_index]))
    if not X:
        raise ValueError(f'No windows in {partition}; reduce input_len/pred_len '
                         'or provide more data per point.')
    metadata = pd.DataFrame(records, columns=[
        'point_id', 'input_start', 'forecast_origin', 'target_time'])
    return np.array(X), np.array(y), np.array(persistence), metadata


def build_test_sequences(df, window_size, horizon, feature_cols, scaler, split_info, protocol='v1'):
    """Reconstruct test windows with saved scaling for either protocol."""
    if protocol == 'v1':
        data = df[df['datetime'] >= pd.Timestamp(split_info['test_start'])]
        return create_sequences(data, window_size, horizon, feature_cols, scaler, 'test')
    if protocol == 'v2':
        return create_sequences(df, window_size, horizon, feature_cols, scaler, 'test',
                                origin_start=split_info['test_start'])
    if protocol == 'v2_fixed':
        data = df[(df.datetime >= pd.Timestamp(split_info['train_start']))
                  & (df.datetime < pd.Timestamp(split_info['test_end_exclusive']))]
        return create_sequences(data, window_size, horizon, feature_cols, scaler, 'fixed-date test',
                                origin_start=split_info['test_start'],
                                target_before=split_info['test_end_exclusive'])
    raise ValueError(f'Unknown protocol: {protocol}')


def prediction_frame(metadata, y_true, y_pred, persistence, run_id, seed, experiment, fold_id='pilot'):
    """Create an auditable one-row-per-test-window prediction table."""
    arrays = [np.asarray(values).reshape(-1) for values in (y_true, y_pred, persistence)]
    if any(len(values) != len(metadata) or not np.isfinite(values).all() for values in arrays):
        raise ValueError('Prediction arrays must be finite and aligned with metadata.')
    frame = metadata.copy()
    frame['y_true'], frame['y_pred'], frame['persistence_pred'] = arrays
    frame['run_id'] = run_id
    frame['seed'] = seed
    frame['experiment'] = experiment
    frame['fold_id'] = fold_id
    frame['lead_hours'] = (pd.to_datetime(frame['target_time'])
                           - pd.to_datetime(frame['forecast_origin'])).dt.total_seconds() / 3600
    return frame


def create_dataset(df, window_size, horizon, feature_cols,
                   split_ratio=TRAIN_SPLIT_RATIO, val_ratio=VAL_SPLIT_RATIO,
                   include_metadata=False, protocol='v1', fold=None):
    """V1 partition-local windows; V2 past context with label boundary embargoes."""
    if protocol not in PROTOCOLS:
        raise ValueError(f'Unknown protocol: {protocol}')
    if (protocol == 'v2_fixed') != (fold is not None):
        raise ValueError('v2_fixed requires a fold, and other protocols must not receive a fold.')
    if fold is not None:
        df = select_fold_panel(df, fold)
    train_df, val_df, test_df, split_info = split_dataframe(df, split_ratio, val_ratio, fold)
    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols])
    X_train, y_train, _, _ = create_sequences(
        train_df, window_size, horizon, feature_cols, scaler, 'training')
    if protocol == 'v1':
        X_val, y_val, _, val_metadata = create_sequences(
            val_df, window_size, horizon, feature_cols, scaler, 'validation')
    else:
        X_val, y_val, _, val_metadata = create_sequences(
            df, window_size, horizon, feature_cols, scaler, 'validation',
            origin_start=split_info['validation_start'], target_before=split_info['test_start'])
    X_test, y_test, persistence, test_metadata = build_test_sequences(
        df, window_size, horizon, feature_cols, scaler, split_info, protocol)
    if fold is not None:
        expected = expected_fold_counts(fold, window_size, horizon, train_df.point_id.nunique())
        if (len(y_train), len(y_val), len(y_test)) != tuple(expected.values()):
            raise ValueError('Generated samples do not match fixed-fold boundary counts.')
    for name, metadata in [('validation', val_metadata), ('test', test_metadata)]:
        for column in ('forecast_origin', 'target_time'):
            split_info[f'{name}_{column}_min'] = str(metadata[column].min())
            split_info[f'{name}_{column}_max'] = str(metadata[column].max())
    print(f'[Data] {split_info}; X_train={X_train.shape}; '
          f'X_val={X_val.shape}; X_test={X_test.shape}')
    result = (X_train, y_train, X_val, y_val, X_test, y_test,
              persistence, scaler, split_info)
    return result + (test_metadata,) if include_metadata else result


def build_model(input_shape, args):
    from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    recurrent = LSTM(64, return_sequences=False)
    if args.model_type == 'bilstm':
        recurrent = Bidirectional(recurrent)
    model = Sequential([Input(shape=input_shape), recurrent,
                        Dropout(args.dropout), Dense(1)])
    model.compile(loss='mse', optimizer=Adam(learning_rate=args.lr), metrics=['mae'])
    return model


def append_results(path, row):
    """Serialize concurrent Slurm writers, including CSV header creation."""
    with path.open('a+', newline='', encoding='utf-8') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        if header is not None and any(key not in header for key in row):
            # Upgrade older result tables without discarding historical experiments.
            previous_rows = list(reader)
            header = header + [key for key in row if key not in header]
            handle.seek(0)
            handle.truncate()
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            writer.writerows(previous_rows)
        handle.seek(0, 2)
        writer = csv.DictWriter(handle, fieldnames=header or list(row))
        if header is None:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main(argv=None):
    args = parse_args(argv)
    import tensorflow as tf
    from tensorflow.keras.callbacks import CSVLogger, EarlyStopping

    tf.keras.utils.set_random_seed(args.seed)
    run_id = f"{args.model_type}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}"
    run_dir = args.save_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    if args.fold_manifest:
        raw_manifest = args.fold_manifest.read_bytes()
        if hashlib.sha256(raw_manifest).hexdigest() != args.fold_manifest_sha256:
            raise ValueError('Fold manifest changed after argument parsing.')
        (run_dir / 'fold_manifest.json').write_bytes(raw_manifest)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(run_id=run_id, target=TARGET_COL, units='m/s (assumed)',
                  train_split_ratio=TRAIN_SPLIT_RATIO if args.fold_spec is None else None,
                  val_split_ratio=VAL_SPLIT_RATIO if args.fold_spec is None else None,
                  evaluation_protocol=PROTOCOLS[args.protocol],
                  tensorflow_version=tf.__version__)
    (run_dir / 'config.json').write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    df = load_and_preprocess_data(args.data_path, args.features)
    (X_train, y_train, X_val, y_val, X_test, y_test,
     persistence, scaler, split_info, test_metadata) = create_dataset(
         df, args.input_len, args.pred_len, args.features, include_metadata=True,
         protocol=args.protocol, fold=FoldSpec.from_dict(args.fold_spec) if args.fold_spec else None)
    config.update(split_info)
    (run_dir / 'config.json').write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    model = build_model(X_train.shape[1:], args)
    model.summary()
    print('[Protocol] Early stopping uses validation only; test is reserved for evaluation.')
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=args.epochs, batch_size=args.batch_size,
        callbacks=[EarlyStopping(monitor='val_loss', patience=args.patience,
                                 restore_best_weights=True),
                   CSVLogger(str(run_dir / 'training_log.csv'))], verbose=1)
    y_pred = model.predict(X_test, batch_size=args.batch_size, verbose=0).reshape(-1)
    predictions = prediction_frame(test_metadata, y_test, y_pred, persistence,
                                   run_id, args.seed, args.experiment, args.fold_id or 'pilot')
    predictions.to_csv(run_dir / 'predictions.csv', index=False)
    metrics = {'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred))),
               'MAE': float(mean_absolute_error(y_test, y_pred)),
               'persistence_RMSE': float(np.sqrt(mean_squared_error(y_test, persistence))),
               'persistence_MAE': float(mean_absolute_error(y_test, persistence))}
    model_path = run_dir / 'model.keras'
    model.save(str(model_path))
    with (run_dir / 'scaler.pkl').open('wb') as handle:
        pickle.dump(scaler, handle)
    (run_dir / 'metrics.json').write_text(
        json.dumps(metrics, indent=2), encoding='utf-8')
    row = {
        'run_id': run_id, 'fold_id': args.fold_id or 'pilot',
        'fold_manifest': str(args.fold_manifest) if args.fold_manifest else None,
        'fold_manifest_sha256': args.fold_manifest_sha256,
        'experiment': args.experiment, 'model_type': args.model_type,
        'input_len': args.input_len, 'pred_len': args.pred_len,
        'features': json.dumps(args.features), 'lr': args.lr,
        'batch_size': args.batch_size, 'epochs': args.epochs,
        'epochs_run': len(history.history['loss']), 'patience': args.patience,
        'dropout': args.dropout, 'seed': args.seed, 'hidden_units': 64,
        'train_split_ratio': config['train_split_ratio'], 'val_split_ratio': config['val_split_ratio'],
        **split_info,
        'train_samples': len(y_train), 'val_samples': len(y_val),
        'test_samples': len(y_test),
        'data_path': str(args.data_path.resolve()),
        'save_dir': str(args.save_dir.resolve()),
        'model_path': str(model_path.resolve()),
        'evaluation_protocol': config['evaluation_protocol'], **metrics,
    }
    append_results(args.save_dir / 'results.csv', row)
    print(f'[Result] {metrics}; artifacts: {run_dir}; '
          f'summary: {args.save_dir / "results.csv"}')


if __name__ == '__main__':
    main()
