"""Static feature contract for the planned P3 paired controls."""

import numpy as np
from sklearn.preprocessing import StandardScaler

STATIC_FEATURES = tuple(f'lc_{code}' for code in (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)) + (
    'dsm_mean_m', 'dsm_sd_m', 'dsm_slope_mean_deg')
RS_GROUPS = {'RS_T0': ('T01', False), 'RS_T1': ('T01', True),
             'RS_S0': ('A01', False), 'RS_S1': ('A01', True)}


def fit_static(table, train_grid_ids):
    """Fit on unique training grids, after explicit QA of coverage."""
    indexed = table.set_index('grid_id', verify_integrity=True)
    training = indexed.loc[sorted(set(train_grid_ids)), list(STATIC_FEATURES)]
    if training.empty or not np.isfinite(training.to_numpy(dtype=float)).all():
        raise ValueError('Static training values must be finite and nonempty.')
    return StandardScaler().fit(training)


def static_inputs(windows, mapping, table, scaler, enabled):
    """Attach features by point-to-grid mapping and preserve window order."""
    if not enabled:
        return np.zeros((len(windows), len(STATIC_FEATURES)), dtype=np.float32)
    grids = mapping.set_index('point_id', verify_integrity=True).loc[windows.point_id, 'grid_id']
    values = table.set_index('grid_id', verify_integrity=True).loc[grids, list(STATIC_FEATURES)]
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError('Static inputs must pass coverage QA before joining.')
    return scaler.transform(values).astype(np.float32)


def build_fusion(dynamic_shape):
    """Both RS switches use this architecture; the switch only changes input data."""
    from tensorflow import keras

    dynamic = keras.Input(shape=dynamic_shape, name='dynamic')
    static = keras.Input(shape=(len(STATIC_FEATURES),), name='static')
    wind = keras.layers.Dropout(0.4)(keras.layers.LSTM(64)(dynamic))
    surface = keras.layers.Dense(8, activation='relu')(static)
    merged = keras.layers.Concatenate()([wind, surface])
    output = keras.layers.Dense(1)(keras.layers.Dense(16, activation='relu')(merged))
    return keras.Model(inputs=[dynamic, static], outputs=output)
