# P1c paired date analysis

Three-day circular blocks, 2000 replicates, RNG seed 20261113. All models and both metrics share the same date indices within each fold. These are post hoc window-conditional intervals from one fitted seed, without multiplicity correction.

| Fold | Model | Reference | Metric | Gain % | 95% date interval |
|---|---|---|---|---|---|
| F01 | D_T1 | D_T0 | RMSE | 2.92 | [0.35, 5.06] |
| F01 | D_T1 | D_T0 | MAE | 2.32 | [-0.24, 4.50] |
| F01 | D_S1 | D_S0 | RMSE | 2.28 | [0.28, 4.30] |
| F01 | D_S1 | D_S0 | MAE | 2.58 | [0.95, 4.36] |
| F01 | D_S1 | D_T1 | RMSE | 5.04 | [2.82, 7.09] |
| F01 | D_S1 | D_T1 | MAE | 5.11 | [2.52, 7.74] |
| F01 | D_T0 | Persistence | RMSE | 3.89 | [0.99, 6.68] |
| F01 | D_T0 | Persistence | MAE | 3.29 | [0.51, 6.07] |
| F01 | D_T1 | Persistence | RMSE | 6.69 | [3.58, 9.22] |
| F01 | D_T1 | Persistence | MAE | 5.53 | [2.60, 8.25] |
| F01 | D_S0 | Persistence | RMSE | 9.33 | [7.77, 10.73] |
| F01 | D_S0 | Persistence | MAE | 7.98 | [6.58, 9.27] |
| F01 | D_S1 | Persistence | RMSE | 11.40 | [9.12, 13.64] |
| F01 | D_S1 | Persistence | MAE | 10.35 | [8.15, 12.30] |
| F02 | D_T1 | D_T0 | RMSE | 0.73 | [-0.74, 2.08] |
| F02 | D_T1 | D_T0 | MAE | 0.54 | [-1.23, 2.16] |
| F02 | D_S1 | D_S0 | RMSE | 0.16 | [-1.13, 1.33] |
| F02 | D_S1 | D_S0 | MAE | 0.07 | [-1.57, 1.59] |
| F02 | D_S1 | D_T1 | RMSE | 4.37 | [2.60, 6.10] |
| F02 | D_S1 | D_T1 | MAE | 4.34 | [2.52, 6.17] |
| F02 | D_T0 | Persistence | RMSE | 13.61 | [11.63, 15.21] |
| F02 | D_T0 | Persistence | MAE | 12.31 | [10.08, 14.32] |
| F02 | D_T1 | Persistence | RMSE | 14.24 | [12.05, 16.01] |
| F02 | D_T1 | Persistence | MAE | 12.78 | [9.77, 15.37] |
| F02 | D_S0 | Persistence | RMSE | 17.85 | [16.04, 19.33] |
| F02 | D_S0 | Persistence | MAE | 16.51 | [14.52, 18.23] |
| F02 | D_S1 | Persistence | RMSE | 17.98 | [16.30, 19.57] |
| F02 | D_S1 | Persistence | MAE | 16.57 | [14.54, 18.31] |
| F03 | D_T1 | D_T0 | RMSE | 2.80 | [-0.15, 5.26] |
| F03 | D_T1 | D_T0 | MAE | 1.55 | [-1.63, 4.28] |
| F03 | D_S1 | D_S0 | RMSE | 0.66 | [-1.48, 2.57] |
| F03 | D_S1 | D_S0 | MAE | 0.19 | [-1.67, 1.82] |
| F03 | D_S1 | D_T1 | RMSE | 2.09 | [-1.13, 4.49] |
| F03 | D_S1 | D_T1 | MAE | 1.86 | [-1.93, 4.93] |
| F03 | D_T0 | Persistence | RMSE | 2.25 | [-0.91, 5.20] |
| F03 | D_T0 | Persistence | MAE | 1.93 | [-1.31, 4.66] |
| F03 | D_T1 | Persistence | RMSE | 4.98 | [1.28, 8.69] |
| F03 | D_T1 | Persistence | MAE | 3.45 | [-0.86, 7.65] |
| F03 | D_S0 | Persistence | RMSE | 6.35 | [3.37, 9.19] |
| F03 | D_S0 | Persistence | MAE | 5.06 | [2.17, 7.83] |
| F03 | D_S1 | Persistence | RMSE | 6.97 | [3.70, 9.98] |
| F03 | D_S1 | Persistence | MAE | 5.24 | [1.99, 8.39] |

Validation selection: direction_enabled=False; equal-fold validation gain=-1.7005%. Test scores and date intervals do not change that rule.
