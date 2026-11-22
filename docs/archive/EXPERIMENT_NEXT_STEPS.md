# CERRA repetitions and diagnostic analysis

For the consolidated experiment history, current results and seasonal plan, see
[EXPERIMENT_RECORD_AND_PLAN.md](../experiments/EXPERIMENT_RECORD_AND_PLAN.md).

All commands below are manual Slurm submissions from this project directory.
The existing 70/10/20 time split, 56-step context, four forecast leads, optimizer,
dropout and early stopping settings remain unchanged. No ConvLSTM is introduced.

## 1. Export and analyze the existing 20 models

```bash
sbatch slurm/run_analysis.sh
```

This CPU job replays existing models with their saved scalers and exports missing
`predictions.csv` files in their original run directories. It never calls fit.
Replayed RMSE/MAE and Persistence scores must match the recorded results within
numerical tolerance. Existing prediction files are reused and validated.
Do not run multiple export jobs over the same directories concurrently.

Predictions contain point ID, input start, forecast origin, target time, lead in
hours, actual wind speed, model prediction, Persistence prediction, run ID, seed
and experiment label. Weather values remain m/s.

The report directory is `result/spatial_analysis/analysis_JOBID/`:

- `per_run_metrics.csv`: original point-weighted and unique-grid overall metrics.
- `per_point_metrics.csv`, `per_grid_metrics.csv`: location-specific metrics.
- `per_time_metrics.csv`: unique-grid metrics for each target timestamp.
- `seed_summary.csv`: mean and sample standard deviation across seeds.
- `paired_seed_summary.csv`: paired percentage improvements relative to T01.
- `paired_grid_improvements.csv`, `grid_improvement_coverage.csv`: where improvements occur.
- `coverage.json`: completed and missing seeds; partial arrays are reported explicitly.
- `source_runs.json`, `analysis_notes.txt`: provenance and interpretation.
- `unique_grid_comparison.png`: comparison curves with seed standard deviations.

100 point IDs map to 98 unique center grid cells. The unique-grid view averages
predictions at equivalent grid/time keys and scores each grid once. This changes
evaluation weighting only; it does not remove duplicates from training data.
Targets and Persistence values at shared cells must agree.

## 2. Add four seeds without repeating seed 42

```bash
sbatch slurm/run_repeats.sh
```

The CPU array has 16 tasks, with at most 4 tasks running simultaneously:

| Array task IDs | Experiment | Seeds in ascending task order |
|---|---|---|
| 0–3 | T01 | 43, 44, 45, 46 |
| 4–7 | S01 | 43, 44, 45, 46 |
| 8–11 | S02 | 43, 44, 45, 46 |
| 12–15 | A01 | 43, 44, 45, 46 |

Each task trains all four forecast leads. Total: 64 additional fits. Each task
has a four-hour limit. Logs are `logs/repeats_ARRAYID_TASKID.out/.err`; outputs are
under `runs/ablation/repeats/EXPERIMENT_ARRAYID_TASKID/`. T00 remains a single-seed
supplementary ablation. New training runs save prediction files automatically.
Add `--account=YOUR_ACCOUNT` to submissions if your allocation requires it.

## 3. Analyze again after the complete array

After the first analysis job is finished, submit another analysis job with the
array job ID returned above (replace ARRAY_JOB_ID):

```bash
sbatch --dependency=afterok:ARRAY_JOB_ID slurm/run_analysis.sh
```

If the first analysis is still running, include both IDs separated by a colon:
`--dependency=afterok:ARRAY_JOB_ID:FIRST_ANALYSIS_JOB_ID`.
This scans original and repeated results together, reuses existing predictions,
and writes a new report directory. On partial failures inspect coverage before
rerunning only failed array tasks, e.g. `sbatch --array=5,9 slurm/run_repeats.sh`.
Do not rerun already successful experiment/seed combinations: the analyzer
rejects duplicates to prevent selecting a favorable run after seeing results.

## Interpretation and next protocol version

Compare matched leads and seeds. Positive improvement means lower error than
T01. One-seed standard deviation is undefined, not evidence of zero variability.
Five seeds measure training variability; they do not establish robustness across
seasons or independent weather periods. Grid improvement fractions describe
spatial coverage, not a significance test.

The short effective validation period remains a pilot limitation. Changing
window context or validation boundaries requires a separately named protocol
and rerunning all comparison groups. Keep this repetition round fixed first.
Select future configurations with validation data, not repeated test-set tuning.
Only consider CNN-LSTM/ConvLSTM after reviewing repeat stability, unique-grid
results, temporal error concentration and the near-versus-distant comparison.
