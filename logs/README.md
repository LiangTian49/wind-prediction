# Slurm logs

All project Slurm scripts write standard output (`.out`) and standard error
(`.err`) here. Existing log files were moved here without changing their names
or contents; `migration_manifest.json` records their previous relative paths
and sizes. If a historical log originated in a subdirectory, that relative
subdirectory is preserved inside logs/.

Patterns:

- `ablation_JOBID.out/.err`
- `extract_JOBID.out/.err`
- `analysis_JOBID.out/.err`
- `repeats_ARRAYID_TASKID.out/.err`
- `v2_ARRAYID_TASKID.out/.err`
- `year_data_JOBID.out/.err`
- `folds_ARRAYID_TASKID.out/.err`
- `fold_repeats_ARRAYID_TASKID.out/.err`
- `confirmation_ARRAYID_TASKID.out/.err`
- `date_blocks_JOBID.out/.err`
- `p0_JOBID.out/.err`
- `p1_ARRAYID_TASKID.out/.err`
- `p1_dates_JOBID.out/.err`
- `direction_JOBID.out/.err`
- `p0b_JOBID.out/.err`
- `p1c_ARRAYID_TASKID.out/.err`
- `p1c_dates_JOBID.out/.err`
- `p2_download_JOBID.out/.err`
- `p2_static_JOBID.out/.err`
- `p3_ARRAYID_TASKID.out/.err`
- `p3b_ARRAYID_TASKID.out/.err`
- `p3b_dates_JOBID.out/.err`
- `newyear_audit_JOBID.out/.err`
- `history_controls_ARRAYID_TASKID.out/.err`
- `horizon_controls_ARRAYID_TASKID.out/.err`
- `temporal_dates_JOBID.out/.err`
- `paper_figures_JOBID.out/.err`

The logs/ directory must exist **before** sbatch submission: Slurm opens these
files before the job script body runs. This README preserves the directory in
version control. Future batch scripts should use absolute output/error paths
inside `/p/project1/sdlrs/tian1/wind_prediction/logs/`.

Example monitoring:

```bash
tail -f logs/v2_912074_1.out
tail -n 80 logs/v2_912074_1.err
```
