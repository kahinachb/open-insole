# Full CV on the LAAS platform (pfcalcul)

Full CV (#45, ADR 0003) of the two candidate models, one Slurm array task per fold (15 Participants):

| Plan | Model |
|---|---|
| `full_cv_mlp_sensor_dropout_w21.json` | MLP, window 21, Sensor dropout (`exp_v3.7.1_sensor_dropout_mlp_r2`) |
| `full_cv_patch_transformer_v8.json` | Patch Transformer v8, window 101, patch 5, Sensor dropout (`exp_v3.7.1_patch_transformer_v8`) |

Each plan holds the exact arguments of its Fixed-split run, `"split": "full_cv"` and `"folds"`, the 15 test Participants in array order. A task runs `full_cv_fold.py`, which trains its fold with the same engine command as `overnight_queue.py`, in the same results tree. The overfit gate is not repeated, since both models passed it on the Fixed split.

## 1. Copy the dataset (laptop, once)

```bash
rsync -azh --progress -e "ssh -J <login>@ssh.laas.fr" \
    dataset_v3.7.1 \
    <login>@pfcalcul.laas.fr:/datasets/GRF2Kine/insole2grfm/
```

The jobs read `dataset_v3.7.1` by default; set `DATA=...` if the copy lands elsewhere. The script stops before submitting when `$DATA/metadata.json` is missing.

## 2. Pull and submit (pfcalcul)

```bash
ssh -J <login>@ssh.laas.fr <login>@pfcalcul.laas.fr
source envs/lstm/lstm_jcp_env/bin/activate      # as usual; the env is exported to the jobs
cd insole2grfm && git fetch && git checkout spec/30-training-recipe && git pull
python3 -c "import torch, numpy, pandas, matplotlib; print(torch.__version__, torch.cuda.is_available())"
PLAN=slurm/laas/full_cv_mlp_sensor_dropout_w21.json sh slurm/laas/submit_full_cv.sh --gpu 1
PLAN=slurm/laas/full_cv_patch_transformer_v8.json sh slurm/laas/submit_full_cv.sh --gpu 1
```

The results go to `results/full_cv_v3.7.1/` (`RESULTS=...` to change). The Slurm logs go to its `logs/` subfolder, one `.out`/`.err` per task.

## 3. Follow

```bash
squeue -u <login>
tail -f results/full_cv_v3.7.1/logs/patch_transformer_v8_<JOBID>_<TASK>.out
ls results/full_cv_v3.7.1/*/all/fold_*/metadata.json | wc -l   # 30 when done
```

To resume a failed or cancelled array, submit the same command again: trained folds are skipped, and interrupted folds are wiped and retrained.

## 4. Evaluate (pfcalcul, one GPU job, inference only)

Once the 30 folds are trained:

```bash
cd insole2grfm && git pull
sh slurm/laas/submit_evaluation.sh --gpu 1
```

The job runs `evaluate_full_cv.sh`:
- `evaluate_grfm.py` on both models with the full configuration, into `evaluation/full_configuration/`;
- `evaluate_sensor_configurations.py` on each model, into `evaluation/configurations/<model>/`. It scores the full configuration, the minimal one and the best one per Patch count of the #42 ranking (frozen copy: `pareto_mlp_sensor_dropout_w21.json`).

The script stops before submitting unless the 30 folds are trained. The log is `logs/full_cv_evaluation_<JOBID>.out`.

## 5. Bring the results back (laptop)

```bash
rsync -azh --progress -e "ssh -J <login>@ssh.laas.fr" \
    --exclude 'configurations/*/*/fold_*' \
    <login>@pfcalcul.laas.fr:results/full_cv_v3.7.1 \
    results/
```

This brings back the fold dirs (checkpoints and full-configuration `predictions.csv`, used by the cycle figures, about 3.6 GB), the logs and every evaluation table. It leaves on the platform the per-configuration predictions (about 20 GB), which are only an intermediate of `evaluate_sensor_configurations.py`.

## 6. Dedicated models of the chosen Sensor configurations (ADR 0004, #44)

Two MLPs trained on a fixed Sensor configuration, without Sensor dropout, with the same arguments as `full_cv_mlp_sensor_dropout_w21.json` otherwise:

| Plan | Sensor configuration | Channels |
|---|---|---|
| `full_cv_mlp_minimal_28ch_w21.json` | heel_gauges midfoot_acc forefoot_gauges toes_gauges: the nested choice, the same in all 15 outer folds | 28 |
| `full_cv_mlp_minimal_34ch_w21.json` | the same plus toes_acc: within 10 % of the full configuration in all 15 folds | 34 |

```bash
cd insole2grfm && git pull
PLAN=slurm/laas/full_cv_mlp_minimal_28ch_w21.json sh slurm/laas/submit_full_cv.sh --gpu 1
PLAN=slurm/laas/full_cv_mlp_minimal_34ch_w21.json sh slurm/laas/submit_full_cv.sh --gpu 1
ls results/full_cv_v3.7.1/mlp_minimal_*/all/fold_*/metadata.json | wc -l   # 30 when done
```

They land next to the other models in the same results tree, so the rsync of step 5 brings them back.

## 7. Mahmud 2026 (Ke2KeNet) under Full CV (#47, workspace ADR 0005)

Ke2KeNet, ported to PyTorch (`ke2kenet.py`), is retrained offline as published: one model per target (12 per fold), 1 024-sample non-causal segments. Two input variants: `gauges` (the 15 gauges of the predicted foot) and `raw` (78 channels).

```bash
cd insole2grfm && git pull
INPUTS=gauges sh slurm/laas/submit_mahmud.sh --gpu 1
INPUTS=raw sh slurm/laas/submit_mahmud.sh --gpu 1
ls results/full_cv_v3.7.1/mahmud_ke2kenet_*/all/fold_*/predictions.csv | wc -l   # 30 when done
ls results/full_cv_v3.7.1/mahmud_ke2kenet_*/all/fold_*/targets/*.npz | wc -l     # 360 when done
```

A fold trains its 12 targets one after the other. Each finished target is saved in `fold_<P>/targets/`, so resubmitting the same command resumes a stopped or timed-out fold. The 2-day limit is meant to cover the 12 targets. The results land in the same results tree, so the rsync of step 5 brings them back.

## 8. Loadsol-like inputs (#34)

The MLP is trained on Sabbah 2024's hardware, emulated: the 4 loadsol-like zonal gauge sums per foot (`--input-features loadsol`), without Sensor dropout, with the same arguments as `full_cv_mlp_sensor_dropout_w21.json` otherwise. Compared with `sabbah_rf_loadsol`, it tests the method at equal inputs.

| Plan | Inputs |
|---|---|
| `full_cv_mlp_loadsol_w21.json` | 4 zones per foot |
| `full_cv_mlp_loadsol_heel_imu_w21.json` | 4 zones per foot and the heel-Patch IMU (`--loadsol-heel-imu`), Sabbah's inputs |

On pfcalcul:

```bash
PLAN=slurm/laas/full_cv_mlp_loadsol_w21.json sh slurm/laas/submit_full_cv.sh --gpu 1
PLAN=slurm/laas/full_cv_mlp_loadsol_heel_imu_w21.json sh slurm/laas/submit_full_cv.sh --gpu 1
```

Locally, one fold after the other:

```bash
for i in $(seq 0 14); do python3 -u full_cv_fold.py slurm/laas/full_cv_mlp_loadsol_w21.json ../dataset_v3.7.1 --results-root ../results/full_cv_v3.7.1 --fold-index $i --device cuda; done
```

## 9. The same MLP without Sensor dropout (robustness to sensor failures)

The plan `full_cv_mlp_no_dropout_w21.json` is `full_cv_mlp_sensor_dropout_w21.json` without `--sensor-dropout`. Scored under the same lost-Patch configurations, it shows what Sensor dropout buys: `evaluate_sensor_configurations.py --without-sensor-dropout` feeds its lost groups their training mean, which is what the Sensor-dropout model is fed.

```bash
PLAN=slurm/laas/full_cv_mlp_no_dropout_w21.json sh slurm/laas/submit_full_cv.sh --gpu 1
ls results/full_cv_v3.7.1/mlp_no_dropout_w21/all/fold_*/metadata.json | wc -l   # 15 when done
```

## 10. One MLP per target (Mahmud's per-target idea, kept real time)

Mahmud 2026 trains one network per target. Here the MLP recipe is kept (`full_cv_mlp_sensor_dropout_w21.json`), and each model learns only its own targets (`--targets`). That makes 11 models per variant:
- **Fz_L and Fz_R together:** `--vertical-share` splits the body-weight-anchored total between the two feet, so it needs both;
- **the 10 other targets alone,** without `--vertical-share`.

There are two variants, with and without Sensor dropout: 22 plans in `per_target/`, 11 × 15 = 165 array tasks per variant.

**Real time:** the 11 MLPs (6.2 M parameters) run in 1.35 ms per prediction on one CPU thread, about 740 predictions/s against the 200 Hz sensor rate. One MLP takes 0.16 ms.

```bash
cd insole2grfm && git pull
for plan in slurm/laas/per_target/mlp_per_target_dropout_*.json slurm/laas/per_target/mlp_per_target_no_dropout_*.json; do
  PLAN=$plan sh slurm/laas/submit_full_cv.sh --gpu 1
done
ls results/full_cv_v3.7.1/mlp_per_target_*/all/fold_*/metadata.json | wc -l   # 330 when done
```

Once rsynced, merge the 11 runs of a variant into one 12-target run, then score it like any run:

```bash
R=../results/full_cv_v3.7.1
python3 merge_target_runs.py $R/mlp_per_target_dropout_*_w21/all --output-dir $R/mlp_per_target_dropout_w21/all
python3 merge_target_runs.py $R/mlp_per_target_no_dropout_*_w21/all --output-dir $R/mlp_per_target_no_dropout_w21/all
```
