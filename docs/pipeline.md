# From raw recordings to a trained model

The pipeline has four stages. Force plates and motion capture are used only to build training targets: they are never model inputs and are not needed at deployment.

| Stage | Command | Output |
|---|---|---|
| 1. Synchronisation | `acquisition/process_manip_v2.py` | `manip/<Subject>/<trial>/synchronized_v2/` |
| 2. Dataset | `insole2grfm/prepare_npy_joint_masked_v3.py` | one NPZ per Trial or Phase, plus `metadata.json` |
| 3. Training (Full CV) | `insole2grfm/full_cv_fold.py` + a plan of `insole2grfm/slurm/laas/` | one model and `predictions.csv` per fold |
| 4. Evaluation | `insole2grfm/evaluate_grfm.py` | `summary.csv`, `per_participant.csv`, … |

`insole2grfm/run_pipeline_v2.py` runs stages 1 and 2 in a row.

## Layout

Commands are run from the repository root, which holds:

- `manip/<Subject>/<trial>/`: the raw recordings of each Subject (insole `.bin`, QTM `.c3d`, the two videos with their timestamps, `trigger_log.csv`), and `manip/<Subject>/tare/`: the tare the Subject's trials are processed with (`tare_fx29_gauche/droit.bin` and `.json`). They are distributed separately, in the dataset of the paper (faces blurred in the videos).
- `calibration_imu/` and `calibration_magnetometer/`: the accelerometer, gyroscope and magnetometer calibrations of both insoles.
- `acquisition/`: decoding and synchronisation (stage 1). Each Subject's own `tare/` folder is used when present; otherwise `acquisition/tare_assignments.json` says which tare recording the Subject uses. It also records where each tare of the dataset comes from, since several Subjects share one tare recording.
- `insole2grfm/`: dataset builder, training engine, evaluation, competitors, characterisation.

Subjects are named by their Participant code: `P01` unloaded, `P01b` loaded (backpack).

## 1. Synchronisation

```bash
python acquisition/process_manip_v2.py manip --dry-run          # list the selection, write nothing
python acquisition/process_manip_v2.py manip                    # whole campaign
python acquisition/process_manip_v2.py manip --subject P12 --trial lateral --force   # one trial, rebuilt
```

For each trial, `synchronized_v2/` holds:

- the tared insole forces and the calibrated IMUs at the insoles' native rate (about 199.4 Hz);
- the force plates and markers of the C3D, on the QTM clock;
- the global plate forces, moments and centre of pressure, computed by the `ezc3d` plate filter;
- `rapport_synchronisation.json` (synchronisation checks, including `platform_cop_quality`) and the per-frame time map of each video.

Rules:

- any packet loss excludes a continuous task; in walking, a loss is tolerated only away from plate contacts;
- the left and right time offsets are estimated separately;
- the tare, accelerometer, gyroscope and magnetometer calibrations are mandatory, and their SHA-256 is recorded in the report.

## 2. Dataset

```bash
python insole2grfm/extract_static_plate_fz.py manip --output insole2grfm/static_plate_fz.csv   # needs ezc3d
python insole2grfm/prepare_npy_joint_masked_v3.py manip \
    --dataset-version 3.7.1 --static-plate-fz insole2grfm/static_plate_fz.csv \
    --output-dir dataset_v3.7.1
```

The dataset follows the insoles' native clock. Left and right packets are paired once on their side-corrected QTM time, without interpolation (5 ms tolerance). Each sample holds:

- **78 inputs**: per foot, 15 Gauges and 4 × (3 accelerometer + 3 gyroscope) channels;
- **12 targets**: the GRFM of each foot, in that foot's frame, with moments transported to the origin of its marker cluster (N·m);
- the **foot state** of each foot (`state_L/R`: 0 Masked, 1 Measured, 2 Airborne) and the **label source** (`label_source_L/R`: 0 none, 1 plate, 2 foot height, 3 protocol window).

Labelling (full rules in `metadata.json["labelling_rules"]`):

- **Measured**: the foot is on a force plate. Each contact is assigned to a foot by the distance between centre of pressure, plate centre and marker cluster, with temporal continuity; a reassignment beyond 250 mm is refused. A sample whose plate-to-foot distance exceeds 300 mm is masked when the lever arm also exceeds 200 mm or the force is below 5 N.
- **Airborne** (target zero): at the same time, the foot's plate is empty, its Foot height is valid and above 15 mm, its markers are reliable (toe marker visible, or at least 3 calibrated markers), and its insole's 15 Gauges sum to at most 120 N; for at least 50 ms. The insole can only veto an Airborne label, never create one.
- **Masked**: anything else, such as a foot loaded off the plates in walking. Masked samples are excluded from training and evaluation.
- In the protocol windows of the continuous tasks (balance, squat, hula hoop, lateral double-support), no sample is Airborne, and low loads are filled in as Measured.
- Trials listed in `lateral_split_trials.json` are split offline into a lateral double-support Phase and a lateral single-support Phase, where the plates show the switch (15 N unloading threshold).

**Body weight.** Each Participant's body weight (without backpack, shared by both Subjects) is the median total plate Fz over the quiet standing of the unloaded static trial. Fallbacks: the loaded static trial minus the nominal backpack (3 kg women, 6 kg men, `participant_sex.json`), then the unloaded Subject's balance trials. Values and sources are in `metadata.json["body_weights"]`; `--refresh-body-weights DATASET_DIR` rewrites them without rebuilding the dataset.

## 3. Training under Full CV

Each fold tests one Participant (both Subjects), early-stops on the next two Participants in alphabetical order, and trains on the other twelve. One plan file fixes the engine arguments of a run; `full_cv_fold.py` runs one fold of it:

```bash
cd insole2grfm
python full_cv_fold.py slurm/laas/full_cv_mlp_sensor_dropout_w21.json ../dataset_v3.7.1 \
    --results-root ../results/full_cv_v3.7.1 --fold-index 0 --device cuda
```

`slurm/laas/README.md` runs the 15 folds as a Slurm array on a cluster.

## 4. Evaluation

```bash
python evaluate_grfm.py --run mlp=../results/full_cv_v3.7.1/mlp_sensor_dropout_w21 \
    --dataset-dir ../dataset_v3.7.1 --output-dir ../results/evaluation/mlp
```

Metrics are computed on Measured samples: NRMSE = RMSE / (p99.5 − p0.5) of the target, per Participant, then mean ± SD over the 15 Participants; forces are also reported in % body weight.
