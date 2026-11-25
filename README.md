# open-insole

Software and trained models of **OpenInsole**, an open-source pair of instrumented insoles that estimates the six-component ground reaction force and moment (6D GRFM) of each foot in real time.

Each insole carries 15 load cells and four IMUs on four rigid patches. A causal multilayer perceptron reads the last 100 ms of both insoles and outputs the three force and three moment components of each foot; one prediction takes 0.2 ms on one CPU thread, far below the 5 ms between two samples.

<p align="center">
  <a href="https://kahinachb.github.io/open-insole-website/"><img src="docs/media/outdoor.webp" width="860" alt="Outside the laboratory: the insoles alone estimate the force and moment under each foot, with the pressure maps and the six components of each foot."></a>
  <br><em>Outside the laboratory, the insoles alone estimate the GRFM of each foot (left foot in blue, right foot in orange).</em>
</p>

Project page and video: https://kahinachb.github.io/open-insole-website/

This repository holds the code that records the insoles, turns the recordings into a dataset, trains and evaluates the models, regenerates every number, table and figure of the paper, and runs the live demo. The hardware (PCBs, 3D-printed parts, bills of materials, firmware and assembly guides) is in a separate repository, `open-insole-hardware`, and the dataset is on Zenodo; both are released upon publication of the paper.

## Layout

| Folder | Content |
|---|---|
| `acquisition/` | Recording of a trial (`record_trial.py`), decoding of the insole stream, synchronisation with the force plates and motion capture |
| `calibration_imu/`, `calibration_magnetometer/` | IMU and magnetometer calibrations of both insoles |
| `insole2grfm/` | Dataset builder, training engine, Full CV, evaluation, competitor reproductions (Sabbah 2024, Mahmud 2026), insole characterisation, visualisation |
| `insole2grfm/slurm/laas/` | The plan of every Full CV run of the paper, and the scripts that ran them on a Slurm cluster |
| `paper/` | The evaluation tables of the paper and `make_artefacts.py`, which regenerates its numbers, tables and figures |
| `demo/` | Live demo: per-foot GRFM estimation and pressure maps in real time, in [Rerun](https://rerun.io) |
| `tools/` | `download_models.py`, the trained models of the paper |
| `docs/` | `pipeline.md` (from raw recordings to a trained model), `guides/recording.md` (recording with the insoles) |

## Installation

Python 3.12:

```bash
python -m venv .venv
.venv/bin/pip install -r insole2grfm/requirements.txt -r acquisition/requirements_synchronization.txt -r demo/requirements.txt
make test PYTHON=.venv/bin/python
```

## Reproducing the paper

Every number, table and figure of the paper is generated from the evaluation tables in `paper/evaluation/`; no number is typed by hand.

```bash
make paper-artefacts PYTHON=.venv/bin/python   # writes paper/artefacts/: numbers.tex, tables, figures
```

Retraining the models needs the dataset (released upon publication): see [`docs/pipeline.md`](docs/pipeline.md).

## Trained models

The models of the paper are assets of the `v1.0` release. Each run is a leave-one-participant-out cross-validation (Full CV): 15 models, each trained without the participant it is named after.

```bash
python tools/download_models.py        # the main model: MLP with sensor dropout, 78 channels
python tools/download_models.py --all  # also the 28- and 34-channel MLPs, the per-component MLPs and the patch Transformer
```

The models are unpacked into `models/`.

The competitor reproductions (Sabbah 2024, Mahmud 2026) are released as code and evaluation tables; their weights are not distributed.

## Live demo

See [`demo/README.md`](demo/README.md): tare the insoles, then estimate and display the GRFM of each foot live, or replay a recorded session.

## Participants

Participants are named by their Participant code (`P01`…`P15`); a loaded Subject adds `b` for backpack (`P01b`). The evaluation tables and the models are pseudonymised; the correspondence between codes and identities is not released.

## Credits

The mechanical design and the ESP-NOW link were developed with Edik Planson (LAAS-CNRS).

## License

BSD 2-Clause, see [`LICENSE`](LICENSE).
