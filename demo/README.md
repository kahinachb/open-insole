# Live demo

This demo reads both insoles from the receiver and estimates the GRFM of each foot in real time. It shows the result in [Rerun](https://rerun.io), with the following views:

- The **insole pressure** view shows the load density of the 15 load cells of each insole, in kPa, with the total load and the recent path of the load centre.
- **One column per foot** shows the following, over the last 10 s:
  - the vertical force Fz, drawn with the sum of the load cells for reference;
  - the shear forces Fx and Fy, in N;
  - the moments Mx, My and Mz, in N·m.

The model is the MLP with Sensor dropout (window of 21 samples, causal). Its components are those of the paper: the GRFM of each foot, in the frame of the foot's marker cluster. One prediction takes about 0.2 ms on one CPU core, and the whole demo runs at 200 Hz on a laptop.

## Setup

```bash
python -m venv .venv-demo
.venv-demo/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv-demo/bin/pip install -r demo/requirements.txt
python tools/download_models.py   # the trained models of the paper, into models/
CHECKPOINT=models/mlp_sensor_dropout_w21/all/fold_P05/models/mlp_test_P05.pt
```

Any checkpoint of `mlp_sensor_dropout_w21` works. The fold of P05 is the median Full CV fold (NRMSE 7.0 %).

## Running a demo

1. Plug the receiver in and switch both insoles on.
2. **Tare.** Put the insoles in their sandals and do not wear them. Leave them still for 3 s:

   ```bash
   .venv-demo/bin/python -m demo.tare --port /dev/ttyACM0
   ```

3. **Live.** The body weight of the wearer is given in N (kg × 9.81):

   ```bash
   .venv-demo/bin/python -m demo.live --port /dev/ttyACM0 --tare-file demo/tare.json \
       --body-weight 650 --checkpoint $CHECKPOINT
   ```

   The Rerun viewer opens with its layout. The session is saved to `demo/sessions/<date>.bin` (raw serial capture) and `.rrd` (Rerun recording). Use `--no-save` to turn this off.

## Replaying a session

Replay a session to make a clean video, or as a fallback if the radio link fails during a live demo:

```bash
# A saved session (same Tare file)
.venv-demo/bin/python -m demo.live --replay demo/sessions/<date>.bin --tare-file demo/tare.json \
    --body-weight 650 --checkpoint $CHECKPOINT

# An acquisition (record_trial.py's dataSemelle.bin), with the Tare file of each foot
.venv-demo/bin/python -m demo.live --replay <trial>/dataSemelle.bin \
    --tare-file <subject>/tare_fx29_gauche.json <subject>/tare_fx29_droit.json \
    --body-weight 744 --checkpoint $CHECKPOINT

# A prepared dataset trial: the force-plate GRFM is drawn under the estimate
# (the processed dataset is released on Zenodo upon publication of the paper)
.venv-demo/bin/python -m demo.live --trial <dataset>/P05_walk.npz \
    --body-weight 744 --checkpoint $CHECKPOINT
```

`--speed 2` replays twice as fast. A saved `.rrd` opens with `rerun demo/sessions/<date>.rrd`.

## Limits

- **Sensor configuration.** The demo assumes the full Sensor configuration. A load cell that fails during a demo is not detected and is passed to the model as is.
- **Packet pairing.** Left and right packets are paired in arrival order. If one insole stops sending, the packets of the other are dropped. The console reports the number of dropped packets every 5 s.
