# Recording with the insoles

The boards, their firmware and the assembly guides are in the hardware repository, `open-insole-hardware`, released upon publication of the paper.

1. Switch on both insoles (in any order).
2. Plug the reception ESP32 into the PC, with a short cable if possible.
3. Start the recording (below).
4. Data should flow within 5 s.
5. Stop with `Ctrl+A` then `Ctrl+Q`.
6. Decode the recording to CSV (below).

## Check an insole

Before a session, or after flashing, plot the Gauge forces live:

```bash
python -m venv .venv-insole
.venv-insole/bin/pip install -r acquisition/requirements_live.txt
cd acquisition && ../.venv-insole/bin/python live_plot_forces.py --port /dev/ttyACM0
```

Plug in the receiver and switch the insole on. You should then see:
- one panel per Patch of the insole switched on; the window title counts the packets, and the rejected bytes should stay near 0;
- the Gauges rising when you press each zone, from the heel to the toes;
- one Gauge of the toe Patch that stays at 0: that Patch carries three Gauges.

Use `--view total` to see the total force of each insole instead. Without `--tare-file` (a JSON file written by `insole_data/calculerTareFX29.py`), every Gauge uses a default Tare of 1000 counts, so the forces are only approximate.

## Commands

Record with [picocom](https://github.com/npat-efault/picocom):

```bash
picocom -b 2000000 /dev/ttyACM0 --log dataSemelle.bin
```

- `2000000` is the baud rate set in the receiver firmware for the ESP32-C3 and C6 (921600 for the ESP WROOM 32).
- `/dev/ttyACM0` is the receiver's port; `ls /dev/ttyACM* /dev/ttyUSB*` after plugging it in lists the candidates.
- `dataSemelle.bin` is the binary recording; name it as you like.

Decode:

```bash
python acquisition/insole_data/decodageToCSV.py dataSemelle.bin dataSemelle.csv
```

## Recording a full trial

The dataset was recorded with `acquisition/record_trial.py`, which writes the files the pipeline expects in one trial directory: the insole stream, two camera videos, and the timestamps that align them.

```bash
pip install -r acquisition/requirements_recording.txt
python acquisition/record_trial.py --output-dir manip/P01/squat --serial-port /dev/ttyACM0 --cameras /dev/video2 /dev/video4
```

- The insoles are recorded continuously into `dataSemelle.bin`.
- The switch wired to D7 of the receiver marks every packet with Signal = 0 or 1. The videos are recorded only while it is on, and its transitions are logged in `trigger_log.csv` with the PC clock, as are the video frames in `cam1_manip_timestamps.csv` and `cam2_manip_timestamps.csv`.
- Press Enter to stop: the script then prints the packet losses, duplicates and left/right balance of the insoles, and the effective frame rate of each camera.

The force plates and the motion capture are recorded by their own software. For the full pipeline, with force plates and motion capture, see [`../pipeline.md`](../pipeline.md).

## Troubleshooting

- **No data:** check the baud rate of the receiver firmware, and the battery charge.
- **Pauses in the data:** lower the acquisition rate.
- **Frames of one Patch missing in the CSV** (for example the left toes): a solder joint between Patches may have broken.
