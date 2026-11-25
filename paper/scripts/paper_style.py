"""The font of every figure of the paper: Nimbus Sans (a free Helvetica), maths included."""

import matplotlib

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "TeX Gyre Heros", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Nimbus Sans",
    "mathtext.it": "Nimbus Sans:italic",
    "mathtext.bf": "Nimbus Sans:bold",
    "pdf.fonttype": 42,
}


def apply() -> None:
    matplotlib.rcParams.update(STYLE)
