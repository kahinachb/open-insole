"""Shared names of the GRFM prediction problem (one source for every module and test).

- ``COMPONENTS`` (Fx..Mz) per foot, ``FEET`` (L, R) and ``TARGETS``, the 12
  targets in the prepared dataset's order (all left-foot components, then the
  right foot's): ``Fx_L, Fy_L, Fz_L, Mx_L, My_L, Mz_L, Fx_R, ..., Mz_R``.
- Foot states of the prepared dataset (``state_L`` / ``state_R``):
  ``STATE_MASKED`` 0 (unknown, never in the loss nor the metrics),
  ``STATE_MEASURED`` 1 (on a force platform), ``STATE_AIRBORNE`` 2 (in the air,
  reference forced to zero).
- Trial names start with the Subject: ``P01b_squat`` -> ``P01b``.
- ``BODY_WEIGHT_FEATURE``: name of the optional body-weight input column
  (engine ``--body-weight-input``, N, constant per Subject).
  ``BODY_WEIGHT_MISSING_FEATURE``: the "body weight missing" flag column
  (engine ``--body-weight-masking``): 1 where the body weight is replaced by
  the default value, else 0.
- Label source per sample and foot (NPZ ``label_source_L/R``, int8):
  ``LABEL_SOURCE_CODES`` code -> name, written by the builder to metadata
  ``labelling_rules.label_source_codes``; readers use the dataset's own names and
  fall back to this map when a dataset does not record them.
  ``MEASURED_LABEL_SOURCES`` / ``AIRBORNE_LABEL_SOURCES``: the names a Measured
  / Airborne label may carry; for Airborne, v3.7's
  ``foot_height`` and the fused rule's ``foot_height_fused`` (v3.7.1, #24).
"""

from __future__ import annotations

import re

COMPONENTS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
FEET = ("L", "R")
TARGETS = tuple(f"{component}_{foot}" for foot in FEET for component in COMPONENTS)
FORCES = tuple(target for target in TARGETS if target.startswith("F"))
BODY_WEIGHT_FEATURE = "body_weight_N"
BODY_WEIGHT_MISSING_FEATURE = "body_weight_missing"

STATE_MASKED = 0
STATE_MEASURED = 1
STATE_AIRBORNE = 2

LABEL_SOURCE_NONE = 0  # Masked
LABEL_SOURCE_PLATE = 1  # Measured, plate contact
LABEL_SOURCE_FOOT_HEIGHT = 2  # Airborne, fused Foot height rule
LABEL_SOURCE_PROTOCOL_WINDOW = 3  # Measured, plate hole filled by the protocol window
LABEL_SOURCE_CODES = {
    LABEL_SOURCE_NONE: "none",
    LABEL_SOURCE_PLATE: "plate",
    LABEL_SOURCE_FOOT_HEIGHT: "foot_height_fused",
    LABEL_SOURCE_PROTOCOL_WINDOW: "protocol_window",
}
MEASURED_LABEL_SOURCES = (LABEL_SOURCE_CODES[LABEL_SOURCE_PLATE], LABEL_SOURCE_CODES[LABEL_SOURCE_PROTOCOL_WINDOW])
AIRBORNE_LABEL_SOURCES = ("foot_height", LABEL_SOURCE_CODES[LABEL_SOURCE_FOOT_HEIGHT])


LOADED_SUFFIX = "b"


def participant_of_subject(subject: str) -> str:
    """Participant of a Subject: ``P01b`` (loaded, backpack) and ``P01`` (unloaded) are ``P01``.

    Any other name is refused: a dataset still named by first names (``Alice2``) would
    otherwise split each Participant into two, and leak it across cross-validation folds.
    """
    match = re.fullmatch(rf"([A-Z]\d{{2,}}){LOADED_SUFFIX}?", subject)
    if match is None:
        raise ValueError(f"{subject!r} is not a Subject code (P01 unloaded, P01{LOADED_SUFFIX} loaded)")
    return match[1]


def subject_of_trial(trial_name: str) -> str:
    """Subject of a Trial = its name before the first underscore (``P01b_squat`` -> ``P01b``)."""
    return str(trial_name).split("_")[0]
