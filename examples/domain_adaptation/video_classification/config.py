"""
Default configurations for action recognition domain adaptation
"""

import os

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.ROOT = "F:/Datasets/EgoAction/"  # "/shared/tale2/Shared"
_C.DATASET.SOURCE = "EPIC"  # dataset options=["EPIC", "GTEA", "ADL", "KITCHEN", "EPIC100"]
_C.DATASET.SRC_TRAINLIST = "epic_D1_train.pkl"
_C.DATASET.SRC_TESTLIST = "epic_D1_test.pkl"
_C.DATASET.TARGET = "EPIC"  # dataset options=["EPIC", "GTEA", "ADL", "KITCHEN", "EPIC100"]
_C.DATASET.TGT_TRAINLIST = "epic_D2_train.pkl"
_C.DATASET.TGT_TESTLIST = "epic_D2_test.pkl"
_C.DATASET.IMAGE_MODALITY = "rgb"  # options=["rgb", "flow", "joint"]
_C.DATASET.INPUT_TYPE = "image"  # options=["image", "feature"]
_C.DATASET.CLASS_TYPE = "verb"  # options=["verb", "verb+noun"]
_C.DATASET.NUM_SEGMENTS = 1  # = 1, if image input; = 8, if feature input.
_C.DATASET.FRAMES_PER_SEGMENT = 16  # = 16, if image input; = 1, if feature input.


def get_cfg_defaults():
    return _C.clone()
