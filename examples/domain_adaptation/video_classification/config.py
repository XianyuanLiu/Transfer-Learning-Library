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

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.LR = 0.001
_C.SOLVER.LR_GAMMA = 0.001
_C.SOLVER.LR_DECAY = 0.75
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 1e-3
_C.SOLVER.WORKERS = 2

_C.SOLVER.NUM_EPOCHS = 20
_C.SOLVER.ITERS_PER_EPOCH = 1000
_C.SOLVER.PRINT_FREQ = 100
_C.SOLVER.SEED = None
_C.SOLVER.PER_CLASS_EVAL = False
_C.SOLVER.LOG_DIR = "dann"
_C.SOLVER.PHASE = "train"  # train, test, analysis

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.ARCH = "simple"
_C.MODEL.BOTTLENECK_DIM = 256
_C.MODEL.NO_POOL = False
_C.MODEL.SCRATCH = True
_C.MODEL.TRADE_OFF = 1.0


# -----------------------------------------------------------------------------
# Comet
# -----------------------------------------------------------------------------
_C.COMET = CN()
_C.COMET.ENABLE = True
_C.COMET.LOG_HISTOGRAMS = False
_C.COMET.LOG_CONFUSION_MATRIX = False
_C.COMET.API_KEY = "fwDWzM3HmQuZuFGFS2q90vLT3"
_C.COMET.PROJECT_NAME = "Action Recognition Domain Adaptation"


def get_cfg_defaults():
    return _C.clone()
