# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            xla_e2e_latency.py
# CREATED:         11/17/2026
# REVISION:        1.0
# ======================================= #

## Iteration = frame i from each view (FRONT, FRONT_LEFT, FRONT_RIGHT, etc)
## Same loop as the other tests, timed with perf_counter per frame and per sub-test.
## No per-stage timing. Views run back to back, one synchronize per frame.

## This file is the same structure as the e2e_latency.py file but this uses TT-XLA to 
## load the CLIP model instead of porting it manually. 
## This will allow us to investigate how TT-XLA performs compared to a ported CLIP model.

import time

import torch
import ttnn
import torch_xla.runtime as xr

from ....common import data
from ... import model_tt as model_lib
from ...tt_clip.device import get_device

