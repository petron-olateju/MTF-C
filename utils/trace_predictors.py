import math
import re
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .spectrum_reconstructors import R_SpetrumSpatioTemporal_Projection_BranchAddition

class Trace_SpetrumSpatioTemporal_Projection_BranchAddition(nn.Module):
    def __init__(self, n_banks, num_channels, num_patches, emb_size):
        super().__init__()
        self.C = num_channels
        self.P = num_patches
        self.F = n_banks

        self.sst_estimator = R_SpetrumSpatioTemporal_Projection_BranchAddition(n_banks, num_channels, num_patches, emb_size)
        self.cf_proposer = nn.Linear(num_channels * n_banks, 2)

    def forward(self, x_spectrum, x_temporal, x_channel):
        sst_hat = self.sst_estimator(x_spectrum, x_temporal, x_channel)
        sst_hat = rearrange(sst_hat, 'b c f p -> b p c f')
        B, P, C, _F = sst_hat.size()

        cf_roi = F.elu(self.cf_proposer(sst_hat.reshape(B, P, -1)))
        x = torch.clamp(cf_roi[:, :, 0], min=0, max=C-1)
        y = torch.clamp(cf_roi[:, :, 1], min=0, max=_F-1)

        x_norm = x / (C - 1)
        y_norm = y / (_F - 1)
        cf_roi = torch.stack((x_norm, y_norm), dim=-1)

        return cf_roi