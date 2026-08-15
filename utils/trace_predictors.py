import math
import re
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .spectrum_reconstructors import (
    R_SpetrumSpatioTemporal_Projection_BranchAddition, 
    CrossAttentionSSTDecoder
)

class CF_Proposer(nn.Module):
    def __init__(self, n_banks, num_channels, num_patches, emb_size):
        super().__init__()
        self.C = num_channels
        self.P = num_patches
        self.F = n_banks

        self.proposer_head = nn.Linear(num_channels * n_banks, 2)

    def forward(self, x):
        B, P, C, _F = x.size()

        cf_roi = F.elu(self.proposer_head(x.reshape(B, P, -1)))
        x = torch.clamp(cf_roi[:, :, 0], min=0, max=C-1)
        y = torch.clamp(cf_roi[:, :, 1], min=0, max=_F-1)

        x_norm = x / (C - 1)
        y_norm = y / (_F - 1)
        cf_roi = torch.stack((x_norm, y_norm), dim=-1)

        return cf_roi


# =============================================================================
# 1. CNN_CF_Proposer
#    Per-patch, independent 1D convolution over the frequency axis.
#    Channels (C) are treated as the conv's input-channel dimension, so the
#    conv mixes across channels while sliding a local window over frequency.
#    No recurrence across patches (same as CF_Proposer in that respect).
# =============================================================================
class CNN_CF_Proposer(nn.Module):
    def __init__(self, n_banks, num_channels, num_patches, emb_size,
                 hidden_channels=32, kernel_size=3):
        super().__init__()
        self.C = num_channels
        self.P = num_patches
        self.F = n_banks

        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(num_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.ELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.ELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden_channels, 2)

    def forward(self, x):
        # x: (B, P, C, F)
        B, P, C, _F = x.size()

        x = x.reshape(B * P, C, _F)              # (B*P, C, F) - conv over F, C as channels
        x = self.conv(x)                          # (B*P, hidden, F)
        x = self.pool(x).squeeze(-1)              # (B*P, hidden)

        cf_roi = F.elu(self.head(x)).reshape(B, P, 2)

        x_coord = torch.clamp(cf_roi[..., 0], min=0, max=C - 1)
        y_coord = torch.clamp(cf_roi[..., 1], min=0, max=_F - 1)
        x_norm = x_coord / (C - 1)
        y_norm = y_coord / (_F - 1)
        cf_roi = torch.stack((x_norm, y_norm), dim=-1)

        return cf_roi


# =============================================================================
# 2. LSTM_CF_Proposer
#    Flattens each patch's (C, F) matrix into a vector (as CF_Proposer does),
#    then processes the sequence of P flattened vectors with an LSTM to add
#    temporal recurrence across patches. No spatial/conv inductive bias.
# =============================================================================
class LSTM_CF_Proposer(nn.Module):
    def __init__(self, n_banks, num_channels, num_patches, emb_size,
                 hidden_size=64, num_layers=1, bidirectional=False):
        super().__init__()
        self.C = num_channels
        self.P = num_patches
        self.F = n_banks

        self.lstm = nn.LSTM(
            input_size=num_channels * n_banks,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_size = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(out_size, 2)

    def forward(self, x):
        # x: (B, P, C, F)
        B, P, C, _F = x.size()

        x = x.reshape(B, P, C * _F)               # flatten (C, F) per patch; P is the sequence axis
        x, _ = self.lstm(x)                         # (B, P, hidden * num_directions)

        cf_roi = F.elu(self.head(x))                # (B, P, 2)

        x_coord = torch.clamp(cf_roi[..., 0], min=0, max=C - 1)
        y_coord = torch.clamp(cf_roi[..., 1], min=0, max=_F - 1)
        x_norm = x_coord / (C - 1)
        y_norm = y_coord / (_F - 1)
        cf_roi = torch.stack((x_norm, y_norm), dim=-1)

        return cf_roi


# =============================================================================
# 3. ConvLSTM_CF_Proposer
#    A genuine ConvLSTM (Shi et al., 2015 style) restricted to 1D: gates are
#    computed via 1D convolution over the frequency axis alone, while the
#    hidden/cell state is itself a (hidden_channels, F) map that is carried
#    and updated recurrently across patches. Channels (C) enter as the input
#    tensor's channel dimension at each step, same as in CNN_CF_Proposer.
# =============================================================================
class ConvLSTMCell1D(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x, state):
        # x: (B, C, F) - single patch's (channel, frequency) map
        # state: (h, c), each (B, hidden_channels, F)
        h, c = state
        combined = torch.cat([x, h], dim=1)          # (B, C + hidden, F)
        gates = self.conv(combined)                    # (B, 4*hidden, F)

        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_state(self, batch_size, freq_len, device, dtype):
        shape = (batch_size, self.hidden_channels, freq_len)
        h0 = torch.zeros(shape, device=device, dtype=dtype)
        c0 = torch.zeros(shape, device=device, dtype=dtype)
        return h0, c0


class ConvLSTM_CF_Proposer(nn.Module):
    def __init__(self, n_banks, num_channels, num_patches, emb_size,
                 hidden_channels=32, kernel_size=3):
        super().__init__()
        self.C = num_channels
        self.P = num_patches
        self.F = n_banks

        self.cell = ConvLSTMCell1D(num_channels, hidden_channels, kernel_size)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden_channels, 2)

    def forward(self, x):
        # x: (B, P, C, F)
        B, P, C, _F = x.size()

        h, c = self.cell.init_state(B, _F, device=x.device, dtype=x.dtype)

        outputs = []
        for p in range(P):
            x_p = x[:, p]                          # (B, C, F)
            h, c = self.cell(x_p, (h, c))          # (B, hidden, F), recurrent across patches
            outputs.append(h.unsqueeze(1))

        h_seq = torch.cat(outputs, dim=1)           # (B, P, hidden, F)
        h_seq = h_seq.reshape(B * P, -1, _F)
        pooled = self.pool(h_seq).squeeze(-1)        # (B*P, hidden)

        cf_roi = F.elu(self.head(pooled)).reshape(B, P, 2)

        x_coord = torch.clamp(cf_roi[..., 0], min=0, max=C - 1)
        y_coord = torch.clamp(cf_roi[..., 1], min=0, max=_F - 1)
        x_norm = x_coord / (C - 1)
        y_norm = y_coord / (_F - 1)
        cf_roi = torch.stack((x_norm, y_norm), dim=-1)

        return cf_roi





class TracePredictor(nn.Module):
    def __init__(
            self, n_banks, num_channels, num_patches, emb_size, 
            sst, 
            cf_proposer='linear', 
            hidden_size=64, num_layers=1,
            hidden_channels=32, kernel_size=3):
        super().__init__()
        self.C = num_channels
        self.P = num_patches
        self.F = n_banks
        self.D = emb_size

        self.sst_estimator = sst(n_banks, num_channels, num_patches, emb_size)

        if cf_proposer.upper() == 'LINEAR':
            print("-------------------------------------------------------")
            print("Using Linear CF_Proposer Head")
            print("-------------------------------------------------------")
            self.cf_proposer = CF_Proposer(self.F, self.C, self.P, self.D)
        elif cf_proposer.upper() == 'LSTM':
            print("-------------------------------------------------------")
            print("Using LSTM CF_Proposer Head")
            print("-------------------------------------------------------")
            self.cf_proposer = LSTM_CF_Proposer(n_banks, num_channels, num_channels, emb_size, hidden_size, num_layers)
        elif cf_proposer.upper() == 'CNN':
            print("-------------------------------------------------------")
            print("Using CNN CF_Proposer Head")
            print("-------------------------------------------------------")
            self.cf_proposer = CNN_CF_Proposer(
                n_banks, num_channels, num_patches, emb_size,
                hidden_channels=hidden_channels, kernel_size=kernel_size)
        elif cf_proposer.upper() == 'CONV_LSTM':
            print("-------------------------------------------------------")
            print("Using ConvLSTM CF_Proposer Head")
            print("-------------------------------------------------------")
            self.cf_proposer = ConvLSTM_CF_Proposer(
                n_banks, num_channels, num_patches, emb_size,
                hidden_channels=hidden_channels, kernel_size=kernel_size)

    def forward(self, x_spectrum, x_temporal, x_channel):
        sst_hat = self.sst_estimator(x_spectrum, x_temporal, x_channel)
        sst_hat = rearrange(sst_hat, 'b c f p -> b p c f')
        B, P, C, _F = sst_hat.size()

        cf_roi = self.cf_proposer(sst_hat)
        return cf_roi


# class Trace_SpectrumSpatioTemporal_Projection_BranchAddition(nn.Module):
#     def __init__(self, n_banks, num_channels, num_patches, emb_size):
#         super().__init__()
#         self.C = num_channels
#         self.P = num_patches
#         self.F = n_banks

#         self.sst_estimator = R_SpetrumSpatioTemporal_Projection_BranchAddition(n_banks, num_channels, num_patches, emb_size)
#         self.cf_proposer = CF_Proposer(n_banks, num_channels, num_patches, emb_size)

#     def forward(self, x_spectrum, x_temporal, x_channel):
#         sst_hat = self.sst_estimator(x_spectrum, x_temporal, x_channel)
#         sst_hat = rearrange(sst_hat, 'b c f p -> b p c f')
#         B, P, C, _F = sst_hat.size()

#         cf_roi = self.cf_proposer(sst_hat)
#         return cf_roi

# class Trace_CrossAttention(nn.Module):
#     def __init__(self, n_banks, num_channels, num_patches, emb_size, n_heads=8, n_layers=2):
#         super().__init__()
#         self.C = num_channels
#         self.P = num_patches
#         self.F = n_banks

#         self.sst_estimator = CrossAttentionSSTDecoder(n_banks, num_channels, num_patches, emb_size, n_heads, n_layers)
