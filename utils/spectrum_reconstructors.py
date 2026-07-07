import math
import re
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# =============================================================================
# Branches Operation/Mixing Layers
# =============================================================================
class SpatioTemporalAddition(nn.Module):
    """Combines temporal and spatial embeddings via broadcasting addition.

    Takes temporal embeddings (B, P, D) and spatial embeddings (B, C, D) and
    combines them through broadcasting addition to produce spatiotemporal
    embeddings (B, C, P, D).

    Args:
        num_channels: Number of channels C.
        num_patches: Number of temporal patches P.

    Input shapes:
        x_temporal: (B, P, D) - temporal embeddings
        x_spatial: (B, C, D) - spatial (channel) embeddings

    Output shape:
        (B, C, P, D) - spatiotemporal embeddings where each element is
        x_spatial[b, c, d] + x_temporal[b, p, d]
    """

    def __init__(self, num_channels, num_patches):
        super().__init__()
        self.C = num_channels
        self.P = num_patches

    def forward(self, x_temporal, x_spatial):
        """Combine temporal and spatial embeddings.

        Args:
            x_temporal: Temporal embeddings (B, P, D).
            x_spatial: Spatial embeddings (B, C, D).

        Returns:
            Spatiotemporal embeddings (B, C, P, D).
        """
        zt = x_temporal.unsqueeze(1).expand(-1, self.C, -1, -1)
        zs = x_spatial.unsqueeze(2).expand(-1, -1, self.P, -1)
        return zs + zt  # (B, C, T, D)
    
class SpectrumSpatioTemporalAddition(SpatioTemporalAddition):
    
    def __init__(self, n_banks, num_channels, num_patches):
        super().__init__(num_channels, num_patches)
        self.F = n_banks

    def forward(self, x_spectrum, x_temporal, x_spatial):
        zs = x_spectrum.unsqueeze(1).unsqueeze(3).expand(-1, self.C, -1, self.P, -1)
        zt = x_temporal.unsqueeze(1).unsqueeze(2).expand(-1, self.C, self.F, -1, -1)
        zc = x_spatial.unsqueeze(2).unsqueeze(3).expand(-1, -1, self.F, self.P, -1)
        return zt + zs + zc     # (B, C, F, P, D)

class SharedProjectionLayer(nn.Module):
    """Shared MLP projection for embedding transformation.

    A two-layer MLP with ELU activation that projects embeddings to an intermediate
    space (2x emb_size) and back, enabling more expressive transformations.
    Shared between temporal and spatial branches to reduce parameters.

    Args:
        emb_size: Input and output embedding dimension.

    Input shape:
        x: (B, *, D) - arbitrary batch of embeddings

    Output shape:
        (B, *, D) - projected embeddings
    """

    def __init__(self, emb_size):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ELU(),
            nn.Linear(emb_size, emb_size),
            nn.ELU(),
        )

    def forward(self, x):
        """Project embeddings through shared MLP.

        Args:
            x: Input embeddings (B, *, D).

        Returns:
            Projected embeddings (B, *, D).
        """
        return self.projection(x)

class EstimatorLayer(nn.Module):
    """Two-layer MLP that maps D-dimensional embeddings to a scalar.

    Args:
        emb_size: Input embedding dimension.

    Input shape:
        x: (..., D)

    Output shape:
        (..., 1)
    """

    def __init__(self, emb_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ELU(),
            nn.Linear(emb_size, 1),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Spectrogram Reconstructors
# =============================================================================
class R_SpatioTemporal_AdditionPerBank(nn.Module):
    def __init__(self, n_banks, num_channels, num_patches, emb_size):
        super().__init__()
        print(f"SST  Target Size: {(num_channels, n_banks, num_patches)}")
        self.n_banks = n_banks
        self.addition = nn.ModuleList([SpatioTemporalAddition(num_channels, num_patches) for n in range(n_banks)])
        self.estimator = EstimatorLayer(emb_size)

    def forward(self, x_spectrum, x_temporal, x_spatial):
        z = []
        for n in range(self.n_banks):
            z_f = self.addition[n](x_temporal, x_spatial)
            z.append(z_f.unsqueeze(dim=2))
        z = torch.concat(z, dim=2)
        z = self.estimator(z).squeeze(-1)
        return z
    
class R_SpatioTemporalProjection_AdditionPerBank(nn.Module):
    def __init__(self, n_banks, num_channels, num_patches, emb_size):
        super().__init__()
        print(f"SST  Target Size: {(num_channels, n_banks, num_patches)}")
        self.n_banks = n_banks
        self.shared_projetion = SharedProjectionLayer(emb_size)
        self.addition = R_SpatioTemporal_AdditionPerBank(n_banks, num_channels, num_patches, emb_size)
        self.estimator = EstimatorLayer(emb_size)

    def forward(self, x_spectrum, x_temporal, x_spatial):
        z_t = self.shared_projetion(x_temporal)
        z_c = self.shared_projetion(x_spatial)
        z = self.addition(x_spectrum, z_t, z_c)
        return z
    
class R_SpatioTemporal_ProjectionAdditionPerBank(nn.Module):
    """Combines temporal and spatial embeddings with shared projection.

    Applies SharedProjectionLayer to both temporal and spatial embeddings before
    combining them via SpatioTemporalAddition. This enables more expressive transformations
    of the embeddings before fusion.

    Args:
        num_channels: Number of channels C.
        num_patches: Number of temporal patches P.
        emb_size: Embedding dimension D.

    Input shapes:
        x_temporal: (B, P, D) - temporal embeddings
        x_spatial: (B, C, D) - spatial embeddings

    Output shape:
        (B, C, P, D) - combined spatiotemporal embeddings
    """

    def __init__(self, n_banks, num_channels, num_patches, emb_size):
        super().__init__()
        print(f"SST  Target Size: {(num_channels, n_banks, num_patches)}")
        self.n_banks = n_banks
        self.shared_projection = nn.ModuleList([SharedProjectionLayer(emb_size) for n in range(n_banks)])
        self.addition = nn.ModuleList([SpatioTemporalAddition(num_channels, num_patches) for n in range(n_banks)])
        self.estimator = EstimatorLayer(emb_size)

    def forward(self, x_spectrum, x_temporal, x_spatial):
        """Project and combine embeddings.

        Args:
            x_temporal: Temporal embeddings (B, P, D).
            x_spatial: Spatial embeddings (B, C, D).

        Returns:
            Combined spatiotemporal embeddings (B, C, P, D).
        """
        z = []

        for n in range(self.n_banks):
            z_t = self.shared_projection[n](x_temporal)
            z_c = self.shared_projection[n](x_spatial)
            z_ct_f = self.addition[n](z_t, z_c)
            z.append(z_ct_f.unsqueeze(dim=2))
        
        z = torch.concat(z, dim=2)
        z = self.estimator(z).squeeze(-1)
        return z
    
class R_SpectrumSpatioTemporal_Addition(nn.Module):
    def __init__(self, n_banks, num_channels, num_patches, emb_size):
        super().__init__()
        print(f"SST  Target Size: {(num_channels, n_banks, num_patches)}")
        self.addition = SpectrumSpatioTemporalAddition(n_banks, num_channels, num_patches)
        self.estimator = EstimatorLayer(emb_size)

    def forward(self, x_spectrum, x_temporal, x_spatial):
        z = self.addition(x_spectrum, x_temporal, x_spatial)
        z = self.estimator(z).squeeze(-1)
        return z
    
class R_SpectrumSpatioTemporal_ProjectionAddition(nn.Module):
    def __init__(self, n_banks, num_channels, num_patches, emb_size):
        super().__init__()
        print(f"SST  Target Size: {(num_channels, n_banks, num_patches)}")
        self.shared_projection = SharedProjectionLayer(emb_size)
        self.addition_estimator = R_SpectrumSpatioTemporal_Addition(n_banks, num_channels, num_patches, emb_size)

    def forward(self, x_spectrum, x_temporal, x_spatial):
        z_s = self.shared_projection(x_spectrum)
        z_t = self.shared_projection(x_temporal)
        z_c = self.shared_projection(x_spatial)

        z = self.addition_estimator(z_s, z_t, z_c).squeeze(-1)
        return z

class R_SpectrumSpatioTemporal_ProjectionAdditionPerBank(nn.Module):
    def __init__(self, n_banks, num_channels, num_patches, emb_size):
        super().__init__()
        print(f"SST  Target Size: {(num_channels, n_banks, num_patches)}")
        self.n_banks = n_banks
        self.shared_projection = nn.ModuleList([SharedProjectionLayer(emb_size) for n in range(self.n_banks)])
        self.addition = nn.ModuleList([SpectrumSpatioTemporalAddition(n_banks, num_channels, num_patches) for n in range(self.n_banks)])
        self.estimator = EstimatorLayer(emb_size)

    def forward(self, x_spectrum, x_temporal, x_spatial):
        z = []

        for n in range(self.n_banks):
            z_s = self.shared_projection[n](x_spectrum)
            z_t = self.shared_projection[n](x_temporal)
            z_c = self.shared_projection[n](x_spatial)
            
            z_sst = self.addition[n](z_s, z_t, z_c)
            z.append(z_sst.unsqueeze(dim=2))
        
        z = torch.concat(z, dim=2)
        z = z.mean(dim=3)
        z = self.estimator(z).squeeze(-1)
        return z
        
    
class CrossAttentionSSTDecoder(nn.Module):
    def __init__(self, n_banks, num_channels, num_patches, emb_size, n_heads=8, n_layers=2):
        super().__init__()
        self.F, self.C, self.P, self.D = n_banks, num_channels, num_patches, emb_size

        self.channel_pos = nn.Parameter(torch.randn(self.C, emb_size) * 0.02)
        self.freq_pos = nn.Parameter(torch.randn(self.F, emb_size) * 0.02)
        self.patch_pos = nn.Parameter(torch.randn(self.P, emb_size) * 0.02)

        self.spectrum_type = nn.Parameter(torch.randn(1, 1, emb_size) * 0.02)
        self.temporal_type = nn.Parameter(torch.randn(1, 1, emb_size) * 0.02)
        self.spatial_type = nn.Parameter(torch.randn(1, 1, emb_size) * 0.02)

        self.cross_attn = nn.ModuleList(
            [nn.MultiheadAttention(emb_size, n_heads, batch_first=True) for _ in range(n_layers)]
        )
        self.norm1 = nn.ModuleList([nn.LayerNorm(emb_size) for _ in range(n_layers)])
        self.ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(emb_size, emb_size * 2), nn.ELU(), nn.Linear(emb_size * 2, emb_size))
            for _ in range(n_layers)
        ])
        self.norm2 = nn.ModuleList([nn.LayerNorm(emb_size) for _ in range(n_layers)])

        self.estimator = EstimatorLayer(emb_size)

    def forward(self, x_spectrum, x_temporal, x_spatial):
        B = x_spectrum.shape[0]

        query = (
            self.channel_pos[:, None, None, :]
            + self.freq_pos[None, :, None, :]
            + self.patch_pos[None, None, :, :]
        ).reshape(self.C * self.F * self.P, self.D).unsqueeze(0).expand(B, -1, -1)

        memory = torch.cat([
            x_spectrum + self.spectrum_type,
            x_temporal + self.temporal_type,
            x_spatial + self.spatial_type,
        ], dim=1)

        z = query
        for attn, n1, ffn, n2 in zip(self.cross_attn, self.norm1, self.ffn, self.norm2):
            attn_out, _ = attn(z, memory, memory)
            z = n1(z + attn_out)
            z = n2(z + ffn(z))

        z = z.view(B, self.C, self.F, self.P, self.D)
        return self.estimator(z).squeeze(-1)   # (B, C, F, P)

