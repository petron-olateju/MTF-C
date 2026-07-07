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
    
class R_Gated_SpectrumSpatioTemporal_ProjectionAddition(R_SpectrumSpatioTemporal_ProjectionAddition):
    def __init__(self, n_banks, num_channels, num_patches, emb_size):
        super().__init__(n_banks, num_channels, num_patches, emb_size)
        self.gates = nn.ModuleDict({
            'spectrum': nn.Sequential(nn.Linear(emb_size, emb_size), nn.Sigmoid()),
            'temporal': nn.Sequential(nn.Linear(emb_size, emb_size), nn.Sigmoid()),
            'spatial': nn.Sequential(nn.Linear(emb_size, emb_size), nn.Sigmoid()),
        })

    def forward(self, x_spectrum, x_temporal, x_spatial):
        x_spectrum = x_spectrum * self.gates['spectrum'](x_spectrum)
        x_temporal = x_temporal * self.gates['temporal'](x_temporal)
        x_spatial = x_spatial * self.gates['spatial'](x_spatial)
        
        return super().forward(x_spectrum, x_temporal, x_spatial)

