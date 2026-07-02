import math
import re
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# =============================================================================
# Spatio-Temporal Mixing Layers
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
        return zs + zt

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
            nn.Linear(emb_size, emb_size * 2),
            nn.ELU(),
            nn.Linear(emb_size * 2, emb_size),
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

class SpatioTemporal_ProjectionAddition_PerBank(nn.Module):
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
        self.n_banks = n_banks
        self.shared_projection = nn.ModuleList([SharedProjectionLayer(emb_size) for n in range(n_banks)])
        self.addition = nn.ModuleList([SpatioTemporalAddition(num_channels, num_patches)] for n in range(n_banks))
        self.estimator = nn.Linear(emb_size, 1)

    def forward(self, x_temporal, x_spatial):
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
            z_ct_f = self.estimator(z_ct_f)
            z_ct_f = z_ct_f.squeeze(dim=-1)
            z.append(z_ct_f.unsqueeze(dim=1))
        
        z = torch.concat(z, dim=1)
        z = rearrange(z, 'b f c t -> b c f t')
        return z


# =============================================================================
# Spectrogram Estimation
# =============================================================================
class SpectrogramEstimator(nn.Module):
    """Maps spatiotemporal embeddings to spectrogram estimates.

    Converts embedding representations to spectrogram format using various methods:
    - st_addition: MLP-based estimation from SpatioTemporalAddition output
    - st_addition_projection: MLP-based estimation from SpatioTemporal_ProjectionAddition_PerBank output
    - stf_attention_temporal_values: Attention-weighted estimation with temporal values
    - filter_banks: Element-wise product of temporal and spatial embeddings

    Args:
        method: Estimation method ('st_addition', 'st_addition_projection',
                'stf_attention_temporal_values', 'filter_banks').
        emb_size: Embedding dimension D.
        n_freqs: Number of frequency bins F.
        num_channels: Number of channels C.
        num_patches: Number of temporal patches P.
        use_ct_shared_projection: Whether to use shared projection for filter_banks.

    Input shapes (vary by method):
        z_st: (B, C, P, D) or (B, F, C, P, D) - spatiotemporal embeddings
        x_embed_temporal: (B, P, D) or (B, F, P, D) - temporal embeddings
        x_embed_spatial: (B, C, D) - spatial embeddings

    Output shape:
        (B, C, P, F) - estimated spectrogram
    """

    def __init__(
        self,
        method,
        emb_size,
        n_freqs,
        num_channels,
        num_patches,
        use_ct_shared_projection=False,
    ):
        super().__init__()
        self.method = method
        self.C = num_channels
        self.P = num_patches
        self.F = n_freqs

        if method in ["st_addition", "st_addition_projection"]:
            self.mlp = nn.Sequential(
                nn.Linear(emb_size, emb_size**2),
                nn.ELU(),
                nn.Linear(emb_size**2, n_freqs),
                nn.ELU(),
            )
        elif method in ["stf_attention_temporal_values", "filter_banks"]:
            self.mlp = nn.Sequential(
                nn.Linear(emb_size, emb_size**2),
                nn.ELU(),
                nn.Linear(emb_size**2, 1),
                nn.ELU(),
            )
            if method == "filter_banks" and use_ct_shared_projection:
                self.ct_shared_projection = SharedProjectionLayer(emb_size)

    def forward(self, z_st, x_embed_temporal, x_embed_spatial):
        """Estimate spectrogram from embeddings.

        Args:
            z_st: Spatiotemporal embeddings (method-dependent shape).
            x_embed_temporal: Temporal embeddings.
            x_embed_spatial: Spatial embeddings.

        Returns:
            Spectrogram estimate of shape (B, C, P, F).
        """
        if self.method in ["st_addition", "st_addition_projection"]:
            return self.mlp(z_st)

        elif self.method == "stf_attention_temporal_values":
            return self.mlp(z_st).squeeze(dim=-1).permute(0, 2, 3, 1)

        elif self.method == "filter_banks":
            if hasattr(self, "ct_shared_projection"):
                z_f = self.ct_shared_projection(z_st)
                z_t = self.ct_shared_projection(x_embed_temporal)
                z_s = self.ct_shared_projection(x_embed_spatial)
            else:
                z_f = z_st
                z_t = x_embed_temporal
                z_s = x_embed_spatial

            z_hat_f = z_f.unsqueeze(2).unsqueeze(1).expand(-1, self.C, -1, self.P, -1)
            z_hat_t = z_t.unsqueeze(1).unsqueeze(2).expand(-1, self.C, self.F, -1, -1)
            z_hat_s = z_s.unsqueeze(2).unsqueeze(3).expand(-1, -1, self.F, self.P, -1)

            z_hat_fts = (z_hat_f + z_hat_t + z_hat_s).contiguous()
            return torch.abs(self.mlp(z_hat_fts)).squeeze(-1).permute(0, 1, 3, 2)

        raise ValueError(f"Unknown spectrogram estimator method: {self.method}")
