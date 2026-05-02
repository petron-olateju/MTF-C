"""Multi-Scale Temporal Frequency Conformer (MTFC) for EEG classification.

This module implements the MTFC architecture which combines spectral-spatio-temporal
embeddings with a transformer-based conformer for EEG signal classification.
"""

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .DBConformer import PatchEmbeddingTemporal, PatchEmbeddingSpatial
from .DBConformer import TransformerEncoder, ClassificationHead


# =============================================================================
# Utility Functions
# =============================================================================


def parse_branch_config(branch):
    """Parse branch configuration string into transformer branches.

    The branch string describes which embeddings are processed by each transformer
    and how they are combined. Format is sequence of letters: 'f' (frequency),
    't' (temporal), 's' (spatial). Each letter represents an embedding type that
    will be processed by a transformer, and underscores indicate concatenation points.

    Args:
        branch: String describing branch configuration. Examples:
            - 'ft_s': one transformer processes freq-temporal (concatenated),
                      then concatenated with spatial
            - 'f_t_s': three separate transformers for freq, temporal, spatial
            - 'all': legacy single transformer for all embeddings (FTS)

    Returns:
        List of tuples, each tuple contains which embedding dimensions to concatenate
        before each transformer. E.g., [('f', 't'), ('s',)] for ft_s.

    Raises:
        ValueError: If branch string contains invalid characters or format.
    """
    if branch == "all":
        return [("fts")]

    valid_chars = set("fts_")
    if not all(c in valid_chars for c in branch):
        raise ValueError(
            f"Invalid branch config '{branch}'. Must contain only 'f', 't', 's', '_'"
        )

    tokens = re.split(r"_", branch)
    tokens = [t for t in tokens if t]

    if not tokens:
        raise ValueError(f"Invalid branch config '{branch}'")

    branch_spec = []
    for token in tokens:
        branch_spec.append(tuple(token))

    return branch_spec


def band_kernel_size(fs, low_freq):
    """Compute kernel size for frequency band filtering.

    Args:
        fs: Sampling frequency in Hz.
        low_freq: Lower frequency bound for the band. If None, defaults to 1 Hz.

    Returns:
        Kernel size (odd number) for symmetric padding in filtering operations.
    """
    if low_freq is None:
        low_freq = 1
    k = int(fs / low_freq / 2)
    return k if k % 2 == 1 else k + 1


def create_filter_banks(n_filter_banks, fs, start_freq=1.0):
    """Create frequency band definitions for filter banks.

    Args:
        n_filter_banks: Number of frequency bands to create.
        fs: Sampling frequency in Hz.
        start_freq: Starting frequency for the first band in Hz.

    Returns:
        Dictionary mapping band index to [start_freq, end_freq] pairs.
    """
    nyquist = fs / 2
    band_width = (nyquist - start_freq) / n_filter_banks
    return {
        i: [
            round(start_freq + i * band_width, 2),
            round(start_freq + (i + 1) * band_width, 2),
        ]
        for i in range(n_filter_banks)
    }


# =============================================================================
# Attention Layers
# =============================================================================


class SpatioTemporal_Temporal_AttentionHead(nn.Module):
    """Cross-attention head for spatiotemporal-to-temporal attention.

    This attention mechanism allows spatiotemporal embeddings (B, C, P, D) to attend
    to temporal embeddings (B, P, D), producing attended spatiotemporal embeddings.
    Used in the stf_attention_temporal_values mode of MTFC.

    Args:
        emb_size: Embedding dimension D.
        num_heads: Number of attention heads.
        dropout: Dropout probability.

    Input shapes:
        query: (B, C, P, D) - spatiotemporal embeddings (channels x patches x dim)
        key_value: (B, P, D) - temporal embeddings (patches x dim)

    Output shape:
        (B, C, P, D) - attended spatiotemporal embeddings
    """

    def __init__(self, emb_size, num_heads=2, dropout=0.2):
        super().__init__()
        assert emb_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads

        self.query_proj = nn.Linear(emb_size, emb_size)
        self.key_proj = nn.Linear(emb_size, emb_size)
        self.value_proj = nn.Linear(emb_size, emb_size)
        self.out_proj = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        """Compute cross-attention from spatiotemporal to temporal embeddings.

        Args:
            query: Spatiotemporal embeddings of shape (B, C, P, D).
            key_value: Temporal embeddings of shape (B, P, D).

        Returns:
            Attended spatiotemporal embeddings of shape (B, C, P, D).
        """
        H, D = self.num_heads, self.head_dim
        B, C, P, _ = query.shape
        P_kv = key_value.shape[1]

        Q = self.query_proj(query).view(B, C, P, H, D)
        K = self.key_proj(key_value).view(B, P_kv, H, D)
        V = self.value_proj(key_value).view(B, P_kv, H, D)

        attn_scores = torch.einsum("BCPHD,BKHD->BCHPK", Q, K) / (D**0.5)
        attn_probs = self.dropout(F.softmax(attn_scores, dim=-2))

        out = torch.einsum("BCHPK,BKHD->BCPHD", attn_probs, V).reshape(B, C, P, H * D)
        return self.out_proj(out)


class N_CrossAttentionHeads(nn.Module):
    """Wrapper for multiple cross-attention heads.

    Wraps F attention heads (e.g., SpatioTemporal_Temporal_AttentionHead) to produce
    output for F frequency components, enabling parallel attention computation.

    Args:
        emb_size: Embedding dimension D.
        num_heads: Number of attention heads per component.
        n_comps: Number of parallel attention components (typically F for frequencies).
        dropout: Dropout probability.
        AttnClass: Attention head class to instantiate.

    Input shapes:
        query: (B, C, P, D) - spatiotemporal embeddings
        key_value: (B, P, D) - temporal embeddings

    Output shape:
        (B, F, C, P, D) - F parallel attended embeddings
    """

    def __init__(
        self,
        emb_size,
        num_heads=2,
        n_comps=7,
        dropout=0.2,
        AttnClass=SpatioTemporal_Temporal_AttentionHead,
    ):
        super().__init__()
        self.component_attn = nn.ModuleList(
            [
                AttnClass(emb_size=emb_size, num_heads=num_heads, dropout=dropout)
                for _ in range(n_comps)
            ]
        )

    def forward(self, query, key_value):
        """Compute parallel attention for multiple components.

        Args:
            query: Spatiotemporal embeddings of shape (B, C, P, D).
            key_value: Temporal embeddings of shape (B, P, D).

        Returns:
            Stacked outputs of shape (B, F, C, P, D).
        """
        return torch.stack(
            [head(query, key_value) for head in self.component_attn], dim=1
        )


# =============================================================================
# Spatio-Temporal Mixing Layers
# =============================================================================


class STAddition(nn.Module):
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


class ST_SharedProjection(nn.Module):
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


class STAdditionProjection(nn.Module):
    """Combines temporal and spatial embeddings with shared projection.

    Applies ST_SharedProjection to both temporal and spatial embeddings before
    combining them via STAddition. This enables more expressive transformations
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

    def __init__(self, num_channels, num_patches, emb_size):
        super().__init__()
        self.shared_projection = ST_SharedProjection(emb_size)
        self.addition = STAddition(num_channels, num_patches)

    def forward(self, x_temporal, x_spatial):
        """Project and combine embeddings.

        Args:
            x_temporal: Temporal embeddings (B, P, D).
            x_spatial: Spatial embeddings (B, C, D).

        Returns:
            Combined spatiotemporal embeddings (B, C, P, D).
        """
        z_t = self.shared_projection(x_temporal)
        z_s = self.shared_projection(x_spatial)
        return self.addition(z_t, z_s)


# =============================================================================
# Spectrogram Estimation
# =============================================================================


class SpectrogramEstimator(nn.Module):
    """Maps spatiotemporal embeddings to spectrogram estimates.

    Converts embedding representations to spectrogram format using various methods:
    - st_addition: MLP-based estimation from STAddition output
    - st_addition_projection: MLP-based estimation from STAdditionProjection output
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
                self.ct_shared_projection = ST_SharedProjection(emb_size)

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


class MultiScaleTemporalPatchEmbedding_FilterBanks(nn.Module):
    """Patch embedding with multiple filter bank branches.

    Creates parallel patch embedding branches, each optimized for a different
    frequency band. Outputs embeddings for all frequency bands simultaneously.

    Args:
        args: Configuration object with data_name, chn, patch_size,
              time_sample_num, class_num.
        n_filter_banks: Number of filter bank branches.
        emb_size: Output embedding dimension.
        fs: Sampling frequency for kernel size computation.

    Input shape:
        x: (B, C, T) - EEG signals (channels x time points)

    Output shape:
        (B, F, P, D) - embeddings for F bands, P patches, D dimensions
    """

    def __init__(self, args, n_filter_banks=6, emb_size=40, fs=250):
        super().__init__()
        self.n_filter_banks = n_filter_banks
        filter_banks = create_filter_banks(n_filter_banks, fs=fs, start_freq=1.0)

        self.patch_embeddings = nn.ModuleList(
            [
                PatchEmbeddingTemporal(
                    data_name=args.data_name,
                    in_planes=args.chn,
                    out_planes=emb_size,
                    kernel_size=band_kernel_size(fs, filter_banks[i][0]),
                    radix=1,
                    patch_size=args.patch_size,
                    time_points=args.time_sample_num,
                    num_classes=args.class_num,
                )
                for i in range(self.n_filter_banks)
            ]
        )

    def forward(self, x):
        """Apply parallel filter bank patch embeddings.

        Args:
            x: EEG signals (B, C, T).

        Returns:
            Filter bank embeddings (B, F, P, D).
        """
        out = torch.cat(
            [
                self.patch_embeddings[i](x).unsqueeze(1)
                for i in range(self.n_filter_banks)
            ],
            dim=1,
        )
        return out


class MultiTemporalConvPool_ChannelsProject_FilterBanks(nn.Module):
    def __init__(
        self,
        n_channels,
        n_filter_banks,
        emb_size=40,
        temporal_kernel=43,
        n_time_points=1000,
        dropout=0.5,
    ):
        super().__init__()
        self.F = n_filter_banks
        self.D = emb_size
        self.C = n_channels

        self.temporal_pool = 3

        T_out = n_time_points
        self.T_out = T_out

        self.filter_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        n_channels,
                        n_channels,
                        kernel_size=temporal_kernel,
                        padding=temporal_kernel // 2,
                        groups=n_channels,
                        bias=False,
                    ),
                    nn.BatchNorm1d(n_channels),
                    nn.ELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(n_filter_banks)
            ]
        )

        self.head = nn.Sequential(
            nn.Conv1d(n_channels, emb_size, kernel_size=1),
            nn.AvgPool1d(kernel_size=3, stride=3),
            nn.Flatten(start_dim=1),
        )

    def forward(self, x):  # x: (B, C, T)
        B, C, T = x.shape
        bank_outputs = []
        for conv in self.filter_convs:
            z = conv(x)  # (B, C, T')
            z = self.head(z)  # (B, D * T_pool) where T_pool = T'//3
            bank_outputs.append(z)

        T_pool = z.shape[1] // self.D
        out = torch.stack(bank_outputs, dim=1)  # (B, F, D * T_pool)
        out = out.view(B, self.F, self.D, T_pool)  # (B, F, D, T_pool)
        out = out.mean(dim=-1)  # (B, F, D) - pool temporal bins
        return out  # (B, F, D)


class MultiTemporalConvFixedPool_ChannelsProject_FilterBanks(nn.Module):
    def __init__(
        self,
        n_channels,
        n_filter_banks,
        emb_size=40,
        temporal_kernel=43,
        n_time_points=1000,
        dropout=0.5,
    ):
        super().__init__()
        self.F = n_filter_banks
        self.D = emb_size
        self.C = n_channels
        self.temporal_pool = 3

        T_out = n_time_points
        self.T_out = T_out

        self.filter_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        n_channels,
                        n_channels,
                        kernel_size=temporal_kernel,
                        padding=temporal_kernel // 2,
                        groups=n_channels,
                        bias=False,
                    ),
                    nn.BatchNorm1d(n_channels),
                    nn.ELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(n_filter_banks)
            ]
        )

        self.head = nn.Sequential(
            nn.Conv1d(n_channels, emb_size, kernel_size=1),
            nn.AdaptiveAvgPool1d(3),
            nn.Flatten(start_dim=1),
        )

    def forward(self, x):  # x: (B, C, T)
        B, C, T = x.shape
        bank_outputs = []
        for conv in self.filter_convs:
            z = conv(x)  # (B, C, T')
            z = self.head(z)  # (B, D*3) - pooled temporal to 3 bins
            bank_outputs.append(z)

        out = torch.stack(bank_outputs, dim=1)  # (B, F, D*3)
        out = out.view(B, self.F, self.D, 3)  # (B, F, D, 3)
        out = out.mean(dim=-1)  # (B, F, D) - pool temporal bins
        return out  # (B, F, D)


class MultiTemporalCollapse_ChannelsProject_FilterBanks(nn.Module):
    def __init__(
        self,
        n_channels,
        n_filter_banks,
        emb_size=40,
        temporal_kernel=43,
        dropout=0.5,
        n_time_points=None,
    ):
        super().__init__()
        self.F = n_filter_banks
        self.D = emb_size

        # F independent temporal filters — each learns different frequency response
        self.filter_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        n_channels,
                        n_channels,
                        kernel_size=temporal_kernel,
                        padding=temporal_kernel // 2,
                        groups=n_channels,
                        bias=False,
                    ),
                    nn.BatchNorm1d(n_channels),
                    nn.ELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(n_filter_banks)
            ]
        )

        # Shared spatial + temporal projection — C channels × T' → emb_size
        # Applied identically per bank after filtering
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # (B*C, T') → (B*C, 1)
            nn.Flatten(),  # (B*C,)
        )
        # After stacking: (B, F, C) → mean over C → (B, F) → Linear → (B, F, D)
        self.proj = nn.Linear(n_channels, emb_size)

    def forward(self, x):  # x: (B, C, T)
        B, C, T = x.shape
        bank_outputs = []
        for conv in self.filter_convs:
            z = conv(x)  # (B, C, T')
            z = z.mean(dim=-1)  # (B, C) — pool T per channel per bank
            bank_outputs.append(z)

        out = torch.stack(bank_outputs, dim=1)  # (B, F, C)
        out = self.proj(out)  # (B, F, D) — Linear(C, D) per bank
        return out  # (B, F, D)


class MultiscaleTemporalCollapse_ChannelsExpand_FilterBanks(nn.Module):
    def __init__(
        self,
        n_channels,
        n_filter_banks,
        emb_size=40,
        dropout=0.5,
        fs=250,
        temporal_kernel=None,
        n_time_points=None,
    ):
        super().__init__()
        self.F = n_filter_banks
        self.D = emb_size

        # Multiple kernel sizes covering different frequency scales
        # Small kernel → sensitive to high frequencies
        # Large kernel → sensitive to low frequencies
        kernel_sizes = [
            max(3, int(fs / 30)),  # ~8 samples @ 250Hz → gamma range
            max(3, int(fs / 13)),  # ~19 samples → beta range
            max(3, int(fs / 8)),  # ~31 samples → alpha range
            max(3, int(fs / 4)),  # ~62 samples → theta range
            max(3, int(fs / 1)),  # ~250 samples → delta range
        ]
        # Make all odd for symmetric padding
        kernel_sizes = [k if k % 2 == 1 else k + 1 for k in kernel_sizes]
        self.n_scales = len(kernel_sizes)

        # One depthwise conv per scale — each sensitive to a different frequency range
        self.scale_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        n_channels,
                        n_channels,
                        kernel_size=k,
                        padding=k // 2,
                        groups=n_channels,
                        bias=False,
                    ),
                    nn.BatchNorm1d(n_channels),
                    nn.ELU(),
                    nn.Dropout(dropout),
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),  # (B, n_channels)
                )
                for k in kernel_sizes
            ]
        )

        # Project concatenated multi-scale features to F*D
        # Input dim: n_channels * n_scales
        self.bank_heads = nn.Linear(
            n_channels * self.n_scales, n_filter_banks * emb_size
        )

    def forward(self, x):  # x: (B, C, T)
        # Extract features at each temporal scale independently
        scale_features = [conv(x) for conv in self.scale_convs]  # list of (B, C)
        z = torch.cat(scale_features, dim=-1)  # (B, C * n_scales)
        out = self.bank_heads(z)  # (B, F*D)
        return out.view(x.size(0), self.F, self.D)  # (B, F, D)


class DualPath_FilterBanks(nn.Module):
    def __init__(
            self, 
            n_channels,
            n_filter_banks,
            emb_size,
            temporal_kernel=25,
            dropout=0.5,
            n_time_points=None
    ):
        self.F = n_filter_banks
        self.D = emb_size

        self.backbone = nn.Sequential(
            nn.Conv1d(
                n_channels,
                n_channels,
                kernel_size=temporal_kernel,
                padding=temporal_kernel // 2,
                groups=n_channels,
                bias=False,
            ),
            nn.BatchNorm1d(n_channels),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.temporal_pool_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(n_channels, n_filter_banks * emb_size),
            nn.ELU()
        )

        self.banks_spatiotemporal_embedding = nn.Sequential([
            nn.Conv2d(
                in_channels = n_filter_banks,
                out_channels = n_filter_banks * emb_size,
                kernel_size = 3, 
                groups = n_filter_banks
            ),
            nn.Conv2d(
                in_channels = n_filter_banks,
                out_channels = n_filter_banks * emb_size,
                kernel_size = 3, 
                groups = n_filter_banks
            ),
            nn.BatchNorm2d(n_filter_banks * emb_size),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d(1, 1)
        ])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, T = x.size()
        z = self.backbone(x)                      # (B, C, T')
        z1 = self.temporal_pool_head(z)           # (B, F*D)
        z1 = rearrange(z1, 'b (f d) -> b f d')    # (B, F, D)   

        z = z.unsqueeze(1).expand(-1, self.F, -1, -1).contiguous
        z2 = self.banks_spatiotemporal_embedding(z)
        z2 = z2.view(B, self.F, self.D)

        return z1 + z2


class TemporalCollapse_ChannelsExpand_FilterBanks(nn.Module):
    def __init__(
        self,
        n_channels,
        n_filter_banks,
        emb_size=40,
        temporal_kernel=25,
        dropout=0.5,
        n_time_points=None,
    ):
        super().__init__()
        self.F = n_filter_banks
        self.D = emb_size

        # Heavy backbone runs once
        self.backbone = nn.Sequential(
            nn.Conv1d(
                n_channels,
                n_channels,
                kernel_size=temporal_kernel,
                padding=temporal_kernel // 2,
                groups=n_channels,
                bias=False,
            ),
            nn.BatchNorm1d(n_channels),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),  # (B, n_channels)
        )
        self.bank_heads = nn.Linear(n_channels, n_filter_banks * emb_size)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):  # x: (B, C, T)
        z = self.backbone(x)  # (B, C)
        out = self.bank_heads(z)  # (B, F*D)
        return out.view(x.size(0), self.F, self.D)  # (B, F, D)


class SpatioTemporalConv_FilterBanks(nn.Module):
    """Learnable 2D filter bank embedding for EEG signals.

    Applies F learned 2D convolutional filters over (C, T) to produce
    F frequency-like embeddings of shape (B, F, D). Each filter learns
    to extract a different spectral-spatial pattern without requiring
    hand-crafted bandpass preprocessing.

    The design uses depthwise-separable convolutions:
      - A spatial conv across channels (kernel: C_kernel x 1) captures
        cross-channel patterns per filter bank.
      - A temporal conv along time (kernel: 1 x T_kernel) captures
        oscillatory structure at a given scale.
      - Global average pooling collapses (C', T') → a single D-dim vector
        per filter bank.

    Args:
        n_channels: Number of EEG channels C.
        n_times: Number of time points T.
        n_filter_banks: Number of learned filter banks F.
        emb_size: Output embedding dimension D.
        spatial_kernel: Kernel height covering channel dimension.
                        Defaults to n_channels (full spatial extent).
        temporal_kernel: Kernel width covering time dimension.
                         Larger → sensitive to lower frequencies.
        dropout: Dropout probability applied after each block.

    Input shape:
        x: (B, C, T)

    Output shape:
        (B, F, D)
    """

    def __init__(
        self,
        n_channels,
        # n_times,
        n_filter_banks,
        emb_size=40,
        spatial_kernel=None,
        temporal_kernel=25,
        dropout=0.5,
        n_time_points=None,
    ):
        super().__init__()
        self.F = n_filter_banks
        self.D = emb_size
        spatial_kernel = spatial_kernel or n_channels  # default: full spatial extent

        # Each filter bank is an independent 2D conv pipeline so filters
        # cannot share weights and are forced to specialise.
        self.filter_banks = nn.ModuleList(
            [
                nn.Sequential(
                    # Treat input as (B, 1, C, T) — single in-channel 2D image
                    # Spatial conv: learns cross-channel weighting
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=emb_size,
                        kernel_size=(spatial_kernel, 1),
                        padding=(spatial_kernel // 2, 0),
                        bias=False,
                    ),
                    nn.BatchNorm2d(emb_size),
                    nn.ELU(),
                    nn.Dropout2d(dropout),
                    # Temporal conv: learns oscillatory structure at this bank's scale
                    nn.Conv2d(
                        in_channels=emb_size,
                        out_channels=emb_size,
                        kernel_size=(1, temporal_kernel),
                        padding=(0, temporal_kernel // 2),
                        groups=emb_size,  # depthwise — each feature evolves independently
                        bias=False,
                    ),
                    nn.BatchNorm2d(emb_size),
                    nn.ELU(),
                    nn.Dropout2d(dropout),
                    # Collapse spatial and temporal dims → single vector per filter bank
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),  # (B, emb_size)
                )
                for _ in range(n_filter_banks)
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d,)):
            nn.init.trunc_normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d,)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):  # x: (B, C, T)
        x = x.unsqueeze(1)  # → (B, 1, C, T)
        out = torch.stack([fb(x) for fb in self.filter_banks], dim=1)  # → (B, F, D)
        return out


class FilterBanksEmbedding(nn.Module):
    """Learnable 2D filter bank embedding for EEG signals.

    Runs all F filter banks in two grouped Conv2d calls instead of
    F sequential forward passes, eliminating the Python loop bottleneck.

    Input:  (B, C, T)
    Output: (B, F, D)
    """

    def __init__(
        self,
        n_channels,
        n_filter_banks,
        emb_size=40,
        spatial_kernel=None,
        temporal_kernel=25,
        dropout=0.5,
        n_time_points=None,
    ):
        super().__init__()
        self.F = n_filter_banks
        self.D = emb_size
        spatial_kernel = spatial_kernel or n_channels

        # All F banks fused: groups=F keeps each bank's filters independent
        # Input treated as (B, F, C, T) — F copies of the same signal
        self.spatial_conv = nn.Conv2d(
            in_channels=self.F,
            out_channels=self.F * emb_size,
            kernel_size=(spatial_kernel, 1),
            padding=(spatial_kernel // 2, 0),
            groups=self.F,
            bias=False,
        )
        self.spatial_bn = nn.BatchNorm2d(self.F * emb_size)
        self.spatial_drop = nn.Dropout2d(dropout)

        self.temporal_conv = nn.Conv2d(
            in_channels=self.F * emb_size,
            out_channels=self.F * emb_size,
            kernel_size=(1, temporal_kernel),
            padding=(0, temporal_kernel // 2),
            groups=self.F * emb_size,  # fully depthwise
            bias=False,
        )
        self.temporal_bn = nn.BatchNorm2d(self.F * emb_size)
        self.temporal_drop = nn.Dropout2d(dropout)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):  # x: (B, C, T)
        B, C, T = x.shape
        # Give each filter bank its own copy of the input: (B, F, C, T)
        x = x.unsqueeze(1).expand(-1, self.F, -1, -1).contiguous()

        x = self.spatial_drop(
            F.elu(self.spatial_bn(self.spatial_conv(x)))
        )  # (B, F*D, C', T)
        x = self.temporal_drop(
            F.elu(self.temporal_bn(self.temporal_conv(x)))
        )  # (B, F*D, 1', T')
        x = self.pool(x)  # (B, F*D, 1, 1)
        x = x.view(B, self.F, self.D)  # (B, F, D)
        return x


# =============================================================================
# Main Model
# =============================================================================


FILTER_BANKS_VARIANTS = {
    "MultiScaleTemporalPatchEmbedding_FilterBanks": MultiScaleTemporalPatchEmbedding_FilterBanks,
    "MultiTemporalConvPool_ChannelsProject_FilterBanks": MultiTemporalConvPool_ChannelsProject_FilterBanks,
    "MultiTemporalConvFixedPool_ChannelsProject_FilterBanks": MultiTemporalConvFixedPool_ChannelsProject_FilterBanks,
    "MultiTemporalCollapse_ChannelsProject_FilterBanks": MultiTemporalCollapse_ChannelsProject_FilterBanks,
    "MultiscaleTemporalCollapse_ChannelsExpand_FilterBanks": MultiscaleTemporalCollapse_ChannelsExpand_FilterBanks,
    "TemporalCollapse_ChannelsExpand_FilterBanks": TemporalCollapse_ChannelsExpand_FilterBanks,
    "SpatioTemporalConv_FilterBanks": SpatioTemporalConv_FilterBanks,
    "FilterBanksEmbedding": FilterBanksEmbedding,
    "DualPath_FilterBanks": DualPath_FilterBanks
}

DEFAULT_FILTER_BANKS_VARIANT = "MultiTemporalConvPool_ChannelsProject_FilterBanks"


class MTFC(nn.Module):
    """Multi-Scale Temporal Frequency Conformer for EEG classification.

    MTFC is a transformer-based architecture that combines spectral-spatio-temporal
    embeddings with frequency-specific attention mechanisms for EEG signal classification.
    It supports multiple spectral-spatio-temporal (SST) methods:
    - st_addition: Simple addition of spatial and temporal embeddings
    - st_addition_projection: Addition of psatial and temporal embeddiongs with shared projection
    - stf_attention_temporal_values: Cross-attention between spatiotemporal and temporal embeddings
    - filter_banks: Multi-band patch embeddings with element-wise combination

    Args:
        args: Configuration object containing:
            - data_name: Dataset name
            - chn: Number of EEG channels
            - time_sample_num: Number of time samples
            - patch_size: Temporal patch size
            - class_num: Number of classification classes
            - gate_flag: Gating mechanism flag
            - posemb_flag: Positional embedding flag
            - branch: Branch configuration string:
                - 'all': single transformer for all embeddings (legacy)
                - 'ft_s': one transformer for freq-temporal, concatenated with spatial
                - 'f_t_s': separate transformers for freq, temporal, spatial, all concatenated
                - 'ft_s': one transformer for freq-temporal, concatenated with spatial
            - chn_attn_flag: Channel attention flag
            - fts_attn_flag: FTS attention flag
            - sst_method: Spectral-spatio-temporal method
            - stft_reconstruction: STFT reconstruction flag
            - spa_dim: Spatial dimension
        n_filter_banks: Number of filter banks.
        wsize_divisor: Window size divisor for STFT.
        freq_downsample: Frequency downsampling factor.
        n_times: Number of time points.
        patch_emb_size: Patch embedding dimension.
        n_heads_patch: Number of attention heads for patch embeddings.
        sst_emb_size: SST embedding dimension (FTS).
        depth: Transformer encoder depth.
        n_classes: Number of output classes.
        fs: Sampling frequency.

    Input shape:
        x: (B, 1, C, T) - EEG signals with dummy frequency dimension

    Output shapes:
        If stft_reconstruction=True:
            loomed_stft: (B, C, F, T) - reconstructed STFT
            x_embed: (B, FTS) - embedding for classification
            out: (B, n_classes) - class logits
        Otherwise:
            None, x_embed, out
    """

    def __init__(
        self,
        args,
        n_filter_banks=4,
        wsize_divisor=2,
        freq_downsample=2,
        n_times=1000,
        patch_emb_size=40,
        n_heads_patch=4,
        sst_emb_size=40,
        depth=5,
        n_classes=2,
        fs=250,
        temporal_kernel=43,
        filter_banks_variant=DEFAULT_FILTER_BANKS_VARIANT,
    ):
        super().__init__()
        self.temporal_kernel = temporal_kernel
        self.P = (args.time_sample_num - 1) // args.patch_size
        self.C = args.chn
        self.D = patch_emb_size
        self.H = n_heads_patch
        self.FTS = sst_emb_size
        self.F = n_filter_banks // freq_downsample
        self.FP = self.F * self.P
        self.fs = fs
        self.n_classes = n_classes
        self.gate_flag = args.gate_flag
        self.posemb_flag = args.posemb_flag
        self.branch = args.branch
        self.chn_attn_flag = args.chn_attn_flag
        self.fts_attn_flag = args.fts_attn_flag
        self.sst_method = args.sst_method
        self.stft_reconstruction = args.stft_reconstruction
        self.ct_shared_projection = getattr(args, "ct_shared_projection", True)
        self.sst_shared_projection = getattr(args, "sst_shared_projection", True)
        self.filter_banks_variant = filter_banks_variant

        self.branch_spec = parse_branch_config(self.branch)
        if self.sst_method is False:
            assert all("f" not in dims for dims in self.branch_spec), (
                "Cannot use frequency branch when sst_method=False"
            )

        wsize = int((self.F - 1) * 2)
        tstep = math.ceil(wsize / wsize_divisor)
        self.stft_length = math.ceil(n_times / tstep)
        self.stft_temporal_loom = nn.Linear(self.P, self.P)

        self._build_sst_layers(args)
        self._build_shared_layers(args)
        self._build_positional_embeddings(args)
        self._build_transformers(depth)
        # self._build_fts_attention_pooling(args)
        self._build_branch_attention_pooling()
        self._build_classifier()

    def _build_branch_attention_pooling(self):
        """Build attention pooling for each branch and embedding type.

        Creates attention pooling layers for each branch that can process
        frequency, temporal, and spatial embeddings separately after transformer.
        """
        self.branch_attention_pool = nn.Sequential(
            nn.Linear(self.transformer_dim, self.transformer_dim),
            nn.Tanh(),
            nn.Linear(self.transformer_dim, 1),
        )

    def _apply_branch_attention_pooling(self, branch_output, branch_dims):
        """Apply attention pooling to transformer output after processing.

        Args:
            branch_output: Transformer output (B, N, D) where N is sequence length
            branch_dims: Tuple of embedding types (e.g., ('f', 't'))

        Returns:
            Summed attention-pooled embeddings (B, transformer_dim)
        """
        pooled_embeddings = []
        seq_idx = 0

        attn_pool = self.branch_attention_pool

        if branch_dims == ("f",):
            seq_len = self.F
        elif branch_dims == ("t",):
            seq_len = self.P
        elif branch_dims == ("s",):
            seq_len = self.C
        elif branch_dims == ("f", "t"):
            seq_len = self.F * self.P
        elif branch_dims == ("f", "s"):
            seq_len = self.F * self.C
        elif branch_dims == ("t", "s"):
            seq_len = self.P * self.C
        elif branch_dims == "fts":
            seq_len = self.F * self.P * self.C

        dim_output = branch_output[:, seq_idx : seq_idx + seq_len]  # type: ignore
        if self.fts_attn_flag:
            attn_weights = torch.softmax(attn_pool(dim_output), dim=1)
            pooled = torch.sum(attn_weights * dim_output, dim=1)
        else:
            pooled = torch.mean(dim_output, dim=1)
        pooled_embeddings.append(pooled)

        return torch.stack(pooled_embeddings, dim=0).sum(dim=0)

    def _build_transformers(self, depth):
        """Build transformer encoders based on branch configuration.

        Creates separate transformer encoders for each branch specified in the
        branch configuration. Each transformer processes concatenated embeddings
        from its specified dimensions.
        """
        self.transformers = nn.ModuleDict()

        transformer_dim = self.FTS if self.sst_shared_projection else self.D

        for i, branch in enumerate(self.branch_spec):
            key = f"branch_{i}"
            self.transformers[key] = TransformerEncoder(depth, transformer_dim)

        self.num_branches = len(self.branch_spec)
        self.transformer_dim = transformer_dim

        self.branch_fusion_transformer = TransformerEncoder(1, transformer_dim)

    def _build_classifier(self):
        """Build classifier head based on total concatenated embedding dimension.

        The classifier input dimension is transformer_dim * num_branches since embeddings
        from all branches are concatenated.
        """
        classifier_input_dim = self.transformer_dim * self.num_branches
        self.classifier = ClassificationHead(classifier_input_dim, self.n_classes)

    def _build_sst_layers(self, args):
        """Build spectral-spatio-temporal method specific layers.

        Creates temporal embeddings, ST mixers, attention heads, and spectrogram
        estimators based on the selected sst_method.
        """
        if self.sst_method in [
            "st_addition",
            "st_addition_projection",
            "stf_attention_temporal_values",
        ]:
            self.frequency_embedding = nn.Sequential(
                nn.Linear(self.C * self.P, self.D),
                nn.ELU(),
                nn.Linear(self.D, self.D),
                nn.ELU(),
            )

        if self.sst_method in [
            "st_addition",
            "st_addition_projection",
            "stf_attention_temporal_values",
        ]:
            self.temporal_embedding = PatchEmbeddingTemporal(
                data_name=args.data_name,
                in_planes=args.chn,
                out_planes=self.D,
                kernel_size=63,
                radix=1,
                patch_size=args.patch_size,
                time_points=args.time_sample_num,
                num_classes=args.class_num,
            )

            if self.sst_method == "st_addition":
                self.st_mixer = STAddition(self.C, self.P)
            elif self.sst_method == "st_addition_projection":
                self.st_mixer = STAdditionProjection(self.C, self.P, self.D)
            elif self.sst_method == "stf_attention_temporal_values":
                self.st_mixer = STAddition(self.C, self.P)
                self.stf_attention_head = N_CrossAttentionHeads(
                    emb_size=self.D,
                    num_heads=self.H,
                    n_comps=self.F,
                    AttnClass=SpatioTemporal_Temporal_AttentionHead,
                )

            self.spectrogram_estimator = SpectrogramEstimator(
                method=self.sst_method,
                emb_size=self.D,
                n_freqs=self.F,
                num_channels=self.C,
                num_patches=self.P,
            )

        elif self.sst_method == "filter_banks":
            self.temporal_embedding = PatchEmbeddingTemporal(
                data_name=args.data_name,
                in_planes=args.chn,
                out_planes=self.D,
                kernel_size=63,
                radix=1,
                patch_size=args.patch_size,
                time_points=args.time_sample_num,
                num_classes=args.class_num,
            )
            FilterBanksClass = FILTER_BANKS_VARIANTS.get(
                self.filter_banks_variant, DEFAULT_FILTER_BANKS_VARIANT
            )
            self.frequency_embedding = FilterBanksClass(
                n_channels=self.C,
                n_filter_banks=self.F,
                emb_size=self.D,
                temporal_kernel=self.temporal_kernel,
                n_time_points=args.time_sample_num,
            )
            if self.stft_reconstruction == "frequency":
                freq_dim = self.FTS if self.sst_shared_projection else self.D
                self.freq_to_bandpowers = nn.Sequential(
                    nn.Linear(freq_dim, freq_dim),
                    nn.ELU(),
                    nn.Linear(freq_dim, 1),
                )
            elif self.stft_reconstruction:
                self.spectrogram_estimator = SpectrogramEstimator(
                    method="filter_banks",
                    emb_size=self.D,
                    n_freqs=self.F,
                    num_channels=self.C,
                    num_patches=self.P,
                    use_ct_shared_projection=self.ct_shared_projection,
                )
        else:
            self.temporal_embedding = PatchEmbeddingTemporal(
                data_name=args.data_name,
                in_planes=args.chn,
                out_planes=self.D,
                kernel_size=63,
                radix=1,
                patch_size=args.patch_size,
                time_points=args.time_sample_num,
                num_classes=args.class_num,
            )

    def _build_shared_layers(self, args):
        """Build spatial embedding and shared projection layers.

        Creates channel embeddings, shared projection networks, and FTS attention
        pooling layers.
        """
        self.channel_embedding = PatchEmbeddingSpatial(
            spa_dim=args.spa_dim, emb_size=self.D
        )

        if self.sst_method is not False and self.sst_shared_projection:
            self.sst_shared_projection_layer = nn.Sequential(
                nn.Linear(self.D, self.D * 2),
                nn.ELU(),
                nn.Linear(self.D * 2, self.FTS),
                nn.ELU(),
            )

    def _build_fts_attention_pooling(self, args):
        """Build FTS attention pooling layer.

        Uses transformer_dim (which is FTS if sst_shared_projection is True, else D)
        to ensure correct dimension matching.
        """
        if args.fts_attn_flag:
            self.fts_attn_pool = nn.Sequential(
                nn.Linear(self.transformer_dim, self.transformer_dim),
                nn.Tanh(),
                nn.Linear(self.transformer_dim, 1),
            )

    def _build_positional_embeddings(self, args):
        """Build positional embeddings if enabled.

        Creates learnable positional embeddings for frequency, temporal,
        and spatial dimensions based on the SST method.
        """
        if args.posemb_flag:
            self.pos_embedding_frequency = nn.Parameter(torch.randn(1, self.F, self.D))
            self.pos_embedding_temporal = nn.Parameter(torch.randn(1, self.P, self.D))
            self.pos_embedding_spatial = nn.Parameter(torch.randn(1, self.C, self.D))

    def forward(self, x):
        """Forward pass of MTFC model.

        Args:
            x: Input EEG signals (B, 1, C, T).

        Returns:
            If stft_reconstruction=True:
                (loomed_stft, x_embed, out)
            Otherwise:
                (None, x_embed, out)
        """
        stft = None

        x_embed_temporal = self.temporal_embedding(x.squeeze(1))
        x_embed_spatial = self.channel_embedding(x.squeeze(1))

        if self.sst_method == "filter_banks":
            x_embed_frequency = self.frequency_embedding(x.squeeze(1))
            if self.stft_reconstruction == "frequency":
                pass  # Will handle in output computation
            elif self.stft_reconstruction:
                stft = self._compute_spectrogram(
                    x_embed_temporal, x_embed_spatial, x_embed_frequency
                )
        else:
            stft = self._compute_spectrogram(x_embed_temporal, x_embed_spatial, None)
            x_embed_frequency = self._compute_frequency_embedding(stft)

        x_embed_temporal, x_embed_spatial, x_embed_frequency = (
            self._apply_positional_encoding(
                x_embed_temporal, x_embed_spatial, x_embed_frequency
            )
        )

        if self.sst_shared_projection and self.sst_method is not False:
            x_embed_frequency = self.sst_shared_projection_layer(x_embed_frequency)
            x_embed_temporal = self.sst_shared_projection_layer(x_embed_temporal)
            x_embed_spatial = self.sst_shared_projection_layer(x_embed_spatial)

        if self.branch == "all":
            z_hat_f = (
                x_embed_frequency.unsqueeze(2)
                .unsqueeze(1)
                .expand(-1, self.C, -1, self.P, -1)
            )
            z_hat_t = (
                x_embed_temporal.unsqueeze(1)
                .unsqueeze(2)
                .expand(-1, self.C, self.F, -1, -1)
            )
            z_hat_s = (
                x_embed_spatial.unsqueeze(2)
                .unsqueeze(3)
                .expand(-1, -1, self.F, self.P, -1)
            )

            x_embed_fts = (z_hat_s + z_hat_f + z_hat_t).contiguous()
            x_embed_fts = rearrange(x_embed_fts, "b c f t d -> b (c f t) d")
            x_embed_fts = self.transformers["branch_0"](x_embed_fts)
            x_embed_fts = self._apply_branch_attention_pooling(x_embed_fts, ("fts"))
            outputs = [x_embed_fts]
        else:
            outputs = []
            for i, branch_dims in enumerate(self.branch_spec):
                branch_key = f"branch_{i}"

                if branch_dims == ("f", "t"):
                    branch_emb_f = x_embed_frequency.unsqueeze(2).expand(
                        -1, -1, self.P, -1
                    )
                    branch_emb_t = x_embed_temporal.unsqueeze(1).expand(
                        -1, self.F, -1, -1
                    )
                    branch_emb = (branch_emb_f + branch_emb_t).contiguous()
                    branch_emb = rearrange(branch_emb, "b f p d -> b (f p) d")
                elif branch_dims == ("t", "s"):
                    branch_emb_t = x_embed_temporal.unsqueeze(1).expand(
                        -1, self.C, -1, -1
                    )
                    branch_emb_s = x_embed_spatial.unsqueeze(2).expand(
                        -1, -1, self.P, -1
                    )
                    branch_emb = (branch_emb_t + branch_emb_s).contiguous()
                    branch_emb = rearrange(branch_emb, "b c p d -> b (c p) d")
                elif branch_dims == ("f", "s"):
                    branch_emb_f = x_embed_frequency.unsqueeze(1).expand(
                        -1, self.C, -1, -1
                    )
                    branch_emb_s = x_embed_spatial.unsqueeze(2).expand(
                        -1, -1, self.F, -1
                    )
                    branch_emb = (branch_emb_f + branch_emb_s).contiguous()
                    branch_emb = rearrange(branch_emb, "b c f d -> b (c f) d")
                elif branch_dims == ("t",):
                    branch_emb = x_embed_temporal
                elif branch_dims == ("f",):
                    branch_emb = x_embed_frequency
                elif branch_dims == ("s",):
                    branch_emb = x_embed_spatial

                branch_emb = self.transformers[branch_key](branch_emb)  # type: ignore
                branch_pooled = self._apply_branch_attention_pooling(
                    branch_emb, branch_dims
                )
                outputs.append(branch_pooled)

        x_embed_fts = torch.cat(outputs, dim=-1)

        x_embed = x_embed_fts
        _, out = self.classifier(x_embed)

        if self.stft_reconstruction == "frequency" and hasattr(
            self, "freq_to_bandpowers"
        ):
            band_powers = self.freq_to_bandpowers(x_embed_frequency).squeeze(-1)
            return band_powers, x_embed, out
        elif self.stft_reconstruction and stft is not None:
            loomed_stft = F.elu(self.stft_temporal_loom(stft.permute(0, 3, 1, 2)))
            loomed_stft = loomed_stft.permute(
                0, 2, 1, 3
            )  # (B, F, C, P) -> (B, C, F, P)
            return loomed_stft, x_embed, out
        return None, x_embed, out

    def _compute_spectrogram(
        self, x_embed_temporal, x_embed_spatial, x_embed_frequency
    ):
        """Compute spectrogram based on SST method.

        Args:
            x_embed_temporal: Temporal embeddings.
            x_embed_spatial: Spatial embeddings.

        Returns:
            Spectrogram estimate (B, C, P, F) or None.
        """
        if self.sst_method in ["st_addition", "st_addition_projection"]:
            z_st = self.st_mixer(x_embed_temporal, x_embed_spatial)
            return self.spectrogram_estimator(z_st, x_embed_temporal, x_embed_spatial)

        elif self.sst_method == "stf_attention_temporal_values":
            z_st = self.st_mixer(x_embed_temporal, x_embed_spatial)
            z_st = self.stf_attention_head(z_st, x_embed_temporal)
            return self.spectrogram_estimator(z_st, x_embed_temporal, x_embed_spatial)

        elif self.sst_method == "filter_banks":
            return self.spectrogram_estimator(
                x_embed_frequency, x_embed_temporal, x_embed_spatial
            )

        return None

    def _compute_frequency_embedding(self, stft):
        """Compute frequency embedding from spectrogram.

        Args:
            stft: Spectrogram estimate (B, C, P, F).

        Returns:
            Frequency embeddings (B, F, D) or None.
        """
        if (
            self.sst_method
            in [
                "st_addition",
                "st_addition_projection",
                "stf_attention_temporal_values",
                "filter_banks",
            ]
            and stft is not None
        ):
            return self.frequency_embedding(rearrange(stft, "b c p f -> b f (c p)"))
        return None

    def _apply_positional_encoding(
        self, x_embed_temporal, x_embed_spatial, x_embed_frequency
    ):
        """Apply positional encodings to embeddings.

        Args:
            x_embed_temporal: Temporal embeddings.
            x_embed_spatial: Spatial embeddings.
            x_embed_frequency: Frequency embeddings.

        Returns:
            Tuple of (temporal, frequency) embeddings with positional encoding.
        """
        if self.posemb_flag:
            x_embed_temporal = x_embed_temporal + self.pos_embedding_temporal
            x_embed_spatial = x_embed_spatial + self.pos_embedding_spatial
            if x_embed_frequency is not None:
                x_embed_frequency = x_embed_frequency + self.pos_embedding_frequency

        return x_embed_temporal, x_embed_spatial, x_embed_frequency
