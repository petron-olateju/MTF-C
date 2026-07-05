"""Multi-Scale Temporal Frequency Conformer (MTFC) for EEG classification.

This module implements the MTFC architecture which combines spectral-spatio-temporal
embeddings with a transformer-based conformer for EEG signal classification.
"""

import math
import re
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .DBConformer import PatchEmbeddingTemporal, PatchEmbeddingSpatial
from .DBConformer import TransformerEncoder, ClassificationHead

# =============================================================================
# Utility Functions
# =============================================================================

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

class MultiscaleTemporalCollapse_ChannelsExpand_FilterBanks(nn.Module):
    """
    - Add grouped spatial mixer before importance scoring:
      channels interact across frequency scales, giving
      the importance encoder access to cross-channel and cross-scale
      contrast (e.g. lateral asymmetry) before per-instance scoring.
    """

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
        self.C = n_channels
        self.F = n_filter_banks
        self.D = emb_size

        kernel_sizes = [
            max(3, int(fs / 30)),
            max(3, int(fs / 13)),
            max(3, int(fs / 8)),
            max(3, int(fs / 4)),
            max(3, int(fs / 1)),
        ]
        kernel_sizes = [k if k % 2 == 1 else k + 1 for k in kernel_sizes]
        self.n_scales = len(kernel_sizes)

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
                    nn.AdaptiveAvgPool1d(4),
                )
                for k in kernel_sizes
            ]
        )

        CN = n_channels * self.n_scales

        # Cross-channel mixing within each frequency scale.
        # groups=n_scales keeps scales separate — channels only
        # interact with channels from the same frequency scale.
        # This lets the encoder see lateral asymmetry per band
        # before importance scoring.
        self.spatial_mixer = nn.Sequential(
            nn.Conv1d(CN, CN, kernel_size=1, groups=1, bias=False),
        )

        # Per-bank importance encoders — now scoring mixed slots
        # so alpha reflects cross-channel contrast, not just
        # individual channel activity.
        self.importance_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(CN, CN, kernel_size=3, padding=1, groups=CN, bias=False),
                    nn.ELU(),
                    nn.Conv1d(CN, CN, kernel_size=4, groups=CN, bias=True),
                )
                for _ in range(n_filter_banks)
            ]
        )

        # F separate embedding heads
        self.embedding_heads = nn.ModuleList(
            [nn.Linear(CN * 4, emb_size) for _ in range(n_filter_banks)]
        )

    def forward(self, x):  # (B, C, T)
        # multi-scale depthwise filtering
        scale_features = [conv(x) for conv in self.scale_convs]  # list of (B, C, 4)
        z = torch.stack(scale_features, dim=2)  # (B, C, N, 4)
        z = rearrange(z, "b c n t -> b (c n) t")  # (B, C*N, 4)

        # cross-channel mixing within each frequency scale
        z = self.spatial_mixer(z)  # (B, C*N, 4)

        # per-bank importance scoring on mixed features + weighted embedding
        bank_outputs = []
        for f in range(self.F):
            alpha = self.importance_encoders[f](z)  # (B, C*N, 1)
            alpha = alpha.squeeze(-1)  # (B, C*N)
            alpha = torch.softmax(alpha, dim=-1)  # (B, C*N)
            alpha = alpha.unsqueeze(-1).expand_as(z)  # (B, C*N, 4)

            z_weighted = z * alpha  # (B, C*N, 4)
            z_flat = rearrange(z_weighted, "b cn t -> b (cn t)")  # (B, C*N*4)
            bank_outputs.append(self.embedding_heads[f](z_flat))  # (B, D)

        z_out = torch.stack(bank_outputs, dim=1)  # (B, F, D)
        return z_out

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

# =============================================================================
# Main Model
# =============================================================================
FILTER_BANKS_VARIANTS = {
    "MultiscaleTemporalCollapse_ChannelsExpand_FilterBanks": MultiscaleTemporalCollapse_ChannelsExpand_FilterBanks,
}
DEFAULT_FILTER_BANKS_VARIANT = "MultiscaleTemporalCollapse_ChannelsExpand_FilterBanks"


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
            - chn_attn_flag: Channel attention flag
            - spectrum_attn_flag: Spectrum attention flag
            - sst_method: Spectrum reconstruction: frequency, STFT, wavelet, Spectral-spatio-temporal method
            - spa_dim: Spatial dimension
        n_filter_banks: Number of filter banks.
        wsize_divisor: Window size divisor for STFT.
        freq_downsample: Frequency downsampling factor.
        n_times: Number of time points.
        patch_emb_size: Patch embedding dimension.
        tem_depth: no of temporal conformer
        chn_depth: no of channel conformer
        spec_depth: no of spectrum conformer
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
        tem_depth=5,
        chn_depth=5,
        spec_depth=5,
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
        self.F = n_filter_banks // freq_downsample
        self.FP = self.F * self.P
        self.fs = fs
        self.n_classes = n_classes
        self.gate_flag = args.gate_flag
        self.posemb_flag = args.posemb_flag
        self.chn_attn_flag = args.chn_attn_flag
        self.spectrum_attn_flag = args.spectrum_attn_flag
        self.temporal_attn_flag = args.temporal_attn_flag
        self.sst_method = args.sst_method
        self.ct_shared_projection = getattr(args, "ct_shared_projection", True)
        self.sst_shared_projection = getattr(args, "sst_shared_projection", True)
        self.filter_banks_variant = filter_banks_variant


        wsize = int((self.F - 1) * 2)
        tstep = math.ceil(wsize / wsize_divisor)
        self.stft_length = math.ceil(n_times / tstep)
        self.stft_temporal_loom = nn.Linear(self.P, self.P)

        self._build_backbones(args)
        self._build_positional_embeddings(args)
        self._build_transformers(tem_depth, chn_depth, spec_depth)
        self._build_branch_attention_pooling()
        self._build_classifier()

    def _build_backbones(self, args):
        """
            - Build channel and spatial embedding adopted from DBConformer
            - Buld Frequency embedding backbone or spectrogram embedding backbone based on self.sst_method
        """

        self.channel_embedding = PatchEmbeddingSpatial(
            spa_dim=args.spa_dim, emb_size=self.D
        )

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

        if self.sst_method in ["frequency_backbone", "channels_frequency_backbone"]:
            FilterBanksClass = FILTER_BANKS_VARIANTS.get(
                self.filter_banks_variant, DEFAULT_FILTER_BANKS_VARIANT
            )
            self.spectrum_embedding = FilterBanksClass(
                n_channels=self.C,
                n_filter_banks=self.F,
                emb_size=self.D,
                temporal_kernel=self.temporal_kernel,
                n_time_points=args.time_sample_num,
            )
            freq_dim = self.D
            self.spectrum_embedding_to_spectrum = nn.Sequential(
                nn.Linear(freq_dim, freq_dim),
                nn.ELU(),
                nn.Linear(freq_dim, 1),
            )
        elif self.sst_method == "spectrogram_backbone":
            pass    # Spectrogram embedding backbone
    
    def _build_positional_embeddings(self, args):
        """Build positional embeddings if enabled.

        Creates learnable positional embeddings for frequency, temporal,
        and spatial dimensions based on the SST method.
        """
        if args.posemb_flag:
            self.pos_embedding_frequency = nn.Parameter(torch.randn(1, self.F, self.D))
            self.pos_embedding_temporal = nn.Parameter(torch.randn(1, self.P, self.D))
            self.pos_embedding_spatial = nn.Parameter(torch.randn(1, self.C, self.D))

    def _build_transformers(self, tem_depth, chn_depth, spec_depth):
        """Build transformer encoders based on branch configuration.

        Creates separate transformer encoders for each branch specified in the
        branch configuration. Each transformer processes concatenated embeddings
        from its specified dimensions.
        """
        self.transformers = nn.ModuleDict()
        if self.sst_method is not None:
            self.transformers['s'] = TransformerEncoder(spec_depth, self.D)
        self.transformers['t'] = TransformerEncoder(tem_depth, self.D)
        self.transformers['c'] = TransformerEncoder(chn_depth, self.D)

    def _build_branch_attention_pooling(self):
        """Build attention pooling for each branch and embedding type.

        Creates attention pooling layers for each branch that can process
        frequency, temporal, and spatial embeddings separately after transformer.
        """
        if self.chn_attn_flag:
            self.channel_attention_pool = nn.Sequential(
                nn.Linear(self.D, self.D),
                nn.Tanh(),
                nn.Linear(self.D, 1),
            )
        if self.spectrum_attn_flag:
            self.spectrum_attn_pool = nn.Sequential(
                    nn.Linear(self.D, self.D),  # D → D
                    nn.Tanh(),
                    nn.Linear(self.D, 1),  # D → 1 (score per spectrum-component embedding)
            )
        if self.temporal_attn_flag:
            self.temporal_attn_pool = nn.Sequential(
                    nn.Linear(self.D, self.D),  # D → D
                    nn.Tanh(),
                    nn.Linear(self.D, 1),  # D → 1 (score per temporal-component embedding)
            )

    def _build_classifier(self):
        """Build classifier head based on total concatenated embedding dimension.

        The classifier input dimension is transformer_dim * num_branches since embeddings
        from all branches are concatenated.
        """
        if self.sst_method is not None:
            classifier_input_dim = self.D * 3
        else:
            classifier_input_dim = self.D * 2
        self.classifier = ClassificationHead(classifier_input_dim, self.n_classes)

    def get_branch_embeddings(self, x):
        if self.sst_method is not None:
            x_embed_spectrum = self.spectrum_embedding(x.squeeze(1))
        else:
            x_embed_spectrum = None
        x_embed_temporal = self.temporal_embedding(x.squeeze(1))
        x_embed_channel = self.channel_embedding(x.squeeze(1))
        x_embed_temporal, x_embed_channel, x_embed_spectrum = (
            self._apply_positional_encoding(
                x_embed_temporal, x_embed_channel, x_embed_spectrum
            )
        )

        if self.sst_method is not None:
            x_s = self.transformers['s'](x_embed_spectrum)
        else:
            x_s = None
        x_t = self.transformers['t'](x_embed_temporal)
        x_c = self.transformers['c'](x_embed_channel)

        return {
            'spectrum': x_s,
            'temporal': x_t,
            'channel': x_c
        }, x_embed_spectrum

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

        branch_embd, x_embed_spectrum = self.get_branch_embeddings(x)
        x_s = branch_embd['spectrum']
        x_t = branch_embd['temporal']
        x_c = branch_embd['channel']

        if self.chn_attn_flag:
            chn_attn_scores = self.channel_attention_pool(x_c)
            chn_attn_weights = F.softmax(chn_attn_scores, dim=1)
            x_c = torch.sum(chn_attn_weights * x_c, dim=1)
        else:
            x_c = x_c.mean(dim=1)

        if self.sst_method is not None:
            if self.spectrum_attn_flag:
                spectrum_attn_scores = self.spectrum_attn_pool(x_s)
                spectrum_attn_weights = F.softmax(spectrum_attn_scores, dim=1)
                x_s = torch.sum(spectrum_attn_weights * x_s, dim=1)
            else:
                x_s = x_s.mean(dim=1)

        if self.temporal_attn_flag:
            temporal_attn_scores = self.temporal_attn_pool(x_t)
            temporal_attn_weights = F.softmax(temporal_attn_scores, dim=1)
            x_t = torch.sum(temporal_attn_weights * x_t, dim=1)
        else:
            x_t = x_t.mean(dim=1)
            
        if self.sst_method is not None:
            x_fused = torch.cat([x_s, x_t, x_c], dim=-1)
        else:
            x_fused = torch.cat([x_t, x_c], dim=-1)

        _, out = self.classifier(x_fused)

        if self.sst_method is not None:
            spectrum = self.spectrum_embedding_to_spectrum(x_embed_spectrum).squeeze(-1)
            return branch_embd, spectrum, x_fused, out
        else:
            return branch_embd, x_fused, out

    def _apply_positional_encoding(
        self, x_embed_temporal, x_embed_channel, x_embed_spectrum
    ):
        """Apply positional encodings to embeddings.

        Args:
            x_embed_temporal: Temporal embeddings.
            x_embed_channel: Spatial embeddings.
            x_embed_spectrum: Spectrum (Frequency, OR Spectrogram or Scalogram) embeddings.

        Returns:
            Tuple of (temporal, channel, spectrum) embeddings with positional encoding
        """
        if self.posemb_flag:
            x_embed_temporal = x_embed_temporal + self.pos_embedding_temporal
            x_embed_channel = x_embed_channel + self.pos_embedding_spatial
            if x_embed_spectrum is not None:
                x_embed_spectrum = x_embed_spectrum + self.pos_embedding_frequency

        return x_embed_temporal, x_embed_channel, x_embed_spectrum
