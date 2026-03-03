import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .DBConformer import PatchEmbeddingTemporal, PatchEmbeddingSpatial
from .DBConformer import TransformerEncoder, ClassificationHead

KERNEL_SIZES = {
    'delta': None,  # will compute below
}

# kernel ~ fs / low_freq / 2, rounded to odd number
def band_kernel_size(fs, low_freq):
    if low_freq is None:
        low_freq = 1  # treat delta as ~1Hz
    k = int(fs / low_freq / 2)
    return k if k % 2 == 1 else k + 1  # ensure odd for symmetric padding

filter_banks = {
    'delta': [1, 4],   # ~large kernel
    'theta': [4, 8],
    'alpha': [8, 12],
    'beta':  [12, 30],
    'gamma': [30, 100],
    'broad': [0.1, 100]  # raw
}


class FilterBanksPatchEmbeddingTemporal(nn.Module):
    def __init__(self, args, n_filter_banks=6, emb_size=40, fs=250):
        super().__init__()

        self.fs = fs
        
        self.n_filter_banks = n_filter_banks
        self.patch_embeddings = nn.ModuleList([
            PatchEmbeddingTemporal(
                data_name=args.data_name,
                in_planes=args.chn,  # number of channels
                out_planes=emb_size,  # Default 40
                kernel_size=band_kernel_size(self.fs, list(filter_banks.values())[i][0]),
                radix=1,
                patch_size=args.patch_size,  # needs to be divisible by the number of time points
                time_points=args.time_sample_num,  # number of time points
                num_classes=args.class_num  # number of classes
            ) for i in range(self.n_filter_banks)
        ])

    def forward(self, x):
        out = [self.patch_embeddings[i](x).unsqueeze(1) for i in range(self.n_filter_banks)]
        out = torch.cat(out, dim=1)
        return out


class MTFC(nn.Module):

    def __init__(self, args, n_filter_banks= 4, wsize_divisor=2, n_times=1000, patch_emb_size=40, sst_emb_size=40, 
            depth=5, n_classes=2, fs=250) -> None:
        super().__init__()

        self.P = args.time_sample_num // args.patch_size  # Example: 1000 // 125 = 8
        self.C = args.chn  # number of channels
        self.D = patch_emb_size
        self.FTS = sst_emb_size
        self.F = n_filter_banks
        self.FP = self.F * self.P
        self.fs = fs
        self.gate_flag = args.gate_flag  # Default False, due to the reduced performance
        self.posemb_flag = args.posemb_flag  # Default True
        self.branch = args.branch  # Default 'all', options=[all, temporal]
        self.chn_atten_flag = args.chn_atten_flag  # Default True
        self.fts_atten_flag = args.fts_atten_flag   # Default True
        self.sst_reconstruction = args.sst_method


        # Match STFT Temporal Length
        if args.sst_method == 'spatio_temporal_embedding':
            wsize = int((self.F - 1) * 2)
            tstep = math.ceil(wsize / wsize_divisor)
            self.stft_length = math.ceil(n_times / tstep)
            self.temporal_loom = nn.Linear(self.P, self.stft_length)


        if (args.sst_method is not False) and (args.sst_method == 'spatio_temporal_embedding'):
            self.spectrogram_generator = nn.Linear(self.D, self.F)
            self.frequency_embedding = nn.Linear(self.C * self.P, self.D)

            self.temporal_embedding = PatchEmbeddingTemporal(
                data_name=args.data_name,
                in_planes=args.chn,  # number of channels
                out_planes=patch_emb_size,  # Default 40
                kernel_size=63,
                radix=1,
                patch_size=args.patch_size,  # needs to be divisible by the number of time points
                time_points=args.time_sample_num,  # number of time points
                num_classes=args.class_num  # number of classes
            )
        elif (args.sst_method is not False) and (args.sst_method == 'filter_banks'):
            self.temporal_embedding = FilterBanksPatchEmbeddingTemporal(args, n_filter_banks=n_filter_banks, emb_size=patch_emb_size, fs=fs)

        self.channel_embedding = PatchEmbeddingSpatial(spa_dim=args.spa_dim, emb_size=patch_emb_size)  # Default 16

        if args.posemb_flag:
            if args.sst_method is not False:
                if self.sst_reconstruction == 'spatio_temporal_embedding':
                    self.pos_embedding_frequency = nn.Parameter(torch.randn(1, self.F, self.D))
                    self.pos_embedding_temporal = nn.Parameter(torch.randn(1, self.P, self.D))
                elif self.sst_reconstruction == 'filter_banks':
                    self.pos_embedding_frequency = nn.Parameter(torch.randn(1, self.F, 1, self.D))
                    self.pos_embedding_temporal = nn.Parameter(torch.randn(1, 1, self.P, self.D))
            else:
                self.pos_embedding_temporal = nn.Parameter(torch.randn(1, 1, self.P, self.D))
            self.pos_embedding_spatial = nn.Parameter(torch.randn(1, self.C, self.D))

        self.sst_projection_space = nn.Linear(self.D, self.FTS)

        if args.fts_atten_flag:
            self.fts_attn_pool = nn.Sequential(
                nn.Linear(self.FTS, self.FTS),
                nn.Tanh(),
                nn.Linear(self.FTS, 1),
            )

        self.fts_transformer = TransformerEncoder(depth, self.FTS)
        self.classifier = ClassificationHead(self.FTS, n_classes)

    def forward(self, x):   # x: (B, F, C, T)
        # x_embed_fp = self.embedding(x[:, :, :self.F, :, :])    # --> (B, F*P, D)
        # x_embed_spatial = self.channel_embedding(x[:, :, -1, :, :].squeeze(1, 2))     # --> (B, C, D)     


        # Get temporal and spatial components embedding
        x_embed_temporal = self.temporal_embedding(x.squeeze(1))    # --> (B, P, D) or (B, F*P, D)
        x_embed_spatial = self.channel_embedding(x.squeeze(1))     # --> (B, C, D)   


        # Get frequency components embdding if frequency components spatio_temporal reconstruction is being used
        if self.sst_reconstruction == 'spatio_temporal_embedding':
            zt = x_embed_temporal.unsqueeze(1).expand(-1, self.C, -1, -1)
            zs = x_embed_spatial.unsqueeze(2).expand(-1, -1, self.P, -1)
            z_st = zs + zt      # Mixing spatial and temporal embeddings
            stft = F.relu(self.spectrogram_generator(z_st))

            x_embed_frequency = rearrange(stft, 'b c p f-> b f (c p)')
            x_embed_frequency = F.relu(self.frequency_embedding(x_embed_frequency))

            loomed_stft = F.relu(self.temporal_loom(rearrange(stft, 'b c p f -> b c f p')))


        # Positional Encoding
        if self.posemb_flag:
            if self.sst_reconstruction == 'spatio_temporal_embedding':
                x_embed_frequency = x_embed_frequency + self.pos_embedding_frequency  # type: ignore # frequency positional encoding
            elif self.sst_reconstruction == 'filter_banks':
                x_embed_temporal = x_embed_temporal + self.pos_embedding_frequency

            x_embed_temporal = x_embed_temporal + self.pos_embedding_temporal  # temporal positional encoding
            x_embed_spatial = x_embed_spatial + self.pos_embedding_spatial  # spatial positional encoding  


        # Project time, space, frequnecy components embedding into the same empedding space
        if self.sst_reconstruction == 'spatio_temporal_embedding':
            x_embed_frequency = self.sst_projection_space(x_embed_frequency)    # type: ignore # --> (B, F, FTS)
        x_embed_temporal = self.sst_projection_space(x_embed_temporal)  # --> (B, P, FTS) or (B, F, P, FTS)
        x_embed_spatial = self.sst_projection_space(x_embed_spatial)    # --> (B, C, FTS)
        

        # Make time-frequency embedding
        if self.sst_reconstruction == 'spatio_temporal_embedding':
            # using spatio-temporal embedding approach from 
            z_hat_f = x_embed_frequency.unsqueeze(2).expand(-1, -1, self.P, -1) # type: ignore
            z_hat_t = x_embed_temporal.unsqueeze(1).expand(-1, self.F, -1, -1)
            x_embed_fp = rearrange(z_hat_f + z_hat_t, 'b f p d -> b (f p) d')
        elif self.sst_reconstruction == 'filter_banks':
            # Squash time and frequency axis if using filter-banks approach
            x_embed_fp = rearrange(x_embed_temporal, 'b f p d -> b (f p) d') 
        else:
            x_embed_fp = x_embed_temporal


        # Mixing TF and Spatial Embeddings
        x_embed_fp = x_embed_fp.unsqueeze(2).expand(-1, -1, self.C, -1)
        x_embed_spatial = x_embed_spatial.unsqueeze(1).expand(-1, self.FP, -1, -1)
        x_embed_fts = (x_embed_fp + x_embed_spatial).contiguous()   # --> (B, F*P, C, FTS)
        x_embed_fts = rearrange(x_embed_fts, 'b t c d -> b (t c) d')


        # Transformer mixed FTS embeddings forward pass
        x_embed_fts = self.fts_transformer(x_embed_fts)


        # Learning FTS Importance
        if self.fts_atten_flag:
            attn_scores = self.fts_attn_pool(x_embed_fts)
            attn_weights = torch.softmax(attn_scores, dim=1)
            x_embed_fts = attn_weights * x_embed_fts
        

        # Classification Head
        x_embed = torch.sum(x_embed_fts, dim=1)
        _, out = self.classifier(x_embed)

        if self.sst_reconstruction == 'spatio_temporal_embedding':
            return loomed_stft, x_embed, out # type: ignore
        return None, x_embed, out

