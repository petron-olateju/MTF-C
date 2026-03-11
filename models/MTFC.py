import math

import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore

from einops import rearrange # type: ignore

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


def create_filter_banks(n_filter_banks, fs, start_freq=1.0):
    nyquist = fs / 2
    band_width = (nyquist - start_freq) / n_filter_banks
    
    filter_banks = {
        i: [round(start_freq + i * band_width, 2), round(start_freq + (i + 1) * band_width, 2)]
        for i in range(n_filter_banks)
    }
    return filter_banks

class st_t_CrossAttentionHead(nn.Module):
    def __init__(self, emb_size, num_heads=2, dropout=0.2):
        super().__init__()
        assert emb_size % num_heads == 0
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads

        self.query_proj = nn.Linear(emb_size, emb_size)
        self.key_proj = nn.Linear(emb_size, emb_size)
        self.value_proj = nn.Linear(emb_size, emb_size)
        self.out_proj = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):  # query: (B, P1, D), key_value: (B, P2, D)
        H, D = self.num_heads, self.head_dim

        Q = self.query_proj(query)
        K = self.key_proj(key_value)
        V = self.value_proj(key_value)

        # Split last dim into (num_heads, head_dim)
        *q_dims, _ = Q.shape
        *k_dims, _ = K.shape

        Q = Q.reshape(*q_dims, H, D)    # (B, F, C, P1, H, D)
        K = K.reshape(*k_dims, H, D)    # (B, P2, H, D)
        V = V.reshape(*k_dims, H, D)    # (B, P2, H, D)

        attn_scores = torch.einsum('bcqhd, bkhd -> bcqhk', Q, K) / (D ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = torch.einsum('bcqhk, bkhd -> bcqhd', attn_probs, V)
        out = out.reshape(*q_dims, self.emb_size)
        out = self.out_proj(out)

        return out

class N_CrossAttentionHeads(nn.Module):
    def __init__(self, emb_size, num_heads=2, n_comps=7, dropout=0.2, AttnClass=st_t_CrossAttentionHead):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.n_comps = n_comps
        self.dropout = dropout

        self.component_attn = nn.ModuleList([
            AttnClass(emb_size=emb_size, num_heads=num_heads, dropout=dropout) 
            for _ in range(self.n_comps)
        ])

    def forward(self, query, key_value):
        out = torch.stack(
            [head(query, key_value) for head in self.component_attn], dim=1
            )
        return out

class FilterBanksPatchEmbeddingTemporal(nn.Module):
    def __init__(self, args, n_filter_banks=6, emb_size=40, fs=250):
        super().__init__()

        self.fs = fs

        filter_banks = create_filter_banks(n_filter_banks, fs=fs, start_freq=1.0)
        
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

    def __init__(self, args, n_filter_banks= 4, wsize_divisor=2, n_times=1000, patch_emb_size=40, n_heads_patch=4, sst_emb_size=40, 
            depth=5, n_classes=2, fs=250) -> None:
        super().__init__()

        self.P = (args.time_sample_num - 1) // args.patch_size  # Example: 1000 // 125 = 8
        self.C = args.chn  # number of channels
        self.D = patch_emb_size
        self.H = n_heads_patch
        self.FTS = sst_emb_size
        self.F = n_filter_banks
        self.FP = self.F * self.P
        self.fs = fs
        self.gate_flag = args.gate_flag  # Default False, due to the reduced performance
        self.posemb_flag = args.posemb_flag  # Default True
        self.branch = args.branch  # Default 'all', options=[all, temporal]
        self.chn_atten_flag = args.chn_atten_flag  # Default True
        self.fts_atten_flag = args.fts_atten_flag   # Default True
        self.sst_method = args.sst_method
        self.stft_reconstruction = args.stft_reconstruction


        # Match STFT Temporal Length
        # if args.stft_reconstruction is True:
        wsize = int((self.F - 1) * 2)
        tstep = math.ceil(wsize / wsize_divisor)
        self.stft_length = math.ceil(n_times / tstep)
        # self.stft_temporal_loom = nn.Linear(self.P, self.stft_length)
        self.stft_temporal_loom = nn.Linear(self.P, self.P)

        
        # Layers Based on Spectral-Spatio-Temporal Method Selected
        if args.sst_method in ['st_addition', 'st_addition_projection', 'stf_attention_temporal_values', 'filter_banks']:
            # self.frequency_embedding = nn.Linear(self.C * self.P, self.D)
            self.frequency_embedding = nn.Sequential(
                nn.Linear(self.C * self.P, self.D),
                nn.ELU(),
                nn.Linear(self.D, self.D),
                nn.ELU())

        if args.sst_method in ['st_addition', 'st_addition_projection', 'stf_attention_temporal_values']:
            if args.sst_method in ['st_addition', 'st_addition_projection']:
                self.spectrogram_generator = nn.Sequential(
                    nn.Linear(self.D, self.D**2),
                    nn.ELU(),
                    nn.Linear(self.D**2, self.F),
                    nn.ELU())
            elif args.sst_method in ['stf_attention_temporal_values']:
                self.spectrogram_generator = nn.Sequential(
                    nn.Linear(self.D, self.D**2),
                    nn.ELU(),
                    nn.Linear(self.D**2, 1),
                    nn.ELU())

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

            if args.sst_method == 'st_addition_projection':
                self.ct_shared_projection = nn.Sequential(
                    nn.Linear(self.D, self.D*2),
                    nn.ELU(),
                    nn.Linear(self.D*2, self.D),
                    nn.ELU()
                )
            elif args.sst_method == 'stf_attention_temporal_values':
                self.stf_attention_head = N_CrossAttentionHeads(
                    emb_size = self.D,
                    num_heads = self.H,
                    n_comps = self.F,
                    AttnClass = st_t_CrossAttentionHead
                )
    
        elif args.sst_method in ['filter_banks']:
            if args.stft_reconstruction:
                self.ct_shared_projection = nn.Sequential(
                    nn.Linear(self.D, self.D*2),
                    nn.ELU(),
                    nn.Linear(self.D*2, self.D),
                    nn.ELU()
                )

            self.temporal_embedding = FilterBanksPatchEmbeddingTemporal(args, n_filter_banks=n_filter_banks, emb_size=patch_emb_size, fs=fs)
        else:
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


        # Spatial Patch Embedding
        self.channel_embedding = PatchEmbeddingSpatial(spa_dim=args.spa_dim, emb_size=patch_emb_size)  # Default 16


        # Positional Encoding based on configurations
        if args.posemb_flag:
            if args.sst_method is not False:
                if self.sst_method in ['st_addition', 'st_addition_projection', 'stf_attention_temporal_values']:
                    self.pos_embedding_frequency = nn.Parameter(torch.randn(1, self.F, self.D))
                    self.pos_embedding_temporal = nn.Parameter(torch.randn(1, self.P, self.D))
                elif self.sst_method in ['filter_banks']:
                    self.pos_embedding_frequency = nn.Parameter(torch.randn(1, self.F, 1, self.D))
                    self.pos_embedding_temporal = nn.Parameter(torch.randn(1, 1, self.P, self.D))
            else:
                self.pos_embedding_temporal = nn.Parameter(torch.randn(1, self.P, self.D))
            self.pos_embedding_spatial = nn.Parameter(torch.randn(1, self.C, self.D))


        # Use Shared Spectral-Spatio-Temporal Space or Not
        if self.sst_method is not False:
            self.sst_shared_projection = nn.Sequential(
                    nn.Linear(self.D, self.D*2),
                    nn.ELU(),
                    nn.Linear(self.D*2, self.FTS),
                    nn.ELU()
                )


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
        if self.sst_method in ['st_addition', 'st_addition_projection', 'stf_attention_temporal_values', 'filter_banks']:
            if self.sst_method == 'st_addition_projection':
                z_t = self.ct_shared_projection(x_embed_temporal)
                z_s = self.ct_shared_projection(x_embed_spatial)

                zt = z_t.unsqueeze(1).expand(-1, self.C, -1, -1)
                zs = z_s.unsqueeze(2).expand(-1, -1, self.P, -1)

                z_st = zs + zt      # Mixing spatial and temporal embeddings
            elif self.sst_method == 'st_addition':
                z_t = x_embed_temporal
                z_s = x_embed_spatial

                zt = z_t.unsqueeze(1).expand(-1, self.C, -1, -1)
                zs = z_s.unsqueeze(2).expand(-1, -1, self.P, -1)

                z_st = zs + zt      # Mixing spatial and temporal embeddings
            elif self.sst_method == 'stf_attention_temporal_values':
                z_t = x_embed_temporal
                z_s = x_embed_spatial.unsqueeze(2).expand(-1, -1, self.P, -1)
                z_st = z_s + x_embed_temporal.unsqueeze(1).expand(-1, self.C, -1, -1)
                z_st = self.stf_attention_head(z_st, z_t)

            # Within Transformer STFT Estimation
            if self.sst_method in ['st_addition', 'st_addition_projection']:
                stft = self.spectrogram_generator(z_st) # type: ignore
            if self.sst_method in ['stf_attention_temporal_values']:
                stft = self.spectrogram_generator(z_st).squeeze(dim=-1) # type: ignore
                stft = rearrange(stft, 'b f c t -> b c t f')

            if self.sst_method in ['filter_banks']:
                z_t = x_embed_temporal.unsqueeze(3).expand(-1, -1, -1, self.C, -1)
                z_s = x_embed_spatial.unsqueeze(1).unsqueeze(2).expand(-1, self.F, self.P, -1, -1)
                if self.stft_reconstruction:
                    z_t = nn.ELU()(self.ct_shared_projection(z_t))
                    z_s = nn.ELU()(self.ct_shared_projection(z_s))

                stft = torch.abs(torch.sum(z_s * z_t, dim=-1))
                stft = rearrange(stft, 'b f p c -> b c p f')


            # Out of Transformer STFT EStimation for STFT Reconstruction
            if self.stft_reconstruction:
                loomed_stft = nn.ELU()(self.stft_temporal_loom(rearrange(stft, 'b c t f -> b c f t'))) # type: ignore
                
            # Generate Embedding for Frequency Components
            x_embed_frequency = rearrange(stft, 'b c p f-> b f (c p)') # type: ignore
            x_embed_frequency = self.frequency_embedding(x_embed_frequency)


        # Positional Encoding
        if self.posemb_flag:
            if self.sst_method in ['st_addition', 'st_addition_projection', 'stf_attention_temporal_values']:
                x_embed_frequency = x_embed_frequency + self.pos_embedding_frequency  # type: ignore # frequency positional encoding
            elif self.sst_method == 'filter_banks':
                x_embed_temporal = x_embed_temporal + self.pos_embedding_frequency

            x_embed_temporal = x_embed_temporal + self.pos_embedding_temporal  # temporal positional encoding
            x_embed_spatial = x_embed_spatial + self.pos_embedding_spatial  # spatial positional encoding  


        # Project time, space, frequency components embedding into the same empedding space
        if self.sst_method in ['st_addition', 'st_addition_projection', 'stf_attention_temporal_values']:
            x_embed_frequency = self.sst_shared_projection(x_embed_frequency)    # type: ignore # --> (B, F, FTS)
        x_embed_temporal = self.sst_shared_projection(x_embed_temporal)  # --> (B, P, FTS) or (B, F, P, FTS)
        x_embed_spatial = self.sst_shared_projection(x_embed_spatial)    # --> (B, C, FTS)
        

        # Make time-frequency embedding
        if self.sst_method in ['st_addition', 'st_addition_projection', 'stf_attention_temporal_values']:
            # using spatio-temporal embedding approach from 
            z_hat_f = x_embed_frequency.unsqueeze(2).expand(-1, -1, self.P, -1) # type: ignore
            z_hat_t = x_embed_temporal.unsqueeze(1).expand(-1, self.F, -1, -1)
            x_embed_fp = rearrange(z_hat_f + z_hat_t, 'b f p d -> b (f p) d')
        elif self.sst_method == 'filter_banks':
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

        if self.stft_reconstruction:
            return loomed_stft, x_embed, out # type: ignore
        return None, x_embed, out

