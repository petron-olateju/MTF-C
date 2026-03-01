import torch
import torch.nn as nn

from einops import rearrange

from .DBConformer import PatchEmbeddingTemporal, PatchEmbeddingSpatial
from .DBConformer import TransformerEncoder, ClassificationHead


class FilterBanksPatchEmbeddingTemporal(nn.Module):
    def __init__(self, args, n_filter_banks=5, emb_size=40):
        super().__init__()
        
        self.n_filter_banks = n_filter_banks
        self.patch_embeddings = nn.ModuleList([
            PatchEmbeddingTemporal(
                data_name=args.data_name,
                in_planes=args.chn,  # number of channels
                out_planes=emb_size,  # Default 40
                kernel_size=63,
                radix=1,
                patch_size=args.patch_size,  # needs to be divisible by the number of time points
                time_points=args.time_sample_num,  # number of time points
                num_classes=args.class_num  # number of classes
            ) for i in range(self.n_filter_banks)
        ])

    def forward(self, x):
        assert x.size(1) == self.n_filter_banks
        out = [self.patch_embeddings[i](x[:, i, :, :]) for i in range(self.n_filter_banks)]
        out = torch.cat(out, dim=1)
        return out


class MTFC(nn.Module):

    def __init__(self, args, n_filter_banks= 5, patch_emb_size=40, sst_emb_size=40, 
            depth=5, n_classes=2) -> None:
        super().__init__()

        self.embedding = FilterBanksPatchEmbeddingTemporal(args, n_filter_banks=n_filter_banks, emb_size=patch_emb_size)
        self.channel_embedding = PatchEmbeddingSpatial(spa_dim=args.spa_dim, emb_size=patch_emb_size)  # Default 16
        self.P = args.time_sample_num // args.patch_size  # Example: 1000 // 125 = 8
        self.C = args.chn  # number of channels
        self.D = patch_emb_size
        self.FTS = sst_emb_size
        self.F = n_filter_banks
        self.FP = self.F * self.P
        self.gate_flag = args.gate_flag  # Default False, due to the reduced performance
        self.posemb_flag = args.posemb_flag  # Default True
        self.branch = args.branch  # Default 'all', options=[all, temporal]
        # self.chn_atten_flag = args.chn_atten_flag  # Default True
        self.fts_atten_flag = args.fts_atten_flag   # Default True

        if args.posemb_flag:
            self.pos_embedding_temporal = nn.Parameter(torch.randn(1, self.F * self.P, self.D))
            self.pos_embedding_spatial = nn.Parameter(torch.randn(1, self.C, self.D))

        self.sst_projection_space = nn.Linear(self.D, self.FST)

        if args.fts_atten_flag:
            self.fts_attn_pool = nn.Sequential(
                nn.Linear(self.FTS, self.FTS),
                nn.Tanh(),
                nn.Linear(self.FTS, 1),
            )

        self.fts_transformer = TransformerEncoder(depth, self.FTS)
        self.classfier = ClassificationHead(self.FTS, n_classes)

    def forward(self, x):   # x: (B, F, C, T)
        x_embed_fp = self.embedding(x[:, :self.F, :, :])    # --> (B, F*P, D)
        x_embed_spatial = self.channel_embedding(x[:, -1, :, :])     # --> (B, C, D)     

        if self.posemb_flag:
            x_embed_fp = x_embed_fp + self.pos_embedding_temporal  # temporal positional encoding
            x_embed_spatial = x_embed_spatial + self.pos_embedding_spatial  # spatial positional encoding   

        x_embed_fp = self.sst_projection_space(x_embed_fp)  # --> (B, F*P, FTS)
        x_embed_spatial = self.sst_projection_space(x_embed_spatial)    # --> (B, C, FTS)


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

        return x_embed, out

