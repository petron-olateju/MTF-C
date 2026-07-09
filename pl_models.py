from argparse import Namespace
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.DBConformer import DBConformer
from models.MTFC import MTFC
from torchmetrics.classification import Accuracy
from torchmetrics.regression import (
    MeanSquaredError,
    NormalizedRootMeanSquaredError
)
from torchmetrics.aggregation import MeanMetric
from torchmetrics.functional import normalized_root_mean_squared_error
from utils.metrics import InterIntraClass_Similarity
from utils.data_loader import DATASET_TASK_MAP
from utils.losses import SupConLoss
from utils.spectrum_reconstructors import (
    R_SpatioTemporal_AdditionPerBank, 
    R_SpatioTemporal_ProjectionAdditionPerBank,
    R_SpatioTemporalProjection_AdditionPerBank,
    R_SpectrumSpatioTemporal_Addition,
    R_SpectrumSpatioTemporal_ProjectionAddition,
    R_SpectrumSpatioTemporal_ProjectionAdditionPerBank,
    R_SpetrumSpatioTemporal_Projection_BranchAddition,
    CrossAttentionSSTDecoder
)
from utils.trace_predictors import (
    Trace_SpetrumSpatioTemporal_Projection_BranchAddition,
)


class SST_Decoder(nn.Module):

    def __init__(self, decoder, n_banks, num_channels, num_patches, emb_size):
        super().__init__()
        self.name = decoder

        if decoder == 'st_addition':
            self.model = R_SpatioTemporal_AdditionPerBank(n_banks, num_channels, num_patches, emb_size)
        elif decoder == 'st_projection+addition':
            self.model = R_SpatioTemporal_ProjectionAdditionPerBank(n_banks, num_channels, num_patches, emb_size)
        elif decoder == 'st_projection_addition':
            self.model = R_SpatioTemporalProjection_AdditionPerBank(n_banks, num_channels, num_patches, emb_size)
        elif decoder =='sst_addition':
            self.model = R_SpectrumSpatioTemporal_Addition(n_banks, num_channels, num_patches, emb_size)
        elif decoder == 'sst_projection+addition':
            self.model = R_SpectrumSpatioTemporal_ProjectionAddition(n_banks, num_channels, num_patches, emb_size)
        elif decoder == 'sst_multi_projection+addition':
            self.model = R_SpectrumSpatioTemporal_ProjectionAdditionPerBank(n_banks, num_channels, num_patches, emb_size)
        elif decoder == 'sst_multi_projection+branch_addition':
            self.model = R_SpetrumSpatioTemporal_Projection_BranchAddition(n_banks, num_channels, num_patches, emb_size)
        elif decoder == 'sst_cross_attention':
            self.model = CrossAttentionSSTDecoder(n_banks, num_channels, num_patches, emb_size)
        else:
            raise ValueError(f"Argument decoder should be one of: [st_addition, st_projection+addition, st_projection_addition, sst_addition, sst_projection+addition, sst_multi_projection+branch_addition, sst_cross_attention]")
    
    def forward(self, x_spectrum, x_temporal, x_spatial):
        z = self.model(x_spectrum, x_temporal, x_spatial)
        return z

class SST_Trace(nn.Module):

    def __init__(self, trace, n_banks, num_channels, num_patches, emb_size):
        super().__init__()
        self.name = trace

        if trace == 'sst_multi_projection+branch_addition':
            self.predictor = Trace_SpetrumSpatioTemporal_Projection_BranchAddition( n_banks, num_channels, num_patches, emb_size)
        else:
            ValueError(f"Argument trace should be one of: [sst_multi_projection+branch_addition, ]")

    def forward(self, x_spectrum, x_temporal, x_spatial):
        z = self.predictor(x_spectrum, x_temporal, x_spatial)
        return z

class db_conformer(pl.LightningModule):

    def __init__(self, MODEL_ARGS):
        super().__init__()
        self.lr = MODEL_ARGS["lr"]
        self.task = DATASET_TASK_MAP[MODEL_ARGS["data_name"]]

        self.sst_method_name = MODEL_ARGS['sst_method']
        self.sst_decoder_name = MODEL_ARGS['sst_decoder']

        args = {
            k: MODEL_ARGS[k]
            for k in [
                "data_name",
                "chn",
                "time_sample_num",
                "class_num",
                "patch_size",
                "spa_dim",
                "gate_flag",
                "posemb_flag",
                "branch",
                "chn_attn_flag",
            ]
        }

        self.model = DBConformer(
            Namespace(**args),
            emb_size=MODEL_ARGS["emb_size"],
            tem_depth=MODEL_ARGS["tem_depth"],
            chn_depth=MODEL_ARGS["chn_depth"],
            chn=MODEL_ARGS["chn"],
            n_classes=MODEL_ARGS["class_num"],
        )

        if self.task == "multiclass":
            self.train_acc = Accuracy(
                task="multiclass", num_classes=MODEL_ARGS["class_num"]
            )
            self.val_acc = Accuracy(
                task="multiclass", num_classes=MODEL_ARGS["class_num"]
            )
            self.test_acc = Accuracy(
                task="multiclass", num_classes=MODEL_ARGS["class_num"]
            )
        elif self.task == "binary":
            self.train_acc = Accuracy(task="binary")
            self.val_acc = Accuracy(task="binary")
            self.test_acc = Accuracy(task="binary")

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        if self.sst_decoder_name is not None:
            if (self.sst_method_name is not None) and (self.sst_method_name.upper() in ['FREQUENCY_BACKBONE']):
                sst, spectrum, x, y = batch
            else:
                sst, x, y = batch
        else:
            x, y = batch
        branch_embeddings, _, x_fused, logits = self(x)

        loss = F.cross_entropy(logits, y)
        return branch_embeddings, x_fused, logits, loss, y

    def training_step(self, batch, batch_idx):
        _, logits, loss, y = self._common_step(batch, batch_idx)

        preds = logits.argmax(dim=1)
        self.train_acc.update(preds, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            _, logits, loss, y = self._common_step(batch, batch_idx)

            preds = logits.argmax(dim=1)
            self.val_acc.update(preds, y)

            self.log("val_loss", loss, prog_bar=True)
            self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            _, logits, loss, y = self._common_step(batch, batch_idx)

            preds = logits.argmax(dim=1)
            self.test_acc.update(preds, y)

            self.log("test_loss", loss, prog_bar=True)
            self.log("test_acc", self.test_acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class db_r_conformer(db_conformer):
    def __init__(self, MODEL_ARGS):
        assert MODEL_ARGS['sst_method'] is not None
        assert MODEL_ARGS['sst_decoder'] is not None
        super().__init__(MODEL_ARGS=MODEL_ARGS)

        self.pretrain = MODEL_ARGS['pretrain']
        if self.pretrain is not False:
            self.pretrain_dir = MODEL_ARGS['pretrain_dir']
            self.encoder = db_conformer.load_from_checkpoint(self.pretrain_dir, MODEL_ARGS=MODEL_ARGS)
            self.encoder.freeze()
            self.encoder.eval()
            print(f"Using Pre-Trained Encoder: {self.pretrain_dir}")
        else:
            self.pretrain_dir = None
            self.encoder = db_conformer(MODEL_ARGS=MODEL_ARGS)
        self.model = self.encoder.model

        if (self.sst_decoder_name is not None) and (self.sst_decoder_name in ['st_addition', 'st_projection+addition', 'st_projection_addition']):
            n_times = MODEL_ARGS['time_sample_num']
            P_cfg = MODEL_ARGS['patch_size']
            n_patches = (n_times - 1) // P_cfg
            
            self.decoder = SST_Decoder(
                decoder=self.sst_decoder_name,
                n_banks=MODEL_ARGS['filter_banks'],
                num_channels=MODEL_ARGS['chn'],
                num_patches=n_patches,
                emb_size=MODEL_ARGS['emb_size']
            )
        else:
            raise ValueError(f"decoder for db_conformer cannot be {self.sst_decoder_name}, can only be one of :{['st_addition', 'st_projection+addition', 'st_projection_addition']}")

        self.train_sst_error = NormalizedRootMeanSquaredError(normalization="l2")
        self.val_sst_error = NormalizedRootMeanSquaredError(normalization="l2")
        self.test_sst_error = NormalizedRootMeanSquaredError(normalization="l2")

    def forward(self, x):
        branch_embeddings, _, x_fused, _ = self.encoder(x)
        x_temporal = branch_embeddings['temporal']
        x_channel = branch_embeddings['channel']

        if self.decoder.name in ['st_addition', 'st_projection+addition', 'st_projection_addition']:
            sst_hat = self.decoder(x_temporal, x_channel)
        else:
            raise ValueError(f"decoder for db_conformer can only be one of :{['st_addition', 'st_projection+addition', 'st_projection_addition']}")
        
        return sst_hat
        

    def _common_step(self, batch, batch_idx):
        branch_embeddings, x_fused, logits, clf_loss, y = self.encoder._common_step(batch, batch_idx)
        x_temporal = branch_embeddings['temporal']
        x_channel = branch_embeddings['channel']

        if self.decoder.name in ['st_addition', 'st_projection+addition', 'st_projection_addition']:
            sst_hat = self.decoder(None, x_temporal, x_channel)
        else:
            raise ValueError(f"decoder for db_conformer can only be one of :{['st_addition', 'st_projection+addition', 'st_projection_addition']}")
        
        if self.sst_decoder_name is not None:
            if (self.sst_method_name is not None) and (self.sst_method_name.upper() in ['FREQUENCY_BACKBONE']):
                sst, spectrum, x, y = batch
            else:
                sst, x, y = batch
        else:
            x, y = batch

        reconstruction_loss = normalized_root_mean_squared_error(sst_hat, sst, normalization="l2")
        if self.pretrain is not False:
            loss = reconstruction_loss
        else:
            loss = reconstruction_loss + clf_loss

        return sst, sst_hat, logits, loss, y
    
    def training_step(self, batch, batch_idx):
        sst, sst_hat, logits, loss, y = self._common_step(batch, batch_idx)

        preds = logits.argmax(dim=1)
        self.train_acc.update(preds, y)

        self.train_sst_error.update(sst_hat, sst)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_sst_error", self.train_sst_error, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            sst, sst_hat, logits, loss, y = self._common_step(batch, batch_idx)

            preds = logits.argmax(dim=1)
            self.val_acc.update(preds, y)

            self.val_sst_error.update(sst_hat, sst)

            self.log("val_loss", loss, prog_bar=True)
            self.log("val_sst_error", self.val_sst_error, prog_bar=True)
            self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            sst, sst_hat, logits, loss, y = self._common_step(batch, batch_idx)

            preds = logits.argmax(dim=1)
            self.test_acc.update(preds, y)

            self.test_sst_error.update(sst_hat, sst)

            self.log("test_loss", loss, prog_bar=True)
            self.log("test_sst_error", self.test_sst_error, prog_bar=True)
            self.log("test_acc", self.test_acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class mtf_c(pl.LightningModule):
    def __init__(self, MODEL_ARGS):
        super().__init__()
        self.lr = MODEL_ARGS["lr"]
        self.task = DATASET_TASK_MAP[MODEL_ARGS["data_name"]]

        self.sst_method_name = MODEL_ARGS['sst_method']
        self.sst_decoder_name = MODEL_ARGS['sst_decoder']
        self.sst_trace_name = MODEL_ARGS['sst_trace']

        args = {
            k: MODEL_ARGS[k]
            for k in [
                "data_name",
                "chn",
                "time_sample_num",
                "patch_size",
                "class_num",
                "gate_flag",
                "posemb_flag",
                "chn_attn_flag",
                "spectrum_attn_flag",
                "temporal_attn_flag",
                "sst_method",
                "spa_dim",
            ]
        }

        self.model = MTFC(
            Namespace(**args),
            n_filter_banks=MODEL_ARGS['filter_banks'],
            wsize_divisor=MODEL_ARGS['wsize_divisor'],
            freq_downsample=MODEL_ARGS['freq_downsample'],
            n_times=MODEL_ARGS['time_sample_num'],
            patch_emb_size=MODEL_ARGS['patch_emb_size'],
            tem_depth=MODEL_ARGS["tem_depth"],
            chn_depth=MODEL_ARGS["chn_depth"],
            spec_depth=MODEL_ARGS["spec_depth"],
            n_classes=MODEL_ARGS["class_num"],
            fs=MODEL_ARGS['fs']
        )

        self.reconstruction_lambda = MODEL_ARGS['reconstruction_lambda']

        if self.task == "multiclass":
            self.train_acc = Accuracy(
                task="multiclass", num_classes=MODEL_ARGS["class_num"]
            )
            self.val_acc = Accuracy(
                task="multiclass", num_classes=MODEL_ARGS["class_num"]
            )
            self.test_acc = Accuracy(
                task="multiclass", num_classes=MODEL_ARGS["class_num"]
            )
        elif self.task == "binary":
            self.train_acc = Accuracy(task="binary")
            self.val_acc = Accuracy(task="binary")
            self.test_acc = Accuracy(task="binary")
        
        if self.model.sst_method is not None:
            self.train_reconstruction = MeanSquaredError()
            self.val_reconstruction = MeanSquaredError()
            self.test_reconstruction = MeanSquaredError()

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        if self.sst_decoder_name is not None:
            if (self.sst_method_name is not None) and (self.sst_method_name.upper() in ['FREQUENCY_BACKBONE']):
                sst, spectrum, x, y = batch
            else:
                sst, x, y = batch
        elif self.sst_method_name is not None:
            spectrum, x, y = batch
        else:
            x, y = batch

        if self.model.sst_method is not None:
            branch_embeddings, spectrum_est, x_fused, logits = self(x)
        else:
            branch_embeddings, x_fused, logits = self(x)

        task_loss = F.cross_entropy(logits, y)
        if self.model.sst_method is not None:
            reconstruction_loss = F.mse_loss(spectrum_est, spectrum)
            loss = task_loss + (self.reconstruction_lambda * reconstruction_loss)
            return branch_embeddings, spectrum_est, spectrum, x_fused, logits, loss, y
        else:
            loss = task_loss
            return branch_embeddings, x_fused, logits, loss, y

    def training_step(self, batch, batch_idx):
        if self.model.sst_method is not None:
            _, spectrum_est, spectrum, _, logits, loss, y = self._common_step(batch, batch_idx)
        else:
            _, _, logits, loss, y = self._common_step(batch, batch_idx)

        preds = logits.argmax(dim=1)
        self.train_acc.update(preds, y)

        if self.model.sst_method is not None:
            self.train_reconstruction.update(spectrum_est, spectrum)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        if self.model.sst_method is not None:
            self.log("train_mse", self.train_reconstruction, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            if self.model.sst_method is not None:
                _, spectrum_est, spectrum, _, logits, loss, y = self._common_step(batch, batch_idx)
            else:
                _, _, logits, loss, y = self._common_step(batch, batch_idx)

            preds = logits.argmax(dim=1)
            self.val_acc.update(preds, y)

            if self.model.sst_method is not None:
                self.val_reconstruction.update(spectrum_est, spectrum)

            self.log("val_loss", loss, prog_bar=True)
            self.log("val_acc", self.val_acc, prog_bar=True)
            if self.model.sst_method is not None:
                self.log("val_mse", self.val_reconstruction, prog_bar=True)

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            if self.model.sst_method is not None:
                _, spectrum_est, spectrum, _, logits, loss, y = self._common_step(batch, batch_idx)
            else:
                _, _, logits, loss, y = self._common_step(batch, batch_idx)

            preds = logits.argmax(dim=1)
            self.test_acc.update(preds, y)
            if self.model.sst_method is not None:
                self.test_reconstruction.update(spectrum_est, spectrum)

            self.log("test_loss", loss, prog_bar=True)
            self.log("test_acc", self.test_acc, prog_bar=True)
            if self.model.sst_method is not None:
                self.log("test_mse", self.test_reconstruction, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class mtf_r_c(mtf_c):

    def __init__(self, MODEL_ARGS):
        assert MODEL_ARGS['sst_method'] is not None
        assert MODEL_ARGS['sst_decoder'] is not None
        super().__init__(MODEL_ARGS=MODEL_ARGS)

        self.pretrain = MODEL_ARGS['pretrain']
        self.trace_pretrain = MODEL_ARGS['trace_pretrain']
        if self.pretrain is not False:
            self.pretrain_dir = MODEL_ARGS['pretrain_dir']
            self.encoder = mtf_c.load_from_checkpoint(self.pretrain_dir, MODEL_ARGS=MODEL_ARGS)
            self.encoder.freeze()
            self.encoder.eval()
            self.model = self.encoder.model
            print(f"Using Pre-Trained Encoder: {self.pretrain_dir}")
        else:
            self.pretrain_dir = None
            self.encoder = mtf_c(MODEL_ARGS=MODEL_ARGS)
            self.model = self.encoder.model

        n_times = MODEL_ARGS['time_sample_num']
        P_cfg = MODEL_ARGS['patch_size']
        n_patches = (n_times - 1) // P_cfg

        self.decoder = SST_Decoder(
            decoder=self.sst_decoder_name,
            n_banks=MODEL_ARGS['filter_banks'],
            num_channels=MODEL_ARGS['chn'],
            num_patches=n_patches,
            emb_size=MODEL_ARGS['patch_emb_size']
        )

        self.train_sst_error = NormalizedRootMeanSquaredError(normalization="l2")
        self.val_sst_error = NormalizedRootMeanSquaredError(normalization="l2")
        self.test_sst_error = NormalizedRootMeanSquaredError(normalization="l2")

    def forward(self, x):
        branch_embeddings, spectrum_est, x_fused, logits = self.encoder(x)
        x_spectrum = branch_embeddings['spectrum']
        x_temporal = branch_embeddings['temporal']
        x_channel = branch_embeddings['channel']

        if self.decoder.name in ['st_addition', 'st_projection+addition', 'st_projection_addition']:
            sst_hat = self.decoder(None, x_temporal, x_channel)
        elif self.decoder.name in ['sst_addition', 'sst_projection+addition', 'sst_cross_attention', 'sst_multi_projection+addition', 'sst_multi_projection+branch_addition']:
            sst_hat = self.decoder(x_spectrum, x_temporal, x_channel)
        else:
            raise ValueError(f"decoder for mtf_c cannot be {self.sst_decoder_name}, can only be one of :{['st_addition', 'st_projection+addition', 'st_projection_addition', 'sst_addition', 'sst_projection+addition', 'sst_cross_attention', 'sst_multi_projection+addition', 'sst_multi_projection+branch_addition']}")

        return sst_hat

    def _common_step(self, batch, batch_idx):
        branch_embeddings, spectrum_est, spectrum, x_fused, logits, encoder_loss, y = self.encoder._common_step(batch, batch_idx)
        x_spectrum = branch_embeddings['spectrum']
        x_temporal = branch_embeddings['temporal']
        x_channel = branch_embeddings['channel']

        if self.decoder.name in ['st_addition', 'st_projection+addition', 'st_projection_addition']:
            sst_hat = self.decoder(None, x_temporal, x_channel)
        elif self.decoder.name in ['sst_addition', 'sst_projection+addition', 'sst_cross_attention', 'sst_multi_projection+addition', 'sst_multi_projection+branch_addition']:
            sst_hat = self.decoder(x_spectrum, x_temporal, x_channel)
        else:
            raise ValueError(f"decoder for mtf_c can only be one of :{['st_addition', 'st_projection+addition', 'st_projection_addition', 'sst_addition', 'sst_projection+addition', 'sst_cross_attention', 'sst_multi_projection+addition', 'sst_multi_projection+branch_addition']}")

        if self.sst_decoder_name is not None:
            if (self.sst_method_name is not None) and (self.sst_method_name.upper() in ['FREQUENCY_BACKBONE']):
                sst, spectrum, x, y = batch
            else:
                sst, x, y = batch
        else:
            x, y = batch

        reconstruction_loss = normalized_root_mean_squared_error(sst_hat, sst, normalization="l2")
        if self.pretrain is not False:
            loss = reconstruction_loss
        else:
            loss = reconstruction_loss + encoder_loss

        return branch_embeddings, sst, sst_hat, logits, loss, y

    def training_step(self, batch, batch_idx):
        _, sst, sst_hat, logits, loss, y = self._common_step(batch, batch_idx)

        preds = logits.argmax(dim=1)
        self.train_acc.update(preds, y)

        self.train_sst_error.update(sst_hat, sst)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_sst_error", self.train_sst_error, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            _, sst, sst_hat, logits, loss, y = self._common_step(batch, batch_idx)

            preds = logits.argmax(dim=1)
            self.val_acc.update(preds, y)

            self.val_sst_error.update(sst_hat, sst)

            self.log("val_loss", loss, prog_bar=True)
            self.log("val_sst_error", self.val_sst_error, prog_bar=True)
            self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            _, sst, sst_hat, logits, loss, y = self._common_step(batch, batch_idx)

            preds = logits.argmax(dim=1)
            self.test_acc.update(preds, y)

            self.test_sst_error.update(sst_hat, sst)

            self.log("test_loss", loss, prog_bar=True)
            self.log("test_sst_error", self.test_sst_error, prog_bar=True)
            self.log("test_acc", self.test_acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    

class mtf_tr_c(mtf_r_c):

    def __init__(self, MODEL_ARGS):
        assert MODEL_ARGS['sst_method'] is not None
        assert MODEL_ARGS['sst_decoder'] is not None
        assert MODEL_ARGS['sst_trace'] is not None
        super().__init__(MODEL_ARGS=MODEL_ARGS)

        self.trace_pretrain = MODEL_ARGS['trace_pretrain']
        self.trace_target = MODEL_ARGS['trace_target']
        self.trace_temperature = MODEL_ARGS['trace_temperature']
        if self.trace_pretrain is not False:
            self.pretrain_dir = MODEL_ARGS['pretrain_dir']
            self.encoder_decoder = mtf_r_c.load_from_checkpoint(self.pretrain_dir, MODEL_ARGS=MODEL_ARGS)
            self.encoder = self.encoder_decoder.encoder
            self.decoder = self.encoder_decoder.decoder
            self.encoder.freeze()
            self.encoder.eval()
            self.decoder.eval()
            self.decoder.eval()
            for p in self.decoder.parameters():
                p.requires_grad = False
            print(f"Using Pre-Trained Encoder: {self.pretrain_dir}")
        else:
            self.pretrain_dir = None
            self.encoder_decoder = mtf_r_c(MODEL_ARGS=MODEL_ARGS)
        # self.model = self.encoder_decoder.encoder.model

        n_times = MODEL_ARGS['time_sample_num']
        P_cfg = MODEL_ARGS['patch_size']
        n_patches = (n_times - 1) // P_cfg

        self.trace = SST_Trace(
            trace=self.sst_trace_name,
            n_banks=MODEL_ARGS['filter_banks'],
            num_channels=MODEL_ARGS['chn'],
            num_patches=n_patches,
            emb_size=MODEL_ARGS['patch_emb_size']
        )
        self.spectrum_normalization = nn.BatchNorm1d(MODEL_ARGS['filter_banks'])
        self.temporal_normalization = nn.BatchNorm1d(n_patches)
        self.channel_normalization = nn.BatchNorm1d(MODEL_ARGS['chn'])

        self.train_inter_class_sim = MeanMetric()
        self.val_inter_class_sim = MeanMetric()
        self.test_inter_class_sim = MeanMetric()
        self.train_intra_class_sim = MeanMetric()
        self.val_intra_class_sim = MeanMetric()
        self.test_intra_class_sim = MeanMetric()

        self.train_decoder_trace_error = MeanSquaredError()
        self.val_decoder_trace_error = MeanSquaredError()
        self.test_decoder_trace_error = MeanSquaredError()

    def forward(self, x):
        branch_embeddings, spectrum_est, x_fused, logits = self.encoder_decoder.encoder(x)
        x_spectrum = branch_embeddings['spectrum']
        x_temporal = branch_embeddings['temporal']
        x_channel = branch_embeddings['channel']

        if self.decoder.name in ['st_addition', 'st_projection+addition', 'st_projection_addition']:
            decoder_sst = self.encoder_decoder.decoder(None, x_temporal, x_channel)
        elif self.decoder.name in ['sst_addition', 'sst_projection+addition', 'sst_cross_attention', 'sst_multi_projection+addition', 'sst_multi_projection+branch_addition']:
            decoder_sst = self.encoder_decoder.decoder(x_spectrum, x_temporal, x_channel)
        else:
            raise ValueError(f"decoder for mtf_c cannot be {self.sst_decoder_name}, can only be one of :{['st_addition', 'st_projection+addition', 'st_projection_addition', 'sst_addition', 'sst_projection+addition', 'sst_cross_attention', 'sst_multi_projection+addition', 'sst_multi_projection+branch_addition']}")

        x_spectrum = self.spectrum_normalization(x_spectrum)
        x_temporal = self.temporal_normalization(x_temporal)
        x_channel = self.channel_normalization(x_channel)

        if self.trace.name in ['st_addition', 'st_projection+addition', 'st_projection_addition']:
            trace_coords = self.trace(None, x_temporal, x_channel)
        elif self.trace.name in ['sst_addition', 'sst_projection+addition', 'sst_cross_attention', 'sst_multi_projection+addition', 'sst_multi_projection+branch_addition']:
            trace_coords = self.trace(x_spectrum, x_temporal, x_channel)
        else:
            raise ValueError(f"trace network for mtf_c cannot be {self.sst_decoder_name}, can only be one of :{['st_addition', 'st_projection+addition', 'st_projection_addition', 'sst_addition', 'sst_projection+addition', 'sst_cross_attention', 'sst_multi_projection+addition', 'sst_multi_projection+branch_addition']}")
        
        trace_coords = rearrange(trace_coords.unsqueeze(1), 'b n p d -> b n (p d)')
        trace_coords = F.normalize(trace_coords, dim=-1)

        return branch_embeddings, logits, decoder_sst, trace_coords
    
    def _common_step(self, batch, batch_idx):
        branch_embeddings, _, _, logits, encoder_decoder_loss, y = self.encoder_decoder._common_step(batch, batch_idx)
        x_spectrum = branch_embeddings['spectrum']
        x_temporal = branch_embeddings['temporal']
        x_channel = branch_embeddings['channel']

        if self.decoder.name in ['st_addition', 'st_projection+addition', 'st_projection_addition']:
            decoder_sst = self.encoder_decoder.decoder(None, x_temporal, x_channel)
        elif self.decoder.name in ['sst_addition', 'sst_projection+addition', 'sst_cross_attention', 'sst_multi_projection+addition', 'sst_multi_projection+branch_addition']:
            decoder_sst = self.encoder_decoder.decoder(x_spectrum, x_temporal, x_channel)
        else:
            raise ValueError(f"decoder for mtf_c cannot be {self.sst_decoder_name}, can only be one of :{['st_addition', 'st_projection+addition', 'st_projection_addition', 'sst_addition', 'sst_projection+addition', 'sst_cross_attention', 'sst_multi_projection+addition', 'sst_multi_projection+branch_addition']}")

        x_spectrum = self.spectrum_normalization(x_spectrum)
        x_temporal = self.temporal_normalization(x_temporal)
        x_channel = self.channel_normalization(x_channel)

        if self.trace.name in ['st_addition', 'st_projection+addition', 'st_projection_addition']:
            trace_coords = self.trace(None, x_temporal, x_channel)
        elif self.trace.name in ['sst_addition', 'sst_projection+addition', 'sst_cross_attention', 'sst_multi_projection+addition', 'sst_multi_projection+branch_addition']:
            trace_coords = self.trace(x_spectrum, x_temporal, x_channel)
        else:
            raise ValueError(f"trace network for mtf_c cannot be {self.sst_decoder_name}, can only be one of :{['st_addition', 'st_projection+addition', 'st_projection_addition', 'sst_addition', 'sst_projection+addition', 'sst_cross_attention', 'sst_multi_projection+addition', 'sst_multi_projection+branch_addition']}")
        
        trace_coords = rearrange(trace_coords.unsqueeze(1), 'b n p d -> b n (p d)')
        trace_coords = F.normalize(trace_coords, dim=-1)

        with torch.no_grad():
            if self.trace_target.upper() == 'HARD_ARGMAX':
                decoder_sst = rearrange(decoder_sst, 'b c f p -> b p c f')
                B, P, C, _F = decoder_sst.shape
                # Flatten the C×F plane
                flat_idx = decoder_sst.reshape(B, P, -1).argmax(dim=-1)  # (B, P)
                # Convert flat indices back to (channel, frequency)
                channel_idx = (flat_idx // _F).float()
                frequency_idx = (flat_idx % _F).float()
                x_coords = channel_idx / (C - 1)
                y_coords = frequency_idx / (_F - 1)
            elif self.trace_target.upper() == 'SOFT_ARGMAX':
                decoder_sst_r = rearrange(decoder_sst, 'b c f p -> b p c f')
                B, P, C, _F = decoder_sst_r.shape
                energy = decoder_sst_r.reshape(B, P, -1)
                weights = F.softmax(energy, dim=-1).reshape(B, P, C, _F)
                chan_weights = weights.sum(dim=-1)  # (B, P, C)
                freq_weights = weights.sum(dim=-2)  # (B, P, F)
                chan_idx = torch.arange(C, device=decoder_sst.device, dtype=decoder_sst.dtype)
                freq_idx = torch.arange(_F, device=decoder_sst.device, dtype=decoder_sst.dtype)
                x_coords = (chan_weights * chan_idx).sum(dim=-1) / (C - 1)
                y_coords = (freq_weights * freq_idx).sum(dim=-1) / (_F - 1)
            else:
                raise ValueError(f'--trace_target should be one of [hard_argmax, soft_argmax] not {self.trace_target}')
            
            decoder_coords = torch.stack([x_coords, y_coords], dim=-1).detach()
            decoder_coords = rearrange(decoder_coords, 'b p d -> b (p d)')
        trace_coords_ = trace_coords.squeeze(dim=1)
        trace_decoder_loss = F.mse_loss(trace_coords_, decoder_coords)

        trace_loss = SupConLoss(
            temperature=self.trace_temperature, 
            base_temperature=self.trace_temperature
        )(trace_coords, y)
        if self.trace_pretrain is not False:
            loss = trace_loss + trace_decoder_loss
        else:
            loss = trace_loss + trace_decoder_loss + encoder_decoder_loss

        return trace_coords, decoder_coords, logits, loss, y
    
    def training_step(self, batch, batch_idx):
        trace_coords, decoder_coords, logits, loss, y = self._common_step(batch, batch_idx)

        preds = logits.argmax(dim=1)
        self.train_acc.update(preds, y)

        inter_class_sim, intra_class_sim = InterIntraClass_Similarity()(trace_coords.squeeze(1), y)
        self.train_inter_class_sim(inter_class_sim)
        self.train_intra_class_sim(intra_class_sim)
        self.train_decoder_trace_error.update(trace_coords.squeeze(1), decoder_coords)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_inter_class_sim", inter_class_sim, prog_bar=True)
        self.log("train_intra_class_sim", intra_class_sim, prog_bar=True)
        self.log("train_trace_target_error", self.train_decoder_trace_error, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            trace_coords, decoder_coords, logits, loss, y = self._common_step(batch, batch_idx)

            preds = logits.argmax(dim=1)
            self.val_acc.update(preds, y)

            inter_class_sim, intra_class_sim = InterIntraClass_Similarity()(trace_coords.squeeze(1), y)
            self.val_inter_class_sim(inter_class_sim)
            self.val_intra_class_sim(intra_class_sim)
            self.val_decoder_trace_error.update(trace_coords.squeeze(1), decoder_coords)

            self.log("val_loss", loss, prog_bar=True)
            self.log("val_inter_class_sim", inter_class_sim, prog_bar=True)
            self.log("val_intra_class_sim", intra_class_sim, prog_bar=True)
            self.log("val_trace_target_error", self.val_decoder_trace_error, prog_bar=True)
            self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            trace_coords, decoder_coords, logits, loss, y = self._common_step(batch, batch_idx)

            preds = logits.argmax(dim=1)
            self.test_acc.update(preds, y)

            inter_class_sim, intra_class_sim = InterIntraClass_Similarity()(trace_coords.squeeze(1), y)
            self.test_inter_class_sim(inter_class_sim)
            self.test_intra_class_sim(intra_class_sim)
            self.test_decoder_trace_error.update(trace_coords.squeeze(1), decoder_coords)

            self.log("test_loss", loss, prog_bar=True)
            self.log("test_inter_class_sim", inter_class_sim, prog_bar=True)
            self.log("test_intra_class_sim", intra_class_sim, prog_bar=True)
            self.log("test_trace_target_error", self.test_decoder_trace_error, prog_bar=True)
            self.log("test_acc", self.test_acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr/2.0e2)

NAME_MODEL_MAP = {
    "db_conformer": db_conformer,
    "db_r_conformer": db_r_conformer,
    "mtf_c": mtf_c,
    "mtf_r_c": mtf_r_c,
    "mtf_tr_c": mtf_tr_c
}
