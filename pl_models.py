from argparse import Namespace
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.DBConformer import DBConformer
from models.MTFC import MTFC
from torchmetrics.classification import Accuracy
from torchmetrics.regression import (
    MeanSquaredError,
    NormalizedRootMeanSquaredError
)
from torchmetrics.functional import normalized_root_mean_squared_error
from utils.data_loader import DATASET_TASK_MAP

from utils.spectrum_reconstructors import (
    R_SpatioTemporal_AdditionPerBank, 
    R_SpatioTemporal_ProjectionAdditionPerBank,
    R_SpatioTemporalProjection_AdditionPerBank,
    R_SpectrumSpatioTemporal_Addition,
    R_SpectrumSpatioTemporal_ProjectionAddition,
    CrossAttentionSSTDecoder
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
        elif decoder == 'sst_cross_attention':
            self.model = CrossAttentionSSTDecoder(n_banks, num_channels, num_patches, emb_size)
        else:
            raise ValueError(f"Argument decoder should be one of: [st_addition, st_projection+addition, st_projection_addition, sst_addition, sst_projection+addition, sst_cross_attention]")
    
    def forward(self, x_spectrum, x_temporal, x_spatial):
        z = self.model(x_spectrum, x_temporal, x_spatial)
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
        if self.pretrain is not False:
            self.pretrain_dir = MODEL_ARGS['pretrain_dir']
            self.encoder = mtf_c.load_from_checkpoint(self.pretrain_dir, MODEL_ARGS=MODEL_ARGS)
            self.encoder.freeze()
            self.encoder.eval()
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
        elif self.decoder.name in ['sst_addition', 'sst_projection+addition', 'sst_cross_attention']:
            sst_hat = self.decoder(x_spectrum, x_temporal, x_channel)
        else:
            raise ValueError(f"decoder for mtf_c cannot be {self.sst_decoder_name}, can only be one of :{['st_addition', 'st_projection+addition', 'st_projection_addition', 'sst_addition', 'sst_projection+addition', 'sst_cross_attention']}")

        return sst_hat

    def _common_step(self, batch, batch_idx):
        branch_embeddings, spectrum_est, spectrum, x_fused, logits, encoder_loss, y = self.encoder._common_step(batch, batch_idx)
        x_spectrum = branch_embeddings['spectrum']
        x_temporal = branch_embeddings['temporal']
        x_channel = branch_embeddings['channel']

        if self.decoder.name in ['st_addition', 'st_projection+addition', 'st_projection_addition']:
            sst_hat = self.decoder(None, x_temporal, x_channel)
        elif self.decoder.name in ['sst_addition', 'sst_projection+addition', 'sst_cross_attention']:
            sst_hat = self.decoder(x_spectrum, x_temporal, x_channel)
        else:
            raise ValueError(f"decoder for mtf_c can only be one of :{['st_addition', 'st_projection+addition', 'st_projection_addition', 'sst_addition', 'sst_projection+addition', 'sst_cross_attention']}")

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



NAME_MODEL_MAP = {
    "db_conformer": db_conformer,
    "db_r_conformer": db_r_conformer,
    "mtf_c": mtf_c,
    "mtf_r_c": mtf_r_c
}
