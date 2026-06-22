from argparse import Namespace
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from models.DBConformer import DBConformer
from models.MTFC import MTFC
from torchmetrics.classification import Accuracy
from torchmetrics.regression import MeanSquaredError
from utils.data_loader import DATASET_TASK_MAP


class db_conformer(pl.LightningModule):

    def __init__(self, MODEL_ARGS):
        super().__init__()
        self.lr = MODEL_ARGS["lr"]
        self.task = DATASET_TASK_MAP[MODEL_ARGS["data_name"]]

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
        x, y = batch
        _, x_fused, logits = self(x)
        loss = F.cross_entropy(logits, y)
        return x_fused, logits, loss, y

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


class mtf_c(pl.LightningModule):
    def __init__(self, MODEL_ARGS):
        super().__init__()
        self.lr = MODEL_ARGS["lr"]
        self.task = DATASET_TASK_MAP[MODEL_ARGS["data_name"]]

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

        self.train_reconstruction = MeanSquaredError()
        self.val_reconstruction = MeanSquaredError()
        self.test_reconstruction = MeanSquaredError()

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        spectrum, x, y = batch
        spectrum_est, x_fused, logits = self(x)

        task_loss = F.cross_entropy(logits, y)
        reconstruction_loss = F.mse_loss(spectrum_est, spectrum)
        loss = task_loss + (self.reconstruction_lambda * reconstruction_loss)
        return spectrum_est, spectrum, x_fused, logits, loss, y

    def training_step(self, batch, batch_idx):
        spectrum_est, spectrum, _, logits, loss, y = self._common_step(batch, batch_idx)

        preds = logits.argmax(dim=1)
        self.train_acc.update(preds, y)

        self.train_reconstruction.update(spectrum_est, spectrum)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        self.log("train_mse", self.train_reconstruction, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            spectrum_est, spectrum, _, logits, loss, y = self._common_step(batch, batch_idx)

            preds = logits.argmax(dim=1)
            self.val_acc.update(preds, y)

            self.val_reconstruction.update(spectrum_est, spectrum)

            self.log("val_loss", loss, prog_bar=True)
            self.log("val_acc", self.val_acc, prog_bar=True)
            self.log("val_mse", self.val_reconstruction, prog_bar=True)

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            spectrum_est, spectrum, _, logits, loss, y = self._common_step(batch, batch_idx)

            preds = logits.argmax(dim=1)
            self.test_acc.update(preds, y)

            self.test_reconstruction.update(spectrum_est, spectrum)

            self.log("test_loss", loss, prog_bar=True)
            self.log("test_acc", self.test_acc, prog_bar=True)
            self.log("test_mse", self.test_reconstruction, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)



NAME_MODEL_MAP = {
    "db_conformer": db_conformer,
    "mtf_c": mtf_c
}
