from argparse import Namespace
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from models.DBConformer import DBConformer
from torchmetrics.classification import Accuracy
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
        self.log("train_performance", self.train_acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            _, logits, loss, y = self._common_step(batch, batch_idx)

            preds = logits.argmax(dim=1)
            self.val_acc.update(preds, y)

            self.log("val_loss", loss, prog_bar=True)
            self.log("val_performance", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            _, logits, loss, y = self._common_step(batch, batch_idx)

            preds = logits.argmax(dim=1)
            self.test_acc.update(preds, y)

            self.log("test_loss", loss, prog_bar=True)
            self.log("test_performance", self.test_acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


NAME_MODEL_MAP = {
    "db_conformer": db_conformer,
}
