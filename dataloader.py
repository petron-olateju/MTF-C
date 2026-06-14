import pytorch_lightning as pl
from utils.data_loader import MI_DATASETS
from utils.data_loader import MI_DataLoader

import torch
from torch.utils.data import random_split, TensorDataset, DataLoader


class TrainValTest_Split_Loader(pl.LightningDataModule):

    def __init__(
        self,
        dataset_name,
        subject,
        batch_size,
        seed=0,
        num_workers=0,
        val_split=0.1,
        test_split=0.0,
        cv=5,
        preprocessing_pipeline=None,
        t0=0.5,
        t1=3.5,
    ):
        super().__init__()

        self.cv = cv
        if (val_split is None) or (val_split == 0.0):
            self.val_split = 1 / self.cv
            self.test_split = 0
        else:
            assert 0 < val_split < 1
            assert test_split + val_split < 1
            self.val_split = val_split
            self.test_split = test_split
        assert 0 <= test_split < 1
        self.train_split = 1 - (self.test_split + self.val_split)

        self.seed = seed

        self.dataset_name = dataset_name
        self.subject = subject
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocessing_pipeline = preprocessing_pipeline
        self.t0 = t0
        self.t1 = t1

        self.setup()

    def setup(self, stage=None):
        if hasattr(self, "train_dataset"):
            return

        args = {
            "dataset_name": self.dataset_name,
            "subject": self.subject,
            "preprocessing_pipeline": self.preprocessing_pipeline,
            "t0": self.t0,
            "t1": self.t1,
        }
        if self.dataset_name in MI_DATASETS:
            X, y, self.info = MI_DataLoader(**args).get_data()
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long)
            n = X.size(0)
            dataset = TensorDataset(X, y)

            train_len = int(n * self.train_split)
            val_len = int(n * self.val_split)
            if self.test_split > 0:
                test_len = n - train_len - val_len
                self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                    dataset,
                    [train_len, val_len, test_len],
                    generator=torch.Generator().manual_seed(self.seed),
                )
            else:
                val_len = n - train_len
                self.train_dataset, self.val_dataset = random_split(
                    dataset,
                    [train_len, val_len],
                    generator=torch.Generator().manual_seed(self.seed),
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        if not hasattr(self, "test_dataset"):
            return []
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
