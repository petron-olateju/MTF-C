import numpy as np
import mne
from utils.preprocessing import compute_band_powers

import pytorch_lightning as pl
from utils.data_loader import get_data_subjects, get_data_loader
from utils.data_loader import MI_DATASETS, SSVEP_DATASETS, RESTING_STATE_DATASETS
from utils.data_loader import MI_DataLoader, SSVEP_DataLoader, RestingState_DataLoader
from utils.preprocessing import train_val_test_split
from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import random_split, TensorDataset, DataLoader

# Helper Functions
def get_data_subjects(dataset_name):
    if dataset_name in MI_DATASETS:
        return MI_DataLoader.get_subjects(dataset_name=dataset_name)
    elif dataset_name in SSVEP_DATASETS:
        return SSVEP_DataLoader.get_subjects(dataset_name=dataset_name)
    elif dataset_name in RESTING_STATE_DATASETS:
        return RestingState_DataLoader.get_subjects(dataset_name=dataset_name)


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
        preprocessing_args=None,
        t0=0.5,
        t1=3.5,
        spectrum=None,
        n_filter_banks = 0
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
        self.preprocessing_args = preprocessing_args
        self.t0 = t0
        self.t1 = t1

        self.spectrum = spectrum
        self.n_filter_banks = n_filter_banks

        self.setup()

    def compute_spectrum(self, X):
        if self.spectrum.upper() == 'FREQUENCY_BACKBONE':
            X_spectrum = compute_band_powers(
                X, 
                n_filter_banks=self.n_filter_banks, 
                fs=self.info['fs']
            )
            X_spectrum = torch.tensor(X_spectrum, dtype=torch.float32)
        
        return X_spectrum

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
        loader = get_data_loader(**args)
        X, y, self.info = loader.get_data()
        
        if self.test_split > 0:
            (X_train, y_train, X_cal, y_val, X_test, y_test) = train_val_test_split(
                X, y,
                self.train_split, self.val_split, self.test_split,
                self.preprocessing_args, self.seed
            )
            
            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.long)
            if self.spectrum is not None:
                X_test_spectrum = self.compute_spectrum(X_test.cpu().detach().numpy())
                self.test_dataset = TensorDataset(X_test_spectrum, X_test, y_test)
            else:
                self.test_dataset = TensorDataset(X_test, y_test)
        else:
            (X_train, y_train, X_val, y_val) = train_val_test_split(
                X, y,
                self.train_split, self.val_split, self.test_split,
                self.preprocessing_args, self.seed
            )

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)
        if self.spectrum is not None:
            X_train_spectrum = self.compute_spectrum(X_train.cpu().detach().numpy())
            X_val_spectrum = self.compute_spectrum(X_val.cpu().detach().numpy())
            self.train_dataset = TensorDataset(X_train_spectrum, X_train, y_train)
            self.val_dataset = TensorDataset(X_val_spectrum, X_val, y_val)
        else:
            self.train_dataset = TensorDataset(X_train, y_train)
            self.val_dataset = TensorDataset(X_val, y_val)

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
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        if not hasattr(self, "test_dataset"):
            return []
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

class StratifiedKFoldDataModule(TrainValTest_Split_Loader):

    def __init__(
        self,
        dataset_name,
        subject,
        batch_size,
        num_workers=0,
        cv=5,
        fold_index=0,
        preprocessing_pipeline=None,
        preprocessing_args=None,
        t0=0.5,
        t1=3.5,
        spectrum=None,
        n_filter_banks=0
    ):
        self.fold_index = fold_index

        super().__init__(
            dataset_name=dataset_name,
            subject=subject,
            batch_size=batch_size,
            num_workers=num_workers,
            cv=cv,
            preprocessing_pipeline=preprocessing_pipeline,
            preprocessing_args=preprocessing_args,
            t0=t0,
            t1=t1,
            spectrum=spectrum,
            n_filter_banks=n_filter_banks
        )

        self.setup()

    def setup(self, stage=None):
        args = {
            "dataset_name": self.dataset_name,
            "subject": self.subject,
            "preprocessing_pipeline": self.preprocessing_pipeline,
            "t0": self.t0,
            "t1": self.t1,
        }
        loader = get_data_loader(**args)
        X, y, self.info = loader.get_data()

        skf = StratifiedKFold(
            n_splits = self.cv,
            shuffle=False
        )
        splits = list(skf.split(range(X.shape[0]), y))
        train_idx, val_idx = splits[self.fold_index]

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        if self.spectrum is not None:
            X_spectrum = self.compute_spectrum(torch.tensor(X))
            X_train_spectrum = X_spectrum[train_idx]
            X_val_spectrum = X_spectrum[val_idx]

            self.train_dataset = TensorDataset(X_train_spectrum, X_train, y_train)
            self.val_dataset = TensorDataset(X_val_spectrum, X_val, y_val)
        else:
            self.train_dataset = TensorDataset(X_train, y_train)
            self.val_dataset = TensorDataset(X_val, y_val)

class LOSO_Loader(TrainValTest_Split_Loader):

    def __init__(
        self,
        dataset_name,
        subject,
        batch_size,
        seed=0,
        num_workers=0,
        preprocessing_pipeline=None,
        preprocessing_args=None,
        t0=0.5,
        t1=3.5,
        spectrum=None,
        n_filter_banks=0
    ):

        super().__init__(
            dataset_name=dataset_name,
            subject=subject,
            batch_size=batch_size,
            seed=seed,
            num_workers=num_workers,
            preprocessing_pipeline=preprocessing_pipeline,
            preprocessing_args=preprocessing_args,
            t0=t0,
            t1=t1,
            spectrum=spectrum,
            n_filter_banks=n_filter_banks
        )

        self.setup()

    def setup(self, stage=None):
        X_train, y_train = [], []
        X_val, y_val = [], []
        for subject in get_data_subjects(self.dataset_name):
            args = {
                "dataset_name": self.dataset_name,
                "subject": subject,
                "preprocessing_pipeline": self.preprocessing_pipeline,
                "t0": self.t0,
                "t1": self.t1,
            }
            loader = get_data_loader(**args)
            X, y, _info = loader.get_data()
            if subject != self.subject:
                X_train.append(X)
                y_train.append(y)
            else:
                X_val.append(X)
                y_val.append(y)
                self.info = _info
        
        X_train = np.concat(X_train, axis=0)
        y_train = np.concat(y_train, axis=-1)
        X_val = np.concat(X_val, axis=0)
        y_val = np.concat(y_val, axis=-1)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)

        if self.spectrum is not None:
            X_train_spectrum = self.compute_spectrum(torch.tensor(X_train))
            X_val_spectrum = self.compute_spectrum(torch.tensor(X_val))

            self.train_dataset = TensorDataset(X_train_spectrum, X_train, y_train)
            self.val_dataset = TensorDataset(X_val_spectrum, X_val, y_val)
        else:
            self.train_dataset = TensorDataset(X_train, y_train)
            self.val_dataset = TensorDataset(X_val, y_val)