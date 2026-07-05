import numpy as np
import mne
from utils.preprocessing import compute_band_powers, compute_channels_band_powers, compute_downsampled_stft

import pytorch_lightning as pl
from utils.data_loader import get_data_subjects, get_data_loader
from utils.data_loader import MI_DATASETS, SSVEP_DATASETS, RESTING_STATE_DATASETS
from utils.data_loader import MI_DataLoader, SSVEP_DataLoader, RestingState_DataLoader
from utils.preprocessing import train_val_test_split, EA, EA_online
from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import random_split, TensorDataset, DataLoader

mne.set_log_level('ERROR')

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
        batch_size,
        num_workers=0,
        val_split=0.1,
        test_split=0.0,
        cv=5,
        preprocessing_pipeline=None,
        preprocessing_args=None,
        t0=0.5,
        t1=3.5,
        spectrum=None,
        sst_decoder=None,
        n_filter_banks = 0,
        patch_size = 100,
        freq_downsample=1,
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

        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocessing_pipeline = preprocessing_pipeline
        self.preprocessing_args = preprocessing_args
        self.t0 = t0
        self.t1 = t1

        self.spectrum = spectrum
        self.sst_decoder = sst_decoder
        self.n_filter_banks = n_filter_banks
        self.patch_size = patch_size
        self.freq_downsample = freq_downsample

    def preload_data(self):
        self.subjects_data = {}
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

            self.subjects_data[subject]={'X': X, 'y': y, 'info': _info}
    
    def update_subject(self, subject, seed, fold=None):
        self.subject = subject
        self.setup_data(seed, fold)

    def compute_spectrum(self, X):
        if self.spectrum.upper() == 'FREQUENCY_BACKBONE':
            X_spectrum = compute_band_powers(
                X, 
                n_filter_banks=self.n_filter_banks, 
                fs=self.info['fs']
            )
            X_spectrum = torch.tensor(X_spectrum, dtype=torch.float32)
        elif self.spectrum.upper() == 'CHANNELS_FREQUENCY_BACKBONE':
            X_spectrum = compute_channels_band_powers(
                X, 
                n_filter_banks=self.n_filter_banks, 
                fs=self.info['fs']
            )
            X_spectrum = torch.tensor(X_spectrum, dtype=torch.float32)
            X_spectrum = torch.mean(X_spectrum, dim=1)
        
        print(f"Spectrum Data Shape: {X_spectrum.size()}")
        return X_spectrum
    
    def compute_sst(self, X, n_times, fs):
        if self.spectrum.upper() in ['STFT', 'FREQUENCY_BACKBONE']:
            X_sst = compute_downsampled_stft(
                X.cpu().numpy(),
                n_filter_banks=self.n_filter_banks,
                patch_size=self.patch_size,
                n_times=n_times,
                freq_downsample=self.freq_downsample,
                fs=fs
            )

            if 'SCALE_SST' in [ppo.upper() for ppo in self.preprocessing_args]:
                min_ = X_sst.min(axis=(1, 2, 3), keepdims=True)
                max_ = X_sst.max(axis=(1, 2,3), keepdims=True)
                X_sst = (X_sst - min_) / (max_ - min_ + 1e-8)

        X_sst = torch.tensor(X_sst, dtype=torch.float32)
        print(f"SST Data Shape: {X_sst.size()}")
        return X_sst

    def setup_data(self, seed=0, fold=None):
        if hasattr(self, "train_dataset"):
            return

        data = self.subjects_data[self.subject]
        X = data['X']
        y = data['y']
        self.info = data['info']
        
        if self.test_split > 0:
            (X_train, y_train, X_val, y_val, X_test, y_test) = train_val_test_split(
                X, y,
                self.train_split, self.val_split, self.test_split,
                self.preprocessing_args, seed
            )
            
            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.long)
            if self.spectrum is not None:
                X_test_spectrum = self.compute_spectrum(X_test.cpu().detach().numpy())
                if self.sst_decoder is None:
                    self.test_dataset = TensorDataset(X_test_spectrum, X_test, y_test)
                else:
                    X_test_sst = self.compute_sst(X_test.cpu(), self.info['n_times'], self.info['fs'])
                    self.test_dataset = TensorDataset(X_test_sst, X_test_spectrum, X_test, y_test)
            else:
                self.test_dataset = TensorDataset(X_test, y_test)
        else:
            (X_train, y_train, X_val, y_val) = train_val_test_split(
                X, y,
                self.train_split, self.val_split, self.test_split,
                self.preprocessing_args, seed
            )

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)

        if (self.spectrum is not None) and (self.spectrum.upper() == 'FREQUENCY_BACKBONE'):
            X_train_spectrum = self.compute_spectrum(torch.tensor(X_train))
            X_val_spectrum = self.compute_spectrum(torch.tensor(X_val))

            if self.sst_decoder is None:
                self.train_dataset = TensorDataset(X_train_spectrum, X_train, y_train)
                self.val_dataset = TensorDataset(X_val_spectrum, X_val, y_val)
            else:
                X_train_sst = self.compute_sst(X_train.cpu(), self.info['n_times'], self.info['fs'])
                X_val_sst = self.compute_sst(X_val.cpu(), self.info['n_times'], self.info['fs'])
                self.train_dataset = TensorDataset(X_train_sst, X_train_spectrum, X_train, y_train)
                self.val_dataset = TensorDataset(X_val_sst, X_val_spectrum, X_val, y_val)

        elif self.sst_decoder is not None:
            X_train_sst = self.compute_sst(X_train.cpu(), self.info['n_times'], self.info['fs'])
            X_val_sst = self.compute_sst(X_val.cpu(), self.info['n_times'], self.info['fs'])
            self.train_dataset = TensorDataset(X_train_sst, X_train, y_train)
            self.val_dataset = TensorDataset(X_val_sst, X_val, y_val)
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
        batch_size,
        num_workers=0,
        cv=5,
        fold_index=0,
        preprocessing_pipeline=None,
        preprocessing_args=None,
        t0=0.5,
        t1=3.5,
        spectrum=None,
        sst_decoder=None,
        n_filter_banks = 0,
        patch_size = 100,
        freq_downsample=1,
    ):
        self.fold_index = fold_index

        super().__init__(
            dataset_name=dataset_name,
            batch_size=batch_size,
            num_workers=num_workers,
            cv=cv,
            preprocessing_pipeline=preprocessing_pipeline,
            preprocessing_args=preprocessing_args,
            t0=t0,
            t1=t1,
            spectrum=spectrum,
            sst_decoder=sst_decoder,
            n_filter_banks=n_filter_banks,
            patch_size=patch_size,
            freq_downsample=freq_downsample
        )

    def setup_data(self, seed=None, fold=0):
        data = self.subjects_data[self.subject]
        X = data['X']
        y = data['y']
        self.info = data['info']

        skf = StratifiedKFold(
            n_splits = self.cv,
            shuffle=False
        )
        splits = list(skf.split(range(X.shape[0]), y))
        train_idx, val_idx = splits[fold]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        if 'EA' in self.preprocessing_args:
            X_train, sqrtRefEA = EA(X_train)
            X_val = EA_online(X_val, sqrtRefEA)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)

        if (self.spectrum is not None) and (self.spectrum.upper() == 'FREQUENCY_BACKBONE'):
            X_train_spectrum = self.compute_spectrum(torch.tensor(X_train))
            X_val_spectrum = self.compute_spectrum(torch.tensor(X_val))

            if self.sst_decoder is None:
                self.train_dataset = TensorDataset(X_train_spectrum, X_train, y_train)
                self.val_dataset = TensorDataset(X_val_spectrum, X_val, y_val)
            else:
                X_train_sst = self.compute_sst(X_train.cpu(), self.info['n_times'], self.info['fs'])
                X_val_sst = self.compute_sst(X_val.cpu(), self.info['n_times'], self.info['fs'])
                self.train_dataset = TensorDataset(X_train_sst, X_train_spectrum, X_train, y_train)
                self.val_dataset = TensorDataset(X_val_sst, X_val_spectrum, X_val, y_val)

        elif self.sst_decoder is not None:
            X_train_sst = self.compute_sst(X_train.cpu(), self.info['n_times'], self.info['fs'])
            X_val_sst = self.compute_sst(X_val.cpu(), self.info['n_times'], self.info['fs'])
            self.train_dataset = TensorDataset(X_train_sst, X_train, y_train)
            self.val_dataset = TensorDataset(X_val_sst, X_val, y_val)
        else:
            self.train_dataset = TensorDataset(X_train, y_train)
            self.val_dataset = TensorDataset(X_val, y_val)

class LOSO_Loader(TrainValTest_Split_Loader):

    def __init__(
        self,
        dataset_name,
        batch_size,
        num_workers=0,
        preprocessing_pipeline=None,
        preprocessing_args=None,
        t0=0.5,
        t1=3.5,
        spectrum=None,
        sst_decoder=None,
        n_filter_banks = 0,
        patch_size = 100,
        freq_downsample=1,
    ):

        super().__init__(
            dataset_name=dataset_name,
            batch_size=batch_size,
            num_workers=num_workers,
            preprocessing_pipeline=preprocessing_pipeline,
            preprocessing_args=preprocessing_args,
            t0=t0,
            t1=t1,
            spectrum=spectrum,
            sst_decoder=sst_decoder,
            n_filter_banks=n_filter_banks,
            patch_size=patch_size,
            freq_downsample=freq_downsample
        )

    def setup_data(self, seed=None, fold=None):
        self.subject 
        X_train, y_train = [], []
        X_val, y_val = [], []
        for subject in get_data_subjects(self.dataset_name):
            data = self.subjects_data[subject]

            if subject != self.subject:
                X_train.append(data['X'])
                y_train.append(data['y'])
            else:
                X_val.append(data['X'])
                y_val.append(data['y'])
                self.info = data['info']
        
        X_train = np.concat(X_train, axis=0)
        y_train = np.concat(y_train, axis=-1)
        X_val = np.concat(X_val, axis=0)
        y_val = np.concat(y_val, axis=-1)

        if 'EA' in self.preprocessing_args:
            X_train, sqrtRefEA = EA(X_train)
            X_val = EA_online(X_val, sqrtRefEA)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)

        if (self.spectrum is not None) and (self.spectrum.upper() == 'FREQUENCY_BACKBONE'):
            X_train_spectrum = self.compute_spectrum(torch.tensor(X_train))
            X_val_spectrum = self.compute_spectrum(torch.tensor(X_val))

            if self.sst_decoder is None:
                self.train_dataset = TensorDataset(X_train_spectrum, X_train, y_train)
                self.val_dataset = TensorDataset(X_val_spectrum, X_val, y_val)
            else:
                X_train_sst = self.compute_sst(X_train.cpu(), self.info['n_times'], self.info['fs'])
                X_val_sst = self.compute_sst(X_val.cpu(), self.info['n_times'], self.info['fs'])
                self.train_dataset = TensorDataset(X_train_sst, X_train_spectrum, X_train, y_train)
                self.val_dataset = TensorDataset(X_val_sst, X_val_spectrum, X_val, y_val)

        elif self.sst_decoder is not None:
            X_train_sst = self.compute_sst(X_train.cpu(), self.info['n_times'], self.info['fs'])
            X_val_sst = self.compute_sst(X_val.cpu(), self.info['n_times'], self.info['fs'])
            self.train_dataset = TensorDataset(X_train_sst, X_train, y_train)
            self.val_dataset = TensorDataset(X_val_sst, X_val, y_val)
        else:
            self.train_dataset = TensorDataset(X_train, y_train)
            self.val_dataset = TensorDataset(X_val, y_val)