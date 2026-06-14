import numpy as np
import torch
from torch.utils.data import Dataset

from typing import Tuple

from moabb.datasets import (
    BNCI2014_001,
    BNCI2014_002,
    BNCI2014_004,
    BNCI2015_001,
    BNCI2015_004,
    Liu2024,
    AlexMI,
    Kalunga2016,
    MAMEM2,
    MAMEM3,
    Nakanishi2015,
    Wang2016,
    Wang2021Combined,
    Cattan2019_PHMD,
    Hinss2021,
    Rodrigues2017,
)
from moabb.paradigms import MotorImagery, SSVEP, RestingStateToP300Adapter

from braindecode.datasets import SleepPhysionet
from braindecode.preprocessing import create_windows_from_events

import os
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import mne

MI_DATASETS = [
    "BNCI2014_001",
    "BNCI2014_002",
    "BNCI2014_004",
    "BNCI2015_001",
    "BNCI2015_004",
    "Liu2024",
    "AlexMI",
]
SSVEP_DATASETS = ["Kalunga2016", "Nakanishi2015", "Wang2021Combined"]
SLEEP_DATASETS = ["SleepPhysionet"]
RESTING_STATE_DATASETS = [
    "Cattan2019_PHMD",
    "Hinss2021",
    "Rodrigues2017",
    "ButtonToneSZ",
]

DATASET_TASK_MAP = {
    "BNCI2014_001": 'binary'
}


class ButtonToneSZ:
    """Custom dataset for Kaggle button-tone-sz (Schizophrenia) EEG data.

    This dataset contains EEG recordings from 81 subjects (49 Schizophrenia patients + 32 Healthy Controls).
    Data is from a basic sensory task involving button-press and playback tones.

    Dataset URL: https://www.kaggle.com/datasets/broach/button-tone-sz
    """

    KAGGLE_DATASET = "broach/button-tone-sz"
    SUBJECTS_INFO = {
        1: "sz",
        2: "sz",
        3: "sz",
        4: "sz",
        5: "sz",
        6: "sz",
        7: "sz",
        8: "sz",
        9: "sz",
        10: "sz",
        11: "sz",
        12: "sz",
        13: "sz",
        14: "sz",
        15: "sz",
        16: "sz",
        17: "sz",
        18: "sz",
        19: "sz",
        20: "sz",
        21: "sz",
        22: "sz",
        23: "sz",
        24: "sz",
        25: "sz",
        26: "sz",
        27: "sz",
        28: "sz",
        29: "sz",
        30: "sz",
        31: "sz",
        32: "sz",
        33: "sz",
        34: "sz",
        35: "sz",
        36: "sz",
        37: "sz",
        38: "sz",
        39: "sz",
        40: "sz",
        41: "sz",
        42: "sz",
        43: "sz",
        44: "sz",
        45: "sz",
        46: "sz",
        47: "sz",
        48: "sz",
        49: "sz",
        50: "hc",
        51: "hc",
        52: "hc",
        53: "hc",
        54: "hc",
        55: "hc",
        56: "hc",
        57: "hc",
        58: "hc",
        59: "hc",
        60: "hc",
        61: "hc",
        62: "hc",
        63: "hc",
        64: "hc",
        65: "hc",
        66: "hc",
        67: "hc",
        68: "hc",
        69: "hc",
        70: "hc",
        71: "hc",
        72: "hc",
        73: "hc",
        74: "hc",
        75: "hc",
        76: "hc",
        77: "hc",
        78: "hc",
        79: "hc",
        80: "hc",
        81: "hc",
    }

    def __init__(self, path_to_data=None):
        """Initialize ButtonToneSZ dataset.

        Args:
            path_to_data: Path to locally downloaded dataset. If None, will check:
                1. /kaggle/input/button-tone-sz/ (Kaggle environment)
                2. ~/.mne_data/ButtonToneSZ (local fallback)
        """
        if path_to_data is None:
            kaggle_path = "/kaggle/input/button-tone-sz"
            if os.path.exists(kaggle_path):
                path_to_data = kaggle_path
            else:
                path_to_data = os.path.expanduser("~/.mne_data/ButtonToneSZ")

        self.path_to_data = path_to_data

        if not os.path.exists(path_to_data):
            self._download()

    def _download(self):
        """Download dataset from Kaggle."""
        os.makedirs(self.path_to_data, exist_ok=True)

        try:
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(
                self.KAGGLE_DATASET, path=self.path_to_data, unzip=True
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download dataset from Kaggle: {e}. "
                "Please ensure Kaggle API is configured. "
                "You can download manually from: https://www.kaggle.com/datasets/broach/button-tone-sz"
            )

    def _find_set_files(self):
        """Find all .set files in the data directory."""
        set_files = []

        for root, dirs, files in os.walk(self.path_to_data):
            for f in files:
                if f.endswith(".set"):
                    set_files.append(os.path.join(root, f))

        return sorted(set_files)

    def get_data(self, subjects=None):
        """Get data for specified subjects.

        Args:
            subjects: List of subject IDs. If None, returns all subjects.

        Returns:
            Dictionary: {subject_id: {session_id: {run_id: Raw}}}
        """
        if subjects is None:
            subjects = list(self.SUBJECTS_INFO.keys())

        data = {}
        set_files = self._find_set_files()

        if not set_files:
            raise FileNotFoundError(
                f"No .set files found in {self.path_to_data}. "
                f"Please ensure the dataset is properly extracted."
            )

        for idx, subject_id in enumerate(subjects):
            if idx < len(set_files):
                set_file = set_files[idx]
                try:
                    raw = mne.io.read_raw_eeglab(set_file, preload=False, verbose=False)
                    data[subject_id] = {"session_0": {"run_0": raw}}
                except Exception as e:
                    print(f"Warning: Could not load {set_file}: {e}")
                    continue

        return data

    @staticmethod
    def get_subjects(path_to_data=None):
        """Get list of available subject IDs.

        Args:
            path_to_data: Path to locally downloaded dataset (unused, for API consistency).

        Returns:
            List of subject IDs (1-81).
        """
        return list(ButtonToneSZ.SUBJECTS_INFO.keys())

    @staticmethod
    def get_subject_label(subject_id):
        """Get label for a subject (sz or hc).

        Args:
            subject_id: Subject ID.

        Returns:
            'sz' for Schizophrenia, 'hc' for Healthy Control.
        """
        return ButtonToneSZ.SUBJECTS_INFO.get(subject_id, None)


def chronological_stratified_kfold(y, n_splits=5):
    """Split each class chronologically into n_splits, then merge."""
    indices = np.arange(len(y))
    fold_indices = [[] for _ in range(n_splits)]
    for class_label in np.unique(y):
        class_idx = indices[y == class_label]
        splits = np.array_split(class_idx, n_splits)
        for fold, split in enumerate(splits):
            fold_indices[fold].extend(split)

    for test_fold in range(n_splits):
        test_idx = np.array(fold_indices[test_fold])
        train_idx = np.concatenate(
            [np.array(fold_indices[i]) for i in range(n_splits) if i != test_fold]
        )
        yield train_idx, test_idx


class EEGDataset(Dataset):
    def __init__(self, data, labels, stft):
        self.signals = torch.tensor(data, dtype=torch.float32)
        self.labels = labels
        self.stft = stft

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx], self.stft[idx]


class MI_DataLoader:
    """Motor Imagery Data Loader class for loading various BCI datasets."""

    DATASETS = {
        "BNCI2014_001": BNCI2014_001,
        "BNCI2014_002": BNCI2014_002,
        "BNCI2014_004": BNCI2014_004,
        "BNCI2015_001": BNCI2015_001,
        "BNCI2015_004": BNCI2015_004,
        "Liu2024": Liu2024,
        "AlexMI": AlexMI,
    }

    def __init__(
        self, dataset_name, subject, preprocessing_pipeline=None, t0=0.5, t1=3.5
    ):
        """Initialize MI_DataLoader.

        Args:
            dataset_name: Name of the dataset ('BNCI2014_001', 'BNCI2014_002', etc.)
            subject: Subject ID (integer)
            preprocessing_pipeline: Optional list of preprocessing functions
            t0: Start time for epoching (seconds)
            t1: End time for epoching (seconds)
        """
        self.dataset_name = dataset_name
        self.subject = subject
        self.preprocessing_pipeline = preprocessing_pipeline
        self.t0 = t0
        self.t1 = t1

        if dataset_name not in self.DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Available: {list(self.DATASETS.keys())}"
            )

        self.dataset = self.DATASETS[dataset_name]()
        self._load_data()

    def _load_data(self):
        """Load data based on dataset name."""
        load_methods = {
            "BNCI2014_001": self._load_BNCI2014_001,
            "BNCI2014_002": self._load_BNCI2014_002,
            "BNCI2014_004": self._load_BNCI2014_004,
            "BNCI2015_001": self._load_BNCI2015_001,
            "BNCI2015_004": self._load_BNCI2015_004,
            "Liu2024": self._load_Liu2024,
            "AlexMI": self._load_AlexMI,
        }

        self.X, self.y, self.info = load_methods[self.dataset_name]()

    def _get_paradigm(self, resample_rate):
        """Get MotorImagery paradigm."""
        return MotorImagery(channels=None, resample=resample_rate)

    def _apply_preprocessing(self, X):
        """Apply preprocessing pipeline to data."""
        if self.preprocessing_pipeline is not None:
            if not isinstance(self.preprocessing_pipeline, list):
                raise ValueError(
                    "preprocessing_pipeline argument should be a list of preprocessing functions"
                )
            for fn in self.preprocessing_pipeline:
                X = fn(X)
        return X

    def _get_fs(self, subject):
        """Get sampling frequency dynamically."""
        raw = self.dataset.get_data(subjects=[subject])
        first_session_key = list(raw[subject].keys())[0]
        first_run_key = list(raw[subject][first_session_key].keys())[0]
        return raw[subject][first_session_key][first_run_key].info["sfreq"]

    def _load_Liu2024(self) -> Tuple:
        """Load Liu2024 dataset."""
        paradigm = self._get_paradigm(250)

        s_x, s_y, metadata = paradigm.get_data(self.dataset, subjects=[self.subject])
        session_keys = metadata["session"].unique()
        first_session = sorted(session_keys)[0]
        session_mask = metadata["session"] == first_session
        s_x = s_x[session_mask]
        s_y = s_y[session_mask]

        fs = self._get_fs(self.subject)
        t0 = int(fs * self.t0)
        t1 = int(fs * self.t1)

        unique_classes = np.unique(s_y)
        class_data = []
        class_labels = []

        for idx, class_name in enumerate(unique_classes):
            class_idx = np.where(s_y == class_name)[0]
            class_data.append(np.array(s_x[class_idx]))
            class_labels.append(np.full(len(class_idx), idx))

        X = np.vstack(class_data)
        X = X[:, :, t0:t1]
        y = np.hstack(class_labels)

        X = self._apply_preprocessing(X)

        n_trials, n_ch, n_times = X.shape
        info = {
            "n_trials": n_trials,
            "n_ch": n_ch,
            "n_times": n_times,
            "n_classes": len(np.unique(y)),
            "fs": fs,
            "class_names": unique_classes.tolist(),
        }

        return (X, y, info)

    def _load_BNCI2014_001(self) -> Tuple:
        """Load BNCI2014_001 dataset."""
        paradigm = self._get_paradigm(250)

        s_x, s_y, metadata = paradigm.get_data(self.dataset, subjects=[self.subject])
        session_mask = metadata["session"].str.contains("0")
        s_x = s_x[session_mask]
        s_y = s_y[session_mask]

        raw = self.dataset.get_data(subjects=[1])
        fs = raw[1]["0train"]["0"].info["sfreq"]
        t0 = int(fs * self.t0)
        t1 = int(fs * self.t1)

        left_imagery_idx = np.where(s_y == "left_hand")[0]
        right_imagery_idx = np.where(s_y == "right_hand")[0]
        left_imagery = np.array(s_x[left_imagery_idx])
        right_imagery = np.array(s_x[right_imagery_idx])

        X = np.vstack((left_imagery, right_imagery))
        X = X[:, :, t0:t1]
        y = np.hstack(
            (np.zeros(left_imagery.shape[0]), np.ones(right_imagery.shape[0]))
        )

        X = self._apply_preprocessing(X)

        n_trials, n_ch, n_times = X.shape
        info = {
            "n_trials": n_trials,
            "n_ch": n_ch,
            "n_times": n_times,
            "n_classes": len(np.unique(y)),
            "fs": fs,
        }

        return (X, y, info)

    def _load_BNCI2014_004(self) -> Tuple:
        """Load BNCI2014_004 dataset."""
        paradigm = self._get_paradigm(250)

        s_x, s_y, metadata = paradigm.get_data(self.dataset, subjects=[self.subject])
        session_keys = metadata["session"].unique()
        first_session = sorted(session_keys)[0]
        session_mask = metadata["session"] == first_session
        s_x = s_x[session_mask]
        s_y = s_y[session_mask]

        fs = self._get_fs(self.subject)
        t0 = int(fs * self.t0)
        t1 = int(fs * self.t1)

        left_imagery_idx = np.where(s_y == "left_hand")[0]
        right_imagery_idx = np.where(s_y == "right_hand")[0]
        left_imagery = np.array(s_x[left_imagery_idx])
        right_imagery = np.array(s_x[right_imagery_idx])

        X = np.vstack((left_imagery, right_imagery))
        X = X[:, :, t0:t1]
        y = np.hstack(
            (np.zeros(left_imagery.shape[0]), np.ones(right_imagery.shape[0]))
        )

        X = self._apply_preprocessing(X)

        n_trials, n_ch, n_times = X.shape
        info = {
            "n_trials": n_trials,
            "n_ch": n_ch,
            "n_times": n_times,
            "n_classes": len(np.unique(y)),
            "fs": fs,
        }

        return (X, y, info)

    def _load_BNCI2015_001(self) -> Tuple:
        """Load BNCI2015_001 dataset."""
        paradigm = self._get_paradigm(512)

        s_x, s_y, metadata = paradigm.get_data(self.dataset, subjects=[self.subject])
        session_keys = metadata["session"].unique()
        first_session = sorted(session_keys)[0]
        session_mask = metadata["session"] == first_session
        s_x = s_x[session_mask]
        s_y = s_y[session_mask]

        fs = self._get_fs(self.subject)
        t0 = int(fs * self.t0)
        t1 = int(fs * self.t1)

        right_imagery_idx = np.where(s_y == "right_hand")[0]
        feet_imagery_idx = np.where(s_y == "feet")[0]

        right_imagery = np.array(s_x[right_imagery_idx])
        feet_imagery = np.array(s_x[feet_imagery_idx])

        X = np.vstack((right_imagery, feet_imagery))
        X = X[:, :, t0:t1]
        y = np.hstack(
            (np.zeros(right_imagery.shape[0]), np.ones(feet_imagery.shape[0]))
        )

        X = self._apply_preprocessing(X)

        n_trials, n_ch, n_times = X.shape
        info = {
            "n_trials": n_trials,
            "n_ch": n_ch,
            "n_times": n_times,
            "n_classes": len(np.unique(y)),
            "fs": fs,
        }

        return (X, y, info)

    def _load_BNCI2014_002(self) -> Tuple:
        """Load BNCI2014_002 dataset."""
        paradigm = self._get_paradigm(512)

        s_x, s_y, metadata = paradigm.get_data(self.dataset, subjects=[self.subject])
        session_keys = metadata["session"].unique()
        first_session = sorted(session_keys)[0]
        session_mask = metadata["session"] == first_session
        s_x = s_x[session_mask]
        s_y = s_y[session_mask]

        fs = self._get_fs(self.subject)
        t0 = int(fs * self.t0)
        t1 = int(fs * self.t1)

        right_imagery_idx = np.where(s_y == "right_hand")[0]
        feet_imagery_idx = np.where(s_y == "feet")[0]

        right_imagery = np.array(s_x[right_imagery_idx])
        feet_imagery = np.array(s_x[feet_imagery_idx])

        X = np.vstack((right_imagery, feet_imagery))
        X = X[:, :, t0:t1]
        y = np.hstack(
            (np.zeros(right_imagery.shape[0]), np.ones(feet_imagery.shape[0]))
        )

        X = self._apply_preprocessing(X)

        n_trials, n_ch, n_times = X.shape
        info = {
            "n_trials": n_trials,
            "n_ch": n_ch,
            "n_times": n_times,
            "n_classes": len(np.unique(y)),
            "fs": fs,
        }

        return (X, y, info)

    def _load_BNCI2015_004(self) -> Tuple:
        """Load BNCI2015_004 dataset."""
        paradigm = self._get_paradigm(256)

        s_x, s_y, metadata = paradigm.get_data(self.dataset, subjects=[self.subject])
        session_keys = metadata["session"].unique()
        first_session = sorted(session_keys)[0]
        session_mask = metadata["session"] == first_session
        s_x = s_x[session_mask]
        s_y = s_y[session_mask]

        fs = self._get_fs(self.subject)
        t0 = int(fs * self.t0)
        t1 = int(fs * self.t1)

        unique_classes = np.unique(s_y)
        class_data = []
        class_labels = []

        for idx, class_name in enumerate(unique_classes):
            class_idx = np.where(s_y == class_name)[0]
            class_data.append(np.array(s_x[class_idx]))
            class_labels.append(np.full(len(class_idx), idx))

        X = np.vstack(class_data)
        X = X[:, :, t0:t1]
        y = np.hstack(class_labels)

        X = self._apply_preprocessing(X)

        n_trials, n_ch, n_times = X.shape
        info = {
            "n_trials": n_trials,
            "n_ch": n_ch,
            "n_times": n_times,
            "n_classes": len(np.unique(y)),
            "fs": fs,
            "class_names": unique_classes.tolist(),
        }

        return (X, y, info)

    def _load_AlexMI(self) -> Tuple:
        """Load AlexMI dataset."""
        paradigm = self._get_paradigm(512)

        s_x, s_y, metadata = paradigm.get_data(self.dataset, subjects=[self.subject])
        session_keys = metadata["session"].unique()
        first_session = sorted(session_keys)[0]
        session_mask = metadata["session"] == first_session
        s_x = s_x[session_mask]
        s_y = s_y[session_mask]

        fs = self._get_fs(self.subject)
        t0 = int(fs * self.t0)
        t1 = int(fs * self.t1)

        unique_classes = np.unique(s_y)
        class_data = []
        class_labels = []

        for idx, class_name in enumerate(unique_classes):
            class_idx = np.where(s_y == class_name)[0]
            class_data.append(np.array(s_x[class_idx]))
            class_labels.append(np.full(len(class_idx), idx))

        X = np.vstack(class_data)
        X = X[:, :, t0:t1]
        y = np.hstack(class_labels)

        X = self._apply_preprocessing(X)

        n_trials, n_ch, n_times = X.shape
        info = {
            "n_trials": n_trials,
            "n_ch": n_ch,
            "n_times": n_times,
            "n_classes": len(np.unique(y)),
            "fs": fs,
            "class_names": unique_classes.tolist(),
        }

        return (X, y, info)

    def get_data(self):
        """Return loaded data.

        Returns:
            Tuple: (X, y, info) where X is the data, y are labels, info is metadata dict
        """
        return self.X, self.y, self.info

    @staticmethod
    def get_subjects(dataset_name):
        """Get list of available subjects for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            List of subject IDs
        """
        if dataset_name not in MI_DataLoader.DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Available: {list(MI_DataLoader.DATASETS.keys())}"
            )

        dataset = MI_DataLoader.DATASETS[dataset_name]()
        return dataset.subject_list

    @staticmethod
    def get_available_datasets():
        """Get list of available dataset names."""
        return list(MI_DataLoader.DATASETS.keys())


class SSVEP_DataLoader:
    """SSVEP Data Loader class for loading various SSVEP datasets."""

    DATASETS = {
        "Kalunga2016": Kalunga2016,
        "MAMEM2": MAMEM2,
        "MAMEM3": MAMEM3,
        "Nakanishi2015": Nakanishi2015,
        "Wang2016": Wang2016,
        "Wang2021Combined": Wang2021Combined,
    }

    def __init__(
        self,
        dataset_name,
        subject,
        preprocessing_pipeline=None,
        t0=0.0,
        tmax=None,
        fmin=7,
        fmax=45,
    ):
        """Initialize SSVEP_DataLoader.

        Args:
            dataset_name: Name of the dataset ('Lee2019_SSVEP', 'Kalunga2016', etc.)
            subject: Subject ID (integer)
            preprocessing_pipeline: Optional list of preprocessing functions
            t0: Start time for epoching (seconds)
            tmax: End time for epoching (seconds, default is None)
            fmin: Low cutoff frequency for bandpass filter (Hz)
            fmax: High cutoff frequency for bandpass filter (Hz)
        """
        self.dataset_name = dataset_name
        self.subject = subject
        self.preprocessing_pipeline = preprocessing_pipeline
        self.t0 = t0
        self.tmax = tmax
        self.fmin = fmin
        self.fmax = fmax

        if dataset_name not in self.DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Available: {list(self.DATASETS.keys())}"
            )

        self.dataset = self.DATASETS[dataset_name]()
        self._load_data()

    def _load_data(self):
        """Load data based on dataset name."""
        load_methods = {
            "Kalunga2016": self._load_Kalunga2016,
            "MAMEM2": self._load_MAMEM2,
            "MAMEM3": self._load_MAMEM3,
            "Nakanishi2015": self._load_Nakanishi2015,
            "Wang2016": self._load_Wang2016,
            "Wang2021Combined": self._load_Wang2021Combined,
        }

        self.X, self.y, self.info = load_methods[self.dataset_name]()

    def _get_paradigm(self, resample_rate):
        """Get SSVEP paradigm."""
        return SSVEP(
            fmin=self.fmin,
            fmax=self.fmax,
            tmin=self.t0,
            tmax=self.tmax,
            resample=resample_rate,
        )

    def _apply_preprocessing(self, X):
        """Apply preprocessing pipeline to data."""
        if self.preprocessing_pipeline is not None:
            if not isinstance(self.preprocessing_pipeline, list):
                raise ValueError(
                    "preprocessing_pipeline argument should be a list of preprocessing functions"
                )
            for fn in self.preprocessing_pipeline:
                X = fn(X)
        return X

    def _get_fs(self, subject):
        """Get sampling frequency dynamically."""
        raw = self.dataset.get_data(subjects=[subject])
        first_session_key = list(raw[subject].keys())[0]
        first_run_key = list(raw[subject][first_session_key].keys())[0]
        return raw[subject][first_session_key][first_run_key].info["sfreq"]

    def _load_Kalunga2016(self) -> Tuple:
        """Load Kalunga2016 dataset."""
        paradigm = self._get_paradigm(256)

        s_x, s_y, metadata = paradigm.get_data(self.dataset, subjects=[self.subject])
        s_x = np.array(s_x)

        fs = self._get_fs(self.subject)
        t0_idx = int(fs * self.t0)
        tmax_idx = int(fs * self.tmax) if self.tmax is not None else s_x.shape[2]

        unique_classes = np.unique(s_y)
        class_data = []
        class_labels = []

        for idx, class_name in enumerate(unique_classes):
            class_idx = np.where(s_y == class_name)[0]
            class_data.append(np.array(s_x[class_idx]))
            class_labels.append(np.full(len(class_idx), idx))

        X = np.vstack(class_data)
        X = X[:, :, t0_idx:tmax_idx]
        y = np.hstack(class_labels)

        X = self._apply_preprocessing(X)

        n_trials, n_ch, n_times = X.shape
        freq_classes = [c for c in unique_classes if c.startswith("freq_")]
        if freq_classes:
            freqs = sorted([float(c.replace("freq_", "")) for c in freq_classes])
        else:
            freqs = []
        info = {
            "n_trials": n_trials,
            "n_ch": n_ch,
            "n_times": n_times,
            "n_classes": len(np.unique(y)),
            "fs": fs,
            "class_names": unique_classes.tolist(),
            "freqs": freqs,
        }

        return (X, y, info)

    def _load_MAMEM2(self) -> Tuple:
        """Load MAMEM2 dataset."""
        paradigm = self._get_paradigm(256)

        s_x, s_y, metadata = paradigm.get_data(self.dataset, subjects=[self.subject])
        s_x = np.array(s_x)

        fs = self._get_fs(self.subject)
        t0_idx = int(fs * self.t0)
        tmax_idx = int(fs * self.tmax) if self.tmax is not None else s_x.shape[2]

        unique_classes = np.unique(s_y)
        class_data = []
        class_labels = []

        for idx, class_name in enumerate(unique_classes):
            class_idx = np.where(s_y == class_name)[0]
            class_data.append(np.array(s_x[class_idx]))
            class_labels.append(np.full(len(class_idx), idx))

        X = np.vstack(class_data)
        X = X[:, :, t0_idx:tmax_idx]
        y = np.hstack(class_labels)

        X = self._apply_preprocessing(X)

        n_trials, n_ch, n_times = X.shape
        freq_classes = [c for c in unique_classes if c.startswith("freq_")]
        if freq_classes:
            freqs = sorted([float(c.replace("freq_", "")) for c in freq_classes])
        else:
            freqs = []
        info = {
            "n_trials": n_trials,
            "n_ch": n_ch,
            "n_times": n_times,
            "n_classes": len(np.unique(y)),
            "fs": fs,
            "class_names": unique_classes.tolist(),
            "freqs": freqs,
        }

        return (X, y, info)

    def _load_MAMEM3(self) -> Tuple:
        """Load MAMEM3 dataset."""
        paradigm = self._get_paradigm(256)

        s_x, s_y, metadata = paradigm.get_data(self.dataset, subjects=[self.subject])
        s_x = np.array(s_x)

        fs = self._get_fs(self.subject)
        t0_idx = int(fs * self.t0)
        tmax_idx = int(fs * self.tmax) if self.tmax is not None else s_x.shape[2]

        unique_classes = np.unique(s_y)
        class_data = []
        class_labels = []

        for idx, class_name in enumerate(unique_classes):
            class_idx = np.where(s_y == class_name)[0]
            class_data.append(np.array(s_x[class_idx]))
            class_labels.append(np.full(len(class_idx), idx))

        X = np.vstack(class_data)
        X = X[:, :, t0_idx:tmax_idx]
        y = np.hstack(class_labels)

        X = self._apply_preprocessing(X)

        n_trials, n_ch, n_times = X.shape
        freq_classes = [c for c in unique_classes if c.startswith("freq_")]
        if freq_classes:
            freqs = sorted([float(c.replace("freq_", "")) for c in freq_classes])
        else:
            freqs = []
        info = {
            "n_trials": n_trials,
            "n_ch": n_ch,
            "n_times": n_times,
            "n_classes": len(np.unique(y)),
            "fs": fs,
            "class_names": unique_classes.tolist(),
            "freqs": freqs,
        }

        return (X, y, info)

    def _load_Nakanishi2015(self) -> Tuple:
        """Load Nakanishi2015 dataset."""
        paradigm = self._get_paradigm(256)

        s_x, s_y, metadata = paradigm.get_data(self.dataset, subjects=[self.subject])
        s_x = np.array(s_x)

        fs = self._get_fs(self.subject)
        t0_idx = int(fs * self.t0)
        tmax_idx = int(fs * self.tmax) if self.tmax is not None else s_x.shape[2]

        unique_classes = np.unique(s_y)
        class_data = []
        class_labels = []

        for idx, class_name in enumerate(unique_classes):
            class_idx = np.where(s_y == class_name)[0]
            class_data.append(np.array(s_x[class_idx]))
            class_labels.append(np.full(len(class_idx), idx))

        X = np.vstack(class_data)
        X = X[:, :, t0_idx:tmax_idx]
        y = np.hstack(class_labels)

        X = self._apply_preprocessing(X)

        n_trials, n_ch, n_times = X.shape
        freq_classes = [c for c in unique_classes if c.startswith("freq_")]
        if freq_classes:
            freqs = sorted([float(c.replace("freq_", "")) for c in freq_classes])
        else:
            freqs = []
        info = {
            "n_trials": n_trials,
            "n_ch": n_ch,
            "n_times": n_times,
            "n_classes": len(np.unique(y)),
            "fs": fs,
            "class_names": unique_classes.tolist(),
            "freqs": freqs,
        }

        return (X, y, info)

    def _load_Wang2016(self) -> Tuple:
        """Load Wang2016 dataset."""
        paradigm = self._get_paradigm(256)

        s_x, s_y, metadata = paradigm.get_data(self.dataset, subjects=[self.subject])
        s_x = np.array(s_x)

        fs = self._get_fs(self.subject)
        t0_idx = int(fs * self.t0)
        tmax_idx = int(fs * self.tmax) if self.tmax is not None else s_x.shape[2]

        unique_classes = np.unique(s_y)
        class_data = []
        class_labels = []

        for idx, class_name in enumerate(unique_classes):
            class_idx = np.where(s_y == class_name)[0]
            class_data.append(np.array(s_x[class_idx]))
            class_labels.append(np.full(len(class_idx), idx))

        X = np.vstack(class_data)
        X = X[:, :, t0_idx:tmax_idx]
        y = np.hstack(class_labels)

        X = self._apply_preprocessing(X)

        n_trials, n_ch, n_times = X.shape
        freq_classes = [c for c in unique_classes if c.startswith("freq_")]
        if freq_classes:
            freqs = sorted([float(c.replace("freq_", "")) for c in freq_classes])
        else:
            freqs = []
        info = {
            "n_trials": n_trials,
            "n_ch": n_ch,
            "n_times": n_times,
            "n_classes": len(np.unique(y)),
            "fs": fs,
            "class_names": unique_classes.tolist(),
            "freqs": freqs,
        }

        return (X, y, info)

    def _load_Wang2021Combined(self) -> Tuple:
        """Load Wang2021Combined dataset."""
        paradigm = self._get_paradigm(256)

        s_x, s_y, metadata = paradigm.get_data(self.dataset, subjects=[self.subject])
        s_x = np.array(s_x)

        fs = self._get_fs(self.subject)
        t0_idx = int(fs * self.t0)
        tmax_idx = int(fs * self.tmax) if self.tmax is not None else s_x.shape[2]

        unique_classes = np.unique(s_y)
        class_data = []
        class_labels = []

        for idx, class_name in enumerate(unique_classes):
            class_idx = np.where(s_y == class_name)[0]
            class_data.append(np.array(s_x[class_idx]))
            class_labels.append(np.full(len(class_idx), idx))

        X = np.vstack(class_data)
        X = X[:, :, t0_idx:tmax_idx]
        y = np.hstack(class_labels)

        X = self._apply_preprocessing(X)

        n_trials, n_ch, n_times = X.shape
        freq_classes = [c for c in unique_classes if c.startswith("freq_")]
        if freq_classes:
            freqs = sorted([float(c.replace("freq_", "")) for c in freq_classes])
        else:
            freqs = []
        info = {
            "n_trials": n_trials,
            "n_ch": n_ch,
            "n_times": n_times,
            "n_classes": len(np.unique(y)),
            "fs": fs,
            "class_names": unique_classes.tolist(),
            "freqs": freqs,
        }

        return (X, y, info)

    def get_data(self):
        """Return loaded data.

        Returns:
            Tuple: (X, y, info) where X is the data, y are labels, info is metadata dict
        """
        return self.X, self.y, self.info

    @staticmethod
    def get_subjects(dataset_name):
        """Get list of available subjects for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            List of subject IDs
        """
        if dataset_name not in SSVEP_DataLoader.DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Available: {list(SSVEP_DataLoader.DATASETS.keys())}"
            )

        dataset = SSVEP_DataLoader.DATASETS[dataset_name]()
        return dataset.subject_list

    @staticmethod
    def get_available_datasets():
        """Get list of available dataset names."""
        return list(SSVEP_DataLoader.DATASETS.keys())


class Sleep_Loader:
    """Sleep Data Loader class for loading sleep stage classification datasets from braindecode."""

    DATASETS = {
        "SleepPhysionet": SleepPhysionet,
    }

    def __init__(
        self,
        dataset_name,
        subject,
        preprocessing_pipeline=None,
        window_size_s=30,
        window_stride_s=30,
        crop_wake_mins=30,
        crop=None,
    ):
        """Initialize Sleep_Loader.

        Args:
            dataset_name: Name of the dataset ('SleepPhysionet')
            subject: Subject ID (integer)
            preprocessing_pipeline: Optional list of preprocessing functions
            window_size_s: Window size in seconds (default: 30)
            window_stride_s: Window stride in seconds (default: 30)
            crop_wake_mins: Minutes of wake time to keep at start/end (default: 30)
            crop: Tuple (start, end) to crop raw files, e.g. (0, 3600*3)
        """
        self.dataset_name = dataset_name
        self.subject = subject
        self.preprocessing_pipeline = preprocessing_pipeline
        self.window_size_s = window_size_s
        self.window_stride_s = window_stride_s
        self.crop_wake_mins = crop_wake_mins
        self.crop = crop

        if dataset_name not in self.DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Available: {list(self.DATASETS.keys())}"
            )

        self.dataset = self.DATASETS[dataset_name](
            subject_ids=[subject],
            crop_wake_mins=crop_wake_mins,
            crop=crop,
        )
        self._load_data()

    def _load_data(self):
        """Load data based on dataset name."""
        load_methods = {
            "SleepPhysionet": self._load_SleepPhysionet,
        }

        self.X, self.y, self.info = load_methods[self.dataset_name]()

    def _apply_preprocessing(self, X):
        """Apply preprocessing pipeline to data."""
        if self.preprocessing_pipeline is not None:
            if not isinstance(self.preprocessing_pipeline, list):
                raise ValueError(
                    "preprocessing_pipeline argument should be a list of preprocessing functions"
                )
            for fn in self.preprocessing_pipeline:
                X = fn(X)
        return X

    def _load_SleepPhysionet(self) -> Tuple:
        """Load SleepPhysionet dataset."""
        sfreq = self.dataset.datasets[0].raw.info["sfreq"]

        windows_dataset = create_windows_from_events(
            self.dataset,
            window_size_s=self.window_size_s,
            window_stride_s=self.window_stride_s,
            preload=True,
        )

        window_idx = windows_dataset.description["original_index"].values
        X = np.array([windows_dataset[i][0] for i in range(len(windows_dataset))])
        y = np.array([windows_dataset[i][1] for i in range(len(windows_dataset))])

        X = self._apply_preprocessing(X)

        n_trials, n_ch, n_times = X.shape
        info = {
            "n_trials": n_trials,
            "n_ch": n_ch,
            "n_times": n_times,
            "n_classes": len(np.unique(y)),
            "fs": sfreq,
        }

        return (X, y, info)

    def get_data(self):
        """Return loaded data.

        Returns:
            Tuple: (X, y, info) where X is the data, y are labels, info is metadata dict
        """
        return self.X, self.y, self.info

    @staticmethod
    def get_subjects(dataset_name):
        """Get list of available subjects for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            List of subject IDs
        """
        if dataset_name not in Sleep_Loader.DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Available: {list(Sleep_Loader.DATASETS.keys())}"
            )

        dataset = Sleep_Loader.DATASETS[dataset_name]()
        return dataset.subject_ids

    @staticmethod
    def get_available_datasets():
        """Get list of available dataset names."""
        return list(Sleep_Loader.DATASETS.keys())


class RestingState_DataLoader:
    """Resting State Data Loader class for loading resting state EEG datasets from moabb."""

    DATASETS = {
        "Cattan2019_PHMD": Cattan2019_PHMD,
        "Hinss2021": Hinss2021,
        "Rodrigues2017": Rodrigues2017,
        "ButtonToneSZ": ButtonToneSZ,
    }

    DEFAULT_EVENTS = {
        "Cattan2019_PHMD": {"off": 1, "on": 2},
        "Hinss2021": {"easy": 2, "diff": 3},
        "Rodrigues2017": {"closed": 1, "open": 2},
        "ButtonToneSZ": {"sz": 1, "hc": 2},
    }

    def __init__(
        self,
        dataset_name,
        subject,
        preprocessing_pipeline=None,
        tmin=10,
        tmax=50,
        fmin=1,
        fmax=35,
        resample=128,
    ):
        """Initialize RestingState_DataLoader.

        Args:
            dataset_name: Name of the dataset ('Cattan2019_PHMD', 'Hinss2021', 'Rodrigues2017', 'ButtonToneSZ')
            subject: Subject ID (integer)
            preprocessing_pipeline: Optional list of preprocessing functions
            tmin: Start time for epoching (seconds, default: 10)
            tmax: End time for epoching (seconds, default: 50)
            fmin: Low cutoff frequency for bandpass filter (Hz, default: 1)
            fmax: High cutoff frequency for bandpass filter (Hz, default: 35)
            resample: Resampling rate (Hz, default: 128)
        """
        self.dataset_name = dataset_name
        self.subject = subject
        self.preprocessing_pipeline = preprocessing_pipeline
        self.tmin = tmin
        self.tmax = tmax
        self.fmin = fmin
        self.fmax = fmax
        self.resample = resample

        if dataset_name not in self.DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Available: {list(self.DATASETS.keys())}"
            )

        self.dataset = self.DATASETS[dataset_name]()
        self._load_data()

    def _load_data(self):
        """Load data based on dataset name."""
        load_methods = {
            "Cattan2019_PHMD": self._load_resting_state,
            "Hinss2021": self._load_resting_state,
            "Rodrigues2017": self._load_resting_state,
            "ButtonToneSZ": self._load_button_tone_sz,
        }

        self.X, self.y, self.info = load_methods[self.dataset_name]()

    def _get_paradigm(self):
        """Get RestingState paradigm."""
        events = self.DEFAULT_EVENTS.get(self.dataset_name)
        return RestingStateToP300Adapter(
            fmin=self.fmin,
            fmax=self.fmax,
            tmin=self.tmin,
            tmax=self.tmax,
            resample=self.resample,
            events=events,
        )

    def _apply_preprocessing(self, X):
        """Apply preprocessing pipeline to data."""
        if self.preprocessing_pipeline is not None:
            if not isinstance(self.preprocessing_pipeline, list):
                raise ValueError(
                    "preprocessing_pipeline argument should be a list of preprocessing functions"
                )
            for fn in self.preprocessing_pipeline:
                X = fn(X)
        return X

    def _get_fs(self, subject):
        """Get sampling frequency dynamically."""
        raw = self.dataset.get_data(subjects=[subject])
        first_session_key = list(raw[subject].keys())[0]
        first_run_key = list(raw[subject][first_session_key].keys())[0]
        return raw[subject][first_session_key][first_run_key].info["sfreq"]

    def _load_resting_state(self) -> Tuple:
        """Load resting state dataset."""
        paradigm = self._get_paradigm()

        s_x, s_y, metadata = paradigm.get_data(self.dataset, subjects=[self.subject])
        s_x = np.array(s_x)

        fs = self._get_fs(self.subject)

        unique_classes = np.unique(s_y)
        class_data = []
        class_labels = []

        for idx, class_name in enumerate(unique_classes):
            class_idx = np.where(s_y == class_name)[0]
            class_data.append(np.array(s_x[class_idx]))
            class_labels.append(np.full(len(class_idx), idx))

        X = np.vstack(class_data)
        y = np.hstack(class_labels)

        X = self._apply_preprocessing(X)

        n_trials, n_ch, n_times = X.shape
        info = {
            "n_trials": n_trials,
            "n_ch": n_ch,
            "n_times": n_times,
            "n_classes": len(np.unique(y)),
            "fs": fs,
            "class_names": unique_classes.tolist(),
        }

        return (X, y, info)

    def _load_button_tone_sz(self) -> Tuple:
        """Load ButtonToneSZ (Schizophrenia) dataset."""
        dataset = ButtonToneSZ()
        data = dataset.get_data(subjects=[self.subject])

        if not data or self.subject not in data:
            raise ValueError(f"No data found for subject {self.subject}")

        raw = data[self.subject]["session_0"]["run_0"]

        sfreq = raw.info["sfreq"]

        if self.resample and self.resample != sfreq:
            raw.resample(self.resample)
            sfreq = self.resample

        epochs = mne.make_fixed_length_epochs(
            raw, duration=self.tmax - self.tmin, preload=True, verbose=False
        )

        X = epochs.get_data()

        label_str = dataset.get_subject_label(self.subject)
        y = np.full(X.shape[0], 1 if label_str == "sz" else 2)

        X = self._apply_preprocessing(X)

        n_trials, n_ch, n_times = X.shape
        info = {
            "n_trials": n_trials,
            "n_ch": n_ch,
            "n_times": n_times,
            "n_classes": 2,
            "fs": sfreq,
            "class_names": ["sz", "hc"],
            "subject_label": label_str,
        }

        return (X, y, info)

    def get_data(self):
        """Return loaded data.

        Returns:
            Tuple: (X, y, info) where X is the data, y are labels, info is metadata dict
        """
        return self.X, self.y, self.info

    @staticmethod
    def get_subjects(dataset_name):
        """Get list of available subjects for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            List of subject IDs
        """
        if dataset_name not in RestingState_DataLoader.DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Available: {list(RestingState_DataLoader.DATASETS.keys())}"
            )

        if dataset_name == "ButtonToneSZ":
            return ButtonToneSZ.get_subjects()

        dataset = RestingState_DataLoader.DATASETS[dataset_name]()
        return dataset.subject_list

    @staticmethod
    def get_available_datasets():
        """Get list of available dataset names."""
        return list(RestingState_DataLoader.DATASETS.keys())


# ============================================================================
# Legacy functions for backward compatibility
# ============================================================================


def load_Liu2024(subject, preprocessing_pipeline=None, t0=0.5, t1=3.5) -> Tuple:
    """Load Liu2024 dataset (legacy function)."""
    loader = MI_DataLoader("Liu2024", subject, preprocessing_pipeline, t0, t1)
    return loader.get_data()


def get_subjects_Liu2024():
    """Get Liu2024 subjects (legacy function)."""
    return MI_DataLoader.get_subjects("Liu2024")


def load_BNCI2014_001(subject, preprocessing_pipeline=None, t0=0.5, t1=3.5) -> Tuple:
    """Load BNCI2014_001 dataset (legacy function)."""
    loader = MI_DataLoader("BNCI2014_001", subject, preprocessing_pipeline, t0, t1)
    return loader.get_data()


def get_subjects_BNCI2014_001():
    """Get BNCI2014_001 subjects (legacy function)."""
    return MI_DataLoader.get_subjects("BNCI2014_001")


def load_BNCI2014_004(subject, preprocessing_pipeline=None, t0=0.5, t1=3.5) -> Tuple:
    """Load BNCI2014_004 dataset (legacy function)."""
    loader = MI_DataLoader("BNCI2014_004", subject, preprocessing_pipeline, t0, t1)
    return loader.get_data()


def get_subjects_BNCI2014_004():
    """Get BNCI2014_004 subjects (legacy function)."""
    return MI_DataLoader.get_subjects("BNCI2014_004")


def load_BNCI2015_001(subject, preprocessing_pipeline=None, t0=0.5, t1=3.5) -> Tuple:
    """Load BNCI2015_001 dataset (legacy function)."""
    loader = MI_DataLoader("BNCI2015_001", subject, preprocessing_pipeline, t0, t1)
    return loader.get_data()


def get_subjects_BNCI2015_001():
    """Get BNCI2015_001 subjects (legacy function)."""
    return MI_DataLoader.get_subjects("BNCI2015_001")


def load_BNCI2014_002(subject, preprocessing_pipeline=None, t0=0.5, t1=3.5) -> Tuple:
    """Load BNCI2014_002 dataset (legacy function)."""
    loader = MI_DataLoader("BNCI2014_002", subject, preprocessing_pipeline, t0, t1)
    return loader.get_data()


def get_subjects_BNCI2014_002():
    """Get BNCI2014_002 subjects (legacy function)."""
    return MI_DataLoader.get_subjects("BNCI2014_002")


def load_BNCI2015_004(subject, preprocessing_pipeline=None, t0=0.5, t1=3.5) -> Tuple:
    """Load BNCI2015_004 dataset (legacy function)."""
    loader = MI_DataLoader("BNCI2015_004", subject, preprocessing_pipeline, t0, t1)
    return loader.get_data()


def get_subjects_BNCI2015_004():
    """Get BNCI2015_004 subjects (legacy function)."""
    return MI_DataLoader.get_subjects("BNCI2015_004")


def load_AlexMI(subject, preprocessing_pipeline=None, t0=0.5, t1=3.5) -> Tuple:
    """Load AlexMI dataset (legacy function)."""
    loader = MI_DataLoader("AlexMI", subject, preprocessing_pipeline, t0, t1)
    return loader.get_data()


def get_subjects_AlexMI():
    """Get AlexMI subjects (legacy function)."""
    return MI_DataLoader.get_subjects("AlexMI")
