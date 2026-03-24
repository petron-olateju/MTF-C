import numpy as np
import torch
from torch.utils.data import Dataset

from typing import Tuple

from moabb.datasets import BNCI2014_001, BNCI2014_002, BNCI2014_004, BNCI2015_001, BNCI2015_004, Liu2024, AlexMI
from moabb.paradigms import MotorImagery

def chronological_stratified_kfold(y, n_splits=5):
    """Split each class chronologically into n_splits, then merge."""
    indices = np.arange(len(y))
    fold_indices = [[] for _ in range(n_splits)]
    for class_label in np.unique(y):
        class_idx = indices[y == class_label]  # already in chronological order
        splits = np.array_split(class_idx, n_splits)
        for fold, split in enumerate(splits):
            fold_indices[fold].extend(split)
    
    for test_fold in range(n_splits):
        test_idx = np.array(fold_indices[test_fold])
        train_idx = np.concatenate([
            np.array(fold_indices[i]) 
            for i in range(n_splits) if i != test_fold
        ])
        yield train_idx, test_idx

class EEGDataset(Dataset):
    def __init__(self, data, labels, stft):
        self.signals = torch.tensor(data, dtype=torch.float32)
        self.labels = labels
        self.stft = stft
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        # Return a single sample
        return self.signals[idx], self.labels[idx], self.stft[idx]

# ============================================================================
# Liu2024
# ============================================================================
def load_Liu2024(subject, preprocessing_pipeline=None, t0=0.5, t1=3.5) -> Tuple:
    """
    Liu2024 Dataset Loader
    - Recent motor imagery dataset
    - Classes and specifications TBD (check MOABB documentation)
    - Using first session only
    """
    dataset = Liu2024()
    paradigm = MotorImagery(channels=None, resample=250)  # Adjust resample if needed
    
    if isinstance(subject, int):
        s_x, s_y, metadata = paradigm.get_data(dataset, subjects=[subject])
        # Use only first session
        session_keys = metadata['session'].unique()
        first_session = sorted(session_keys)[0]
        session_mask = metadata['session'] == first_session
        s_x = s_x[session_mask]
        s_y = s_y[session_mask]
    else:
        raise ValueError('subject argument should be an integer within valid range on MOABB site')

    # Get sampling frequency dynamically
    raw = dataset.get_data(subjects=[subject])
    first_session_key = list(raw[subject].keys())[0]
    first_run_key = list(raw[subject][first_session_key].keys())[0]
    fs = raw[subject][first_session_key][first_run_key].info['sfreq']
    
    t0 = int(fs * t0)
    t1 = int(fs * t1)

    # Get unique classes and create mappings
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
    s_x, s_y = 0, 0

    if preprocessing_pipeline is not None:
        if not isinstance(preprocessing_pipeline, list):
            raise ValueError("preprocessing_pipeline argument should be a list of preprocessing functions")
        for fn in preprocessing_pipeline:
            X = fn(X)

    n_trials, n_ch, n_times = X.shape
    info = {
        "n_trials": n_trials,
        "n_ch": n_ch,
        "n_times": n_times,
        "n_classes": len(np.unique(y)),
        "fs": fs,
        "class_names": unique_classes.tolist()
    }
    
    return (X, y, info)

def get_subjects_Liu2024():
    dataset = Liu2024()
    return dataset.subject_list

# ============================================================================
# BNCI2014_001 (Already provided, included for completeness)
# ============================================================================
def load_BNCI2014_001(subject, preprocessing_pipeline=None, t0=0.5, t1=3.5) -> Tuple:
    """
    BNCI2014_001 Dataset Loader
    - 9 subjects
    - 2 classes: left_hand, right_hand
    - 22 EEG channels
    - 250 Hz sampling rate
    - 2 sessions (using first session only)
    """
    dataset = BNCI2014_001()
    paradigm = MotorImagery(channels=None, resample=250)
    
    if isinstance(subject, int):
        s_x, s_y, metadata = paradigm.get_data(dataset, subjects=[subject])
        session_mask = metadata['session'].str.contains('0')  # Session T only
        s_x = s_x[session_mask]
        s_y = s_y[session_mask]
    else:
        raise ValueError('subject argument should be an integer within valid range on MOABB site')

    raw = dataset.get_data(subjects=[1])
    fs = raw[1]['0train']['0'].info['sfreq']
    t0 = int(fs * t0)
    t1 = int(fs * t1)

    left_imagery_idx = np.where(s_y == 'left_hand')[0]
    right_imagery_idx = np.where(s_y == 'right_hand')[0]
    left_imagery = np.array(s_x[left_imagery_idx])
    right_imagery = np.array(s_x[right_imagery_idx])
    s_x, s_y = 0, 0

    X = np.vstack((left_imagery, right_imagery))
    X = X[:, :, t0:t1]
    y = np.hstack((np.zeros(left_imagery.shape[0]), np.ones(right_imagery.shape[0])))

    if preprocessing_pipeline is not None:
        if isinstance(preprocessing_pipeline, list) == 0:
            raise ValueError("preprocessing_pipeline argument should be a list of preprocessing functions")
        else:
            for fn in preprocessing_pipeline:
                X = fn(X)

    n_trials, n_ch, n_times = X.shape
    info = {
        "n_trials": n_trials,
        "n_ch": n_ch,
        "n_times": n_times,
        "n_classes": len(np.unique(y)),
        "fs": fs
    }
    
    return (X, y, info)

def get_subjects_BNCI2014_001():
    dataset = BNCI2014_001()
    return dataset.subject_list


# ============================================================================
# BNCI2014_004
# ============================================================================
def load_BNCI2014_004(subject, preprocessing_pipeline=None, t0=0.5, t1=3.5) -> Tuple:
    """
    BNCI2014_004 Dataset Loader
    - 9 subjects
    - 2 classes: left_hand, right_hand
    - 3 EEG channels (C3, Cz, C4)
    - 250 Hz sampling rate
    - 5 sessions (using first session only)
    """
    dataset = BNCI2014_004()
    paradigm = MotorImagery(channels=None, resample=250)
    
    if isinstance(subject, int):
        s_x, s_y, metadata = paradigm.get_data(dataset, subjects=[subject])
        # Use only first session
        session_keys = metadata['session'].unique()
        first_session = sorted(session_keys)[0]
        session_mask = metadata['session'] == first_session
        s_x = s_x[session_mask]
        s_y = s_y[session_mask]
    else:
        raise ValueError('subject argument should be an integer within valid range on MOABB site')

    # Get sampling frequency dynamically
    raw = dataset.get_data(subjects=[subject])
    first_session_key = list(raw[subject].keys())[0]
    first_run_key = list(raw[subject][first_session_key].keys())[0]
    fs = raw[subject][first_session_key][first_run_key].info['sfreq']
    
    t0 = int(fs * t0)
    t1 = int(fs * t1)

    left_imagery_idx = np.where(s_y == 'left_hand')[0]
    right_imagery_idx = np.where(s_y == 'right_hand')[0]
    left_imagery = np.array(s_x[left_imagery_idx])
    right_imagery = np.array(s_x[right_imagery_idx])
    s_x, s_y = 0, 0

    X = np.vstack((left_imagery, right_imagery))
    X = X[:, :, t0:t1]
    y = np.hstack((np.zeros(left_imagery.shape[0]), np.ones(right_imagery.shape[0])))

    if preprocessing_pipeline is not None:
        if isinstance(preprocessing_pipeline, list) == 0:
            raise ValueError("preprocessing_pipeline argument should be a list of preprocessing functions")
        else:
            for fn in preprocessing_pipeline:
                X = fn(X)

    n_trials, n_ch, n_times = X.shape
    info = {
        "n_trials": n_trials,
        "n_ch": n_ch,
        "n_times": n_times,
        "n_classes": len(np.unique(y)),
        "fs": fs
    }
    
    return (X, y, info)

def get_subjects_BNCI2014_004():
    dataset = BNCI2014_004()
    return dataset.subject_list


# ============================================================================
# BNCI2015_001
# ============================================================================
def load_BNCI2015_001(subject, preprocessing_pipeline=None, t0=0.5, t1=3.5) -> Tuple:
    """
    BNCI2015_001 Dataset Loader
    - 12 subjects
    - 2 classes: right_hand, feet
    - 13 EEG channels (Laplacian derivations around C3, Cz, C4)
    - 512 Hz sampling rate
    - 2-3 sessions per subject (using first session only)
    """
    dataset = BNCI2015_001()
    paradigm = MotorImagery(channels=None, resample=512)
    
    if isinstance(subject, int):
        s_x, s_y, metadata = paradigm.get_data(dataset, subjects=[subject])
        # Use only first session
        session_keys = metadata['session'].unique()
        first_session = sorted(session_keys)[0]
        session_mask = metadata['session'] == first_session
        s_x = s_x[session_mask]
        s_y = s_y[session_mask]
    else:
        raise ValueError('subject argument should be an integer within valid range on MOABB site')

    # Get sampling frequency dynamically
    raw = dataset.get_data(subjects=[subject])
    first_session_key = list(raw[subject].keys())[0]
    first_run_key = list(raw[subject][first_session_key].keys())[0]
    fs = raw[subject][first_session_key][first_run_key].info['sfreq']
    
    t0 = int(fs * t0)
    t1 = int(fs * t1)

    # BNCI2015_001 uses right_hand and feet
    right_imagery_idx = np.where(s_y == 'right_hand')[0]
    feet_imagery_idx = np.where(s_y == 'feet')[0]
    
    right_imagery = np.array(s_x[right_imagery_idx])
    feet_imagery = np.array(s_x[feet_imagery_idx])
    s_x, s_y = 0, 0

    X = np.vstack((right_imagery, feet_imagery))
    X = X[:, :, t0:t1]
    y = np.hstack((
        np.zeros(right_imagery.shape[0]),  # 0: right_hand
        np.ones(feet_imagery.shape[0])     # 1: feet
    ))

    if preprocessing_pipeline is not None:
        if isinstance(preprocessing_pipeline, list) == 0:
            raise ValueError("preprocessing_pipeline argument should be a list of preprocessing functions")
        else:
            for fn in preprocessing_pipeline:
                X = fn(X)

    n_trials, n_ch, n_times = X.shape
    info = {
        "n_trials": n_trials,
        "n_ch": n_ch,
        "n_times": n_times,
        "n_classes": len(np.unique(y)),
        "fs": fs
    }
    
    return (X, y, info)

def get_subjects_BNCI2015_001():
    dataset = BNCI2015_001()
    return dataset.subject_list


# ============================================================================
# BNCI2014_002
# ============================================================================
def load_BNCI2014_002(subject, preprocessing_pipeline=None, t0=0.5, t1=3.5) -> Tuple:
    """
    BNCI2014_002 Dataset Loader
    - 14 subjects
    - 2 classes: right_hand, feet
    - 15 EEG channels
    - 512 Hz sampling rate
    - 2 sessions (using first session only)
    """
    dataset = BNCI2014_002()
    paradigm = MotorImagery(channels=None, resample=512)
    
    if isinstance(subject, int):
        s_x, s_y, metadata = paradigm.get_data(dataset, subjects=[subject])
        # Use only first session
        session_keys = metadata['session'].unique()
        first_session = sorted(session_keys)[0]
        session_mask = metadata['session'] == first_session
        s_x = s_x[session_mask]
        s_y = s_y[session_mask]
    else:
        raise ValueError('subject argument should be an integer within valid range on MOABB site')

    # Get sampling frequency from the paradigm's resample rate or first available session
    raw = dataset.get_data(subjects=[subject])
    # Get first session and run dynamically
    first_session_key = list(raw[subject].keys())[0]
    first_run_key = list(raw[subject][first_session_key].keys())[0]
    fs = raw[subject][first_session_key][first_run_key].info['sfreq']
    
    t0 = int(fs * t0)
    t1 = int(fs * t1)

    # BNCI2014_002 uses right_hand and feet
    right_imagery_idx = np.where(s_y == 'right_hand')[0]
    feet_imagery_idx = np.where(s_y == 'feet')[0]
    
    right_imagery = np.array(s_x[right_imagery_idx])
    feet_imagery = np.array(s_x[feet_imagery_idx])
    s_x, s_y = 0, 0

    X = np.vstack((right_imagery, feet_imagery))
    X = X[:, :, t0:t1]
    y = np.hstack((
        np.zeros(right_imagery.shape[0]),  # 0: right_hand
        np.ones(feet_imagery.shape[0])     # 1: feet
    ))

    if preprocessing_pipeline is not None:
        if isinstance(preprocessing_pipeline, list) == 0:
            raise ValueError("preprocessing_pipeline argument should be a list of preprocessing functions")
        else:
            for fn in preprocessing_pipeline:
                X = fn(X)

    n_trials, n_ch, n_times = X.shape
    info = {
        "n_trials": n_trials,
        "n_ch": n_ch,
        "n_times": n_times,
        "n_classes": len(np.unique(y)),
        "fs": fs
    }
    
    return (X, y, info)

def get_subjects_BNCI2014_002():
    dataset = BNCI2014_002()
    return dataset.subject_list


# ============================================================================
# BNCI2015_004
# ============================================================================
def load_BNCI2015_004(subject, preprocessing_pipeline=None, t0=0.5, t1=3.5) -> Tuple:
    """
    BNCI2015_004 Dataset Loader
    - 9 subjects (users with disability - spinal cord injury and stroke)
    - 5 classes: WORD (mental word association), SUB (mental subtraction), 
                  NAV (spatial navigation), HAND (right hand MI), FEET (feet MI)
    - 30 EEG channels
    - 256 Hz sampling rate
    - 2 sessions (using first session only)
    """
    dataset = BNCI2015_004()
    paradigm = MotorImagery(channels=None, resample=256)
    
    if isinstance(subject, int):
        s_x, s_y, metadata = paradigm.get_data(dataset, subjects=[subject])
        # Use only first session
        session_keys = metadata['session'].unique()
        first_session = sorted(session_keys)[0]
        session_mask = metadata['session'] == first_session
        s_x = s_x[session_mask]
        s_y = s_y[session_mask]
    else:
        raise ValueError('subject argument should be an integer within valid range on MOABB site')

    # Get sampling frequency dynamically
    raw = dataset.get_data(subjects=[subject])
    first_session_key = list(raw[subject].keys())[0]
    first_run_key = list(raw[subject][first_session_key].keys())[0]
    fs = raw[subject][first_session_key][first_run_key].info['sfreq']
    
    t0 = int(fs * t0)
    t1 = int(fs * t1)

    # Get unique classes and create mappings
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
    s_x, s_y = 0, 0

    if preprocessing_pipeline is not None:
        if isinstance(preprocessing_pipeline, list) == 0:
            raise ValueError("preprocessing_pipeline argument should be a list of preprocessing functions")
        else:
            for fn in preprocessing_pipeline:
                X = fn(X)

    n_trials, n_ch, n_times = X.shape
    info = {
        "n_trials": n_trials,
        "n_ch": n_ch,
        "n_times": n_times,
        "n_classes": len(np.unique(y)),
        "fs": fs,
        "class_names": unique_classes.tolist()
    }
    
    return (X, y, info)

def get_subjects_BNCI2015_004():
    dataset = BNCI2015_004()
    return dataset.subject_list


# ============================================================================
# AlexMI
# ============================================================================
def load_AlexMI(subject, preprocessing_pipeline=None, t0=0.5, t1=3.5) -> Tuple:
    """
    AlexMI Dataset Loader (Alexandre Motor Imagery)
    - 8 subjects
    - 3 classes: right_hand, feet, rest
    - 16 EEG channels
    - 512 Hz sampling rate
    - Multiple sessions (using first session only)
    """
    dataset = AlexMI()
    paradigm = MotorImagery(channels=None, resample=512)
    
    if isinstance(subject, int):
        s_x, s_y, metadata = paradigm.get_data(dataset, subjects=[subject])
        # Use only first session
        session_keys = metadata['session'].unique()
        first_session = sorted(session_keys)[0]
        session_mask = metadata['session'] == first_session
        s_x = s_x[session_mask]
        s_y = s_y[session_mask]
    else:
        raise ValueError('subject argument should be an integer within valid range on MOABB site')

    # Get sampling frequency dynamically
    raw = dataset.get_data(subjects=[subject])
    first_session_key = list(raw[subject].keys())[0]
    first_run_key = list(raw[subject][first_session_key].keys())[0]
    fs = raw[subject][first_session_key][first_run_key].info['sfreq']
    
    t0 = int(fs * t0)
    t1 = int(fs * t1)

    # Get unique classes and create mappings
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
    s_x, s_y = 0, 0

    if preprocessing_pipeline is not None:
        if not isinstance(preprocessing_pipeline, list):
            raise ValueError("preprocessing_pipeline argument should be a list of preprocessing functions")
        for fn in preprocessing_pipeline:
            X = fn(X)

    n_trials, n_ch, n_times = X.shape
    info = {
        "n_trials": n_trials,
        "n_ch": n_ch,
        "n_times": n_times,
        "n_classes": len(np.unique(y)),
        "fs": fs,
        "class_names": unique_classes.tolist()
    }
    
    return (X, y, info)

def get_subjects_AlexMI():
    dataset = AlexMI()
    return dataset.subject_list