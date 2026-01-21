import numpy as np
import torch
from torch.utils.data import Dataset

from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.signals = torch.tensor(data, dtype=torch.float32)
        self.labels = labels
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        # Return a single sample
        return self.signals[idx], self.labels[idx]

def load_BNCI2014_001(subject, preprocessing_pipeline=None, t0=0.5, t1=3.5):
    dataset = BNCI2014_001()
    paradigm = MotorImagery(channels=None, resample=250)
    if isinstance(subject, int):
        s_x, s_y, metadata = paradigm.get_data(dataset, subjects=[subject])
    else:
        raise ValueError('subject argument should be an interger within valid range on MOABB site')

    raw = dataset.get_data(subjects=[1])
    fs = raw[1]['0train']['0'].info['sfreq']
    t0 = int(fs*t0)
    t1 = int(fs*t1)

    left_imagery_idx = np.where(s_y=='left_hand')[0]
    right_imagery_idx = np.where(s_y=='right_hand')[0]
    left_imagery = np.array(s_x[left_imagery_idx])
    right_imagery = np.array( s_x[right_imagery_idx])
    s_x, s_y = 0, 0
    X = np.vstack((left_imagery, right_imagery))
    X = X[:, :, t0:t1]
    y = np.hstack((np.zeros(left_imagery.shape[0]), np.ones(right_imagery.shape[0])))    

    if preprocessing_pipeline is not None:
        if isinstance(preprocessing_pipeline, list) == 0:
            raise ValueError("preprocessing_pipeline argument should be a list of preprocseeing functions")
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
    
    return X, y, info

def get_subjects_BNCI2014_001():
    dataset = BNCI2014_001()
    return dataset.subject_list