import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.signal import butter, sosfiltfilt

from mne.decoding import Scaler

def EA(x):
    """
    Parameters
    ----------
    x : numpy array
        data of shape (num_samples, num_channels, num_time_samples)

    Returns
    ----------
    XEA : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    """
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA, sqrtRefEA

def EA_online(x, sqrtRefEA):
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA

def bandpass_filtering(X, low=8.0, high=30.0, fs=250):
    sos = butter(4, [low, high], btype='bandpass', fs=fs, output='sos')

    # Apply filter to each trial and channel
    X_filtered = np.zeros_like(X)
    for i in range(X.shape[0]):  # trials
        for j in range(X.shape[1]):  # channels
            X_filtered[i, j, :] = sosfiltfilt(sos, X[i, j, :])

    return X_filtered