import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.signal import butter, sosfiltfilt

from mne.decoding import Scaler


def compute_band_powers(x, n_filter_banks, fs):
    """
    Compute log band powers for n_filter_banks equally spaced bands.
    
    Args:
        x:              (B, C, T) numpy array of raw EEG signal
        n_filter_banks: number of frequency bands F
        fs:             sampling frequency in Hz
    
    Returns:
        (B, F) log-normalised mean band power averaged over channels
    """
    B, C, T = x.shape

    fft    = np.fft.rfft(x, axis=-1)       # (B, C, T//2+1)
    power  = np.abs(fft) ** 2              # (B, C, T//2+1)
    power  = power.mean(axis=1)            # (B, T//2+1) — avg over channels

    freqs     = np.fft.rfftfreq(T, d=1.0/fs)   # (T//2+1,) in Hz
    nyquist   = fs / 2.0
    band_width = nyquist / n_filter_banks

    band_powers = []
    for i in range(n_filter_banks):
        low  = i * band_width
        high = (i + 1) * band_width
        mask = (freqs >= low) & (freqs < high)

        if mask.sum() == 0:
            # frequency resolution coarser than band width — take nearest bin
            mid_freq = (low + high) / 2.0
            nearest  = np.argmin(np.abs(freqs - mid_freq))
            mask     = np.zeros(len(freqs), dtype=bool)
            mask[nearest] = True

        band_powers.append(power[:, mask].mean(axis=-1))  # (B,)

    band_powers = np.stack(band_powers, axis=1)  # (B, F)
    return np.log1p(band_powers)                 # (B, F) log-normalised

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