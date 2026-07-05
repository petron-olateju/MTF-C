import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.signal import butter, sosfiltfilt

import mne
from mne.decoding import Scaler

from sklearn.model_selection import train_test_split

mne.set_log_level('ERROR')

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

    fft = np.fft.rfft(x, axis=-1)  # (B, C, T//2+1)
    power = np.abs(fft) ** 2  # (B, C, T//2+1)
    power = power.mean(axis=1)  # (B, T//2+1) — avg over channels

    freqs = np.fft.rfftfreq(T, d=1.0 / fs)  # (T//2+1,) in Hz
    band_width = 29 / n_filter_banks

    band_powers = []
    for i in range(n_filter_banks):
        low = i * band_width
        high = (i + 1) * band_width
        mask = (freqs >= low) & (freqs < high)

        if mask.sum() == 0:
            # frequency resolution coarser than band width — take nearest bin
            mid_freq = (low + high) / 2.0
            nearest = np.argmin(np.abs(freqs - mid_freq))
            mask = np.zeros(len(freqs), dtype=bool)
            mask[nearest] = True

        band_powers.append(power[:, mask].mean(axis=-1))  # (B,)

    band_powers = np.stack(band_powers, axis=1)  # (B, F)
    return np.log1p(band_powers)  # (B, F) log-normalised

def compute_channels_band_powers(x, n_filter_banks, fs):
    """
    Compute log band powers for n_filter_banks equally spaced bands.

    Args:
        x:              (B, C, T) numpy array of raw EEG signal
        n_filter_banks: number of frequency bands F
        fs:             sampling frequency in Hz

    Returns:
        (B, C, F) log-normalised mean band power per-channel
    """
    B, C, T = x.shape

    fft = np.fft.rfft(x, axis=-1)  # (B, C, T//2+1)
    power = np.abs(fft) ** 2  # (B, C, T//2+1)

    freqs = np.fft.rfftfreq(T, d=1.0 / fs)  # (T//2+1,) in Hz
    band_width = 29 / n_filter_banks

    band_powers = []
    for i in range(n_filter_banks):
        low = i * band_width
        high = (i + 1) * band_width
        mask = (freqs >= low) & (freqs < high)

        if mask.sum() == 0:
            # frequency resolution coarser than band width — take nearest bin
            mid_freq = (low + high) / 2.0
            nearest = np.argmin(np.abs(freqs - mid_freq))
            mask = np.zeros(len(freqs), dtype=bool)
            mask[nearest] = True

        band_powers.append(power[:, :, mask].mean(axis=-1))  # (B, C)

    band_powers = np.stack(band_powers, axis=-1)  # (B, C, F)
    return np.log1p(band_powers)  # (B, C, F) log-normalised


def compute_downsampled_stft(x, n_filter_banks, patch_size, n_times, freq_downsample, fs):
    F_cfg = n_filter_banks
    P_cfg = patch_size
    wsize = int((F_cfg - 1) * 2)
    assert wsize % 2 == 0
    tstep = wsize // 2

    # STFT per sample -> shape (n_samples, n_channels, n_freqs, n_steps)
    # n_freqs == wsize//2 + 1 == F_cfg
    stft = np.abs(np.array([
        mne.time_frequency.stft(x_sample, wsize, tstep)
        for x_sample in x
    ]))

    # ---- Frequency axis: keep bins 1..min(F_cfg, 30), then average down ----
    n_freqs_avail = stft.shape[-2]
    hi = min(F_cfg, 30)
    stft = stft[..., 0:hi + 1, :]  # drop DC bin, clip to 30 bins max

    if freq_downsample and freq_downsample > 1:
        n_freqs = stft.shape[-2]
        n_freqs_trim = (n_freqs // freq_downsample) * freq_downsample
        stft = stft[..., :n_freqs_trim, :]
        new_shape = stft.shape[:-2] + (n_freqs_trim // freq_downsample, freq_downsample, stft.shape[-1])
        stft = stft.reshape(new_shape).mean(axis=-2)

    # ---- Time axis: average-pool n_steps -> target_T ----
    target_T = (n_times - 1) // P_cfg
    n_steps = stft.shape[-1]
    edges = np.linspace(0, n_steps, target_T + 1).astype(int)
    edges[1:] = np.maximum(edges[1:], edges[:-1] + 1)  # guarantee non-empty bins

    pooled = np.stack(
        [stft[..., edges[i]:edges[i + 1]].mean(axis=-1) for i in range(target_T)],
        axis=-1,
    )

    return pooled


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


def train_val_test_split(X, y, train_split, val_split, test_split, ppo, seed):
    n = X.shape[0]
    train_len = int(n * train_split)
    val_len = int(n * val_split)
    test_len = int(n * test_split)

    if test_split > 0:
        train_split = n - val_len - test_len
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_len,
            stratify=y,
            random_state=seed,
            shuffle=True
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_len,
            stratify=y_temp,
            random_state=seed,
            shuffle=True
        )
    else:
        val_len = n - train_len
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=val_len,
            stratify=y,
            random_state=seed,
            shuffle=True
        )

    if 'EA' in ppo:
        print("Using Euclidean Alignment")
        X_train, sqrtRefEA = EA(X_train)
        X_val = EA_online(X_val, sqrtRefEA)
        if test_split > 0:
            X_test = EA_online(X_test, sqrtRefEA)
            return (X_train, y_train, X_val, y_val, X_test, y_test)
        return (X_train, y_train, X_val, y_val)


def bandpass_filtering(X, low=8.0, high=30.0, fs=250):
    sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")

    # Apply filter to each trial and channel
    X_filtered = np.zeros_like(X)
    for i in range(X.shape[0]):  # trials
        for j in range(X.shape[1]):  # channels
            X_filtered[i, j, :] = sosfiltfilt(sos, X[i, j, :])

    return X_filtered
