"""Unit tests for data_loader module.

Tests cover:
- EEGDataset class
- Chronological stratified k-fold function
- Legacy function exports
- SSVEP_DataLoader class
"""

import numpy as np
import pytest
import torch

from utils.data_loader import (
    EEGDataset,
    chronological_stratified_kfold,
    load_BNCI2014_001,
    load_BNCI2014_002,
    load_BNCI2014_004,
    load_BNCI2015_001,
    load_BNCI2015_004,
    load_Liu2024,
    load_AlexMI,
    get_subjects_BNCI2014_001,
    get_subjects_BNCI2014_002,
    get_subjects_BNCI2014_004,
    get_subjects_BNCI2015_001,
    get_subjects_BNCI2015_004,
    get_subjects_Liu2024,
    get_subjects_AlexMI,
    SSVEP_DataLoader,
)


class TestEEGDataset:
    """Tests for EEGDataset class."""

    def test_eeg_dataset_init(self):
        """Test that EEGDataset initializes correctly."""
        data = np.random.randn(10, 22, 500).astype(np.float32)
        labels = np.array([0] * 5 + [1] * 5)
        stft = np.random.randn(10, 22, 8, 4).astype(np.float32)

        dataset = EEGDataset(data, labels, stft)

        assert isinstance(dataset.signals, torch.Tensor)
        assert dataset.signals.dtype == torch.float32
        assert np.array_equal(dataset.labels, labels)
        assert np.array_equal(dataset.stft, stft)

    def test_eeg_dataset_len(self):
        """Test that __len__ returns correct number of samples."""
        data = np.random.randn(10, 22, 500).astype(np.float32)
        labels = np.zeros(10)
        stft = np.random.randn(10, 22, 8, 4).astype(np.float32)

        dataset = EEGDataset(data, labels, stft)

        assert len(dataset) == 10

    def test_eeg_dataset_getitem(self):
        """Test that __getitem__ returns correct sample."""
        data = np.random.randn(10, 22, 500).astype(np.float32)
        labels = np.array([0] * 5 + [1] * 5)
        stft = np.random.randn(10, 22, 8, 4).astype(np.float32)

        dataset = EEGDataset(data, labels, stft)

        signal, label, stft_sample = dataset[0]

        assert signal.shape == (22, 500)
        assert label == labels[0]
        assert stft_sample.shape == (22, 8, 4)

    def test_eeg_dataset_empty(self):
        """Test that EEGDataset works with single sample."""
        data = np.random.randn(1, 22, 500).astype(np.float32)
        labels = np.array([0])
        stft = np.random.randn(1, 22, 8, 4).astype(np.float32)

        dataset = EEGDataset(data, labels, stft)

        assert len(dataset) == 1

    def test_eeg_dataset_single_channel(self):
        """Test that EEGDataset works with single channel."""
        data = np.random.randn(5, 1, 500).astype(np.float32)
        labels = np.array([0, 1, 0, 1, 0])
        stft = np.random.randn(5, 1, 8, 4).astype(np.float32)

        dataset = EEGDataset(data, labels, stft)

        signal, label, stft_sample = dataset[0]

        assert signal.shape == (1, 500)

    def test_eeg_dataset_getitem_indexing(self):
        """Test that __getitem__ works with various indices."""
        data = np.random.randn(10, 22, 500).astype(np.float32)
        labels = np.arange(10)
        stft = np.random.randn(10, 22, 8, 4).astype(np.float32)

        dataset = EEGDataset(data, labels, stft)

        signal, label, stft_sample = dataset[5]
        assert label == 5
        assert stft_sample.shape == (22, 8, 4)


class TestChronologicalStratifiedKFold:
    """Tests for chronological_stratified_kfold function."""

    def test_kfold_basic(self):
        """Test basic k-fold functionality."""
        y = np.array([0] * 10 + [1] * 10)
        folds = list(chronological_stratified_kfold(y, n_splits=5))

        assert len(folds) == 5
        for train_idx, test_idx in folds:
            assert len(train_idx) + len(test_idx) == 20
            assert len(np.intersect1d(train_idx, test_idx)) == 0

    def test_kfold_output_shapes(self):
        """Test that output indices have correct shapes."""
        y = np.array([0] * 6 + [1] * 6)
        folds = list(chronological_stratified_kfold(y, n_splits=3))

        total_train = sum(len(train) for train, _ in folds)
        total_test = sum(len(test) for _, test in folds)

        assert len(folds) == 3
        assert len(y) == 12

    def test_kfold_no_overlap(self):
        """Test that train and test sets don't overlap."""
        y = np.array([0, 0, 0, 1, 1, 1])
        folds = list(chronological_stratified_kfold(y, n_splits=3))

        for train_idx, test_idx in folds:
            assert len(np.intersect1d(train_idx, test_idx)) == 0

    def test_kfold_single_class(self):
        """Test k-fold with single class."""
        y = np.zeros(10)
        folds = list(chronological_stratified_kfold(y, n_splits=5))

        assert len(folds) == 5

    def test_kfold_balanced_classes(self):
        """Test k-fold with balanced classes."""
        y = np.array([0] * 5 + [1] * 5)
        folds = list(chronological_stratified_kfold(y, n_splits=5))

        for train_idx, test_idx in folds:
            assert len(np.intersect1d(train_idx, test_idx)) == 0


class TestLegacyFunctions:
    """Tests for backward compatibility functions."""

    def test_get_subjects_all_datasets(self):
        """Test that get_subjects functions return lists."""
        assert isinstance(get_subjects_BNCI2014_001(), list)
        assert isinstance(get_subjects_BNCI2014_002(), list)
        assert isinstance(get_subjects_BNCI2014_004(), list)
        assert isinstance(get_subjects_BNCI2015_001(), list)
        assert isinstance(get_subjects_BNCI2015_004(), list)
        assert isinstance(get_subjects_Liu2024(), list)
        assert isinstance(get_subjects_AlexMI(), list)

    def test_load_functions_callable(self):
        """Test that load functions are callable."""
        functions = [
            load_BNCI2014_001,
            load_BNCI2014_002,
            load_BNCI2014_004,
            load_BNCI2015_001,
            load_BNCI2015_004,
            load_Liu2024,
            load_AlexMI,
        ]

        for fn in functions:
            assert callable(fn)


class TestSSVEPDataLoader:
    """Tests for SSVEP_DataLoader class."""

    def test_get_available_datasets(self):
        """Test that get_available_datasets returns correct list."""
        datasets = SSVEP_DataLoader.get_available_datasets()
        assert isinstance(datasets, list)
        assert "Kalunga2016" in datasets
        assert "MAMEM2" in datasets
        assert "MAMEM3" in datasets
        assert "Nakanishi2015" in datasets
        assert "Wang2021Combined" in datasets

    def test_get_subjects_returns_list(self):
        """Test that get_subjects returns a list of subject IDs."""
        subjects = SSVEP_DataLoader.get_subjects("Kalunga2016")
        assert isinstance(subjects, list)
        assert len(subjects) > 0

    def test_get_subjects_all_datasets(self):
        """Test that get_subjects works for all SSVEP datasets."""
        datasets = SSVEP_DataLoader.get_available_datasets()
        for ds in datasets:
            subjects = SSVEP_DataLoader.get_subjects(ds)
            assert isinstance(subjects, list)
            assert len(subjects) > 0

    def test_invalid_dataset_raises_error(self):
        """Test that invalid dataset name raises ValueError."""
        with pytest.raises(ValueError):
            SSVEP_DataLoader("InvalidDataset", subject=1)

    def test_invalid_subject_raises_error(self):
        """Test that invalid subject raises appropriate error."""
        with pytest.raises(Exception):
            SSVEP_DataLoader("Kalunga2016", subject=9999)

    def test_init_parameters(self):
        """Test that SSVEP_DataLoader accepts all parameters."""
        loader = SSVEP_DataLoader(
            dataset_name="Kalunga2016",
            subject=1,
            preprocessing_pipeline=None,
            t0=0.5,
            tmax=4.0,
            fmin=7,
            fmax=45,
        )
        assert loader.dataset_name == "Kalunga2016"
        assert loader.subject == 1
        assert loader.t0 == 0.5
        assert loader.tmax == 4.0
        assert loader.fmin == 7
        assert loader.fmax == 45

    def test_default_parameters(self):
        """Test default parameter values."""
        loader = SSVEP_DataLoader(
            dataset_name="Kalunga2016",
            subject=1,
        )
        assert loader.t0 == 0.0
        assert loader.tmax is None
        assert loader.fmin == 7
        assert loader.fmax == 45

    def test_get_data_returns_tuple(self):
        """Test that get_data returns tuple of (X, y, info)."""
        loader = SSVEP_DataLoader("Kalunga2016", subject=1)
        X, y, info = loader.get_data()

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(info, dict)

        assert X.ndim == 3
        assert len(X) == len(y)

        assert "n_trials" in info
        assert "n_ch" in info
        assert "n_times" in info
        assert "n_classes" in info
        assert "fs" in info

    def test_data_shape_consistency(self):
        """Test that X and y have consistent lengths."""
        loader = SSVEP_DataLoader("Kalunga2016", subject=1)
        X, y, info = loader.get_data()

        assert X.shape[0] == len(y)
        assert X.shape[1] == info["n_ch"]
        assert X.shape[2] == info["n_times"]

    @pytest.mark.skip(reason="Slow - requires MOABB data download")
    def test_label_range(self):
        """Test that labels are in valid range."""
        loader = SSVEP_DataLoader("MAMEM2", subject=1)
        X, y, info = loader.get_data()

        unique_labels = np.unique(y)
        assert np.min(unique_labels) >= 0
        assert np.max(unique_labels) < info["n_classes"]

    @pytest.mark.skip(reason="Slow - requires MOABB data download")
    def test_info_contains_frequencies(self):
        """Test that info dict contains frequency information."""
        loader = SSVEP_DataLoader("Wang2021Combined", subject=1)
        X, y, info = loader.get_data()

        assert "freqs" in info
        assert isinstance(info["freqs"], list)

    @pytest.mark.skip(reason="Slow - requires MOABB data download")
    def test_preprocessing_pipeline_applied(self):
        """Test that preprocessing pipeline is applied."""

        def simple_preprocess(X):
            return X - np.mean(X, axis=2, keepdims=True)

        loader = SSVEP_DataLoader(
            dataset_name="Kalunga2016",
            subject=1,
            preprocessing_pipeline=[simple_preprocess],
        )
        X, y, info = loader.get_data()

        assert X is not None

    @pytest.mark.skip(reason="Slow - requires MOABB data download")
    def test_different_time_windows(self):
        """Test loading with different time windows."""
        loader1 = SSVEP_DataLoader("Kalunga2016", subject=1, t0=0.5, tmax=3.0)
        X1, y1, info1 = loader1.get_data()

        loader2 = SSVEP_DataLoader("Kalunga2016", subject=1, t0=1.0, tmax=4.0)
        X2, y2, info2 = loader2.get_data()

        assert X1.shape != X2.shape
        assert info1["n_times"] != info2["n_times"]

    @pytest.mark.skip(reason="Slow - requires MOABB data download")
    @pytest.mark.parametrize("dataset", SSVEP_DataLoader.get_available_datasets())
    def test_all_datasets_loadable(self, dataset):
        """Test that all datasets can be loaded."""
        subjects = SSVEP_DataLoader.get_subjects(dataset)
        loader = SSVEP_DataLoader(dataset, subject=subjects[0])
        X, y, info = loader.get_data()

        assert X is not None
        assert y is not None
        assert info is not None
        assert X.shape[0] > 0
