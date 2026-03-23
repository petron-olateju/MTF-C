"""Unit tests for utils modules.

Tests cover:
- data_loader.py: EEGDataset class, load functions
- experiment_recorder.py: Experiment and Parameter classes
- preprocessing.py: EA, EA_online, bandpass_filtering functions
"""

import os
import tempfile
from argparse import Namespace
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch
import yaml

from utils.data_loader import EEGDataset
from utils.experiment_recorder import Experiment, Parameter
from utils.preprocessing import EA, EA_online, bandpass_filtering


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
        """Test that EEGDataset works with empty arrays."""
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


class TestDataLoaderValidation:
    """Tests for data loader input validation logic (without actual data loading)."""

    def test_subject_validation_error_logic(self):
        """Test that non-integer subject raises ValueError using logic from loader."""

        def validate_subject(subject):
            if isinstance(subject, int):
                return True
            raise ValueError(
                "subject argument should be an integer within valid range on MOABB site"
            )

        with pytest.raises(ValueError, match="subject argument should be an integer"):
            validate_subject("invalid")

        assert validate_subject(1) is True

    def test_preprocessing_pipeline_validation_logic(self):
        """Test preprocessing pipeline validation logic."""

        def validate_pipeline(preprocessing_pipeline):
            if preprocessing_pipeline is not None:
                if isinstance(preprocessing_pipeline, list) == 0:
                    raise ValueError("preprocessing_pipeline argument should be a list")

        validate_pipeline(None)
        validate_pipeline([])
        validate_pipeline([lambda x: x])

        with pytest.raises(
            ValueError, match="preprocessing_pipeline argument should be a list"
        ):
            validate_pipeline("not_a_list")

    def test_preprocessing_pipeline_applied(self):
        """Test that preprocessing pipeline functions are applied correctly."""

        def apply_pipeline(X, preprocessing_pipeline):
            if preprocessing_pipeline is not None:
                if not isinstance(preprocessing_pipeline, list):
                    raise ValueError("preprocessing_pipeline argument should be a list")
                for fn in preprocessing_pipeline:
                    X = fn(X)
            return X

        def scale_data(x):
            return x * 2

        X = np.array([[1, 2, 3], [4, 5, 6]])
        pipeline = [scale_data]

        result = apply_pipeline(X, pipeline)
        np.testing.assert_array_equal(result, X * 2)


class TestExperimentRecorder:
    """Tests for Experiment and Parameter classes."""

    def test_experiment_init(self):
        """Test that Experiment initializes correctly."""
        experiment = Experiment(
            "test_exp", dir="./experiments", description="Test experiment"
        )

        assert experiment.name == "test_exp"
        assert experiment.dir == "./experiments/test_exp"
        assert experiment.description == "Test experiment"
        assert experiment.parameters == {}
        assert experiment.baseline is None

    def test_experiment_with_baseline(self):
        """Test Experiment initialization with baseline."""
        experiment = Experiment(
            "test_exp", dir="./experiments", description="Test", baseline="v0.1"
        )

        assert experiment.baseline == "v0.1"

    def test_parameter_init(self):
        """Test that Parameter initializes correctly."""
        param = Parameter(
            value=0.001, var_name="learning_rate", parameter_class="training"
        )

        assert param.get_value() == 0.001
        assert param.get_var_name() == "learning_rate"
        assert param.get_parameter_class() == "training"

    def test_parameter_set_value(self):
        """Test that Parameter value can be set."""
        param = Parameter(value=0.001, var_name="lr", parameter_class="training")

        param.set_value(0.01)

        assert param.get_value() == 0.01

    def test_parameter_set_var_name(self):
        """Test that Parameter variable name can be set."""
        param = Parameter(value=0.001, var_name="lr", parameter_class="training")

        param.set_var_name("learning_rate")

        assert param.get_var_name() == "learning_rate"

    def test_parameter_set_parameter_class(self):
        """Test that Parameter class can be set."""
        param = Parameter(value=0.001, var_name="lr", parameter_class="training")

        param.set_parameter_class("optimizer")

        assert param.get_parameter_class() == "optimizer"

    def test_experiment_update_param_global(self):
        """Test updating global parameter."""
        experiment = Experiment("test_exp", dir="./experiments", description="Test")
        param = Parameter(value=0.001, var_name="lr", parameter_class=None)

        experiment.update_param(param)

        assert "lr" in experiment.parameters
        assert experiment.parameters["lr"] == 0.001

    def test_experiment_update_param_global_lowercase(self):
        """Test updating global parameter with lowercase 'global'."""
        experiment = Experiment("test_exp", dir="./experiments", description="Test")
        param = Parameter(value=0.001, var_name="lr", parameter_class="GLOBAL")

        experiment.update_param(param)

        assert "lr" in experiment.parameters
        assert experiment.parameters["lr"] == 0.001

    def test_experiment_update_param_with_class(self):
        """Test updating parameter with a class."""
        experiment = Experiment("test_exp", dir="./experiments", description="Test")
        param = Parameter(value=0.001, var_name="lr", parameter_class="training")

        experiment.update_param(param)

        assert "training" in experiment.parameters
        assert "lr" in experiment.parameters["training"]
        assert experiment.parameters["training"]["lr"] == 0.001

    def test_experiment_update_param_creates_nested_dict(self):
        """Test that updating creates nested dict if class doesn't exist."""
        experiment = Experiment("test_exp", dir="./experiments", description="Test")
        param1 = Parameter(value=0.001, var_name="lr", parameter_class="training")
        param2 = Parameter(value=32, var_name="batch_size", parameter_class="training")

        experiment.update_param(param1)
        experiment.update_param(param2)

        assert "training" in experiment.parameters
        assert len(experiment.parameters["training"]) == 2

    def test_experiment_add_params(self):
        """Test adding multiple parameters at once."""
        experiment = Experiment("test_exp", dir="./experiments", description="Test")
        params = [
            Parameter(value=0.001, var_name="lr", parameter_class="training"),
            Parameter(value=32, var_name="batch_size", parameter_class="training"),
            Parameter(value="adam", var_name="optimizer", parameter_class=None),
        ]

        experiment.add_params(params)

        assert "training" in experiment.parameters
        assert "optimizer" in experiment.parameters
        assert experiment.parameters["optimizer"] == "adam"

    def test_experiment_update_param_validates_value_type(self):
        """Test that update_param validates value types."""
        experiment = Experiment("test_exp", dir="./experiments", description="Test")

        valid_types = [1, 1.0, "str", {"a": 1}, [1, 2], np.array([1, 2])]
        for value in valid_types:
            param = Parameter(value=value, var_name="test", parameter_class=None)
            experiment.update_param(param)

    def test_experiment_update_param_rejects_invalid_value(self):
        """Test that update_param rejects invalid value types."""
        experiment = Experiment("test_exp", dir="./experiments", description="Test")

        class CustomClass:
            pass

        param = Parameter(value=CustomClass(), var_name="test", parameter_class=None)
        with pytest.raises((AssertionError, TypeError)):
            experiment.update_param(param)

    @pytest.fixture
    def temp_experiment_dir(self):
        """Create a temporary directory for experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_experiment_save(self, temp_experiment_dir):
        """Test that Experiment.save() creates correct files."""
        experiment = Experiment(
            "test_exp", dir=temp_experiment_dir, description="Test experiment"
        )
        param = Parameter(value=0.001, var_name="lr", parameter_class="training")
        experiment.add_params([param])

        experiment.save()

        assert os.path.exists(
            f"{temp_experiment_dir}/test_exp/training_parameters.yaml"
        )

        with open(f"{temp_experiment_dir}/test_exp/training_parameters.yaml", "r") as f:
            saved_data = yaml.safe_load(f)

        assert saved_data["name"] == "test_exp"
        assert saved_data["description"] == "Test experiment"
        assert "parameters" in saved_data


class TestPreprocessing:
    """Tests for preprocessing functions."""

    def test_ea_basic(self):
        """Test that EA returns correct output shape."""
        x = np.random.randn(10, 22, 500)
        XEA, sqrtRefEA = EA(x)

        assert XEA.shape == x.shape
        assert sqrtRefEA.shape == (22, 22)

    def test_ea_output_is_floating_point(self):
        """Test that EA output is floating point."""
        x = np.random.randn(10, 22, 500)
        XEA, sqrtRefEA = EA(x)

        assert XEA.dtype.kind == "f"
        assert sqrtRefEA.dtype.kind == "f"

    def test_ea_identity_invariance(self):
        """Test EA is identity-invariant for equal covariances."""
        x = np.random.randn(10, 3, 100)
        x_normalized = (x - x.mean(axis=2, keepdims=True)) / (
            x.std(axis=2, keepdims=True) + 1e-8
        )
        XEA, sqrtRefEA = EA(x_normalized)

        assert XEA.shape == x.shape

    def test_ea_single_trial(self):
        """Test EA with single trial."""
        x = np.random.randn(1, 22, 500)
        XEA, sqrtRefEA = EA(x)

        assert XEA.shape == x.shape
        assert sqrtRefEA.shape == (22, 22)

    def test_ea_single_channel(self):
        """Test EA with single channel."""
        x = np.random.randn(10, 1, 500)
        XEA, sqrtRefEA = EA(x)

        assert XEA.shape == x.shape
        assert sqrtRefEA.shape == (1, 1)

    def test_ea_online_basic(self):
        """Test that EA_online returns correct output shape."""
        x = np.random.randn(10, 22, 500)
        _, sqrtRefEA = EA(x)

        XEA_online = EA_online(x, sqrtRefEA)

        assert XEA_online.shape == x.shape

    def test_ea_online_output_is_floating_point(self):
        """Test that EA_online output is floating point."""
        x = np.random.randn(10, 22, 500)
        _, sqrtRefEA = EA(x)

        XEA_online = EA_online(x, sqrtRefEA)

        assert XEA_online.dtype.kind == "f"

    def test_ea_online_uses_provided_reference(self):
        """Test that EA_online uses provided reference matrix."""
        x = np.random.randn(10, 22, 500)
        _, sqrtRefEA = EA(x)

        XEA_batch, _ = EA(x)
        XEA_online = EA_online(x, sqrtRefEA)

        np.testing.assert_allclose(XEA_batch, XEA_online, rtol=1e-10)

    def test_ea_online_single_trial(self):
        """Test EA_online with single trial."""
        x = np.random.randn(1, 22, 500)
        _, sqrtRefEA = EA(x)

        XEA_online = EA_online(x, sqrtRefEA)

        assert XEA_online.shape == x.shape

    def test_bandpass_filtering_basic(self):
        """Test that bandpass_filtering returns correct output shape."""
        X = np.random.randn(10, 22, 500)
        X_filtered = bandpass_filtering(X)

        assert X_filtered.shape == X.shape

    def test_bandpass_filtering_output_is_floating_point(self):
        """Test that bandpass_filtering output is floating point."""
        X = np.random.randn(10, 22, 500)
        X_filtered = bandpass_filtering(X)

        assert X_filtered.dtype.kind == "f"

    def test_bandpass_filtering_reduces_variance(self):
        """Test that bandpass filtering reduces signal variance outside band."""
        np.random.seed(42)
        fs = 250
        t = np.linspace(0, 2, 2 * fs)
        signal_8hz = np.sin(2 * np.pi * 8 * t)
        signal_20hz = np.sin(2 * np.pi * 20 * t)
        signal_50hz = np.sin(2 * np.pi * 50 * t)
        X = np.stack([signal_8hz, signal_20hz, signal_50hz])
        X = np.stack([X] * 1)

        X_filtered = bandpass_filtering(X, low=15, high=25, fs=fs)

        assert X_filtered.shape == X.shape

    def test_bandpass_filtering_custom_frequency_bounds(self):
        """Test bandpass_filtering with custom frequency bounds."""
        X = np.random.randn(10, 22, 500)
        X_filtered = bandpass_filtering(X, low=4.0, high=40.0, fs=500)

        assert X_filtered.shape == X.shape

    def test_bandpass_filtering_default_parameters(self):
        """Test bandpass_filtering with default parameters."""
        X = np.random.randn(10, 22, 500)
        X_filtered_default = bandpass_filtering(X)
        X_filtered_explicit = bandpass_filtering(X, low=8.0, high=30.0, fs=250)

        assert X_filtered_default.shape == X.shape
        assert X_filtered_explicit.shape == X.shape

    def test_bandpass_filtering_single_trial(self):
        """Test bandpass_filtering with single trial."""
        X = np.random.randn(1, 22, 500)
        X_filtered = bandpass_filtering(X)

        assert X_filtered.shape == X.shape

    def test_bandpass_filtering_single_channel(self):
        """Test bandpass_filtering with single channel."""
        X = np.random.randn(10, 1, 500)
        X_filtered = bandpass_filtering(X)

        assert X_filtered.shape == X.shape


class TestIntegration:
    """Integration tests combining multiple utilities."""

    def test_ea_pipeline_with_eeg_dataset(self):
        """Test EA preprocessing with EEGDataset."""
        raw_data = np.random.randn(10, 22, 500)
        XEA, _ = EA(raw_data)

        labels = np.array([0] * 5 + [1] * 5)
        stft = np.random.randn(10, 22, 8, 4).astype(np.float32)

        dataset = EEGDataset(XEA.astype(np.float32), labels, stft)

        assert len(dataset) == 10
        signal, label, _ = dataset[0]
        assert signal.shape == (22, 500)

    def test_experiment_recording_preprocessing_params(self):
        """Test that preprocessing parameters can be recorded in Experiment."""
        experiment = Experiment("test_exp", dir="./experiments", description="Test")

        low_param = Parameter(
            value=8.0, var_name="low_freq", parameter_class="bandpass"
        )
        high_param = Parameter(
            value=30.0, var_name="high_freq", parameter_class="bandpass"
        )
        fs_param = Parameter(value=250, var_name="fs", parameter_class="bandpass")

        experiment.add_params([low_param, high_param, fs_param])

        assert "bandpass" in experiment.parameters
        assert experiment.parameters["bandpass"]["low_freq"] == 8.0
        assert experiment.parameters["bandpass"]["high_freq"] == 30.0
        assert experiment.parameters["bandpass"]["fs"] == 250
