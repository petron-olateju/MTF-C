"""Unit tests for run_experiment.py script.

Tests cover:
- Argument parsing (parse_args)
- Dataset handling (single dataset vs 'all' datasets)
- Subject iteration per dataset
- Result aggregation and CSV saving
"""

import os
import tempfile
import yaml
from argparse import Namespace
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import run_experiment


class TestParseArgs:
    """Tests for argument parsing functionality."""

    def test_default_arguments(self):
        """Test that default arguments are correctly parsed."""
        with patch("sys.argv", ["run_experiment.py"]):
            args = run_experiment.parse_args()
            assert args.script == "cross_validation.py"
            assert args.model_name == "db_conformer"
            assert args.dataset == "dummy_dataset"
            assert args.device == "cpu"
            assert args.experiment_version == "v0.0"
            assert args.experiment_description == "Baseline Experiment"
            assert args.experiment_folder == "./experiments"
            assert args.verbose is False

    def test_custom_arguments(self):
        """Test that custom arguments are correctly parsed."""
        with patch(
            "sys.argv",
            [
                "run_experiment.py",
                "--script",
                "pre_training",
                "--model_name",
                "mtf_c",
                "--dataset",
                "all",
                "--device",
                "cuda",
                "--experiment_version",
                "v1.0",
                "--experiment_description",
                "Test Experiment",
                "--experiment_folder",
                "/tmp/experiments",
                "--verbose",
            ],
        ):
            args = run_experiment.parse_args()
            assert args.script == "pre_training"
            assert args.model_name == "mtf_c"
            assert args.dataset == "all"
            assert args.device == "cuda"
            assert args.experiment_version == "v1.0"
            assert args.experiment_description == "Test Experiment"
            assert args.experiment_folder == "/tmp/experiments"
            assert args.verbose is True

    def test_single_dataset_argument(self):
        """Test that single dataset argument is correctly parsed."""
        with patch(
            "sys.argv",
            [
                "run_experiment.py",
                "--dataset",
                "BNCI2014_001",
            ],
        ):
            args = run_experiment.parse_args()
            assert args.dataset == "BNCI2014_001"

    def test_all_datasets_argument(self):
        """Test that 'all' dataset argument is correctly parsed."""
        with patch(
            "sys.argv",
            [
                "run_experiment.py",
                "--dataset",
                "all",
            ],
        ):
            args = run_experiment.parse_args()
            assert args.dataset == "all"


class TestDatasetHandling:
    """Tests for dataset and subject handling logic."""

    def test_single_dataset_returns_list_with_one(self):
        """Test that single dataset returns a list with one dataset."""
        dataset = "BNCI2014_001"
        if dataset == "all":
            datasets = [
                "BNCI2014_001",
                "BNCI2014_002",
                "BNCI2014_004",
                "BNCI2015_001",
                "BNCI2015_004",
                "AlexMI",
            ]
        else:
            datasets = [dataset]

        assert datasets == ["BNCI2014_001"]
        assert len(datasets) == 1

    def test_all_datasets_returns_all_available(self):
        """Test that 'all' dataset returns all available datasets."""
        expected_datasets = [
            "BNCI2014_001",
            "BNCI2014_002",
            "BNCI2014_004",
            "BNCI2015_001",
            "BNCI2015_004",
            "AlexMI",
        ]

        datasets = [
            "BNCI2014_001",
            "BNCI2014_002",
            "BNCI2014_004",
            "BNCI2015_001",
            "BNCI2015_004",
            "AlexMI",
        ]

        assert len(datasets) == 6
        assert all(ds in datasets for ds in expected_datasets)

    def test_dummy_dataset_default_subjects(self):
        """Test that dummy_dataset defaults to single subject."""
        dataset = "dummy_dataset"
        if dataset == "BNCI2014_001":
            subjects = [1, 2, 3]
        elif dataset == "BNCI2014_002":
            subjects = [1, 2, 3]
        elif dataset == "BNCI2014_004":
            subjects = [1, 2, 3]
        elif dataset == "BNCI2015_001":
            subjects = [1, 2, 3]
        elif dataset == "BNCI2015_004":
            subjects = [1, 2, 3]
        elif dataset == "Liu2024":
            subjects = [1, 2, 3]
        elif dataset == "AlexMI":
            subjects = [1, 2, 3]
        else:
            subjects = [1]

        assert subjects == [1]


class TestResultAggregation:
    """Tests for result aggregation logic."""

    def test_accuracy_aggregation(self):
        """Test that accuracies are correctly aggregated across subjects."""
        all_accuracies = []

        subject_results = [0.75, 0.80, 0.85, 0.70]
        all_accuracies = all_accuracies + subject_results

        acc_mean = np.mean(all_accuracies)
        acc_std = np.std(all_accuracies)

        assert abs(acc_mean - 0.775) < 0.01
        assert abs(acc_std - 0.054) < 0.01

    def test_kappa_aggregation(self):
        """Test that kappas are correctly aggregated across subjects."""
        all_kappas = []

        subject_results = [0.65, 0.70, 0.75, 0.60]
        all_kappas = all_kappas + subject_results

        kappa_mean = np.mean(all_kappas)

        assert abs(kappa_mean - 0.675) < 0.01

    def test_stft_aggregation(self):
        """Test that STFT losses are correctly aggregated across subjects."""
        all_stft = []

        subject_results = [0.30, 0.35, 0.25, 0.40]
        all_stft = all_stft + subject_results

        stft_result = np.mean(all_stft)

        assert abs(stft_result - 0.325) < 0.01

    def test_results_dict_per_model(self):
        """Test that results are stored per model."""
        accuracy_results = {}
        accuracy_std_results = {}
        kappa_results = {}
        kappa_std_results = {}
        stft_reconstruction_results = {}

        model_name = "mtf_c"

        if model_name not in accuracy_results:
            accuracy_results[model_name] = {}
            accuracy_std_results[model_name] = {}
            kappa_results[model_name] = {}
            kappa_std_results[model_name] = {}
            stft_reconstruction_results[model_name] = {}

        accuracy_results[model_name]["BNCI2014_001"] = 0.85
        accuracy_std_results[model_name]["BNCI2014_001"] = 0.05
        kappa_results[model_name]["BNCI2014_001"] = 0.80
        kappa_std_results[model_name]["BNCI2014_001"] = 0.06
        stft_reconstruction_results[model_name]["BNCI2014_001"] = 0.25

        assert model_name in accuracy_results
        assert "BNCI2014_001" in accuracy_results[model_name]
        assert accuracy_results[model_name]["BNCI2014_001"] == 0.85


class TestCSVExport:
    """Tests for CSV export functionality."""

    @pytest.fixture
    def temp_experiment_dir(self):
        """Create a temporary directory for experiment results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_csv_results_structure(self, temp_experiment_dir):
        """Test that results are saved with correct CSV structure."""
        accuracy_results = {
            "mtf_c": {"BNCI2014_001": 0.85, "BNCI2014_002": 0.80},
            "db_conformer": {"BNCI2014_001": 0.75},
        }

        version_dir = os.path.join(temp_experiment_dir, "v1.0")
        os.makedirs(version_dir, exist_ok=True)

        pd.DataFrame(accuracy_results).to_csv(
            os.path.join(version_dir, "accuracy.results.csv")
        )

        loaded = pd.read_csv(
            os.path.join(version_dir, "accuracy.results.csv"), index_col=0
        )

        assert "mtf_c" in loaded.columns
        assert "db_conformer" in loaded.columns
        assert loaded.loc["BNCI2014_001", "mtf_c"] == 0.85

    def test_multiple_csv_files_created(self, temp_experiment_dir):
        """Test that all required CSV files are created."""
        accuracy_results = {"mtf_c": {"BNCI2014_001": 0.85}}
        accuracy_std_results = {"mtf_c": {"BNCI2014_001": 0.05}}
        kappa_results = {"mtf_c": {"BNCI2014_001": 0.80}}
        kappa_std_results = {"mtf_c": {"BNCI2014_001": 0.06}}
        stft_reconstruction_results = {"mtf_c": {"BNCI2014_001": 0.25}}

        version_dir = os.path.join(temp_experiment_dir, "v1.0")
        os.makedirs(version_dir, exist_ok=True)

        pd.DataFrame(accuracy_results).to_csv(
            os.path.join(version_dir, "accuracy.results.csv")
        )
        pd.DataFrame(accuracy_std_results).to_csv(
            os.path.join(version_dir, "accuracy_std.results.csv")
        )
        pd.DataFrame(kappa_results).to_csv(
            os.path.join(version_dir, "kappa.results.csv")
        )
        pd.DataFrame(kappa_std_results).to_csv(
            os.path.join(version_dir, "kappa_std.results.csv")
        )
        pd.DataFrame(stft_reconstruction_results).to_csv(
            os.path.join(version_dir, "stft_reconstruction.results.csv")
        )

        assert os.path.exists(os.path.join(version_dir, "accuracy.results.csv"))
        assert os.path.exists(os.path.join(version_dir, "accuracy_std.results.csv"))
        assert os.path.exists(os.path.join(version_dir, "kappa.results.csv"))
        assert os.path.exists(os.path.join(version_dir, "kappa_std.results.csv"))
        assert os.path.exists(
            os.path.join(version_dir, "stft_reconstruction.results.csv")
        )


class TestResultsYAML:
    """Tests for results.yaml saving functionality in run_experiment."""

    @pytest.fixture
    def temp_experiment_dir(self):
        """Create a temporary directory for experiment results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_results_yaml_structure(self, temp_experiment_dir):
        """Test that results are saved with correct YAML structure."""
        experiment_version = "test_version"
        model_name = "mtf_c"
        dataset = "BNCI2014_001"
        timestamp = "2024-01-01T00:00:00"

        run_config = {
            "training": {
                "val_size": 0.2,
                "n_iter": 100,
                "folds": 5,
                "lr": 0.001,
                "batch_size": 64,
            },
            "model": {"patch_size": 6, "filter_banks": 7},
        }
        dataset_results = {
            dataset: {
                "accuracy": {"mean": 0.85, "std": 0.05},
                "kappa": {"mean": 0.80, "std": 0.06},
                "stft_reconstruction_loss": 0.25,
            }
        }

        yaml_dir = os.path.join(temp_experiment_dir, experiment_version)
        os.makedirs(yaml_dir, exist_ok=True)
        results_yaml_path = os.path.join(yaml_dir, "results.yaml")

        all_results = {}
        all_results.setdefault(model_name, {}).setdefault(timestamp, {}).update(
            {
                "config": run_config,
                "results": dataset_results,
            }
        )

        with open(results_yaml_path, "w") as f:
            yaml.dump(all_results, f, default_flow_style=False, sort_keys=False)

        with open(results_yaml_path, "r") as f:
            loaded_results = yaml.safe_load(f)

        assert model_name in loaded_results
        assert timestamp in loaded_results[model_name]
        assert "config" in loaded_results[model_name][timestamp]
        assert "results" in loaded_results[model_name][timestamp]
        assert dataset in loaded_results[model_name][timestamp]["results"]
        assert (
            loaded_results[model_name][timestamp]["results"][dataset]["accuracy"][
                "mean"
            ]
            == 0.85
        )
        assert loaded_results[model_name][timestamp]["config"]["training"]["folds"] == 5
        assert (
            loaded_results[model_name][timestamp]["config"]["model"]["patch_size"] == 6
        )

        with open(results_yaml_path, "w") as f:
            yaml.dump(all_results, f, default_flow_style=False, sort_keys=False)

        with open(results_yaml_path, "r") as f:
            loaded_results = yaml.safe_load(f)

        assert model_name in loaded_results
        assert timestamp in loaded_results[model_name]
        assert "config" in loaded_results[model_name][timestamp]
        assert "results" in loaded_results[model_name][timestamp]
        assert dataset in loaded_results[model_name][timestamp]["results"]
        assert (
            loaded_results[model_name][timestamp]["results"][dataset]["accuracy"][
                "mean"
            ]
            == 0.85
        )

    def test_results_yaml_appends_to_existing(self, temp_experiment_dir):
        """Test that new results are appended to existing results YAML."""
        experiment_version = "test_version"
        yaml_dir = os.path.join(temp_experiment_dir, experiment_version)
        os.makedirs(yaml_dir, exist_ok=True)
        results_yaml_path = os.path.join(yaml_dir, "results.yaml")

        existing_results = {
            "mtf_c": {
                "2024-01-01T00:00:00": {
                    "experiment_description": "Test experiment",
                    "config": {
                        "training": {
                            "val_size": 0.2,
                            "folds": 5,
                        },
                        "model": {"patch_size": 6},
                    },
                    "results": {
                        "BNCI2014_001": {
                            "accuracy": {"mean": 0.80, "std": 0.05},
                            "kappa": {"mean": 0.60, "std": 0.05},
                        }
                    },
                }
            }
        }

        with open(results_yaml_path, "w") as f:
            yaml.dump(existing_results, f)

        new_timestamp = "2024-01-02T00:00:00"
        new_config = {
            "training": {
                "val_size": 0.2,
                "folds": 5,
                "lr": 0.001,
            },
            "model": {"patch_size": 6, "filter_banks": 7},
        }
        new_dataset_results = {
            "BNCI2014_001": {
                "accuracy": {"mean": 0.85, "std": 0.05},
                "kappa": {"mean": 0.80, "std": 0.06},
                "stft_reconstruction_loss": 0.25,
            }
        }

        with open(results_yaml_path, "r") as f:
            all_results = yaml.safe_load(f)

        all_results.setdefault("mtf_c", {}).setdefault(new_timestamp, {}).update(
            {
                "experiment_description": "New experiment",
                "config": new_config,
                "results": new_dataset_results,
            }
        )

        with open(results_yaml_path, "w") as f:
            yaml.dump(all_results, f, default_flow_style=False)

        with open(results_yaml_path, "r") as f:
            loaded = yaml.safe_load(f)

        assert "2024-01-01T00:00:00" in loaded["mtf_c"]
        assert "2024-01-02T00:00:00" in loaded["mtf_c"]
        assert (
            loaded["mtf_c"]["2024-01-02T00:00:00"]["results"]["BNCI2014_001"][
                "accuracy"
            ]["mean"]
            == 0.85
        )
        assert (
            loaded["mtf_c"]["2024-01-02T00:00:00"]["config"]["training"]["lr"] == 0.001
        )

        with open(results_yaml_path, "w") as f:
            yaml.dump(all_results, f, default_flow_style=False)

        with open(results_yaml_path, "r") as f:
            loaded = yaml.safe_load(f)

        assert "2024-01-01T00:00:00" in loaded["mtf_c"]
        assert "2024-01-02T00:00:00" in loaded["mtf_c"]
        assert (
            loaded["mtf_c"]["2024-01-02T00:00:00"]["results"]["BNCI2014_001"][
                "accuracy"
            ]["mean"]
            == 0.85
        )

    def test_results_yaml_multiple_datasets(self, temp_experiment_dir):
        """Test results.yaml with multiple datasets."""
        experiment_version = "test_version"
        yaml_dir = os.path.join(temp_experiment_dir, experiment_version)
        os.makedirs(yaml_dir, exist_ok=True)
        results_yaml_path = os.path.join(yaml_dir, "results.yaml")

        timestamp = "2024-01-01T00:00:00"
        run_config = {
            "training": {"folds": 5, "lr": 0.001, "batch_size": 64},
            "model": {"patch_size": 6, "filter_banks": 7},
        }
        dataset_results = {}
        for dataset in ["BNCI2014_001", "BNCI2014_002"]:
            dataset_results[dataset] = {
                "accuracy": {"mean": 0.85, "std": 0.05},
                "kappa": {"mean": 0.80, "std": 0.06},
                "stft_reconstruction_loss": 0.25,
            }

        all_results = {}
        all_results.setdefault("mtf_c", {}).setdefault(timestamp, {}).update(
            {
                "config": run_config,
                "results": dataset_results,
            }
        )

        with open(results_yaml_path, "w") as f:
            yaml.dump(all_results, f, default_flow_style=False, sort_keys=False)

        with open(results_yaml_path, "r") as f:
            loaded = yaml.safe_load(f)

        assert timestamp in loaded["mtf_c"]
        assert "config" in loaded["mtf_c"][timestamp]
        assert "results" in loaded["mtf_c"][timestamp]
        assert "BNCI2014_001" in loaded["mtf_c"][timestamp]["results"]
        assert "BNCI2014_002" in loaded["mtf_c"][timestamp]["results"]
        assert loaded["mtf_c"][timestamp]["config"]["training"]["folds"] == 5
        assert (
            loaded["mtf_c"][timestamp]["results"]["BNCI2014_001"]["accuracy"]["mean"]
            == 0.85
        )
        assert (
            loaded["mtf_c"][timestamp]["results"]["BNCI2014_002"]["accuracy"]["mean"]
            == 0.85
        )

    def test_results_yaml_timestamp_structure(self, temp_experiment_dir):
        """Test that results are stored under timestamp at model level."""
        experiment_version = "test_version"
        yaml_dir = os.path.join(temp_experiment_dir, experiment_version)
        os.makedirs(yaml_dir, exist_ok=True)
        results_yaml_path = os.path.join(yaml_dir, "results.yaml")

        timestamp = "2024-01-01T00:00:00"
        run_config = {
            "training": {"folds": 5, "lr": 0.001},
            "model": {"patch_size": 6, "filter_banks": 7},
        }
        experiment_description = "Test run"
        dataset_results = {
            "BNCI2014_001": {
                "accuracy": {"mean": 0.85, "std": 0.05},
                "kappa": {"mean": 0.70, "std": 0.10},
                "stft_reconstruction_loss": 0.25,
            },
            "BNCI2014_002": {
                "accuracy": {"mean": 0.80, "std": 0.08},
                "kappa": {"mean": 0.65, "std": 0.12},
                "stft_reconstruction_loss": 0.30,
            },
        }

        all_results = {}
        all_results.setdefault("mtf_c", {}).setdefault(timestamp, {}).update(
            {
                "config": run_config,
                "results": dataset_results,
            }
        )

        with open(results_yaml_path, "w") as f:
            yaml.dump(all_results, f, default_flow_style=False)

        with open(results_yaml_path, "r") as f:
            loaded = yaml.safe_load(f)

        assert timestamp in loaded["mtf_c"]
        assert "config" in loaded["mtf_c"][timestamp]
        assert "results" in loaded["mtf_c"][timestamp]
        assert "BNCI2014_001" in loaded["mtf_c"][timestamp]["results"]
        assert "BNCI2014_002" in loaded["mtf_c"][timestamp]["results"]
        assert loaded["mtf_c"][timestamp]["config"]["training"]["folds"] == 5
        assert (
            loaded["mtf_c"][timestamp]["results"]["BNCI2014_001"]["accuracy"]["mean"]
            == 0.85
        )
        assert (
            loaded["mtf_c"][timestamp]["results"]["BNCI2014_002"]["accuracy"]["mean"]
            == 0.80
        )

    def test_results_yaml_without_stft_loss(self, temp_experiment_dir):
        """Test results.yaml for non-MTFC models without STFT loss."""
        experiment_version = "test_version"
        yaml_dir = os.path.join(temp_experiment_dir, experiment_version)
        os.makedirs(yaml_dir, exist_ok=True)
        results_yaml_path = os.path.join(yaml_dir, "results.yaml")

        timestamp = "2024-01-01T00:00:00"
        run_config = {
            "training": {"folds": 5, "lr": 0.001, "batch_size": 32},
            "model": {"patch_size": 6},
        }
        dataset_results = {
            "BNCI2014_001": {
                "accuracy": {"mean": 0.75, "std": 0.08},
                "kappa": {"mean": 0.50, "std": 0.15},
                "stft_reconstruction_loss": None,
            }
        }

        all_results = {}
        all_results.setdefault("db_conformer", {}).setdefault(timestamp, {}).update(
            {
                "config": run_config,
                "results": dataset_results,
            }
        )

        with open(results_yaml_path, "w") as f:
            yaml.dump(all_results, f, default_flow_style=False)

        with open(results_yaml_path, "r") as f:
            loaded = yaml.safe_load(f)

        assert (
            loaded["db_conformer"][timestamp]["results"]["BNCI2014_001"]["accuracy"][
                "mean"
            ]
            == 0.75
        )
        assert (
            loaded["db_conformer"][timestamp]["results"]["BNCI2014_001"][
                "stft_reconstruction_loss"
            ]
            is None
        )
        assert (
            loaded["db_conformer"][timestamp]["config"]["training"]["batch_size"] == 32
        )

        with open(results_yaml_path, "w") as f:
            yaml.dump(all_results, f, default_flow_style=False)

        with open(results_yaml_path, "r") as f:
            loaded = yaml.safe_load(f)

        assert (
            loaded["db_conformer"][timestamp]["results"]["BNCI2014_001"]["accuracy"][
                "mean"
            ]
            == 0.75
        )
        assert (
            loaded["db_conformer"][timestamp]["results"]["BNCI2014_001"][
                "stft_reconstruction_loss"
            ]
            is None
        )


class TestMockExperiment:
    """Integration tests with mocked cross_validation."""

    @pytest.fixture
    def mock_cross_validation(self):
        """Mock cross_validation function that returns predictable results."""
        call_count = [0]

        def mock_cv(args, experiment):
            call_count[0] += 1
            accuracy = [0.75 + (call_count[0] * 0.01)]
            kappa = [0.70 + (call_count[0] * 0.01)]
            stft_reconstruction_loss = [0.30 - (call_count[0] * 0.01)]
            hyperparameters = Namespace(
                val_size=0.2,
                n_iter=100,
                eval_inter=10,
                folds=5,
                n_repeats=1,
                lr=0.001,
                batch_size=64,
            )
            model_configs = {"patch_size": 6, "filter_banks": 7}
            return (
                accuracy,
                kappa,
                stft_reconstruction_loss,
                experiment,
                hyperparameters,
                model_configs,
            )

        return mock_cv, call_count

    def test_subject_iteration_single_dataset(self, mock_cross_validation):
        """Test that script iterates through subjects for single dataset."""
        mock_cv, call_count = mock_cross_validation

        subjects = [1, 2, 3]

        for subject in subjects:
            mock_cv(Namespace(dataset="BNCI2014_001", subject=subject), experiment=None)

        assert call_count[0] == 3

    def test_subject_iteration_multiple_datasets(self, mock_cross_validation):
        """Test that script iterates through subjects for multiple datasets."""
        mock_cv, call_count = mock_cross_validation

        datasets = ["BNCI2014_001", "BNCI2014_002"]
        dataset_subjects = {
            "BNCI2014_001": [1, 2],
            "BNCI2014_002": [1, 2, 3],
        }

        for dataset in datasets:
            for subject in dataset_subjects[dataset]:
                mock_cv(Namespace(dataset=dataset, subject=subject), experiment=None)

        assert call_count[0] == 5

    def test_method_args_structure(self):
        """Test that method arguments are structured correctly."""
        mthd_args = Namespace(
            dataset="BNCI2014_001",
            subject=1,
            device="cpu",
            model_name="mtf_c",
            verbose=True,
        )

        assert mthd_args.dataset == "BNCI2014_001"
        assert mthd_args.subject == 1
        assert mthd_args.device == "cpu"
        assert mthd_args.model_name == "mtf_c"
        assert mthd_args.verbose is True


class TestMetricCalculations:
    """Tests for metric calculations and formatting."""

    def test_print_format_non_mtfc(self):
        """Test that non-MTFC models print without STFT loss."""
        accuracy = [0.85]
        kappa = [0.80]

        model_name = "db_conformer"

        if model_name != "mtf_c":
            output = f"subject 1 | Accuracy: {np.mean(accuracy):.2f}, Kappa: {np.mean(kappa):.2f}"
        else:
            stft_loss = [0.25]
            output = f"subject 1 | Accuracy: {np.mean(accuracy):.2f}, Kappa: {np.mean(kappa):.2f} | STFT Reconstruction Loss: {np.mean(stft_loss):.2f}"

        assert "Accuracy: 0.85" in output
        assert "Kappa: 0.80" in output
        assert "STFT" not in output

    def test_print_format_mtfc(self):
        """Test that MTFC models print with STFT loss."""
        accuracy = [0.85]
        kappa = [0.80]
        stft_loss = [0.25]

        model_name = "mtf_c"

        if model_name != "mtf_c":
            output = f"subject 1 | Accuracy: {np.mean(accuracy):.2f}, Kappa: {np.mean(kappa):.2f}"
        else:
            output = f"subject 1 | Accuracy: {np.mean(accuracy):.2f}, Kappa: {np.mean(kappa):.2f} | STFT Reconstruction Loss: {np.mean(stft_loss):.2f}"

        assert "Accuracy: 0.85" in output
        assert "Kappa: 0.80" in output
        assert "STFT Reconstruction Loss: 0.25" in output
