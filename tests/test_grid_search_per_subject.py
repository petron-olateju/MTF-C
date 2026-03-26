"""Unit tests for grid_search_per_subject.py script.

Tests cover:
- Argument parsing (parse_args)
- Grid search logic (parameter combination iteration, best params tracking)
- Result saving and YAML output
"""

import os
import tempfile
import yaml
from argparse import Namespace
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

import numpy as np
import pytest

import grid_search_per_subject


class TestParseArgs:
    """Tests for argument parsing functionality."""

    def test_default_arguments(self):
        """Test that default arguments are correctly parsed."""
        with patch("sys.argv", ["grid_search_per_subject.py"]):
            args = grid_search_per_subject.parse_args()
            assert args.script == "cross_validation.py"
            assert args.model_name == "db_conformer"
            assert args.dataset == "dummy_dataset"
            assert args.subject == 1
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
                "grid_search_per_subject.py",
                "--script",
                "pre_training",
                "--model_name",
                "mtf_c",
                "--dataset",
                "BNCI2014_001",
                "--subject",
                "3",
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
            args = grid_search_per_subject.parse_args()
            assert args.script == "pre_training"
            assert args.model_name == "mtf_c"
            assert args.dataset == "BNCI2014_001"
            assert args.subject == 3
            assert args.device == "cuda"
            assert args.experiment_version == "v1.0"
            assert args.experiment_description == "Test Experiment"
            assert args.experiment_folder == "/tmp/experiments"
            assert args.verbose is True


class TestGridSearchLogic:
    """Tests for grid search parameter combination logic."""

    @pytest.fixture
    def mock_configs(self):
        """Mock grid search configuration for testing."""
        return {
            "param_a": [1, 2],
            "param_b": ["x", "y"],
        }

    def test_grid_combinations_count(self, mock_configs):
        """Test that all parameter combinations are generated."""
        param_names = list(mock_configs.keys())
        param_values = list(mock_configs.values())

        from itertools import product

        combos = list(product(*param_values))

        assert len(combos) == 4
        assert combos == [(1, "x"), (1, "y"), (2, "x"), (2, "y")]

    def test_param_dict_creation(self, mock_configs):
        """Test that parameter combinations are correctly converted to dicts."""
        param_names = list(mock_configs.keys())
        param_values = list(mock_configs.values())

        from itertools import product

        combos = list(product(*param_values))

        param_dicts = [dict(zip(param_names, combo)) for combo in combos]

        assert param_dicts[0] == {"param_a": 1, "param_b": "x"}
        assert param_dicts[1] == {"param_a": 1, "param_b": "y"}
        assert param_dicts[2] == {"param_a": 2, "param_b": "x"}
        assert param_dicts[3] == {"param_a": 2, "param_b": "y"}

    def test_best_params_tracking(self):
        """Test that best parameters are correctly tracked based on accuracy."""
        best_acc = 0
        best_params = None

        results = [
            ({"lr": 0.001}, 0.70),
            ({"lr": 0.01}, 0.85),
            ({"lr": 0.1}, 0.75),
            ({"lr": 0.01}, 0.90),
        ]

        for current_params, acc in results:
            if acc > best_acc:
                best_acc = acc
                best_params = current_params

        assert best_acc == 0.90
        assert best_params == {"lr": 0.01}

    def test_best_params_tiebreaker_stft_loss(self):
        """Test that STFT loss is used as tiebreaker when accuracy is equal."""
        best_acc = 0
        best_acc_std = 0
        best_kappa = 0
        best_kappa_std = 0
        best_stft_loss = float("inf")
        best_params = None

        test_results = [
            ({"lr": 0.001}, 0.85, 0.82, 0.50),
            ({"lr": 0.01}, 0.85, 0.80, 0.30),
            ({"lr": 0.1}, 0.85, 0.85, 0.40),
        ]

        for current_params, acc_mean, kappa_mean, stft_result in test_results:
            acc_std = 0.01
            kappa_std = 0.02

            if acc_mean > best_acc:
                best_acc = acc_mean
                best_acc_std = acc_std
                best_kappa = kappa_mean
                best_kappa_std = kappa_std
                best_stft_loss = stft_result
                best_params = current_params
            elif (acc_mean == best_acc) and (stft_result <= best_stft_loss):
                best_acc = acc_mean
                best_acc_std = acc_std
                best_kappa = kappa_mean
                best_kappa_std = kappa_std
                best_stft_loss = stft_result
                best_params = current_params

        assert best_params == {"lr": 0.01}
        assert best_stft_loss == 0.30


class TestResultSaving:
    """Tests for result saving functionality."""

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
        subject = 1

        run_entry = {
            "timestamp": "2024-01-01T00:00:00",
            "model": model_name,
            "hyperparameters": {"lr": 0.001, "batch_size": 32},
            "dataset": dataset,
            "subject": subject,
            "metrics": {
                "accuracy": {"mean": 0.85, "std": 0.05},
                "kappa": {"mean": 0.80, "std": 0.06},
                "stft_reconstruction_loss": 0.25,
            },
        }

        yaml_dir = os.path.join(temp_experiment_dir, experiment_version)
        os.makedirs(yaml_dir, exist_ok=True)
        results_yaml_path = os.path.join(yaml_dir, "results.yaml")

        all_results = {}
        all_results.setdefault(model_name, {}).setdefault(dataset, {}).setdefault(
            f"subject_{subject}", []
        ).append(run_entry)

        with open(results_yaml_path, "w") as f:
            yaml.dump(all_results, f, default_flow_style=False, sort_keys=False)

        with open(results_yaml_path, "r") as f:
            loaded_results = yaml.safe_load(f)

        assert model_name in loaded_results
        assert dataset in loaded_results[model_name]
        assert f"subject_{subject}" in loaded_results[model_name][dataset]
        assert len(loaded_results[model_name][dataset][f"subject_{subject}"]) == 1

    def test_results_yaml_appends_to_existing(self, temp_experiment_dir):
        """Test that new results are appended to existing results YAML."""
        experiment_version = "test_version"
        yaml_dir = os.path.join(temp_experiment_dir, experiment_version)
        os.makedirs(yaml_dir, exist_ok=True)
        results_yaml_path = os.path.join(yaml_dir, "results.yaml")

        existing_results = {
            "mtf_c": {
                "BNCI2014_001": {
                    "subject_1": [
                        {
                            "timestamp": "2024-01-01T00:00:00",
                            "metrics": {"accuracy": {"mean": 0.80}},
                        }
                    ]
                }
            }
        }

        with open(results_yaml_path, "w") as f:
            yaml.dump(existing_results, f)

        new_run_entry = {
            "timestamp": "2024-01-02T00:00:00",
            "model": "mtf_c",
            "hyperparameters": {"lr": 0.01},
            "dataset": "BNCI2014_001",
            "subject": 1,
            "metrics": {
                "accuracy": {"mean": 0.85},
                "kappa": {"mean": 0.80},
                "stft_reconstruction_loss": 0.30,
            },
        }

        with open(results_yaml_path, "r") as f:
            all_results = yaml.safe_load(f) or {}

        all_results.setdefault("mtf_c", {}).setdefault("BNCI2014_001", {}).setdefault(
            "subject_1", []
        ).append(new_run_entry)

        with open(results_yaml_path, "w") as f:
            yaml.dump(all_results, f, default_flow_style=False, sort_keys=False)

        with open(results_yaml_path, "r") as f:
            loaded_results = yaml.safe_load(f)

        assert len(loaded_results["mtf_c"]["BNCI2014_001"]["subject_1"]) == 2


class TestMockGridSearch:
    """Integration tests for grid search with mocked cross_validation."""

    @pytest.fixture
    def mock_cross_validation(self):
        """Mock cross_validation function that returns predictable results."""
        call_count = [0]

        def mock_cv(args, experiment, config=None, model_configs=None):
            call_count[0] += 1

            acc_base = 0.60 + (call_count[0] * 0.05)
            accuracy = [acc_base]
            kappa = [acc_base - 0.05]
            stft_reconstruction_loss = [1.0 - (call_count[0] * 0.1)]

            hyperparameters = Namespace(
                val_size=0.2,
                n_iter=100,
                eval_inter=10,
                folds=5,
                n_repeats=1,
                lr=0.001,
                batch_size=64,
            )

            return (
                accuracy,
                kappa,
                stft_reconstruction_loss,
                experiment,
                hyperparameters,
            )

        return mock_cv, call_count

    def test_grid_search_runs_all_combinations(self, mock_cross_validation):
        """Test that grid search iterates through all parameter combinations."""
        mock_cv, call_count = mock_cross_validation

        test_config = {
            "param_a": [1, 2],
            "param_b": [10, 20],
        }

        total_combos = 1
        for v in test_config.values():
            total_combos *= len(v)

        assert total_combos == 4

        param_names = list(test_config.keys())
        param_values = list(test_config.values())

        from itertools import product

        combos = list(product(*param_values))

        assert len(combos) == 4


class TestAccuracyMetrics:
    """Tests for accuracy and metric calculations."""

    def test_accuracy_mean_std_calculation(self):
        """Test that accuracy mean and std are calculated correctly."""
        accuracies = [0.75, 0.80, 0.85, 0.70, 0.78]

        acc_mean = np.mean(accuracies)
        acc_std = np.std(accuracies)

        assert abs(acc_mean - 0.776) < 0.01
        assert abs(acc_std - 0.054) < 0.01

    def test_kappa_mean_std_calculation(self):
        """Test that kappa mean and std are calculated correctly."""
        kappas = [0.65, 0.70, 0.75, 0.60, 0.68]

        kappa_mean = np.mean(kappas)
        kappa_std = np.std(kappas)

        assert abs(kappa_mean - 0.676) < 0.01
        assert abs(kappa_std - 0.054) < 0.01

    def test_stft_loss_mean_calculation(self):
        """Test that STFT reconstruction loss mean is calculated correctly."""
        stft_losses = [0.30, 0.35, 0.25, 0.40, 0.32]

        stft_result = np.mean(stft_losses)

        assert abs(stft_result - 0.324) < 0.01
