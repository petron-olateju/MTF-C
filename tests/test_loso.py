"""Unit tests for LOSO (Leave-One-Subject-Out) module."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LOSO import parse_args, save_loso_csv, load_all_subjects_data


class TestParseArgs:
    """Tests for LOSO argument parsing."""

    def test_default_arguments(self):
        """Test default argument values."""
        import sys

        old_argv = sys.argv
        try:
            sys.argv = ["LOSO.py"]
            args = parse_args()
            assert args.dataset == "dummy_dataset"
            assert args.model_name == "db_conformer"
            assert args.device == "cpu"
        finally:
            sys.argv = old_argv

    def test_script_choices(self):
        """Test that loso is a valid script choice."""
        import sys
        from run_experiment import parse_args as re_parse_args

        old_argv = sys.argv
        try:
            sys.argv = [
                "run_experiment.py",
                "--script",
                "loso",
                "--dataset",
                "dummy_dataset",
            ]
            args = re_parse_args()
            assert args.script == "loso"
        finally:
            sys.argv = old_argv

    def test_subjects_list_argument(self):
        """Test --subjects-list argument parsing."""
        import sys
        from LOSO import parse_args

        old_argv = sys.argv
        try:
            sys.argv = [
                "LOSO.py",
                "--subjects-list",
                "1,2,3,4,5",
                "--dataset",
                "dummy_dataset",
            ]
            args = parse_args()
            assert args.subjects_list == "1,2,3,4,5"
        finally:
            sys.argv = old_argv


class TestLoadAllSubjectsData:
    """Tests for loading all subjects data."""

    def test_dummy_dataset(self):
        """Test loading dummy dataset returns expected structure."""
        data = load_all_subjects_data("dummy_dataset", [])

        assert isinstance(data, dict)
        assert 1 in data
        X, y, info = data[1]
        assert X.shape == (5, 3, 1000)
        assert y.shape == (5,)
        assert info["n_ch"] == 3
        assert info["n_times"] == 1000
        assert info["n_classes"] == 2


class TestSaveLosoCsv:
    """Tests for saving LOSO results to CSV."""

    def test_save_csv(self, tmp_path):
        """Test that CSV is saved correctly."""
        subject_accuracies = [0.80, 0.70, 0.90, 0.75]
        subject_kappas = [0.60, 0.40, 0.80, 0.50]
        subjects = [1, 2, 3, 4]

        output_path = tmp_path / "test_loso.csv"
        save_loso_csv(subject_accuracies, subject_kappas, subjects, str(output_path))

        assert output_path.exists()

        content = output_path.read_text()
        lines = content.strip().split("\n")

        assert lines[0] == "subject,accuracy,kappa"
        assert lines[1] == "1,0.8000,0.6000"
        assert lines[2] == "2,0.7000,0.4000"
        assert lines[3] == "3,0.9000,0.8000"
        assert lines[4] == "4,0.7500,0.5000"

    def test_csv_format(self, tmp_path):
        """Test CSV has correct number of rows."""
        subject_accuracies = [0.5, 0.6, 0.7]
        subject_kappas = [0.1, 0.2, 0.3]
        subjects = [10, 20, 30]

        output_path = tmp_path / "test_loso2.csv"
        save_loso_csv(subject_accuracies, subject_kappas, subjects, str(output_path))

        content = output_path.read_text()
        lines = content.strip().split("\n")

        assert len(lines) == 4  # header + 3 subjects


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
