# MTF-C: Multi-Scale Temporal Frequency Conformer

A deep learning framework for EEG signal classification supporting multiple BCI paradigms including Motor Imagery (MI), SSVEP, Sleep staging, and Resting State.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Validation Methods](#validation-methods)
- [Command Line Arguments](#command-line-arguments)
- [Configuration Files](#configuration-files)
- [Available Datasets](#available-datasets)
- [Examples](#examples)

---

## Overview

MTF-C provides neural network architectures for EEG-based brain-computer interface classification:

- **MTFC** (Multi-Scale Temporal Frequency Conformer): Combines spectral-spatio-temporal embeddings with transformer-based conformer
- **DBConformer**: Dual-branch Conformer for EEG classification
- **Dual TSST**: Dual Time-Scale Spatial Temporal network

Supported paradigms:
- Motor Imagery (MI)
- Steady-State Visual Evoked Potentials (SSVEP)
- Sleep staging
- Resting State analysis

---

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv sync

# Using pip
pip install -e .
```

### Basic Usage

```bash
# Run cross-validation experiment
python run_experiment.py --script cross_validation --model_name mtf_c --dataset BNCI2014_001

# Run LOSO experiment
python run_experiment.py --script loso --model_name db_conformer --dataset BNCI2014_001 --subject 1

# Run on multiple datasets
python run_experiment.py --model_name mtf_c --dataset all --experiment_version v1.0
```

---

## Validation Methods

### Cross-Validation (`cross_validation`)

Within-subject k-fold cross-validation using StratifiedKFold:

- Splits subject data into `folds` number of folds
- Repeats the process `n_repeats` times with different random seeds
- Uses StratifiedKFold to maintain class distribution across folds
- Applies Euclidean Alignment (EA) for preprocessing:
  - `EA()` on training data to compute reference covariance
  - `EA_online()` on test data using the computed reference
- For MTFC model, optionally computes STFT reconstruction loss during training
- Reports accuracy and Cohen's Kappa for each fold and repeat

### Leave-One-Subject-Out (`loso`)

Cross-subject evaluation paradigm:

- Trains on all subjects except one held-out test subject
- Tests on the held-out subject to evaluate cross-subject generalization
- Uses Euclidean Alignment (EA) with training set reference applied to test data
- Runs `n_repeats` training cycles with different seeds
- More challenging but more realistic for BCI deployment scenarios

### Pre-Training (`pre_training`)

Pre-training mode (placeholder for future implementation).

---

## Command Line Arguments

The main entry point is `run_experiment.py`:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--script` | str | `cross_validation` | Validation method: `cross_validation`, `pre_training`, `loso` |
| `--model_name` | str | `db_conformer` | Model architecture: `db_conformer`, `mtf_c`, `dual_tsst` |
| `--dataset` | str | `dummy_dataset` | Dataset name (see Available Datasets section) |
| `--device` | str | `cpu` | Compute device: `cpu`, `cuda` |
| `--experiment_version` | str | `v0.0` | Version identifier for experiment results |
| `--experiment_description` | str | `Baseline Experiment` | Human-readable description of experiment |
| `--messages` | str | `""` | Messages to save in results.yaml (split by double space) |
| `--experiment_folder` | str | `./experiments` | Directory to save experiment results |
| `--verbose` | flag | - | Print detailed training progress |

### Dataset Group Shortcuts

- `all`: All MI datasets (BNCI2014_001, BNCI2014_002, BNCI2014_004)
- `mi`: All Motor Imagery datasets
- `ssvep`: All SSVEP datasets (Kalunga2016, Nakanishi2015, Wang2021Combined)
- `sleep`: Sleep staging dataset (SleepPhysionet)
- `resting_state`: Resting state datasets (Cattan2019_PHMD, Rodrigues2017)

---

## Configuration Files

Configuration files are located in the `configs/` directory:

- `cross_validation.yaml` - Default config for cross-validation experiments
- `loso.yaml` - Default config for LOSO experiments
- `grid_search.yaml` - Config with hyperparameter search grids
- `test_cv_config.yaml` - Minimal config for testing

### Training Parameters

All training parameters are under the `training:` section:

| Parameter | Type | Possible Values | Default | Description |
|-----------|------|-----------------|---------|-------------|
| `val_size` | float | `0.0` to `1.0` | `0.0` | Validation split ratio (0.0 = no validation split) |
| `n_iter` | int | Positive integer | `100` | Number of training epochs |
| `eval_inter` | int | Positive integer | `5` | Evaluation interval (every N epochs) |
| `folds` | int | Positive integer >= 2 | `5` | Number of k-fold cross-validation folds |
| `n_repeats` | int | Positive integer | `1` | Number of times to repeat k-fold CV |
| `lr` | float | Positive float | `1.0e-3` | Learning rate |
| `batch_size` | int | Positive integer | `32` (cv), `64` (loso) | Batch size |

### Model: DBConformer Parameters

All DBConformer parameters are under the `db_conformer:` section:

| Parameter | Type | Possible Values | Default | Description |
|-----------|------|-----------------|---------|-------------|
| `patch_size` | int | `20, 40, 60, 80, 100, 120, 140` | `100` | Patch size for temporal segmentation |
| `filter_banks` | int | `7, 9, 11, 13, 15` | `11` | Number of filter banks for frequency decomposition |
| `freq_downsample` | int | Positive integer | `1` | Frequency downsampling factor |
| `wsize_divisor` | int | `2, 4` | `2` | Window size divisor for STFT |
| `spa_dim` | int | Positive integer | `16` | Spatial embedding dimension |
| `gate_flag` | bool | `true`, `false` | `false` | Enable gated fusion between branches |
| `posemb_flag` | bool | `true`, `false` | `true` | Use positional embeddings |
| `branch` | str | `all` | `all` | Branch configuration (how embeddings are combined) |
| `chn_attn_flag` | bool | `true`, `false` | `true` | Enable channel attention |
| `fts_attn_flag` | bool | `true`, `false` | `false` | Enable feature attention |
| `sst_method` | bool | `true`, `false` | `false` | Use spectral-spatio-temporal method |
| `stft_reconstruction` | bool/str | `true`, `false`, `STFT`, `frequency` | `false` | Enable STFT reconstruction loss |
| `emb_size` | int | Positive integer | `40` | Embedding size |
| `tem_depth` | int | Positive integer | `1` | Temporal transformer depth |
| `chn_depth` | int | Positive integer | `1` | Channel transformer depth |

### Model: MTFC Parameters

All MTFC parameters are under the `mtfc:` section:

| Parameter | Type | Possible Values | Default | Description |
|-----------|------|-----------------|---------|-------------|
| `patch_size` | int | Positive integer | `100` | Patch size for temporal segmentation |
| `filter_banks` | int | Positive integer | `11` | Number of filter banks |
| `filter_banks_variant` | str | `MultiTemporalConvPool_ChannelsProject_FilterBanks` | `MultiTemporalConvPool_ChannelsProject_FilterBanks` | Filter bank variant |
| `freq_downsample` | int | Positive integer | `1` | Frequency downsampling factor |
| `wsize_divisor` | int | `2, 4` | `2` | Window size divisor |
| `spa_dim` | int | Positive integer | `16` | Spatial embedding dimension |
| `gate_flag` | bool | `true`, `false` | `false` | Enable gated fusion |
| `posemb_flag` | bool | `true`, `false` | `true` | Use positional embeddings |
| `branch` | str | `all`, `ft_s`, `fs_t`, `f_t_s`, `f_ts` | `f_t_s` | Branch configuration (see below) |
| `chn_attn_flag` | bool | `true`, `false` | `false` | Enable channel attention |
| `fts_attn_flag` | bool | `true`, `false` | `false` | Enable feature attention |
| `sst_method` | str | `false`, `filter_banks`, `stf_attention_temporal_values`, `st_addition_projection` | `filter_banks` | SST computation method |
| `stft_reconstruction` | bool/str | `true`, `false`, `STFT`, `frequency` | `frequency` | Enable STFT reconstruction loss |
| `ct_shared_projection` | bool | `true`, `false` | `false` | Use shared projection in spectrogram estimator |
| `sst_shared_projection` | bool | `true`, `false` | `false` | Project embeddings to SST dimension |
| `patch_emb_size` | int | Positive integer | `40` | Patch embedding dimension |
| `n_heads_patch` | int | Positive integer | `4` | Number of attention heads for patch transformer |
| `sst_emb_size` | int | Positive integer | `40` | SST embedding dimension |
| `tem_depth` | int | Positive integer | `1` | Temporal transformer depth |
| `chn_depth` | int | Positive integer | `1` | Channel transformer depth |
| `temporal_kernel` | int | Positive odd integer | `43` | Temporal convolution kernel size |

### Branch Configuration Options

The `branch` parameter defines how frequency (f), temporal (t), and spatial (s) embeddings are processed:

| Value | Description |
|-------|-------------|
| `all` | Legacy single transformer for all embeddings (FTS concatenated) |
| `ft_s` | One transformer processes freq-temporal (concatenated), then concatenated with spatial |
| `f_t_s` | Three separate transformers for freq, temporal, spatial, then concatenated |
| `fs_t` | One transformer processes freq-spatial, then concatenated with temporal |
| `f_ts` | One transformer processes freq, then concatenated with temporal-spatial |

### SST Method Options

| Value | Description |
|-------|-------------|
| `false` | No SST method |
| `filter_banks` | Use filter banks for spectral decomposition |
| `stf_attention_temporal_values` | Use spatio-temporal to temporal cross-attention |
| `st_addition_projection` | Add and project SST embeddings |

### STFT Reconstruction Options

| Value | Description |
|-------|-------------|
| `false` | No reconstruction loss |
| `true` | Same as `STFT` |
| `STFT` | Use Short-Time Fourier Transform for reconstruction |
| `frequency` | Use band power features for reconstruction |

### Grid Search Configuration

The `grid_search.yaml` allows specifying multiple values for hyperparameters:

```yaml
db_conformer:
  patch_size: [20, 40, 60, 80, 100, 120, 140]
  filter_banks: [7, 9, 11, 13, 15]
  wsize_divisor: [2, 4]
  ...

mtf_c:
  patch_size: [150]
  filter_banks: [151]
  sst_method: ['st_addition_projection']
  ...
```

---

## Available Datasets

### Motor Imagery (MI)

| Dataset ID | Description | Sessions | Movements |
|------------|-------------|----------|-----------|
| `BNCI2014_001` | BCI Competition IV dataset 2a (left/right hand MI) | 2 | 4 (left hand, right hand, foot, tongue) |
| `BNCI2014_002` | BCI Competition IV dataset 2b | 2 | 2 (left/right hand) |
| `BNCI2014_004` | BCI Competition IV dataset 2a (extended) | 2 | 4 |
| `BNCI2015_001` | BCI Competition V dataset 2a | 2 | 2 (left/right hand) |
| `BNCI2015_004` | BCI Competition V dataset 2b | 2 | 2 (left/right hand) |
| `AlexMI` | AlexNet-based MI dataset | 2 | - |
| `Liu2024` | Liu et al. 2024 dataset | - | - |

### SSVEP

| Dataset ID | Description | Target Frequencies |
|------------|-------------|-------------------|
| `Kalunga2016` | Kalunga SSVEP dataset | - |
| `MAMEM2` | MAMEM SSVEP dataset 2 | - |
| `MAMEM3` | MAMEM SSVEP dataset 3 | - |
| `Nakanishi2015` | Nakanishi SSVEP dataset | - |
| `Wang2021Combined` | Wang 2021 combined SSVEP | - |

### Sleep Staging

| Dataset ID | Description | Stages |
|------------|-------------|--------|
| `SleepPhysionet` | PhysioNet Sleep EDF dataset | Wake, N1, N2, N3, REM |

### Resting State

| Dataset ID | Description |
|------------|-------------|
| `Cattan2019_PHMD` | Cattan 2019 PHMD dataset |
| `Hinss2021` | Hinss 2021 dataset |
| `Rodrigues2017` | Rodrigues 2017 dataset |
| `ButtonToneSZ` | Button Tone dataset |

### Other

| Dataset ID | Description |
|------------|-------------|
| `dummy_dataset` | Randomly generated dummy data for testing (5 samples, 3 channels, 1000 timepoints) |

---

## Examples

### Example 1: Basic Cross-Validation

```bash
python run_experiment.py \
  --script cross_validation \
  --model_name mtf_c \
  --dataset BNCI2014_001 \
  --experiment_version v1.0
```

### Example 2: LOSO with CUDA

```bash
python run_experiment.py \
  --script loso \
  --model_name db_conformer \
  --dataset BNCI2014_001 \
  --subject 1 \
  --device cuda \
  --experiment_version v1.0_loso
```

### Example 3: Run Multiple MI Datasets

```bash
python run_experiment.py \
  --model_name mtf_c \
  --dataset mi \
  --experiment_description "Motor Imagery Benchmark" \
  --experiment_version v2.0
```

### Example 4: Run All Datasets

```bash
python run_experiment.py \
  --model_name db_conformer \
  --dataset all \
  --experiment_version v_full_benchmark
```

### Example 5: Custom Experiment Folder

```bash
python run_experiment.py \
  --script cross_validation \
  --model_name mtf_c \
  --dataset BNCI2014_001 \
  --experiment_folder ./my_experiments \
  --experiment_version custom_v1
```

### Example 6: SSVEP Dataset

```bash
python run_experiment.py \
  --script cross_validation \
  --model_name db_conformer \
  --dataset Nakanishi2015 \
  --verbose
```

### Example 7: Sleep Dataset

```bash
python run_experiment.py \
  --script loso \
  --model_name mtf_c \
  --dataset SleepPhysionet \
  --experiment_version sleep_v1
```

### Example 8: Run with Custom Configuration

Edit the config file in `configs/cross_validation.yaml` or `configs/loso.yaml` before running:

```bash
# Modify configs/cross_validation.yaml
# Then run
python run_experiment.py \
  --script cross_validation \
  --model_name mtf_c \
  --dataset BNCI2014_001
```

---

## Output Files

Experiment results are saved to `./experiments/{experiment_version}/`:

- `accuracy.results.csv` - Mean accuracy per dataset and model
- `accuracy_std.results.csv` - Standard deviation of accuracy
- `kappa.results.csv` - Cohen's Kappa scores
- `kappa_std.results.csv` - Standard deviation of Kappa
- `stft_reconstruction.results.csv` - STFT reconstruction loss (MTFC only)
- `results.yaml` - Full results in YAML format including hyperparameters and messages

---

## Requirements

- Python >= 3.12
- PyTorch
- MNE >= 1.11.0
- Braindecode >= 1.0.0
- MOABB >= 1.5.0
- scikit-learn >= 1.8.0
- einops >= 0.8.2
- timm >= 1.0.25

See `pyproject.toml` for full dependencies.
