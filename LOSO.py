import yaml
from tqdm import tqdm
from typing import Tuple, List, Union, Dict
import csv

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import mne

mne.set_log_level("WARNING")
import scipy
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.metrics import accuracy_score
from utils.preprocessing import EA, EA_online, bandpass_filtering, compute_band_powers
from utils.data_loader import (
    EEGDataset,
    MI_DataLoader,
    SSVEP_DataLoader,
    Sleep_Loader,
    RestingState_DataLoader,
)
from utils.experiment_recorder import Parameter, Experiment
from models.DBConformer import DBConformer
from models.MTFC import MTFC

import argparse
from argparse import Namespace


def parse_args():
    parser = argparse.ArgumentParser(
        description="Model, Hyperparameters, ExperimentLogger options"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="db_conformer",
        choices=["db_conformer", "mtf_c", "dual_tsst"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dummy_dataset",
        choices=[
            "dummy_dataset",
            "BNCI2014_001",
            "BNCI2014_002",
            "BNCI2014_004",
            "BNCI2015_001",
            "BNCI2015_004",
            "Liu2024",
            "AlexMI",
            "Kalunga2016",
            "MAMEM2",
            "MAMEM3",
            "Nakanishi2015",
            "Wang2021Combined",
            "SleepPhysionet",
            "Cattan2019_PHMD",
            "Hinss2021",
            "Rodrigues2017",
            "ButtonToneSZ",
        ],
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--verbose", action="store_true", help="Print training progress"
    )
    parser.add_argument(
        "--experiment_folder",
        type=str,
        default="./experiments",
        help="Folder to save experiment results",
    )
    parser.add_argument(
        "--experiment_version",
        type=str,
        default=None,
        help="Experiment version name for versioning results",
    )
    parser.add_argument(
        "--subjects-list",
        type=str,
        default=None,
        help="Comma-separated list of subject IDs to use (e.g., '1,2,3'). If None, uses all subjects.",
    )

    return parser.parse_args()


def load_all_subjects_data(dataset_name: str, preprocessing_pipeline: List) -> Dict:
    """Load data for all subjects in a dataset.

    Args:
        dataset_name: Name of the dataset
        preprocessing_pipeline: List of preprocessing functions

    Returns:
        Dictionary mapping subject_id to (X, y, dataset_info)
    """
    RESTING_STATE_DATASETS = [
        "Cattan2019_PHMD",
        "Hinss2021",
        "Rodrigues2017",
        "ButtonToneSZ",
    ]
    SSVEP_DATASETS = ["Kalunga2016", "Nakanishi2015", "Wang2021Combined"]
    SLEEP_DATASETS = ["SleepPhysionet"]

    all_data = {}

    if dataset_name == "dummy_dataset":
        return {
            1: (
                np.random.randn(5, 3, 1000),
                np.random.randint(0, 2, size=5),
                {"n_ch": 3, "n_times": 1000, "n_classes": 2, "fs": 250},
            )
        }

    if dataset_name in RESTING_STATE_DATASETS:
        subjects = RestingState_DataLoader.get_subjects(dataset_name)
        for subject in tqdm(subjects, desc=f"Loading {dataset_name} subjects"):
            try:
                loader = RestingState_DataLoader(
                    dataset_name=dataset_name,
                    subject=subject,
                    preprocessing_pipeline=preprocessing_pipeline,
                    tmin=10,
                    tmax=50,
                    fmin=1,
                    fmax=35,
                    resample=128,
                )
                X, y, dataset_info = loader.get_data()
                all_data[subject] = (X, y, dataset_info)
            except Exception as e:
                print(f"Warning: Could not load subject {subject}: {e}")
                continue

    elif dataset_name in SLEEP_DATASETS:
        subjects = Sleep_Loader.get_subjects(dataset_name)
        for subject in tqdm(subjects, desc=f"Loading {dataset_name} subjects"):
            try:
                loader = Sleep_Loader(
                    dataset_name=dataset_name,
                    subject=subject,
                    preprocessing_pipeline=preprocessing_pipeline,
                )
                X, y, dataset_info = loader.get_data()
                all_data[subject] = (X, y, dataset_info)
            except Exception as e:
                print(f"Warning: Could not load subject {subject}: {e}")
                continue

    elif dataset_name in SSVEP_DATASETS:
        subjects = SSVEP_DataLoader.get_subjects(dataset_name)
        for subject in tqdm(subjects, desc=f"Loading {dataset_name} subjects"):
            try:
                loader = SSVEP_DataLoader(
                    dataset_name=dataset_name,
                    subject=subject,
                    preprocessing_pipeline=preprocessing_pipeline,
                    t0=0.0,
                    tmax=4.0,
                )
                X, y, dataset_info = loader.get_data()
                all_data[subject] = (X, y, dataset_info)
            except Exception as e:
                print(f"Warning: Could not load subject {subject}: {e}")
                continue

    else:
        subjects = MI_DataLoader.get_subjects(dataset_name)
        for subject in tqdm(subjects, desc=f"Loading {dataset_name} subjects"):
            try:
                loader = MI_DataLoader(
                    dataset_name=dataset_name,
                    subject=subject,
                    preprocessing_pipeline=preprocessing_pipeline,
                    t0=0.5,
                    t1=3.5,
                )
                X, y, dataset_info = loader.get_data()
                all_data[subject] = (X, y, dataset_info)
            except Exception as e:
                print(f"Warning: Could not load subject {subject}: {e}")
                continue

    return all_data


def main(
    args=None,
    experiment: Union["Experiment", None] = None,
    config=None,
    model_configs=None,
    save_yaml: bool = True,
):
    if args is None:
        args = parse_args()
    device = args.device

    PREPROCESSING = [bandpass_filtering]

    RESTING_STATE_DATASETS = [
        "Cattan2019_PHMD",
        "Hinss2021",
        "Rodrigues2017",
        "ButtonToneSZ",
    ]
    SSVEP_DATASETS = ["Kalunga2016", "Nakanishi2015", "Wang2021Combined"]
    SLEEP_DATASETS = ["SleepPhysionet"]

    all_subject_data = load_all_subjects_data(args.dataset, PREPROCESSING)

    subjects = sorted(all_subject_data.keys())

    if getattr(args, "subjects_list", None) is not None:
        subject_ids = [int(s.strip()) for s in args.subjects_list.split(",")]
        subjects = [s for s in subjects if s in subject_ids]

    if len(subjects) == 0:
        raise ValueError("No subjects available for LOSO validation")

    reference_info = all_subject_data[subjects[0]][2]

    if config is None:
        with open("./configs/loso.yaml", "r") as f:
            configs = yaml.safe_load(f)

        training_configs = configs["training"]
        model_configs = configs[args.model_name]
    else:
        with open(f"./configs/{config}.yaml", "r") as f:
            configs = yaml.safe_load(f)

        training_configs = configs["training"]
        if model_configs is None:
            model_configs = configs[args.model_name]

    hyperparameters = Namespace(
        val_size=training_configs["val_size"],
        n_iter=training_configs["n_iter"],
        eval_inter=training_configs["eval_inter"],
        folds=training_configs["folds"],
        n_repeats=training_configs["n_repeats"],
        lr=training_configs["lr"],
        batch_size=training_configs["batch_size"],
    )

    if experiment is not None:
        experiment.add_params(
            [
                Parameter(
                    hyperparameters.val_size,
                    "validation_size",
                    "Training-Hyperparameters",
                ),
                Parameter(hyperparameters.n_iter, "n_iter", "Training-Hyperparameters"),
                Parameter(
                    hyperparameters.eval_inter,
                    "eval_inter",
                    "Training-Hyperparameters",
                ),
                Parameter(hyperparameters.folds, "k-folds", "Training-Hyperparameters"),
                Parameter(
                    hyperparameters.n_repeats,
                    "k-folds-repeat",
                    "Training-Hyperparameters",
                ),
                Parameter(
                    hyperparameters.lr, "learning_rate", "Training-Hyperparameters"
                ),
                Parameter(
                    hyperparameters.batch_size,
                    "batch_size",
                    "Training-Hyperparameters",
                ),
                Parameter(
                    {"n_subjects": len(subjects)}, args.dataset, "Dataset-Details"
                ),
                Parameter(reference_info, args.dataset, "Dataset-Details"),
            ]
        )

        if args.model_name == "db_conformer":
            experiment.add_params(
                [
                    Parameter(
                        model_configs["emb_size"],
                        "embedding_size",
                        "Model-Hyperparameters",
                    )
                ]
            )
        elif args.model_name == "mtf_c":
            experiment.add_params(
                [
                    Parameter(
                        model_configs["patch_emb_size"],
                        "patch_embedding_size",
                        "Model-Hyperparameters",
                    ),
                    Parameter(
                        model_configs["sst_emb_size"],
                        "sst_embedding_size",
                        "Model-Hyperparameters",
                    ),
                    Parameter(
                        model_configs["filter_banks"],
                        "filter_banks",
                        "Model-Hyperparameters",
                    ),
                    Parameter(
                        model_configs["wsize_divisor"],
                        "wsize_divisor",
                        "Model-Hyperparameters",
                    ),
                    Parameter(
                        model_configs["freq_downsample"],
                        "freq_downsample",
                        "Model-Hyperparameters",
                    ),
                    Parameter(
                        model_configs["sst_method"],
                        "sst_method",
                        "Model-Hyperparameters",
                    ),
                ]
            )

    model_args = Namespace(
        data_name=args.dataset,
        chn=reference_info["n_ch"],
        time_sample_num=reference_info["n_times"],
        class_num=reference_info["n_classes"],
        patch_size=model_configs["patch_size"],
        spa_dim=model_configs["spa_dim"],
        gate_flag=model_configs["gate_flag"],
        posemb_flag=model_configs["posemb_flag"],
        branch=model_configs["branch"],
        chn_attn_flag=model_configs["chn_attn_flag"],
        fts_attn_flag=model_configs["fts_attn_flag"],
        sst_method=model_configs["sst_method"],
        stft_reconstruction=model_configs["stft_reconstruction"],
        ct_shared_projection=model_configs.get("ct_shared_projection", True),
        sst_shared_projection=model_configs.get("sst_shared_projection", True),
    )

    if experiment is not None:
        experiment.add_params(
            [
                Parameter(model_args.patch_size, "patch_size", "Model-Hyperparameters"),
                Parameter(
                    model_args.spa_dim,
                    "spatial_dimensionality",
                    "Model-Hyperparameters",
                ),
                Parameter(
                    model_configs["tem_depth"],
                    "temporal_depth",
                    "Model-Hyperparameters",
                ),
                Parameter(
                    model_configs["chn_depth"], "channel_depth", "Model-Hyperparameters"
                ),
            ]
        )

    val_size = hyperparameters.val_size
    n_iter = hyperparameters.n_iter
    eval_inter = hyperparameters.eval_inter

    all_subject_accuracies = []
    all_subject_kappas = []

    for seed in tqdm(
        range(1, hyperparameters.n_repeats + 1),
        total=hyperparameters.n_repeats,
        desc="LOSO Repeats",
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        subject_accuracies = []
        subject_kappas = []

        for test_subject in tqdm(subjects, desc="LOSO Validation"):
            train_subjects = [s for s in subjects if s != test_subject]

            X_train_list = [all_subject_data[s][0] for s in train_subjects]
            y_train_list = [all_subject_data[s][1] for s in train_subjects]

            X_train = np.vstack(X_train_list)
            y_train = np.concatenate(y_train_list)

            X_test, y_test = all_subject_data[test_subject][:2]

            if val_size > 0.0:
                sss = StratifiedShuffleSplit(
                    n_splits=1, test_size=val_size, random_state=42
                )
                for train_idx, val_idx in sss.split(X_train, y_train):
                    X_tr, X_val = X_train[train_idx], X_train[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]
            else:
                X_tr, y_tr = X_train, y_train

            if args.model_name == "db_conformer":
                model = DBConformer(
                    model_args,
                    emb_size=model_configs["emb_size"],
                    tem_depth=model_configs["tem_depth"],
                    chn_depth=model_configs["chn_depth"],
                    chn=reference_info["n_ch"],
                    n_classes=reference_info["n_classes"],
                )
                model = model.to(device)
            elif args.model_name == "mtf_c":
                model = MTFC(
                    model_args,
                    n_filter_banks=model_configs["filter_banks"],
                    patch_emb_size=model_configs["patch_emb_size"],
                    n_heads_patch=model_configs["n_heads_patch"],
                    wsize_divisor=model_configs["wsize_divisor"],
                    freq_downsample=model_configs["freq_downsample"],
                    n_times=reference_info["n_times"],
                    sst_emb_size=model_configs["sst_emb_size"],
                    depth=model_configs["tem_depth"],
                    n_classes=reference_info["n_classes"],
                    fs=reference_info["fs"],
                )
                model = model.to(device)
            else:
                model = DBConformer(
                    model_args,
                    emb_size=model_configs["emb_size"],
                    tem_depth=model_configs["tem_depth"],
                    chn_depth=model_configs["chn_depth"],
                    chn=reference_info["n_ch"],
                    n_classes=reference_info["n_classes"],
                )
                model = model.to(device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=hyperparameters.lr,
                betas=(0.9, 0.99),
                weight_decay=0,
            )

            _x_test = X_test
            _y_test = y_test

            _x_train, _y_train = X_tr, y_tr

            loss_fn = nn.CrossEntropyLoss()

            _x_train, sqrtRefEA = EA(_x_train)
            _x_test = EA_online(_x_test, sqrtRefEA)

            stft_reconstruction_type = model_configs.get("stft_reconstruction", False)

            if stft_reconstruction_type == "STFT" or stft_reconstruction_type is True:
                F_cfg = model_configs["filter_banks"]
                P_cfg = model_configs["patch_size"]
                wsize = int((F_cfg - 1) * 2)
                assert wsize % 2 == 0
                tstep = wsize // 2
                _stft_train = np.array(
                    [mne.time_frequency.stft(x, wsize, tstep) for x in _x_train]
                )
                _stft_test = np.array(
                    [mne.time_frequency.stft(x, wsize, tstep) for x in _x_test]
                )
                _stft_train = abs(_stft_train)
                _stft_test = abs(_stft_test)
                target_T = (reference_info["n_times"] - 1) // P_cfg
                _stft_train = _stft_train[:, :, :, :target_T]
                _stft_test = _stft_test[:, :, :, :target_T]
                freq_downsample = model_configs["freq_downsample"]
                F_bins_trimmed = (
                    _stft_train.shape[2] // freq_downsample
                ) * freq_downsample
                _stft_train = (
                    _stft_train[:, :, :F_bins_trimmed, :]
                    .reshape(
                        _stft_train.shape[0],
                        _stft_train.shape[1],
                        -1,
                        freq_downsample,
                        _stft_train.shape[3],
                    )
                    .mean(axis=3)
                )
                _stft_test = (
                    _stft_test[:, :, :F_bins_trimmed, :]
                    .reshape(
                        _stft_test.shape[0],
                        _stft_test.shape[1],
                        -1,
                        freq_downsample,
                        _stft_test.shape[3],
                    )
                    .mean(axis=3)
                )
            elif stft_reconstruction_type == "frequency":
                F_cfg = model_configs["filter_banks"]
                _stft_train = compute_band_powers(
                    _x_train, n_filter_banks=F_cfg, fs=reference_info["fs"]
                )
                _stft_test = compute_band_powers(
                    _x_test, n_filter_banks=F_cfg, fs=reference_info["fs"]
                )
            else:
                _stft_train = np.zeros((len(_x_train), 1, 1, 1), dtype=np.float32)
                _stft_test = np.zeros((len(_x_test), 1, 1, 1), dtype=np.float32)

            train_loader = DataLoader(
                EEGDataset(_x_train, _y_train, _stft_train),
                batch_size=hyperparameters.batch_size,
                shuffle=True,
            )
            test_loader = DataLoader(
                EEGDataset(_x_test, _y_test, _stft_test),
                batch_size=hyperparameters.batch_size,
                shuffle=True,
            )

            last_acc = 0
            last_kappa = -1

            for i in range(n_iter):
                model.train()
                train_loss = 0
                train_acc = 0
                for j, (x, y, y_stft) in enumerate(train_loader):
                    x = torch.unsqueeze(x, 1)
                    x, y, y_stft = (
                        x.to(device),
                        y.to(device),
                        y_stft.to(device=device, dtype=torch.float),
                    )
                    y = y.long()
                    optimizer.zero_grad()
                    stft, representations, logits = model(x)
                    if stft is not None:
                        stft_loss = nn.MSELoss()(stft, y_stft)
                    else:
                        stft_loss = None
                    acc = accuracy_score(logits, y.cpu().detach().numpy())

                    loss = loss_fn(logits, y)
                    if stft_loss is not None:
                        loss += stft_loss
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    train_acc += acc.item()

                train_loss /= len(train_loader)
                train_acc /= len(train_loader)

                if i == 0 or (i + 1) % eval_inter == 0 or i == n_iter - 1:
                    model.eval()
                    test_accs = []
                    test_losses = []
                    for j, (x, y, y_stft) in enumerate(test_loader):
                        if x.size(0) < 2:
                            continue
                        x = torch.unsqueeze(x, 1)
                        x, y, y_stft = (
                            x.to(device),
                            y.to(device),
                            y_stft.to(device=device, dtype=torch.float),
                        )
                        y = y.long()
                        with torch.no_grad():
                            stft, representations, logits = model(x)
                            if stft is not None:
                                stft_loss = nn.MSELoss()(stft, y_stft)
                            else:
                                stft_loss = None
                            acc = accuracy_score(logits, y.cpu().detach().numpy())
                            loss = loss_fn(logits, y)
                            if stft_loss is not None:
                                loss += stft_loss

                            test_accs.append(acc)
                            test_losses.append(loss.item())

                    fold_acc = np.mean(test_accs).item()
                    fold_kappa = (fold_acc - 0.5) / (1 - 0.5)

                    last_acc = fold_acc
                    last_kappa = fold_kappa

            subject_accuracies.append(last_acc)
            subject_kappas.append(last_kappa)

        all_subject_accuracies.append(subject_accuracies)
        all_subject_kappas.append(subject_kappas)

    mean_accuracy = float(np.mean(all_subject_accuracies))
    std_accuracy = float(np.std(all_subject_accuracies))
    mean_kappa = float(np.mean(all_subject_kappas))
    std_kappa = float(np.std(all_subject_kappas))

    print(f"LOSO Results (across {hyperparameters.n_repeats} repeats):")
    print(f"  Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"  Kappa:    {mean_kappa:.4f} ± {std_kappa:.4f}")

    return (
        all_subject_accuracies,
        all_subject_kappas,
        subjects,
        mean_accuracy,
        std_accuracy,
        mean_kappa,
        std_kappa,
        experiment,
        hyperparameters,
        model_configs,
    )


def save_loso_csv(
    subject_accuracies: List[float],
    subject_kappas: List[float],
    subjects: List[int],
    output_path: str,
):
    """Save LOSO results to CSV with per-subject metrics."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "accuracy", "kappa"])
        for subject, acc, kappa in zip(subjects, subject_accuracies, subject_kappas):
            writer.writerow([subject, f"{np.mean(acc):.4f}", f"{np.mean(kappa):.4f}"])


if __name__ == "__main__":
    (
        accuracy,
        kappa,
        subjects,
        mean_acc,
        std_acc,
        mean_kappa,
        std_kappa,
        experiment,
        hyperparameters,
        model_configs,
    ) = main()
