import yaml
from tqdm import tqdm
from typing import Tuple, List, Union

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import mne

mne.set_log_level("WARNING")  # suppress INFO logs
import scipy

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import CosineAnnealingLR

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
    parser.add_argument("--subject", type=int, default=1)
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

    return parser.parse_args()


def main(
    args=None,
    experiment: Union["Experiment", None] = None,
    config=None,
    model_configs=None,
    save_yaml: bool = True,
):
    if args is None:
        args = parse_args()
    device = args.device  # --> Update to Parameter object
    # verbose = args.verbose  # --> set to command line argument

    # ====================
    # DATA LOADING & SPLITTING
    # ====================
    PREPROCESSING = [
        bandpass_filtering,
    ]

    SSVEP_DATASETS = ["Kalunga2016", "Nakanishi2015", "Wang2021Combined"]
    SLEEP_DATASETS = ["SleepPhysionet"]
    RESTING_STATE_DATASETS = [
        "Cattan2019_PHMD",
        "Hinss2021",
        "Rodrigues2017",
        "ButtonToneSZ",
    ]

    if args.dataset == "dummy_dataset":
        X = np.random.randn(5, 3, 1000)
        y = np.random.randint(0, 2, size=5)

        dataset_info = {"n_ch": 3, "n_times": 1000, "n_classes": 2, "fs": 250}
    elif args.dataset in RESTING_STATE_DATASETS:
        loader = RestingState_DataLoader(
            dataset_name=args.dataset,
            subject=args.subject,
            preprocessing_pipeline=PREPROCESSING,
            tmin=10,
            tmax=50,
            fmin=1,
            fmax=35,
            resample=128,
        )
        X, y, dataset_info = loader.get_data()
    elif args.dataset in SLEEP_DATASETS:
        loader = Sleep_Loader(
            dataset_name=args.dataset,
            subject=args.subject,
            preprocessing_pipeline=PREPROCESSING,
        )
        X, y, dataset_info = loader.get_data()
    elif args.dataset in SSVEP_DATASETS:
        loader = SSVEP_DataLoader(
            dataset_name=args.dataset,
            subject=args.subject,
            preprocessing_pipeline=PREPROCESSING,
            t0=0.0,
            tmax=4.0,
        )
        X, y, dataset_info = loader.get_data()
    else:
        loader = MI_DataLoader(
            dataset_name=args.dataset,
            subject=args.subject,
            preprocessing_pipeline=PREPROCESSING,
            t0=0.5,
            t1=3.5,
        )
        X, y, dataset_info = loader.get_data()

    # ====================
    # CONFIGS & HYPERPARAMETERS
    # ====================
    if config is None:
        with open("./configs/cross_validation.yaml", "r") as f:
            configs = yaml.safe_load(f)

        training_configs = configs["training"]
        model_configs = configs[args.model_name]
    else:
        with open(f"./configs/{config}.yaml", "r") as f:
            configs = yaml.safe_load(f)

        training_configs = configs["training"]
        if model_configs is None:
            model_configs = configs[args.model_name]

    # --> Training Hyperparameters + update experiment tracker
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
                    "Traininig-Hyperparameters",
                ),
                Parameter(
                    hyperparameters.n_iter, "n_iter", "Traininig-Hyperparameters"
                ),
                Parameter(
                    hyperparameters.eval_inter,
                    "eval_inter",
                    "Traininig-Hyperparameters",
                ),
                Parameter(
                    hyperparameters.folds, "k-folds", "Traininig-Hyperparameters"
                ),
                Parameter(
                    hyperparameters.n_repeats,
                    "k-folds-repeat",
                    "Traininig-Hyperparameters",
                ),
                Parameter(
                    hyperparameters.lr, "learning_rate", "Traininig-Hyperparameters"
                ),
                Parameter(
                    hyperparameters.batch_size,
                    "batch_size",
                    "Traininig-Hyperparameters",
                ),
                Parameter(dataset_info, args.dataset, "Dataset-Details"),
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

    # --> Model parameters + update experiment tracker
    model_args = Namespace(
        # Data configuration
        data_name=args.dataset,
        chn=dataset_info["n_ch"],  # Number of channels (matches your X shape)
        time_sample_num=dataset_info[
            "n_times"
        ],  # Number of time points (matches your X shape)
        class_num=dataset_info["n_classes"],  # Binary classification
        # Patch configuration
        patch_size=model_configs["patch_size"],
        spa_dim=model_configs["spa_dim"],  # Spatial dimension for channel embedding
        # Model flags
        gate_flag=model_configs["gate_flag"],  # Use gated fusion (paper default: False)
        posemb_flag=model_configs[
            "posemb_flag"
        ],  # Use positional embeddings (paper default: True)
        branch=model_configs[
            "branch"
        ],  # Options: 'all', 'temporal', 'spatial' (paper default: 'all')
        chn_attn_flag=model_configs[
            "chn_attn_flag"
        ],  # Use channel attention (paper default: True)
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

    if val_size > 0.0:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
        for train_idx, val_idx in sss.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
    else:
        X_train = X
        y_train = y

    # --> Repeat 5-fold CV per subject for 5 cycles (n_repeats)
    all_accuracies = []  # accruacies for 5 folds * 5 cycle
    all_kappas = []  # kappa values for 5 folds * 5 cycle
    all_stft_reconstruction_loss = []
    for seed in tqdm(
        range(1, hyperparameters.n_repeats + 1), total=hyperparameters.n_repeats
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        k = hyperparameters.folds
        skf = StratifiedKFold(n_splits=k, shuffle=False)

        start_stft_loss = 0

        folds_acc = []  # folds accuracies for current cycle
        folds_kappa = []  # folds kappa values for current cycle
        folds_stft_reconstruction_loss = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_train, y_train)):
            # --> Instantiate new model per -fold
            if args.model_name == "db_conformer":
                model = DBConformer(  # --> Update ARgs to Parameter object
                    model_args,
                    emb_size=model_configs["emb_size"],
                    tem_depth=model_configs["tem_depth"],
                    chn_depth=model_configs["chn_depth"],
                    chn=dataset_info[
                        "n_ch"
                    ],  # Number of channels (redundant but needed)
                    n_classes=dataset_info[
                        "n_classes"
                    ],  # Number of classes (redundant but needed)
                )
                model = model.to(device)
            elif args.model_name == "mtf_c":
                model = MTFC(  # --> Update ARgs to Parameter object
                    model_args,
                    n_filter_banks=model_configs["filter_banks"],
                    patch_emb_size=model_configs["patch_emb_size"],
                    n_heads_patch=model_configs["n_heads_patch"],
                    wsize_divisor=model_configs["wsize_divisor"],
                    freq_downsample=model_configs["freq_downsample"],
                    n_times=dataset_info["n_times"],
                    sst_emb_size=model_configs["sst_emb_size"],
                    depth=model_configs["tem_depth"],
                    n_classes=dataset_info["n_classes"],
                    fs=dataset_info["fs"],
                    temporal_kernel=model_configs.get("temporal_kernel", 43),
                )
                model = model.to(device)
            else:
                model = DBConformer(  # --> Update ARgs to Parameter object
                    model_args,
                    emb_size=model_configs["emb_size"],
                    tem_depth=model_configs["tem_depth"],
                    chn_depth=model_configs["chn_depth"],
                    chn=dataset_info[
                        "n_ch"
                    ],  # Number of channels (redundant but needed)
                    n_classes=dataset_info[
                        "n_classes"
                    ],  # Number of classes (redundant but needed)
                )
                model = model.to(device)

            # --> Optimizer + learning_rate scheduling scheme
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=hyperparameters.lr,
                betas=(0.9, 0.99),
                weight_decay=0,
            )  # --> Update Args to Parameter objects
            # scheduler = CosineAnnealingLR(optimizer, T_max=n_iter, eta_min=1E-6)

            # --> Split data for fold +  class imbalance reweighting + euclidean alignment + dataloader definitiion
            _x_test, _y_test = X_train[test_idx], y_train[test_idx]
            _x_train, _y_train = X_train[train_idx], y_train[train_idx]

            # class_counts = np.bincount(
            #     _y_train.astype(int), minlength=dataset_info["n_classes"]
            # )
            # total_samples = len(_y_train)
            # class_weights = total_samples / (dataset_info["n_classes"] * class_counts)
            # class_weights = torch.FloatTensor(class_weights).to(device)

            # loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            loss_fn = nn.CrossEntropyLoss()

            # Euclidean Alignemnt of epochs
            _x_train, sqrtRefEA = EA(_x_train)
            _x_test = EA_online(_x_test, sqrtRefEA)

            # Compute STFT or band powers for each epoch
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
                target_T = (dataset_info["n_times"] - 1) // P_cfg
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
                    _x_train, n_filter_banks=F_cfg, fs=dataset_info["fs"]
                )
                _stft_test = compute_band_powers(
                    _x_test, n_filter_banks=F_cfg, fs=dataset_info["fs"]
                )
            else:
                _stft_train = np.zeros((len(_x_train), 1, 1, 1), dtype=np.float32)
                _stft_test = np.zeros((len(_x_test), 1, 1, 1), dtype=np.float32)

            # if args.model_name=='mtf_c':
            #     mne.set_log_level('WARNING')  # suppress INFO logs
            #     filter_banks = {
            #         'delta': [None, 4],
            #         'theta': [4, 8],
            #         'alpha': [8, 12],
            #         'beta': [12, 30],
            #         'gamma': [30, 100]
            #     }

            #     train = []
            #     test = []
            #     for band, corner_freqs in filter_banks.items():
            #         train.append(mne.filter.filter_data(
            #             _x_train, sfreq=dataset_info['fs'],
            #             l_freq=corner_freqs[0], h_freq=corner_freqs[1]
            #             )[:, np.newaxis, :, :])
            #         test.append(mne.filter.filter_data(
            #             _x_test, sfreq=dataset_info['fs'],
            #             l_freq=corner_freqs[0], h_freq=corner_freqs[1]
            #             )[:, np.newaxis, :, :])

            #     train.append(_x_train[:, np.newaxis, :, :])
            #     test.append(_x_test[:, np.newaxis, :, :])

            #     _x_train = np.concatenate(train, axis=1)
            #     _x_test = np.concatenate(test, axis=1)

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

            # Training and Evaluate CV-fold for n_iter epochs
            # best_acc_per_fold = 0
            # best_kappa_per_fold = -1
            # best_stft_reconstruction_loss_per_fold = math.inf
            last_acc_per_fold = 0
            last_kappa_per_fold = -1
            last_stft_reconstruction_loss_per_fold = math.inf
            for i in range(
                n_iter
            ):  # tqdm(range(n_iter), total=n_iter): # desc=f"Training: fold {fold+1}/{k}"
                # Train and upadte train folds performance
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
                    if stft_loss is not None:  # type: ignore
                        loss += stft_loss  # type: ignore
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

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
                            # logits, loss, acc = fine_tune_run(encoder, model, clf_head, x, y, LBL_SMOOTH=LBL_SMOOTH)
                            stft, representations, logits = model(x)
                            if stft is not None:
                                stft_loss = nn.MSELoss()(stft, y_stft)
                            else:
                                stft_loss = None
                            acc = accuracy_score(logits, y.cpu().detach().numpy())
                            loss = loss_fn(logits, y)
                            if stft_loss is not None:  # type: ignore
                                loss += stft_loss  # type: ignore

                            test_accs.append(acc)
                            test_losses.append(loss.item())

                    fold_acc = np.mean(test_accs).item()
                    fold_kappa = (fold_acc - 0.5) / (1 - 0.5)
                    if (args.model_name == "mtf_c") and (stft is not None):
                        fold_stft_loss = stft_loss

                    last_acc_per_fold = fold_acc
                    last_kappa_per_fold = fold_kappa
                    if (args.model_name == "mtf_c") and (stft is not None):
                        last_stft_reconstruction_loss_per_fold = fold_stft_loss
                        start_stft_loss += fold_stft_loss.item()

                    # if fold_acc > best_acc_per_fold:
                    #     best_acc_per_fold = fold_acc
                    #     best_kappa_per_fold = fold_kappa
                    #     if (args.model_name == "mtf_c") and (stft is not None):
                    #         best_stft_reconstruction_loss_per_fold = fold_stft_loss
                    #         start_stft_loss += fold_stft_loss.item()

                    # if verbose:
                    #     print(f"Acc:{fold_acc}, Kappa:{fold_kappa}")

            folds_acc.append(last_acc_per_fold)
            folds_kappa.append(last_kappa_per_fold)
            folds_stft_reconstruction_loss.append(
                last_stft_reconstruction_loss_per_fold
            )

            # folds_acc.append(best_acc_per_fold)
            # folds_kappa.append(best_kappa_per_fold)
            # folds_stft_reconstruction_loss.append(
            #     best_stft_reconstruction_loss_per_fold
            # )

        accuracy = np.mean(folds_acc)
        kappa = np.mean(folds_kappa)
        if (args.model_name == "mtf_c") and (stft is not None):  # type: ignore
            _folds_stft_loss = [f.item() for f in folds_stft_reconstruction_loss]
            stft_loss = np.mean(_folds_stft_loss)
        else:
            stft_loss = 0

        all_accuracies.append(accuracy)
        all_kappas.append(kappa)
        all_stft_reconstruction_loss.append(stft_loss)

    # accuracy = np.mean(all_accuracies)
    # kappa = np.mean(all_kappas)
    # print(f"subject - {args.subject}")
    # print(f"Accuracy: {np.mean(accuracy):.2f}")
    # print(f"Kappa: {np.mean(kappa):.2f}")

    print(
        f"START STFT RECONSTRUCTION LOSS: {start_stft_loss / (hyperparameters.n_repeats * hyperparameters.folds)}"
    )

    return (
        all_accuracies,
        all_kappas,
        all_stft_reconstruction_loss,
        experiment,
        hyperparameters,
        model_configs,
    )


if __name__ == "__main__":
    accuracy, kappa, stft_reconstruction_loss, experiment = main()
