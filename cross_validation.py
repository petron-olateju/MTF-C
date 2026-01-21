from tqdm import tqdm

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, KFold

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.metrics import accuracy_score
from utils.preprocessing import EA, bandpass_filtering, exponential_moving_standardization
from utils.data_loader import EEGDataset, load_BNCI2014_001
from models.DBConformer import DBConformer

import argparse
from argparse import Namespace

def parse_args():
    parser = argparse.ArgumentParser(description='Model, Hyperparameters, ExperimentLogger options')

    parser.add_argument('--model_name', type=str, default='db_conformer',
        choices=[
            'db_conformer',
            'mtf_c',
            'dual_tsst'
        ]
        )
    parser.add_argument('--dataset', type=str, default='dummy_dataset',
        choices=['dummy_dataset', 'BNCI2014_001']
    )
    parser.add_argument('--subject', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu',
        choices = ['cpu', 'cuda']
    )
    parser.add_argument('--verbose', action='store_true',
        help='Print training progress')

    return parser.parse_args()

def main(args=None):
    if args is None:
        args = parse_args()
    device = args.device  # --> Update to Parameter object
    verbose = args.verbose  # --> set to command line argument

    # ====================
    # DATA LOADING & SPLITTING
    # ====================
    PREPROCESSING = [
        bandpass_filtering,
        # exponential_moving_standardization,
    ]
    if args.dataset == 'dummy_dataset':
        X = np.random.randn(100, 1, 3, 1062)      # --> Replacce with loader class from utils.dataset_loader 
        y = np.random.randint(0, 2, size=100)

        dataset_info = {
        "n_ch": 3,
        "n_times": 1062,
        "n_classes": 2
    }
    else:
        if args.dataset == 'BNCI2014_001':
            X, y, dataset_info = load_BNCI2014_001(
                subject=args.subject, 
                preprocessing_pipeline=PREPROCESSING
                )

    # --> Replace with laod from yaml file
    hyperparameters = Namespace(
        val_size=0.0,
        n_iter=100,
        eval_inter=5,
        folds=5
    )

    val_size = hyperparameters.val_size # -- > replace with valsize Parameter object for experiment logging
    if val_size > 0.0:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
        for train_idx, val_idx in sss.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
    else:
        X_train = X
        y_train = y

    # ====================
    # Within Subject 5-CV
    # ====================
    n_iter = hyperparameters.n_iter    # --> Update to Parameter object
    eval_inter = hyperparameters.eval_inter  # --> Update to Parameter object

    folds_acc = []
    folds_kappa = []

    # --> Replace with laod from yaml file
    model_args = Namespace(
        # Data configuration
        data_name=args.dataset,
        chn=dataset_info["n_ch"],                    # Number of channels (matches your X shape)
        time_sample_num=dataset_info["n_times"],     # Number of time points (matches your X shape)
        class_num=dataset_info["n_classes"],              # Binary classification
        
        # Patch configuration
        patch_size=100,           
        spa_dim=16,               # Spatial dimension for channel embedding
        
        # Model flags
        gate_flag=False,          # Use gated fusion (paper default: False)
        posemb_flag=True,         # Use positional embeddings (paper default: True)
        branch='all',             # Options: 'all', 'temporal', 'spatial' (paper default: 'all')
        chn_atten_flag=True       # Use channel attention (paper default: True)
    )

    k = hyperparameters.folds
    skf = StratifiedKFold(n_splits=k, shuffle=False)
    # kf = KFold(n_splits=5, shuffle=False)
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_train, y_train)):
        model = DBConformer(# --> Update ARgs to Parameter object
            model_args,
            emb_size=40,      # Embedding dimension (paper default)
            tem_depth=2,      # Temporal transformer depth (paper default: 5-6)
            chn_depth=2,      # Spatial transformer depth (paper default: 5-6)
            chn=dataset_info["n_ch"],            # Number of channels (redundant but needed)
            n_classes=dataset_info["n_classes"]       # Number of classes (redundant but needed)
            ) 
        optimizer = torch.optim.Adam(model.parameters(), lr=1E-3, betas=(0.9, 0.99), weight_decay=0)     # --> Update Args to Parameter objects
        loss_fn = nn.CrossEntropyLoss()

        _x_test, _y_test = X_train[test_idx], y_train[test_idx]
        _x_train, _y_train = X_train[train_idx], y_train[train_idx]

        _x_train = EA(_x_train)
        _x_test = EA(_x_test)
        train_loader = DataLoader(
            EEGDataset(_x_train, _y_train),
            batch_size=32,
            shuffle=True
            )
        test_loader = DataLoader(
            EEGDataset(_x_test, _y_test),
            batch_size=32,
            shuffle=True
            )

        # Training and Evaluate CV-folds for n_iter epochs
        best_acc_per_fold = 0
        best_kappa_per_fold = -1
        for i in tqdm(range(n_iter), desc=f"Training: fold {fold+1}/{k}"):
            # Train and upadte train folds performance
            model.train
            train_loss = 0
            train_acc = 0
            for j, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                y = y.long()
                optimizer.zero_grad()
                representations, logits = model(x)
                acc = accuracy_score(logits, y.cpu().detach().numpy())

                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_acc += acc.item()

            train_loss /= len(train_loader)
            train_acc /= len(train_loader)

            if i==0 or (i+1)%eval_inter==0 or i==n_iter-1:
                model.eval()
                test_accs = []
                test_losses = []
                for j, (x, y) in enumerate(test_loader):
                    x, y = x.to(device), y.to(device)
                    y = y.long()
                    with torch.no_grad():
                        # logits, loss, acc = fine_tune_run(encoder, model, clf_head, x, y, LBL_SMOOTH=LBL_SMOOTH)
                        representations, logits = model(x)
                        acc = accuracy_score(logits, y.cpu().detach().numpy())
                        loss = loss_fn(logits, y)

                        test_accs.append(acc)
                        test_losses.append(loss.item())

                fold_acc = np.mean(test_accs).item()
                fold_kappa = (fold_acc - 0.5) / (1 - 0.5)

                if fold_acc > best_acc_per_fold:
                    best_acc_per_fold = fold_acc
                    best_kappa_per_fold = fold_kappa
                    
                if verbose:
                    print(f"Acc:{fold_acc}, Kappa:{fold_kappa}")

        folds_acc.append(best_acc_per_fold)
        folds_kappa.append(best_kappa_per_fold)
    
    accuracy = np.mean(folds_acc)
    kappa = np.mean(folds_kappa)

    print("\n \n \n")
    print(f"Accuracy: {np.mean(accuracy):.2f}")
    print(f"Kappa: {np.mean(kappa):.2f}")

    return accuracy, kappa

if __name__ == '__main__':
    accuracy, kappa = main()