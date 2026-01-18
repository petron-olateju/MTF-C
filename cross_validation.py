from tqdm import tqdm

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.metrics import accuracy_score
from utils.data_loader import EEGDataset
from models.DBConformer import DBConformer

device = 'cpu'  # --> Update to Parameter object
verbose = True  # --> set to command line argument

# ====================
# DATA LOADING & SPLITTING
# ====================
X = np.random.randn((100, 1, 9, 1000))      # --> Replacce with loader class from utils.dataset_loader 
y = np.random.randint(0, 1, size=100)

val_size = 0.2 # -- > replace with valsize Parameter object for experiment logging
if val_size > 0.0:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, val_idx in sss.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
else:
    X_train = X
    y_train = y

# ====================
# Within Subject 5-CV
# ====================
n_iter = 100    # --> Update to Parameter object
eval_inter = 5  # --> Update to Parameter object

folds_acc = []
folds_kappa = []

k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(skf.split(X_train, y_train)):
    model = DBConformer()       # --> Update ARgs to Parameter object
    optimizer = torch.optim.Adam(model.parameters(), lr=3E-4, betas=(0.9, 0.99), weight_decay=1E-5)     # --> Update Args to Parameter objects
    loss_fn = nn.CrossEntropyLoss()

    x_test, y_test = X_train[test_idx], y_train[test_idx]
    x_train, y_train = X_train[train_idx], y_train[train_idx]
    train_loader = DataLoader(
        EEGDataset(x_train, y_train),
        batch_size=32,
        shuffle=True
        )
    test_loader = DataLoader(
        EEGDataset(x_test, y_test),
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
            optimizer.zero_grad()
            representations, logits = model(x)
            acc = accuracy_score(logits, y.cou().detach().numpy())

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
                with torch.no_grad():
                    # logits, loss, acc = fine_tune_run(encoder, model, clf_head, x, y, LBL_SMOOTH=LBL_SMOOTH)
                    representations, logits = model(x)
                    acc = accuracy_score(logits, y.cpu().detach().numpy())
                    loss = loss_fn(logits, y)

                    test_accs.append(acc)
                    test_losses.append(loss.item())

            fold_acc = torch.mean(test_accs).item()
            fold_kappa = (fold_acc - 0.5) / (1 - 0.5)

            if fold_acc > best_acc_per_fold:
                best_acc_per_fold = fold_acc
                best_kappa_per_fold = fold_kappa
                
            if verbose:
                print(f"Acc:{fold_acc}, Kappa:{fold_kappa}")

    folds_acc.append(best_acc_per_fold)
    folds_kappa.append(best_kappa_per_fold)