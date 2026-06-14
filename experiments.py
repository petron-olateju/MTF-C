import yaml
from argparse import Namespace

import numpy as np
import torch

from dataloader import TrainValTest_Split_Loader
from utils.data_loader import (
    MI_DATASETS,
    SSVEP_DATASETS,
    SLEEP_DATASETS,
    MI_DataLoader,
    SSVEP_DataLoader,
    Sleep_Loader,
)
from pl_models import db_conformer
from pl_models import NAME_MODEL_MAP

import pytorch_lightning as pl


# Helper Functions
def get_data_subjects(dataset_name):
    if dataset_name in MI_DATASETS:
        return MI_DataLoader.get_subjects(dataset_name=dataset_name)
    elif dataset_name in SSVEP_DATASETS:
        return SSVEP_DataLoader.get_subjects(dataset_name=dataset_name)
    elif dataset_name in SLEEP_DATASETS:
        return Sleep_Loader.get_subjects(dataset_name=dataset_name)


def get_data_loader(dataset_name, subject, preprocessing_pipeline, t0, t1):
    if dataset_name in MI_DATASETS:
        return MI_DataLoader(dataset_name, subject, preprocessing_pipeline, t0, t1)
    elif dataset_name in SSVEP_DATASETS:
        return SSVEP_DataLoader(dataset_name, subject, preprocessing_pipeline, t0, t1)
    elif dataset_name in SLEEP_DATASETS:
        return Sleep_Loader(dataset_name, subject, preprocessing_pipeline, t0, t1)


def get_model(model_name, dataset_name, dataset_info):
    with open("configs/model_params.yaml", "r") as f:
        MODEL_PARAMS = yaml.safe_load(f)[model_name]

    with open("configs/training_params.yaml", "r") as f:
        TRAINING_PARAMS = yaml.safe_load(f)

    MODEL_PARAMS["data_name"] = dataset_name
    MODEL_PARAMS["chn"] = dataset_info["n_ch"]
    MODEL_PARAMS["time_sample_num"] = dataset_info["n_times"]
    MODEL_PARAMS["class_num"] = dataset_info["n_classes"]
    MODEL_PARAMS["lr"] = TRAINING_PARAMS["lr"]

    return NAME_MODEL_MAP[model_name](MODEL_PARAMS), MODEL_PARAMS


# Cross Validation Experiment
def cross_validation(
    dataset_name,
    model_name,
    batch_size,
    n_epochs,
    n_folds,
    n_repeats,
    preprocessing_pipeline,
    t0,
    t1,
    experiment_seed,
):
    np.random.seed(experiment_seed)
    torch.manual_seed(experiment_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(experiment_seed)

    SUBJECTS = get_data_subjects(dataset_name=dataset_name)
    SUBJECTS = SUBJECTS[0:1]  # Remove after testing code restructuring

    performance = []
    loss = []
    subjects_performance = {}

    for subject in SUBJECTS:
        sub_performance = []
        sub_loss = []
        for repeat in range(n_repeats):
            for fold in range(n_folds):

                dm = TrainValTest_Split_Loader(
                    dataset_name=dataset_name,
                    subject=subject,
                    batch_size=batch_size,
                    seed=fold,
                    num_workers=0,
                    val_split=None,
                    cv=n_folds,
                    preprocessing_pipeline=preprocessing_pipeline,
                    t0=t0,
                    t1=t1,
                )

                model, model_params = get_model(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    dataset_info=dm.info,
                )

                trainer = pl.Trainer(
                    max_epochs=n_epochs,
                )

                trainer.fit(model, datamodule=dm)
                val_metrics = trainer.validate(model, dm)[0]
                print(
                    f"Subject-{subject} ({fold+1} / {n_folds}) Performance:",
                    val_metrics,
                )

                sub_performance.append(val_metrics["val_performance"])
                sub_loss.append(val_metrics["val_loss"])

        sub_performance = np.mean(sub_performance).item()
        sub_loss = np.mean(sub_loss).item()
        subjects_performance[subject] = sub_performance

        performance.append(sub_performance)
        loss.append(sub_loss)

    performance_mean = np.mean(performance).item()
    performance_std = np.std(performance).item()
    loss = np.mean(loss).item()

    return {
        "model_params": model_params,
        "mean_performance": performance_mean,
        "std_performance": performance_std,
        "loss": loss,
        "subjects_performance": subjects_performance,
    }
