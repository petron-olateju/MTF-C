import yaml
from argparse import Namespace

import numpy as np
import torch

from dataloader import TrainValTest_Split_Loader
from utils.data_loader import (
    MI_DATASETS,
    SSVEP_DATASETS,
    RESTING_STATE_DATASETS,
    MI_DataLoader,
    SSVEP_DataLoader,
    RestingState_DataLoader,
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
    elif dataset_name in RESTING_STATE_DATASETS:
        return RestingState_DataLoader.get_subjects(dataset_name=dataset_name)


def get_data_loader(dataset_name, subject, preprocessing_pipeline, t0, t1):
    if dataset_name in MI_DATASETS:
        return MI_DataLoader(dataset_name, subject, preprocessing_pipeline, t0, t1)
    elif dataset_name in SSVEP_DATASETS:
        return SSVEP_DataLoader(dataset_name, subject, preprocessing_pipeline, t0, t1)
    elif dataset_name in RESTING_STATE_DATASETS:
        return RestingState_DataLoader(dataset_name, subject, preprocessing_pipeline, t0, t1)


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
    MODEL_PARAMS["fs"] = dataset_info['fs']

    return NAME_MODEL_MAP[model_name](MODEL_PARAMS), MODEL_PARAMS

def get_model_params(model_name):
    with open("configs/model_params.yaml", "r") as f:
        MODEL_PARAMS = yaml.safe_load(f)[model_name]
    return MODEL_PARAMS




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

    SUBJECTS = get_data_subjects(dataset_name=dataset_name)
    SUBJECTS = SUBJECTS[0:1]  # Remove after testing code restructuring

    acc = []
    reconstruction = []
    loss = []
    subjects_acc = {}
    subjects_reconstruction = {}

    for subject in SUBJECTS:

        sub_acc = []
        sub_reconstruction = []
        sub_loss = []

        for repeat in range(n_repeats):
            np.random.seed(experiment_seed+repeat)
            torch.manual_seed(experiment_seed+repeat)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(experiment_seed+repeat)

            for fold in range(n_folds):

                if model_name == 'mtf_c':
                    model_params = get_model_params('mtf_c')
                    spectrum = model_params['sst_method']
                    n_filter_banks = model_params['filter_banks']
                elif model_name == 'db_conformer':
                    spectrum = None
                    n_filter_banks = 0

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
                    spectrum = spectrum,
                    n_filter_banks = n_filter_banks
                )

                model, model_params = get_model(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    dataset_info=dm.info,
                )

                trainer = pl.Trainer(
                    max_epochs=n_epochs,
                    accelerator="auto",
                    devices="auto"
                )

                trainer.fit(model, datamodule=dm)
                val_metrics = trainer.validate(model, dm)[0]
                print(
                    f"Subject-{subject} ({fold+1} / {n_folds}) Performance:",
                    val_metrics,
                )

                sub_acc.append(val_metrics["val_acc"])
                sub_loss.append(val_metrics["val_loss"])
                if "val_reconstruction" in val_metrics:
                    sub_reconstruction.append(val_metrics["val_reconstruction"])

        sub_acc = np.mean(sub_acc).item()
        sub_loss = np.mean(sub_loss).item()
        subjects_acc[subject] = sub_acc
        if len(sub_reconstruction) > 0:
            sub_reconstruction = np.mean(sub_reconstruction).item()
            subjects_reconstruction[subject] = sub_reconstruction
            reconstruction.append(sub_reconstruction)

        acc.append(sub_acc)
        loss.append(sub_loss)

    acc_mean = np.mean(acc).item()
    acc_std = np.std(acc).item()
    loss = np.mean(loss).item()
    
    if len(reconstruction) > 0:
        reconstruction_mean = np.mean(reconstruction).item()
        reconstruction_std = np.std(reconstruction).item()

        return {
            "model_params": model_params,
            "mean_acc": acc_mean,
            "std_acc": acc_std,
            "mean_spectrum_mse": reconstruction_mean,
            "std_spectrum_mse": reconstruction_std,
            "loss": loss,
            "subjects_acc": subjects_acc,
        }
    
    return {
        "model_params": model_params,
        "mean_acc": acc_mean,
        "std_acc": acc_std,
        "loss": loss,
        "subjects_acc": subjects_acc,
    }
