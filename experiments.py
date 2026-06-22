import yaml
from argparse import Namespace

import numpy as np
import torch

from utils.data_loader import get_data_subjects
from dataloader import (
    TrainValTest_Split_Loader, 
    StratifiedKFoldDataModule,
    LOSO_Loader
)
from utils.preprocessing import bandpass_filtering
from pl_models import NAME_MODEL_MAP

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


# Helper Functions
def get_model(model_name, dataset_name, dataset_info, lr):
    with open("configs/model_params.yaml", "r") as f:
        MODEL_PARAMS = yaml.safe_load(f)[model_name]

    MODEL_PARAMS["data_name"] = dataset_name
    MODEL_PARAMS["chn"] = dataset_info["n_ch"]
    MODEL_PARAMS["time_sample_num"] = dataset_info["n_times"]
    MODEL_PARAMS["class_num"] = dataset_info["n_classes"]
    MODEL_PARAMS["lr"] = lr
    MODEL_PARAMS["fs"] = dataset_info['fs']

    return NAME_MODEL_MAP[model_name](MODEL_PARAMS), MODEL_PARAMS

def get_model_params(model_name):
    with open("configs/model_params.yaml", "r") as f:
        MODEL_PARAMS = yaml.safe_load(f)[model_name]
    return MODEL_PARAMS

def make_preprocessing_pipeline(preprocessing_arg):
    pipeline = []
    if 'bandpass' in preprocessing_arg:
        print("Using bandpass filter")
        pipeline.append(bandpass_filtering)


# Across subject evaluation experiments
def across_subjects_evaluation(
    dataset_name,
    model_name,
    t0,
    t1,
    experiment_path,
    run_timestamp,
    experiment_seed,
    batch_size,
    lr,
    n_epochs,
    n_folds,
    val_split,
    test_split,
    n_repeats,
    preprocessing_args,
    validation_strategy='cv',
):

    SUBJECTS = get_data_subjects(dataset_name=dataset_name)
    preprocessing_pipeline = make_preprocessing_pipeline(preprocessing_args)

    subjects_acc = {}
    subjects_reconstruction = {}
    sub_acc = []
    sub_reconstruction = []
    sub_loss = []

    if model_name == 'mtf_c':
        model_params = get_model_params('mtf_c')
        spectrum = model_params['sst_method']
        n_filter_banks = model_params['filter_banks']
    elif model_name == 'db_conformer':
        spectrum = None
        n_filter_banks = 0

    if validation_strategy == 'cv':
        dm = StratifiedKFoldDataModule(
            dataset_name=dataset_name,
            batch_size=batch_size,
            num_workers=0,
            cv=n_folds,
            preprocessing_pipeline=preprocessing_pipeline,
            preprocessing_args=preprocessing_args,
            t0=t0,
            t1=t1,
            spectrum = spectrum,
            n_filter_banks = n_filter_banks
        )
    elif validation_strategy == 'train_test':
        dm = TrainValTest_Split_Loader(
            dataset_name=dataset_name,
            batch_size=batch_size,
            num_workers=0,
            val_split=val_split,
            test_split=test_split,
            preprocessing_pipeline=preprocessing_pipeline,
            preprocessing_args=preprocessing_args,
            t0=t0,
            t1=t1,
            spectrum = spectrum,
            n_filter_banks = n_filter_banks
        )
    elif validation_strategy == 'loso':
        dm = LOSO_Loader(
            dataset_name=dataset_name,
            batch_size=batch_size,
            num_workers=0,
            preprocessing_pipeline=preprocessing_pipeline,
            preprocessing_args=preprocessing_args,
            t0=t0,
            t1=t1,
            spectrum=spectrum,
            n_filter_banks=n_filter_banks
        )
    dm.preload_data()
    
    for subject in SUBJECTS:

        for repeat in range(1, n_repeats+1):
            np.random.seed(experiment_seed+repeat)
            torch.manual_seed(experiment_seed+repeat)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(experiment_seed+repeat)

            folds_acc = []
            folds_reconstruction = []
            folds_loss = []

            for fold in range(n_folds):
                dm.update_subject(subject=subject, seed=experiment_seed+repeat, fold=fold)
                
                model, model_params = get_model(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    dataset_info=dm.info,
                    lr=lr
                )
                checkpoint_callback = ModelCheckpoint(
                    monitor="val_acc",
                    mode="max",
                    save_top_k=1,
                    dirpath=experiment_path,
                    filename=f'{run_timestamp}|model:{model_name}|subject:{subject}'
                )
                trainer = pl.Trainer(
                    max_epochs=n_epochs,
                    accelerator="auto",
                    devices="auto",
                    callbacks=[checkpoint_callback]
                )

                trainer.fit(model, datamodule=dm)
                val_metrics = trainer.validate(model, dm, ckpt_path="best")[0]

                folds_acc.append(val_metrics["val_acc"])
                folds_loss.append(val_metrics["val_loss"])
                if "val_reconstruction" in val_metrics:
                    folds_reconstruction.append(val_metrics["val_reconstruction"])
                
            folds_acc = np.mean(folds_acc)
            folds_loss = np.mean(folds_loss)
            sub_acc.append(folds_acc)
            sub_loss.append(folds_loss)
            if len(folds_reconstruction) > 0:
                folds_reconstruction = np.mean(folds_reconstruction)
                sub_reconstruction.append(folds_reconstruction)

        subjects_acc[subject] = np.mean(sub_acc[-n_repeats:]).item()
        if len(sub_reconstruction) > 0:
            subjects_reconstruction[subject] = np.mean(sub_reconstruction[-n_repeats:]).item()

        print(
            f"Subject-{subject} Performance:",
            {'acc': subjects_acc[subject]},
        )

    acc_mean = np.mean(sub_acc).item()
    acc_std = np.std(sub_acc).item()
    loss = np.mean(sub_loss).item()
    
    if len(sub_reconstruction) > 0:
        reconstruction_mean = np.mean(sub_reconstruction).item()
        reconstruction_std = np.std(sub_reconstruction).item()

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
