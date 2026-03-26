from tqdm import tqdm
from datetime import datetime

import numpy as np
import pandas as pd
import argparse
import yaml
import os
from argparse import Namespace

from utils.data_loader import (
    get_subjects_BNCI2014_001,
    get_subjects_BNCI2014_002,
    get_subjects_BNCI2014_004,
)
from utils.data_loader import (
    get_subjects_BNCI2015_001,
    get_subjects_BNCI2015_004,
    get_subjects_Liu2024,
)
from utils.data_loader import (
    get_subjects_AlexMI,
    SSVEP_DataLoader,
    Sleep_Loader,
    RestingState_DataLoader,
)
from utils.experiment_recorder import Parameter, Experiment
from cross_validation import main as cross_validation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment Script runner: Model, Hyperparameters, ExperimentLogger options"
    )

    parser.add_argument(
        "--script",
        type=str,
        default="cross_validation.py",
        choices=["cross_validation", "pre_training"],
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
            "AlexMI",
            "Liu2024",
            "Kalunga2016",
            "MAMEM2",
            "MAMEM3",
            "Nakanishi2015",
            "Wang2021Combined",
            "SleepPhysionet",
            "Cattan2019_PHMD",
            "Hinss2021",
            "Rodrigues2017",
            "all",
            "mi",
            "ssvep",
            "sleep",
            "resting_state",
        ],
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--experiment_version", type=str, default="v0.0")
    parser.add_argument(
        "--experiment_description", type=str, default="Baseline Experiment"
    )
    parser.add_argument(
        "--experiment_folder",
        type=str,
        default="./experiments",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print training progress"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    experiment = Experiment(
        args.experiment_version,
        dir=args.experiment_folder,
        description=args.experiment_description,
    )

    script = args.script
    model_name = args.model_name
    if (args.dataset == "all") or (args.dataset == "mi"):
        datasets = ["BNCI2014_001", "BNCI2014_002", "BNCI2014_004"]
    elif args.dataset == "ssvep":
        datasets = ["Kalunga2016", "Nakanishi2015", "Wang2021Combined"]
    elif args.dataset == "sleep":
        datasets = ["SleepPhysionet"]
    elif args.dataset == "resting_state":
        datasets = ["Cattan2019_PHMD"]     # Excluded Hinss2021, too short (2s length), Rodrigues2017 (too short for five-fold CV)
    else:
        datasets = [args.dataset]
    device = args.device

    all_accuracies = []
    all_kappas = []
    all_stft = []

    accuracy_results = {}
    accuracy_std_results = {}
    kappa_results = {}
    kappa_std_results = {}
    stft_reconstruction_results = {}

    run_timestamp = datetime.now().isoformat()
    dataset_results = {}
    run_config = {}

    for dataset in datasets:
        if dataset == "BNCI2014_001":
            subjects = get_subjects_BNCI2014_001()
            print(f"{dataset} subjects: {subjects}")
        elif dataset == "BNCI2014_002":
            subjects = get_subjects_BNCI2014_002()
            print(f"{dataset} subjects: {subjects}")
        elif dataset == "BNCI2014_004":
            subjects = get_subjects_BNCI2014_004()
            print(f"{dataset} subjects: {subjects}")
        elif dataset == "BNCI2015_001":
            subjects = get_subjects_BNCI2015_001()
            print(f"{dataset} subjects: {subjects}")
        elif dataset == "BNCI2015_004":
            subjects = get_subjects_BNCI2015_004()
            print(f"{dataset} subjects: {subjects}")
        elif dataset == "Liu2024":
            subjects = get_subjects_Liu2024()
            print(f"{dataset} subjects: {subjects}")
        elif dataset == "AlexMI":
            subjects = get_subjects_AlexMI()
            print(f"{dataset} subjects: {subjects}")
        elif dataset == "Kalunga2016":
            subjects = SSVEP_DataLoader.get_subjects("Kalunga2016")
            print(f"{dataset} subjects: {subjects}")
        elif dataset == "MAMEM2":
            subjects = SSVEP_DataLoader.get_subjects("MAMEM2")
            print(f"{dataset} subjects: {subjects}")
        elif dataset == "MAMEM3":
            subjects = SSVEP_DataLoader.get_subjects("MAMEM3")
            print(f"{dataset} subjects: {subjects}")
        elif dataset == "Nakanishi2015":
            subjects = SSVEP_DataLoader.get_subjects("Nakanishi2015")
            print(f"{dataset} subjects: {subjects}")
        elif dataset == "Wang2021Combined":
            subjects = SSVEP_DataLoader.get_subjects("Wang2021Combined")
            print(f"{dataset} subjects: {subjects}")
        elif dataset == "SleepPhysionet":
            subjects = Sleep_Loader.get_subjects("SleepPhysionet")
            print(f"{dataset} subjects: {subjects}")
        elif dataset in ["Cattan2019_PHMD", "Hinss2021", "Rodrigues2017"]:
            subjects = RestingState_DataLoader.get_subjects(dataset)
            print(f"{dataset} subjects: {subjects}")
        else:
            subjects = [1]  # Default for dummy_dataset or unknown datasets

        for subject in tqdm(subjects, total=len(subjects), desc="Training"):
            mthd_args = Namespace(
                dataset=dataset,
                subject=subject,
                device=device,
                model_name=model_name,
                verbose=True,
                experiment_folder=args.experiment_folder,
                experiment_version=args.experiment_version,
            )

            if script == "cross_validation":
                (
                    accuracy,
                    kappa,
                    stft_reconstruction_loss,
                    experiment,
                    hyperparameters,
                    model_configs,
                ) = cross_validation(mthd_args, experiment)
                all_accuracies = all_accuracies + accuracy
                all_kappas = all_kappas + kappa
                all_stft = all_stft + stft_reconstruction_loss

                if not run_config:
                    run_config.update(
                        {
                            "training": {
                                "val_size": hyperparameters.val_size,
                                "n_iter": hyperparameters.n_iter,
                                "eval_inter": hyperparameters.eval_inter,
                                "folds": hyperparameters.folds,
                                "n_repeats": hyperparameters.n_repeats,
                                "lr": hyperparameters.lr,
                                "batch_size": hyperparameters.batch_size,
                            },
                            "model": model_configs,
                        }
                    )

            if args.model_name != "mtf_c":
                print(
                    f"subject {subject} | Accuracy: {np.mean(accuracy):.2f}, Kappa: {np.mean(kappa):.2f}"
                )  # type: ignore
                print("============================================")
            else:
                print(
                    f"subject {subject} | Accuracy: {np.mean(accuracy):.2f}, Kappa: {np.mean(kappa):.2f} | STFT Reconstruction Loss: {np.mean(stft_reconstruction_loss):.2f}"
                )  # type: ignore
                print("============================================")

        acc_mean = np.mean(all_accuracies)
        acc_std = np.std(all_accuracies)
        kappa_mean = np.mean(all_kappas)
        kappa_std = np.std(all_kappas)
        stft_result = np.mean(all_stft)

        dataset_results[dataset] = {
            "accuracy": {
                "mean": float(acc_mean),
                "std": float(acc_std),
            },
            "kappa": {
                "mean": float(kappa_mean),
                "std": float(kappa_std),
            },
            "stft_reconstruction_loss": float(stft_result) if stft_result else None,
        }

        if args.model_name not in accuracy_results:
            accuracy_results[args.model_name] = {}
            accuracy_std_results[args.model_name] = {}
            kappa_results[args.model_name] = {}
            kappa_std_results[args.model_name] = {}
            stft_reconstruction_results[args.model_name] = {}

        accuracy_results[args.model_name][dataset] = acc_mean
        accuracy_std_results[args.model_name][dataset] = acc_std
        kappa_results[args.model_name][dataset] = kappa_mean
        kappa_std_results[args.model_name][dataset] = kappa_std
        stft_reconstruction_results[args.model_name][dataset] = stft_result

        print("========================================")
        print(f"\n {dataset} dataset. All subjects completed!!!")
        print(
            f"Accuracy: {np.mean(all_accuracies):.2f} +- {np.std(all_accuracies):.2f}"
        )
        print(f"Kappa: {np.mean(all_kappas):.2f} +- {np.std(all_kappas):.2f}")

        if experiment is not None:
            yaml_dir = f"{args.experiment_folder}/{args.experiment_version}"
            os.makedirs(yaml_dir, exist_ok=True)
            results_yaml_path = f"{yaml_dir}/results.yaml"

            if os.path.exists(results_yaml_path):
                with open(results_yaml_path, "r") as f:
                    all_results = yaml.safe_load(f) or {}
            else:
                all_results = {}

            all_results.setdefault(args.model_name, {}).setdefault(
                run_timestamp, {}
            ).update(
                {
                    "experiment_description": args.experiment_description,
                    "config": run_config,
                    "results": dataset_results,
                }
            )

            with open(results_yaml_path, "w") as f:
                yaml.dump(all_results, f, default_flow_style=False, sort_keys=False)

    if experiment is not None:
        experiment.save()
        pd.DataFrame(accuracy_results).to_csv(
            f"{args.experiment_folder}/{args.experiment_version}/accuracy.results.csv"
        )
        pd.DataFrame(accuracy_std_results).to_csv(
            f"{args.experiment_folder}/{args.experiment_version}/accuracy_std.results.csv"
        )
        pd.DataFrame(kappa_results).to_csv(
            f"{args.experiment_folder}/{args.experiment_version}/kappa.results.csv"
        )
        pd.DataFrame(kappa_std_results).to_csv(
            f"{args.experiment_folder}/{args.experiment_version}/kappa_std.results.csv"
        )
        pd.DataFrame(stft_reconstruction_results).to_csv(
            f"{args.experiment_folder}/{args.experiment_version}/stft_reconstruction.results.csv"
        )


if __name__ == "__main__":
    main()
