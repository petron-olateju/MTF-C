from tqdm import tqdm

import numpy as np
import pandas as pd
import argparse
from argparse import Namespace

from utils.data_loader import get_subjects_BNCI2014_001, get_subjects_BNCI2014_002, get_subjects_BNCI2014_004
from utils.data_loader import get_subjects_BNCI2015_001, get_subjects_BNCI2015_004, get_subjects_Liu2024
from utils.data_loader import get_subjects_AlexMI
from utils.experiment_recorder import Parameter, Experiment
from cross_validation import main as cross_validation

def parse_args():
    parser = argparse.ArgumentParser(description='Experiment Script runner: Model, Hyperparameters, ExperimentLogger options')

    parser.add_argument('--script', type=str, default='cross_validation.py', 
        choices=[
            'cross_validation',
            'pre_training'
        ])

    parser.add_argument('--model_name', type=str, default='db_conformer',
        choices=[
            'db_conformer',
            'mtf_c',
            'dual_tsst'
        ])
    parser.add_argument('--dataset', type=str, default='dummy_dataset',
        choices=[
            'dummy_dataset', 'BNCI2014_001', 'BNCI2014_002', 'BNCI2014_004', 
            'BNCI2015_001', 'BNCI2015_004', 'AlexMI', 'all']
    )
    parser.add_argument('--device', type=str, default='cpu',
        choices = ['cpu', 'cuda']
    )
    parser.add_argument('--experiment_version', type=str, default='v0.0')
    parser.add_argument('--experiment_description', type=str, default='Baseline Experiment')
    parser.add_argument('--experiment_folder', type=str, default='./experiments',)
    parser.add_argument('--verbose', action='store_true',
        help='Print training progress')

    return parser.parse_args()

def main():
    args = parse_args()

    experiment = Experiment(
        args.experiment_version, 
        dir = args.experiment_folder,
        description = args.experiment_description
    )
    
    script = args.script
    model_name = args.model_name
    if args.dataset == 'all':
        datasets = ['BNCI2014_001', 'BNCI2014_002', 'BNCI2014_004']
        # datasets = ['AlexMI',]
    else:
        datasets = [args.dataset]
    device = args.device

    all_accuracies = []
    all_kappas = []
    all_stft = []

    accuracy_results = { }
    accuracy_std_results = { }
    kappa_results = { }
    kappa_std_results = { }
    stft_reconstruction_results = { }

    for dataset in datasets:
        if dataset == 'BNCI2014_001':
            subjects = get_subjects_BNCI2014_001()
            print(f'{dataset} subjects: {subjects}')
        elif dataset == 'BNCI2014_002':
            subjects = get_subjects_BNCI2014_002()
            print(f'{dataset} subjects: {subjects}')
        elif dataset == 'BNCI2014_004':
            subjects = get_subjects_BNCI2014_004()
            print(f'{dataset} subjects: {subjects}')
        elif dataset == 'BNCI2015_001':
            subjects = get_subjects_BNCI2015_001()
            print(f'{dataset} subjects: {subjects}')
        elif dataset == 'BNCI2015_004':
            subjects = get_subjects_BNCI2015_004()
            print(f'{dataset} subjects: {subjects}')
        elif dataset == 'Liu2024':
            subjects = get_subjects_Liu2024()
            print(f'{dataset} subjects: {subjects}')
        elif dataset == 'AlexMI':
            subjects = get_subjects_AlexMI()
            print(f'{dataset} subjects: {subjects}')
        else:
            subjects = [1]  # Default for dummy_dataset or unknown datasets

        for subject in tqdm(subjects, total=len(subjects), desc="Training"):

            mthd_args = Namespace(
                dataset=dataset,
                subject=subject,
                device=device,
                model_name=model_name,
                verbose=True
            )
           
            if script == 'cross_validation':
                accuracy, kappa, stft_reconstruction_loss, experiment = cross_validation(mthd_args, experiment)
                all_accuracies = all_accuracies + accuracy
                all_kappas = all_kappas + kappa
                all_stft = all_stft + stft_reconstruction_loss
            if args.model_name != 'mtf_c':
                print(f"subject {subject} | Accuracy: {np.mean(accuracy):.2f}, Kappa: {np.mean(kappa):.2f}") # type: ignore
                print("============================================")
            else:
                print(f"subject {subject} | Accuracy: {np.mean(accuracy):.2f}, Kappa: {np.mean(kappa):.2f} | STFT Reconstruction Loss: {np.mean(stft_reconstruction_loss):.2f}") # type: ignore
                print("============================================")


        acc_mean = np.mean(all_accuracies)
        acc_std = np.std(all_accuracies)
        kappa_mean = np.mean(all_kappas)
        kappa_std = np.mean(all_kappas)
        stft_result = np.mean(all_stft)

        if args.model_name not in accuracy_results:
            accuracy_results[args.model_name] = { }
            accuracy_std_results[args.model_name] = { }
            kappa_results[args.model_name] = { }
            kappa_std_results[args.model_name] = { }
            stft_reconstruction_results[args.model_name] = { }

        accuracy_results[args.model_name][dataset] = acc_mean
        accuracy_std_results[args.model_name][dataset] = acc_std
        kappa_results[args.model_name][dataset] = kappa_mean
        kappa_std_results[args.model_name][dataset] = kappa_std
        stft_reconstruction_results[args.model_name][dataset] = stft_result

        print("========================================")
        print(f"\n {dataset} dataset. All subjects completed!!!")
        print(f"Accuracy: {np.mean(all_accuracies):.2f} +- {np.std(all_accuracies):.2f}")
        print(f"Kappa: {np.mean(all_kappas):.2f} +- {np.std(all_kappas):.2f}")
    
    if experiment is not None:
        experiment.save()
        pd.DataFrame(accuracy_results).to_csv(f'{args.experiment_folder}/{args.experiment_version}/accuracy.results.csv')
        pd.DataFrame(accuracy_std_results).to_csv(f'{args.experiment_folder}/{args.experiment_version}/accuracy_std.results.csv')
        pd.DataFrame(kappa_results).to_csv(f'{args.experiment_folder}/{args.experiment_version}/kappa.results.csv')
        pd.DataFrame(kappa_std_results).to_csv(f'{args.experiment_folder}/{args.experiment_version}/kappa_std.results.csv')
        pd.DataFrame(stft_reconstruction_results).to_csv(f'{args.experiment_folder}/{args.experiment_version}/stft_reconstruction.results.csv')


if __name__ == '__main__':
    main()