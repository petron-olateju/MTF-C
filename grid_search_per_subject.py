from tqdm import tqdm

import math
import yaml
import os
from datetime import datetime
from itertools import product

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
            'BNCI2015_001', 'BNCI2015_004', 'AlexMI']
    )
    parser.add_argument('--subject', type=int, default=1)
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
    with open('configs/grid_search.yaml', 'r') as f:
        configs = yaml.safe_load(f)
    grid_params = configs[args.model_name]

    experiment = Experiment(
        args.experiment_version, 
        dir = args.experiment_folder,
        description = args.experiment_description
    )
    
    script = args.script
    model_name = args.model_name
    if args.dataset == 'all':
        raise ValueError('--dataset argument has to be single dataset when running grid search per subject')
    device = args.device

    accuracy_results = { }
    accuracy_std_results = { }
    kappa_results = { }
    kappa_std_results = { }
    stft_reconstruction_results = { }

    # Build all combinations for the current model
    param_names = list(grid_params.keys())
    param_values = list(grid_params.values())

    best_acc = 0
    best_acc_std = 0
    best_kappa = 0
    best_kappa_std = 0
    best_stft_loss = math.inf
    best_params = None

    combos = product(*param_values)
    total_combos = 1
    for v in param_values:
        total_combos *= len(v)
    for combo in tqdm(combos, total=total_combos, desc='Grid Search'):
        current_param = dict(zip(param_names, combo))

        mthd_args = Namespace(
            dataset=args.dataset,
            subject=args.subject,
            device=device,
            model_name=model_name,
            verbose=True,
        )
    
        if script == 'cross_validation':
            accuracy, kappa, stft_reconstruction_loss, experiment = cross_validation(mthd_args, experiment, config='grid_search', model_configs=current_param)

        if args.model_name != 'mtf_c':
            print(f"subject {args.subject} | Accuracy: {np.mean(accuracy):.2f}, Kappa: {np.mean(kappa):.2f}")
            print("============================================")
        else:
            print(f"subject {args.subject} | Accuracy: {np.mean(accuracy):.2f}, Kappa: {np.mean(kappa):.2f} | STFT Reconstruction Loss: {np.mean(stft_reconstruction_loss):.2f}")
            print("============================================")

        acc_mean = np.mean(accuracy)
        acc_std = np.std(accuracy)
        kappa_mean = np.mean(kappa)
        kappa_std = np.mean(kappa)
        stft_result = np.mean(stft_reconstruction_loss)

        if acc_mean > best_acc:
            best_acc = acc_mean
            best_acc_std = acc_std
            best_kappa = kappa_mean
            best_kappa_std = kappa_std
            best_stft_loss = stft_result
            best_params = current_param
        elif (acc_mean == best_acc) and (stft_result <= best_stft_loss):
            best_acc = acc_mean
            best_acc_std = acc_std
            best_kappa = kappa_mean
            best_kappa_std = kappa_std
            best_stft_loss = stft_result
            best_params = current_param

    if args.model_name not in accuracy_results:
        accuracy_results[args.model_name] = { }
        accuracy_std_results[args.model_name] = { }
        kappa_results[args.model_name] = { }
        kappa_std_results[args.model_name] = { }
        stft_reconstruction_results[args.model_name] = { }

    accuracy_results[args.model_name][args.dataset] = acc_mean
    accuracy_std_results[args.model_name][args.dataset] = acc_std
    kappa_results[args.model_name][args.dataset] = kappa_mean
    kappa_std_results[args.model_name][args.dataset] = kappa_std
    stft_reconstruction_results[args.model_name][args.dataset] = stft_result

    print("========================================")
    print(f"\n Grid search for subject {args.subject} {args.dataset} dataset completed!!!")
    print(f"Accuracy: {best_acc:.2f} +- {best_acc_std:.2f}")
    print(f"Kappa: {best_kappa:.2f} +- {best_kappa_std:.2f}")
    if args.model_name == 'mtf_c':
        print(f"STFT MSE Loss: {best_stft_loss:.2f}")

    if experiment is not None:
        experiment.save()

        yaml_dir = f'{args.experiment_folder}/{args.experiment_version}'
        os.makedirs(yaml_dir, exist_ok=True)

        run_entry = {
            'timestamp': datetime.now().isoformat(),
            'model': args.model_name,
            'hyperparameters': best_params,
            'dataset': args.dataset,
            'subject': args.subject,
            'metrics': {
                'accuracy': {
                    'mean': float(acc_mean),
                    'std': float(acc_std)
                },
                'kappa': {
                    'mean': float(kappa_mean),
                    'std': float(kappa_std)
                },
                'stft_reconstruction_loss': float(stft_result)
            }
        }

        results_yaml_path = f'{yaml_dir}/results.yaml'

        # Load existing YAML or start fresh
        if os.path.exists(results_yaml_path):
            with open(results_yaml_path, 'r') as f:
                all_results = yaml.safe_load(f) or {}
        else:
            all_results = {}

        # Nested structure: model -> dataset -> subject -> list of runs
        all_results \
            .setdefault(args.model_name, {}) \
            .setdefault(args.dataset, {}) \
            .setdefault(f'subject_{args.subject}', []) \
            .append(run_entry)

        with open(results_yaml_path, 'w') as f:
            yaml.dump(all_results, f, default_flow_style=False, sort_keys=False)

        print(f"Results saved to {results_yaml_path}")


if __name__ == '__main__':
    main()