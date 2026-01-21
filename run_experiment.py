from tqdm import tqdm
import csv

import numpy as np
import argparse
from argparse import Namespace

from utils.data_loader import get_subjects_BNCI2014_001
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
        choices=['dummy_dataset', 'BNCI2014_001']
    )
    parser.add_argument('--device', type=str, default='cpu',
        choices = ['cpu', 'cuda']
    )
    parser.add_argument('--csv_output', type=str, default='./',)
    parser.add_argument('--verbose', action='store_true',
        help='Print training progress')

    return parser.parse_args()

def main():
    args = parse_args()

    if args.dataset == 'BNCI2014_001':
        subjects = get_subjects_BNCI2014_001()
    else:
        subjects = [1]
    
    script = args.script
    model_name = args.model_name
    dataset = args.dataset
    device = args.device

    accuracies = []
    kappas = []

    for subject in tqdm(subjects, total=len(subjects), desc="Training"):
        print(f"\n{'='*50}")
        print(f"Running Subject {subject}")
        print(f"{'='*50}\n")

        args = Namespace(
            dataset=dataset,
            subject=subject,
            device=device,
            model_name=model_name,
            verbose=True
        )
        if script == 'cross_validation':
            accuracy, kappa = cross_validation(args)
        else:
            accuracy = 0.5
            kappa = 0.0
        accuracies.append(accuracy)
        kappas.append(kappa)
        print(f"subject {subject} | Accuracy: {accuracy:.2f}, Kappa: {kappa:.2f}")
    print("\n \n \n")
    print("\nAll subjects completed!")
    print(f"Accuracy: {np.mean(accuracies):.2f} +- {np.std(accuracies):.2f}")
    print(f"Kappa: {np.mean(kappa):.2f} +- {np.std(kappas):.2f}")


if __name__ == '__main__':
    main()