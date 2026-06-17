import os
import yaml
import argparse
from datetime import datetime

from dataloader import TrainValTest_Split_Loader
from experiments import across_subjects_evaluation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment parameters: [model_name, dataset_name, validation_strategy]"
    )

    parser.add_argument("--model_name", type=str, default="mtf_c")
    parser.add_argument("--dataset_name", type=str, default="BNCI2014_001")
    parser.add_argument(
        "--validation_strategy", type=str, choices=["cv", "loso", "train_test"], default="cv"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="experiments")

    return parser.parse_args()


def main():
    args = parse_args()
    experiment_name = f"dataset={args.dataset_name} | validation_strategy={args.validation_strategy}"
    run_timestamp = datetime.now().isoformat()
    experiment_path = os.path.join(args.output_dir, experiment_name)
    

    with open("configs/training_params.yaml", "r") as f:
        TRAINING_PARAMS = yaml.safe_load(f)
    with open("configs/dataset_params.yaml", "r") as f:
        DATASET_PARAMS = yaml.safe_load(f)
    t0 = DATASET_PARAMS[args.dataset_name]["epoch_start"]
    t1 = DATASET_PARAMS[args.dataset_name]["epoch_end"]

    experiment = {
        'model': args.model_name,
        'experiment_seed': args.seed,
        'taining_params': TRAINING_PARAMS,
        'dataset_params': DATASET_PARAMS[args.dataset_name]
    }

    performance = across_subjects_evaluation(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        experiment_seed=args.seed,
        validation_strategy=args.validation_strategy,
        t0=t0,
        t1=t1,
        experiment_path=experiment_path,
        run_timestamp=run_timestamp,
        **TRAINING_PARAMS
    )

    experiment = {**experiment, **performance}
    os.makedirs(experiment_path, exist_ok=True)
    if os.path.exists(f"{experiment_path}/history.yaml"):
        with open(f"{experiment_path}/history.yaml", "r") as f:
            history = yaml.safe_load(f)
    else:
        history = {}
    history[run_timestamp] = experiment
    with open(f"{experiment_path}/history.yaml", "w") as f:
        yaml.dump(history, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()
