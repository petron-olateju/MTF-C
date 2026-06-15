import os
import yaml
import argparse
from datetime import datetime

from dataloader import TrainValTest_Split_Loader
from experiments import cross_validation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment parameters: [model_name, dataset_name, validation_strategy]"
    )

    parser.add_argument("--model_name", type=str, default="mtf_c")
    parser.add_argument("--dataset_name", type=str, default="BNCI2014_001")
    parser.add_argument(
        "--validation_strategy", type=str, choices=["cv", "loso"], default="cv"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="experiments")

    return parser.parse_args()


def main():
    args = parse_args()
    experiment_name = f"dataset={args.dataset_name} | validation_strategy={args.validation_strategy}"
    run_timestamp = datetime.now().isoformat()

    with open("configs/training_params.yaml", "r") as f:
        TRAINING_PARAMS = yaml.safe_load(f)
    N_REPEATS = TRAINING_PARAMS["n_repeats"]

    with open("configs/model_params.yaml", "r") as f:
        MODEL_PARAMS = yaml.safe_load(f)

    if args.validation_strategy == "cv":
        pass
        N_FOLDS = TRAINING_PARAMS["n_folds"]
        BATCH_SIZE = TRAINING_PARAMS["batch_size"]
        N_EPOCHS = TRAINING_PARAMS["n_epochs"]
        t0 = MODEL_PARAMS[args.dataset_name]["epoch_start"]
        t1 = MODEL_PARAMS[args.dataset_name]["epoch_end"]

        experiment = {
            'model': args.model_name,
            'experiment_seed': args.seed,
            'taining_params': TRAINING_PARAMS
        }

        performance = cross_validation(
            dataset_name=args.dataset_name,
            model_name=args.model_name,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            n_folds=N_FOLDS,
            n_repeats=N_REPEATS,
            preprocessing_pipeline=None,
            t0=t0,
            t1=t1,
            experiment_seed=args.seed,
        )

    experiment = {**experiment, **performance}
    experiment_path = os.path.join(args.output_dir, experiment_name)
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
