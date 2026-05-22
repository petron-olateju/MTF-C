#!/bin/bash

source .venv/bin/activate

uv run python run_experiment.py \
    --script "loso" \
    --model_name "db_conformer" \
    --dataset "mi" \
    --device "cuda" \
    --experiment_version "DBConformer_MI_LOSO" \
    --experiment_description "DBConformer MOTOR_IMAGERY benchmark on REST datasets" \
    --messages "use 5 repeats of LOSO validation with different seeds for confirmation of DBConformer."