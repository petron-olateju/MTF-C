#!/bin/bash

source .venv/bin/activate

uv run python run_experiment.py \
    --script "cross_validation" \
    --model_name "mtf_c" \
    --dataset "mi" \
    --device "cuda" \
    --experiment_version "MTFC+FilterBanks_MI_CV" \
    --experiment_description "MTFC MOTOR_IMAGERY benchmark on MI datasets in 4 fold CV mode" \
    --messages "use 5 repeats (seeds) of five-fold CV per subject"