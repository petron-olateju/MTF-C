#!/bin/bash
set -x

source .venv/bin/activate
uv run python main.py --validation_strategy loso --dataset_name Cattan2019_PHMD --model_name mtf_c --experiment_details "Increase batch size to 64  frequency spectrum reconstruction (lambda=0.1) with channel attention pooling  fix:restrict compute_bandpowers frequency range to 1-30Hz"