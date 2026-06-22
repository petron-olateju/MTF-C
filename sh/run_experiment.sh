#!/bin/bash
set -x

source .venv/bin/activate
uv run python main.py --validation_strategy loso --dataset_name Cattan2019_PHMD --model_name mtf_c --experiment_details "Increase batch size to 64  disable spectrum reconstruction  enable channel attention pooling to mimick DBConformer dual-path"