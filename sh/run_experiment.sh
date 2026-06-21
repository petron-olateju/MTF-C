#!/bin/bash
set -x

source .venv/bin/activate
uv run python main.py --validation_strategy loso --dataset_name Cattan2019_PHMD --model_name mtf_c