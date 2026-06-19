#!/bin/bash
set -x

source .venv/bin/activate
uv run python main.py --validation_strategy loso --dataset_name BNCI2014_004 --model_name db_conformer