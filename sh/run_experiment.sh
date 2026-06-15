#!/bin/bash

source .venv/bin/activate
uv run python main.py --validation_strategy train_test --dataset_name BNCI2014_001 --model_name mtf_c