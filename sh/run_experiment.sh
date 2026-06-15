#!/bin/bash

source .venv/bin/activate
uv run python main.py --validation_strategy cv --dataset_name BNCI2014_001 --model_name db_conformer