#!/bin/bash
set -x

source .venv/bin/activate

models=("db_conformer" "mtf_c")
datasets=("BNCI2014_001" "BNCI2014_002" "BNCI2014_004") 
validation="loso"

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        uv run python main.py --validation_strategy "$validation" --dataset_name "$dataset" --model_name "$model"
    done
done