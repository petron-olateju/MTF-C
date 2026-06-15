#!/bin/bash
set -x

source .venv/bin/activate

models=("db_conformer" "mtf_c")
datasets=("BNCI2014_001" "BNCI2014_002" "BNCI2014_004" "BNCI2015_001" "BNCI2015_004" "Liu2024" "AlexMI" "Kalunga2016" "Nakanishi2015" "Wang2021Combined" "Cattan2019_PHMD" "Hinss2021" "Rodrigues2017")
validation="cv"

for model in "${models[@]}"; do
    for dataset in "${datasets}"; do
        uv run python main.py --validation_strategy "$validation" --dataset_name "$dataset" --model_name "$model"
    done
done