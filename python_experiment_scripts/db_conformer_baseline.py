import yaml
import subprocess
from tqdm import tqdm

with open('configs/model_params.yaml', 'r') as f:
    MODEL_PARAMS = yaml.safe_load(f)
DATASETS = [
    # "Cattan2019_PHMD", "Rodrigues2017", 
    # "Nakanishi2015", "Kalunga2016", 
    # "BNCI2014_001", 
    "BNCI2014_002", "BNCI2014_004"
]
VALIDATION = "loso"

for dataset in tqdm(DATASETS, total=len(DATASETS)):
    subprocess.run(
        [
            "python",
            "main.py",
            "--validation_strategy", VALIDATION,
            "--dataset_name", dataset,
            "--model_name", "db_conformer",
            "--experiment_details", "db_conformer baseline  make transformer depth=1 across all branches  use just one LOSO repeat",
            "--output_dir", "experiments/classification/frequency_backbone"
        ],
        check=True,
    )