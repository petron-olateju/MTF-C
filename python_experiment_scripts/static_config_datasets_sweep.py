import yaml
import subprocess
from tqdm import tqdm

DATASETS = [
    # "BNCI2014_002"
    "Cattan2019_PHMD", "Rodrigues2017", 
    "Nakanishi2015", "Kalunga2016", 
    "BNCI2014_001", "BNCI2014_004"
]
VALIDATION = "loso"
OUTPUT_DIR = "experiments/sst_decoding+NRMSE_Loss"

for dataset in tqdm(DATASETS, total=len(DATASETS)):

    EXPERIMENT_DETAILS = f"Transformer depth=2 across all branches  One LOSO repeat  channel_attn_pooling alone  sst_decoder=sst_multi_projection+branch_addition"
    
    subprocess.run(
            [
                "python",
                "main.py",
                "--validation_strategy", VALIDATION,
                "--dataset_name", dataset,
                "--model_name", "mtf_r_c",
                "--experiment_details", EXPERIMENT_DETAILS,
            "--output_dir", OUTPUT_DIR
            ],
            check=True,
        )