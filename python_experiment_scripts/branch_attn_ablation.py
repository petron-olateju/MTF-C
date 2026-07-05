import yaml
import subprocess
from tqdm import tqdm

with open('configs/model_params.yaml', 'r') as f:
    MODEL_PARAMS = yaml.safe_load(f)
DATASETS = [
    "Cattan2019_PHMD", "Rodrigues2017", 
    "Nakanishi2015", "Kalunga2016", 
    "BNCI2014_001", "BNCI2014_002", "BNCI2014_004"
]
VALIDATION = "loso"

for dataset in tqdm(DATASETS, total=len(DATASETS)):

    for spectrum_attn in [True, False]:
        MODEL_PARAMS['mtf_c']['spectrum_attn_flag'] = spectrum_attn
        for temporal_attn in [True, False]:
            MODEL_PARAMS['mtf_c']['temporal_attn_flag'] = temporal_attn

            if (spectrum_attn is False) and (temporal_attn is False):
                continue

            with open('configs/model_params.yaml', 'w') as f:
                yaml.dump(MODEL_PARAMS, f)

            EXPERIMENT_DETAILS = f"Frequency spectrum reconstruction, lambda=0.001  Transformer depth=2 across all branches  One LOSO repeat  spectrum_attn_pooling={spectrum_attn}  temporal_attn_pooling={temporal_attn}"

            subprocess.run(
                [
                    "python",
                    "main.py",
                    "--validation_strategy", VALIDATION,
                    "--dataset_name", dataset,
                    "--model_name", "mtf_c",
                    "--experiment_details", EXPERIMENT_DETAILS,
                "--output_dir", "experiments/classification/frequency_backbone"
                ],
                check=True,
            )