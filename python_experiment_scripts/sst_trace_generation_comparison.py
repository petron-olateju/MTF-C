import yaml
import subprocess
from tqdm import tqdm

with open('configs/model_params.yaml', 'r') as f:
    MODEL_PARAMS = yaml.safe_load(f)
DATASETS = [
    # "BNCI2014_001",
    "Cattan2019_PHMD", "Rodrigues2017", 
    "Nakanishi2015", "Kalunga2016", 
    "BNCI2014_001", "BNCI2014_002", "BNCI2014_004"
]
VALIDATION = "loso"
OUTPUT_DIR = "experiments/sst_trace_generation"

for dataset in tqdm(DATASETS, total=len(DATASETS)):
        
    # for sst_trace in ['sst_addition', 'sst_projection+addition', 'gated_sst_projection+addition']:
    for sst_trace in ['sst_multi_projection+branch_addition']:
        MODEL_PARAMS['mtf_c']['sst_trace'] = sst_trace

        for trace_target in ['soft_argmax']:
            MODEL_PARAMS['mtf_c']['trace_target'] = trace_target
            with open('configs/model_params.yaml', 'w') as f:
                yaml.dump(MODEL_PARAMS, f)

            EXPERIMENT_DETAILS = f"Transformer depth=2 across all branches  One LOSO repeat  channel_attn_pooling alone  sst_decoder=sst_projection+branch_addition  sst_trace={sst_trace}"
            
            subprocess.run(
                    [
                        "python",
                        "main.py",
                        "--validation_strategy", VALIDATION,
                        "--dataset_name", dataset,
                        "--model_name", "mtf_tr_c",
                        "--experiment_details", EXPERIMENT_DETAILS,
                    "--output_dir", OUTPUT_DIR
                    ],
                    check=True,
                )