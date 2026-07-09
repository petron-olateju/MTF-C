import yaml
import subprocess
from tqdm import tqdm

with open('configs/model_params.yaml', 'r') as f:
    MODEL_PARAMS = yaml.safe_load(f)
DATASETS = [
    "Cattan2019_PHMD",
    # "Cattan2019_PHMD", "Rodrigues2017", 
    # "Nakanishi2015", "Kalunga2016", 
    # "BNCI2014_001", "BNCI2014_002", "BNCI2014_004"
]
VALIDATION = "loso"
OUTPUT_DIR = "experiments/sst_trace_generation"

for dataset in tqdm(DATASETS, total=len(DATASETS)):

    # for sst_decoder in ['st_addition', 'st_projection+addition', 'st_projection_addition']:
    #     MODEL_PARAMS['db_conformer']['sst_decoder'] = sst_decoder
    #     MODEL_PARAMS['mtf_c']['sst_decoder'] = sst_decoder
    #     with open('configs/model_params.yaml', 'w') as f:
    #         yaml.dump(MODEL_PARAMS, f)

    #     EXPERIMENT_DETAILS = f"Transformer depth=2 across all branches  One LOSO repeat  channel_attn_pooling alone  sst_decoder={sst_decoder}"

    #     subprocess.run(
    #             [
    #                 "python",
    #                 "main.py",
    #                 "--validation_strategy", VALIDATION,
    #                 "--dataset_name", dataset,
    #                 "--model_name", "db_r_conformer",
    #                 "--experiment_details", EXPERIMENT_DETAILS,
    #             "--output_dir", OUTPUT_DIR
    #             ],
    #             check=True,
    #         )
        
    #     subprocess.run(
    #             [
    #                 "python",
    #                 "main.py",
    #                 "--validation_strategy", VALIDATION,
    #                 "--dataset_name", dataset,
    #                 "--model_name", "mtf_r_c",
    #                 "--experiment_details", EXPERIMENT_DETAILS,
    #             "--output_dir", OUTPUT_DIR
    #             ],
    #             check=True,
    #         )
        
    # for sst_trace in ['sst_addition', 'sst_projection+addition', 'gated_sst_projection+addition']:
    for sst_trace in ['sst_multi_projection+branch_addition']:
        MODEL_PARAMS['mtf_c']['sst_trace'] = sst_trace
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