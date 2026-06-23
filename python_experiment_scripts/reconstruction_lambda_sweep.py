import yaml
import subprocess

with open('configs/model_params.yaml', 'r') as f:
    MODEL_PARAMS = yaml.safe_load(f)
DATASETS = [
    # "Cattan2019_PHMD", 
    "Rodrigues2017", 
    "Nakinishi2015", "Kalunga2016", "Wang2021Combined", 
    "BNCI2014_001", "BNCI2014_002", "BNCI2014_004"
]
LAMBDAS = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1.0]

VALIDATION = "loso"

for dataset in DATASETS:
    MODEL_PARAMS['mtf_c']['sst_method'] = None
    MODEL_PARAMS['mtf_c']['reconstruction_lambda'] = 0.0
    with open('configs/model_params.yaml', 'w') as f:
        yaml.dump(MODEL_PARAMS, f)
    for lambda_ in LAMBDAS:
        MODEL_PARAMS['mtf_c']['reconstruction_lambda'] = lambda_
        with open('configs/model_params.yaml', 'w') as f:
            yaml.dump(MODEL_PARAMS, f)

        EXPERIMENT_DETAILS = f"Increase batch size to 64  frequency spectrum reconstruction (lambda={lambda_}) with channel attention pooling"

        subprocess.run(
            [
                "python",
                "main.py",
                "--validation_strategy", VALIDATION,
                "--dataset_name", dataset,
                "--model_name", "mtf_c",
                "--experiment_details", EXPERIMENT_DETAILS,
            ],
            check=True,
        )