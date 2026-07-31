import os
import subprocess
import yaml

with open("secrets.yaml") as f:
    secrets = yaml.safe_load(f)

creds = secrets["CloudfareR2"]["cloud_access_token"]

env = os.environ.copy()
env["AWS_ACCESS_KEY_ID"] = creds["access_key_ID"]
env["AWS_SECRET_ACCESS_KEY"] = creds["secret_access_key"]
env["AWS_DEFAULT_REGION"] = "auto"
print("AWS CREDENTIALS LOADED")

subprocess.run([
    "aws", "s3", "sync",
    "experiments/",
    "s3://mtfc-model-checkpoints/experiments/",
    "--exclude", "*",
    "--include", "**/dataset*/*.ckpt",
    "--endpoint-url",
    "https://462a449f422520aa7ab25f587cc14d39.r2.cloudflarestorage.com",
], env=env, check=True)
print("MTFC MODEL CHECKPOINTS UPLOADED")