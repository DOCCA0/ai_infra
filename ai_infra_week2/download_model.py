from huggingface_hub import snapshot_download
import os

model_id = "gpt2"
local_dir = "/home/wu/code/ai_infra/ai_infra_week2/gpt2"

print(f"Downloading {model_id} to {local_dir}...")
snapshot_download(repo_id=model_id, local_dir=local_dir)
print("Download complete.")