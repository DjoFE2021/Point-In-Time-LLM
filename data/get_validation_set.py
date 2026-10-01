"""Download the FineWeb10B validation shard fineweb_val_000000.bin (GPT-2 tokens) from the HF Hub.

Usage: python data/get_validation_set.py [LOCAL_DIR]   (default: ~/fineweb_pit/fineweb10B)
"""
import os
import sys
from huggingface_hub import hf_hub_download

# Download one file of kjj0/fineweb10B-gpt2 into local_dir unless it is already there

def get(fname, local_dir):
    if not os.path.exists(os.path.join(local_dir, fname)):
        hf_hub_download(repo_id="kjj0/fineweb10B-gpt2", filename=fname,
                        repo_type="dataset", local_dir=local_dir)
        print(f"Downloaded: {fname}")
    else:
        print(f"Already exists: {fname}")

if __name__ == "__main__":
    # Target directory: first CLI argument, else ~/fineweb_pit/fineweb10B
    default_dir = os.path.expanduser("~/fineweb_pit/fineweb10B")
    os.makedirs(default_dir, exist_ok=True)
    local_dir = sys.argv[1] if len(sys.argv) > 1 else default_dir
    
    os.makedirs(local_dir, exist_ok=True)
    print(f"Downloading to: {local_dir}")
    
    # Only the validation shard is needed; training data comes from get_train_set.py
    get("fineweb_val_%06d.bin" % 0, local_dir)
