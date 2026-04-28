import os
import sys
from huggingface_hub import hf_hub_download

# Download the GPT-2 tokens of Fineweb10B from huggingface. This
# saves about an hour of startup time compared to regenerating them.

def get(fname, local_dir):
    if not os.path.exists(os.path.join(local_dir, fname)):
        hf_hub_download(repo_id="kjj0/fineweb10B-gpt2", filename=fname,
                        repo_type="dataset", local_dir=local_dir)
        print(f"Downloaded: {fname}")
    else:
        print(f"Already exists: {fname}")

if __name__ == "__main__":
    # Default to home directory
    default_dir = os.path.expanduser("~/fineweb_pit/fineweb10B")
    os.makedirs(default_dir, exist_ok=True)
    local_dir = sys.argv[1] if len(sys.argv) > 1 else default_dir
    
    os.makedirs(local_dir, exist_ok=True)
    print(f"Downloading to: {local_dir}")
    
    # Always download validation file
    get("fineweb_val_%06d.bin" % 0, local_dir)
