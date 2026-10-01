#!/bin/bash -l
# SLURM job: embed all news articles with ChronoGPT on 4 GPUs, then aggregate them monthly.
# Usage (from the repository root): mkdir -p logs && sbatch embeddings/experiments/chronoGPT/submit.sh
#   (base model: sbatch --export=ALL,MODEL_TYPE=base embeddings/experiments/chronoGPT/submit.sh)
# Paths are read from embeddings/experiments/.env; set VENV=/path/to/venv to use that environment (optional).
#SBATCH --job-name=chronogpt-embed
#SBATCH --partition=h100
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=360G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Adjust module names for your cluster (see: module spider python / module spider cuda)
if command -v module >/dev/null 2>&1; then
    module purge
    module load python cuda 2>/dev/null || true
fi

if [ -n "${VENV:-}" ]; then
    export PATH="$VENV/bin:$PATH"
fi

# Run from the repository root (the directory sbatch was called from)
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR" || exit 1
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Snapshots download to the default Hugging Face cache (honours HF_HOME) or CHRONOGPT_CACHE_DIR
export PYTHONUNBUFFERED=1

echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MODEL_TYPE=${MODEL_TYPE:-instruct}  # "instruct" or "base"
echo "Model type: $MODEL_TYPE"

srun python3 embeddings/experiments/chronoGPT/main.py --model-type "$MODEL_TYPE"

echo "Finished : $(date)"
