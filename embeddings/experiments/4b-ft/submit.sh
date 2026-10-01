#!/bin/bash -l
# SLURM job: embed all DJN articles with the time-varying SFT'd 4B checkpoints on 4 GPUs.
# Usage (from the repository root): mkdir -p logs && sbatch embeddings/experiments/4b-ft/submit.sh
# Paths are read from embeddings/experiments/.env; set VENV=/path/to/venv to use that environment (optional).
#SBATCH --job-name=4b-ft-embed
#SBATCH --partition=h100
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=360G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Load modules where the cluster has them (adjust names for your cluster)
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
export PYTHONUNBUFFERED=1

echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun python3 embeddings/experiments/4b-ft/main.py

echo "Finished : $(date)"
