#!/bin/bash -l
#SBATCH --job-name=4b-ft-embed-test
#SBATCH --partition=l40s
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=46G
#SBATCH --time=01:00:00
#SBATCH --output=/home/jschwab/FinGPT/embeddings/4b-ft/logs/%j.out
#SBATCH --error=/home/jschwab/FinGPT/embeddings/4b-ft/logs/%j.err

module purge
module load python cuda 2>/dev/null || true

VENV=/scratch/jschwab/venvs/chronogpt
export PATH="$VENV/bin:$PATH"

SCRIPT_DIR=/home/jschwab/FinGPT/embeddings/4b-ft
cd "$SCRIPT_DIR"
export PYTHONPATH="/home/jschwab/FinGPT:$PYTHONPATH"
export PYTHONUNBUFFERED=1

mkdir -p /home/jschwab/FinGPT/embeddings/4b-ft/logs

echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 main.py --test

echo "Finished : $(date)"
