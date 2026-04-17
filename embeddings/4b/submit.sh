#!/bin/bash -l
#SBATCH --job-name=4b-embed
#SBATCH --partition=l40s
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=360G
#SBATCH --time=48:00:00
#SBATCH --output=/home/jschwab/FinGPT/embeddings/4b/logs/%j.out
#SBATCH --error=/home/jschwab/FinGPT/embeddings/4b/logs/%j.err

module purge
module load python cuda 2>/dev/null || true

VENV=/scratch/jschwab/venvs/chronogpt
source "$VENV/bin/activate"

SCRIPT_DIR=/home/jschwab/FinGPT/embeddings/4b
cd "$SCRIPT_DIR"
export PYTHONPATH="/home/jschwab/FinGPT:$PYTHONPATH"
export PYTHONUNBUFFERED=1

mkdir -p /home/jschwab/FinGPT/embeddings/4b/logs
mkdir -p /scratch/jschwab/embeddings/4b

echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun python3 main.py

echo "Finished : $(date)"
