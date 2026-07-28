#!/bin/bash -l
#SBATCH --job-name=4b-full-embed
#SBATCH --partition=h100
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=360G
#SBATCH --time=48:00:00
#SBATCH --output=/home/jschwab/FinGPT/embeddings/4b/logs/%j.out
#SBATCH --error=/home/jschwab/FinGPT/embeddings/4b/logs/%j.err

module purge
module load python cuda 2>/dev/null || true

VENV=/scratch/jschwab/venvs/chronogpt

if [ ! -d "$VENV" ]; then
    echo "Creating venv at $VENV ..."
    python -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install torch tiktoken pandas huggingface_hub
fi

export PATH="$VENV/bin:$PATH"

export PATH="$VENV/bin:$PATH"

SCRIPT_DIR=/home/jschwab/FinGPT/embeddings/4b
cd "$SCRIPT_DIR"
export PYTHONPATH="/home/jschwab/FinGPT:$PYTHONPATH"
export PYTHONUNBUFFERED=1

mkdir -p /home/jschwab/FinGPT/embeddings/4b/logs
mkdir -p /scratch/jschwab/embeddings/4b-full-v2

echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun python3 main.py --last_ckpt

echo "Finished : $(date)"
