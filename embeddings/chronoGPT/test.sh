#!/bin/bash -l
#SBATCH --job-name=chronogpt-embed
#SBATCH --partition=l40s
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/jschwab/FinGPT/embeddings/chronoGPT/logs/%j.out
#SBATCH --error=/home/jschwab/FinGPT/embeddings/chronoGPT/logs/%j.err

module purge
# Fill in exact names from: module spider python && module spider cuda
module load python cuda 2>/dev/null || true

VENV=/scratch/jschwab/venvs/chronogpt

if [ ! -d "$VENV" ]; then
    echo "Creating venv at $VENV ..."
    python -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install torch tiktoken pandas huggingface_hub
fi

source "$VENV/bin/activate"

SCRIPT_DIR=/home/jschwab/FinGPT/embeddings/chronoGPT
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

export HF_HOME=/scratch/jschwab/hf
unset TRANSFORMERS_CACHE
export PYTHONUNBUFFERED=1

mkdir -p /home/jschwab/FinGPT/embeddings/chronoGPT/logs
mkdir -p /scratch/jschwab/hf
mkdir -p /scratch/jschwab/embeddings/chronogpt_instruct
mkdir -p /scratch/jschwab/embeddings/chronogpt_base

echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MODEL_TYPE=instruct   # change to "base" for the base model
echo "Model type: $MODEL_TYPE"

python3 main.py --model-type "$MODEL_TYPE" --test


echo "Finished : $(date)"
