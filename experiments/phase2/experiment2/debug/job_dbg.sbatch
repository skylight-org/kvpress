#!/bin/bash
#SBATCH --job-name=p2e2dbg
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/home/rethinkingai-self/23b1835/phase2/experiment2/logs/%x-%j.out
set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate kvpress
export CONDA_OVERRIDE_CUDA=12.9 HF_HOME="$HOME/hf" TMPDIR="$HOME/tmp" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd "$HOME/phase2/experiment2"
python debug/${SCRIPT:-dbg_e2e.py} 2>&1 | grep -vE "Loading weights|KVzipPress requires|it/s\]"
