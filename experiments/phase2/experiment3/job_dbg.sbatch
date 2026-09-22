#!/bin/bash
#SBATCH --job-name=p2e3-dbg
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:40:00
#SBATCH --output=/home/rethinkingai-self/23b1835/phase2/experiment3/logs/%x-%j.out
set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate kvpress
export CONDA_OVERRIDE_CUDA=12.9 HF_HOME="$HOME/hf" TMPDIR="$HOME/tmp" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd "$HOME/phase2/experiment3"; python ${SCRIPT:-debug_kv.py} 2>&1 | grep --line-buffered -vE "Loading weights|it/s\]"
