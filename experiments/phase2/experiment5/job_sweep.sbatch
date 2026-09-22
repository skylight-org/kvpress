#!/bin/bash
#SBATCH --job-name=p2e5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rethinkingai-self/23b1835/phase2/experiment5/logs/%x-%j.out
set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate kvpress-pr
# The kvpress-pr env has src/kvpress-skylight editable-installed; E5 must use the e5 worktree
# instead, which carries VKvWRPress, the micro-metric logging and the head-stats metric.
export PYTHONPATH="$HOME/src/kvpress-e5"
export CONDA_OVERRIDE_CUDA=12.9 HF_HOME="$HOME/hf" TMPDIR="$HOME/tmp" \
       TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# SCBench's scorer imports nltk -> sqlite3, which needs CXXABI_1.3.15 from the conda libstdc++
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$HOME/phase2/experiment5"
python run_sweep.py --sweep "$SWEEP"
