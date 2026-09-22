#!/bin/bash
#SBATCH --job-name=p2e4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/home/rethinkingai-self/23b1835/phase2/experiment4/logs/%x-%j.out
# SWEEP=sweeps/smoke.yaml sbatch -p l40 --qos=l40 --job-name=p2e4-smoke job_sweep.sbatch
# (a40 and l40 cap --time at 4d/2d; 24h fits every partition used here)
set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate kvpress
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# compute nodes have no DNS: without these every from_pretrained burns ~40s on retries
export CONDA_OVERRIDE_CUDA=12.9 HF_HOME="$HOME/hf" TMPDIR="$HOME/tmp" TOKENIZERS_PARALLELISM=false \
       HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
# SCBench's scorer path imports nltk -> sqlite3, which needs CXXABI_1.3.15 from the conda libstdc++
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$HOME/phase2/experiment4"
python run_sweep.py --sweep "${SWEEP:?set SWEEP}" 2>&1 | grep --line-buffered -v "Loading weights"
