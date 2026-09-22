#!/bin/bash
#SBATCH --job-name=p2e2
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/rethinkingai-self/23b1835/phase2/experiment2/logs/%x-%j.out
# SWEEP=sweeps/P1.yaml sbatch job_sweep.sbatch      (SMOKE=1 for the smoke run)
set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate kvpress
# compute nodes have no DNS: without these, every from_pretrained burns ~40s on retries first
export CONDA_OVERRIDE_CUDA=12.9 HF_HOME="$HOME/hf" TMPDIR="$HOME/tmp" TOKENIZERS_PARALLELISM=false \
       HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
# RULER's registry imports loogle -> nltk -> sqlite3, which needs CXXABI_1.3.15. The compute node's
# /lib64/libstdc++.so.6 does not have it; the conda env's does. E8/E9 never hit this because
# LongBench's scorer imports no nltk.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd "$HOME/phase2/experiment2"
python run_sweep.py --sweep "${SWEEP:?set SWEEP}" ${SMOKE:+--smoke} ${GROUP:+--group "$GROUP"} 2>&1 | grep --line-buffered -v "Loading weights"
