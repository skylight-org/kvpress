#!/bin/bash
#SBATCH --job-name=p2e3
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rethinkingai-self/23b1835/phase2/experiment3/logs/%x-%j.out
# SWEEP=sweeps/probe.yaml sbatch job_sweep.sbatch     (GROUP=... to override results subdir)
set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate kvpress
# compute nodes have no DNS: without these every from_pretrained burns ~40s on retries
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CONDA_OVERRIDE_CUDA=12.9 HF_HOME="$HOME/hf" TMPDIR="$HOME/tmp" TOKENIZERS_PARALLELISM=false \
       HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
# SCBench's scorer path imports nltk -> sqlite3, which needs CXXABI_1.3.15 from the conda libstdc++
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$HOME/phase2/experiment3"
python run_sweep.py --sweep "${SWEEP:?set SWEEP}" ${GROUP:+--group "$GROUP"} 2>&1 \
  | grep --line-buffered -v "Loading weights"
