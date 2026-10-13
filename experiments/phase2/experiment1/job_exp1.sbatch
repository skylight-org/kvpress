#!/bin/bash
#SBATCH --job-name=p2e1
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/rethinkingai-self/23b1835/phase2/experiment1/logs/%x-%j.out
# SMOKE from the submitting shell: SMOKE=1 sbatch job_exp1.sbatch
set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate llm
# HF_HUB_OFFLINE: compute nodes have no DNS, so every from_pretrained would burn ~40s on retries
export CONDA_OVERRIDE_CUDA=12.9 HF_HOME="$HOME/hf" TMPDIR="$HOME/tmp" TOKENIZERS_PARALLELISM=false \
       HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$HOME/phase2/experiment1"
python run_exp1.py ${SMOKE:+--smoke}
python report.py ${SMOKE:+--smoke}
