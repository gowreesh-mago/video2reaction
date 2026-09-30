#!/usr/bin/env bash
#SBATCH --job-name=v2r-feature_cache
#SBATCH --account=gusr133332
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@60
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from a frozen release}"
exec bash scripts/run_experiment.sh feature_cache
