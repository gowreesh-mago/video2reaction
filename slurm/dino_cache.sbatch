#!/usr/bin/env bash
#SBATCH --job-name=v2r-dino_cache
#SBATCH --account=gusr133332
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@30
set -euo pipefail
bash scripts/run_experiment.sh dino_cache
