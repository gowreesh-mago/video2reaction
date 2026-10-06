#!/usr/bin/env bash
#SBATCH --job-name=v2r-on_vad_sparse_gate050_s44
#SBATCH --account=gusr133332
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --signal=B:TERM@30
set -euo pipefail
bash scripts/run_experiment.sh on_vad_sparse_gate050_s44
