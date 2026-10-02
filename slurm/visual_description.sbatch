#!/usr/bin/env bash
#SBATCH --job-name=v2r-visual_description
#SBATCH --account=gusr133332
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --signal=B:TERM@60
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from a frozen release}"
exec bash scripts/run_experiment.sh visual_description
