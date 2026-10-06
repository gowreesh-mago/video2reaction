#!/usr/bin/env bash
#SBATCH --job-name=v2r-morning-status
#SBATCH --account=gusr133332
#SBATCH --partition=staging
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Frozen release required}"
source configs/clusters/snellius.env
python3 scripts/overnight_status.py
