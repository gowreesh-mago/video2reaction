#!/usr/bin/env bash
#SBATCH --job-name=v2r-overnight-launcher
#SBATCH --account=gusr133332
#SBATCH --partition=staging
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Frozen release required}"
source configs/clusters/snellius.env
python3 scripts/submit_trajectory_overnight.py --completed-cache-job 27682121 --completed-smoke-jobs "$@"
