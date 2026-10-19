#!/usr/bin/env bash
#SBATCH --job-name=v2r-previous-errors
#SBATCH --account=gusr133332
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source configs/clusters/snellius.env
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
uv run --frozen --no-sync python -m pytest -q tests/test_previous_run_analysis.py
uv run --frozen --no-sync python scripts/analyze_previous_runs.py \
  --root "$V2R_ROOT" --out "$V2R_ROOT/reports/2026-10-19-error-analysis"
