#!/usr/bin/env bash
#SBATCH --job-name=v2r-vad-dino
#SBATCH --account=gusr133332
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Frozen source release required}"
source configs/clusters/snellius.env
export UV_PROJECT_ENVIRONMENT="$V2R_ROOT/envs/projection-cpu-20261021"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/v2r-mpl-$SLURM_JOB_ID"
out="$V2R_ROOT/reports/2026-10-21-completion/projection"
if [[ "${1:-}" == smoke ]]; then
  "$HOME/.local/bin/uv" run --frozen --no-sync python -m pytest -q tests/test_vad_dino_projection.py tests/test_previous_run_analysis.py
  "$HOME/.local/bin/uv" run --frozen --no-sync python scripts/run_vad_dino_projection.py --root "$V2R_ROOT" --out "$out/smoke_$SLURM_JOB_ID" --seed 42 --smoke
else
  seed=$((42 + SLURM_ARRAY_TASK_ID))
  "$HOME/.local/bin/uv" run --frozen --no-sync python scripts/run_vad_dino_projection.py --root "$V2R_ROOT" --out "$out/seed${seed}_$SLURM_ARRAY_JOB_ID" --seed "$seed"
fi
