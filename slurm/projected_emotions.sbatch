#!/usr/bin/env bash
#SBATCH --job-name=v2r-emotion-anchors
#SBATCH --account=gusr133332
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Frozen release required}"
source configs/clusters/snellius.env
export UV_PROJECT_ENVIRONMENT="$V2R_ROOT/envs/projection-cpu-20261021"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
args=()
if [[ "${1:-}" == smoke ]]; then
  seed=42
  run="smoke_$SLURM_JOB_ID"
  args+=(--smoke)
else
  seed=$((42+SLURM_ARRAY_TASK_ID))
  run="seed${seed}_$SLURM_ARRAY_JOB_ID"
fi
"$HOME/.local/bin/uv" run --frozen --no-sync python scripts/evaluate_projected_emotions.py \
  --root "$V2R_ROOT" --projection "$V2R_ROOT/reports/2026-10-21-completion/projection/seed${seed}_28448337" \
  --out "$V2R_ROOT/reports/2026-10-21-completion/projected_emotions/$run" "${args[@]}"
