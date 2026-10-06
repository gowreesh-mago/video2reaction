#!/usr/bin/env bash
#SBATCH --job-name=v2r-smoke-overnight-3
#SBATCH --account=gusr133332
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --signal=B:TERM@30
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the frozen release}"
source configs/clusters/snellius.env
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
EXP="smoke_trajectory_overnight_3videos_${SLURM_JOB_ID}"
mkdir -p "$V2R_ROOT/logs/smoke/$EXP"
exec > >(tee -a "$V2R_ROOT/logs/smoke/$EXP/stdout.log") 2>&1
finish() {
  rc=$?
  if (( rc != 0 )); then
    .venv/bin/python scripts/update_registry.py --experiment "$EXP" --status failed --error "SLURM shell exited with status $rc" || true
  fi
}
trap finish EXIT
trap 'exit 143' TERM INT
date -u
hostname
if [[ "${V2R_WARM_IMPORTS:-1}" == 1 ]]; then
  bash scripts/warm_imports.sh
fi
uv run --frozen --no-sync python scripts/check_gpu.py
uv run --frozen --no-sync python -m pytest tests -q
uv run --frozen --no-sync python scripts/smoke_trajectory.py --run-name "$EXP" --batch overnight --shard-index 3 --shard-count 4 --epochs 20
ln -s "$V2R_ROOT/logs/smoke/$EXP/stdout.log" "$V2R_OUTPUT_DIR/smoke/$EXP/stdout.log"
