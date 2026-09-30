#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from a frozen release directory}"
source configs/clusters/snellius.env
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
name=${1:?Experiment name required}
export V2R_RUN_LOG="$V2R_ROOT/logs/experiments/${name}_${SLURM_JOB_ID}.log"
run="${name}_${SLURM_JOB_ID}"
mkdir -p "$V2R_ROOT/logs/experiments"
exec > >(tee -a "$V2R_RUN_LOG") 2>&1
finish() {
  rc=$?
  if (( rc != 0 )); then
    .venv/bin/python scripts/update_registry.py --experiment "$run" --status failed --error "SLURM shell exited with status $rc; inspect $V2R_RUN_LOG" || true
  fi
}
trap finish EXIT
trap 'exit 143' TERM INT
date -u
hostname
# The shared uv environment is already locked/provisioned; jobs never mutate it.
if [[ "${V2R_WARM_IMPORTS:-1}" == 1 ]]; then
  bash scripts/warm_imports.sh
fi
args=(--config "configs/experiments/$name.yaml" --run-name "$run")
if [[ -n "${V2R_RESUME_FROM:-}" ]]; then
  args+=(--resume-from "$V2R_RESUME_FROM")
fi
uv run --frozen --no-sync python train.py "${args[@]}"
