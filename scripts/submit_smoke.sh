#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source configs/clusters/snellius.env
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$V2R_ROOT/logs/slurm"
squeue -u "$USER"
sbatch --test-only slurm/smoke_3videos.sbatch
job=$(sbatch --parsable --output="$V2R_ROOT/logs/slurm/smoke-%j.out" slurm/smoke_3videos.sbatch)
job=${job%%;*}
uv run --frozen python scripts/update_registry.py --experiment "smoke_3videos_$job" --status submitted --job-id "$job"
echo "Submitted smoke job $job"
