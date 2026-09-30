#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source configs/clusters/snellius.env
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$V2R_ROOT/logs/slurm"
mkdir -p "$V2R_ROOT/results/submissions"
# Idempotent per synchronized source snapshot, even with simultaneous launchers.
exec 9>"$V2R_ROOT/results/submissions/smoke.lock"
flock 9
snapshot=$(python3 -c 'import json; print(json.load(open("code_version.json"))["source_sha256"])')
record="$V2R_ROOT/results/submissions/smoke-$snapshot.jobid"
if [[ -s "$record" ]]; then
  echo "This source snapshot already has smoke job $(cat "$record"); inspect it with sacct."
  exit 0
fi
squeue -u "$USER"
sbatch --test-only slurm/smoke_3videos.sbatch
job=$(sbatch --parsable --output="$V2R_ROOT/logs/slurm/smoke-%j.out" slurm/smoke_3videos.sbatch)
job=${job%%;*}
printf '%s\n' "$job" > "$record"
python3 scripts/update_registry.py --experiment "smoke_3videos_$job" --status submitted --job-id "$job"
echo "Submitted smoke job $job"
