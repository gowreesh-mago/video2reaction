#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source configs/clusters/snellius.env
export PATH="$HOME/.local/bin:$PATH"
name=${1:-smoke_3videos}
case "$name" in
  smoke_3videos) prefix=smoke ;;
  smoke_emotion_3videos) prefix=smoke_emotion ;;
  smoke_descriptions_3videos) prefix=smoke_descriptions ;;
  *) echo "Unknown smoke test: $name" >&2; exit 2 ;;
esac
mkdir -p "$V2R_ROOT/logs/slurm"
mkdir -p "$V2R_ROOT/results/submissions"
# Idempotent per synchronized source snapshot, even with simultaneous launchers.
exec 9>"$V2R_ROOT/results/submissions/smoke.lock"
flock 9
snapshot=$(python3 -c 'import json; print(json.load(open("code_version.json"))["source_sha256"])')
record="$V2R_ROOT/results/submissions/$prefix-$snapshot.jobid"
if [[ -s "$record" ]]; then
  echo "This source snapshot already has smoke job $(cat "$record"); inspect it with sacct."
  exit 0
fi
squeue -u "$USER"
sbatch --test-only "slurm/$name.sbatch"
job=$(sbatch --parsable --output="$V2R_ROOT/logs/slurm/$prefix-%j.out" "slurm/$name.sbatch")
job=${job%%;*}
printf '%s\n' "$job" > "$record"
python3 scripts/update_registry.py --experiment "${name}_$job" --status submitted --job-id "$job"
echo "Submitted smoke job $job"
