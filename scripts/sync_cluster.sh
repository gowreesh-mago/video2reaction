#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
host=${V2R_HOST:-snellius.surf.nl}
remote=${V2R_REMOTE_REPO:-/scratch-shared/gmago/video2reaction/code}
python3 scripts/write_code_manifest.py
# No --delete: outputs, datasets and remote environments must survive code syncs.
rsync -az --itemize-changes --exclude=.git --exclude=.venv --exclude=__pycache__ \
  --exclude=.pytest_cache --exclude=.DS_Store --exclude=.env --exclude=.aws --exclude=.codex --exclude=.agents \
  --exclude=outputs --exclude=results --exclude=logs --exclude=data --exclude=cluster.env \
  --exclude=assets/emotion_vad.json \
  ./ "$host:$remote/"
