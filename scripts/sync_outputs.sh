#!/usr/bin/env bash
# Pull generated run artifacts; the dataset and shared feature cache stay remote.
set -euo pipefail
cd "$(dirname "$0")/.."
host=${V2R_HOST:-snellius.surf.nl}
remote=${V2R_REMOTE_ROOT:-/scratch-shared/gmago/video2reaction}
for directory in outputs logs results; do
  mkdir -p "$directory"
  # Materialize stdout.log links so they work on the Mac. Stage received files
  # before replacement, exclude transient writes/locks, and preserve local history.
  rsync -azL --delay-updates --itemize-changes \
    --exclude='*features*.pt' --exclude='*features*.npy' \
    --exclude='*.lock' --exclude='.*' \
    "$host:$remote/$directory/" "$directory/"
done
python3 - "$host" "$remote" <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

summary = {
    'synced_at': datetime.now(timezone.utc).isoformat(),
    'source_host': sys.argv[1],
    'source_root': sys.argv[2],
    'directories': {},
    'excluded': ['dataset/data directory', 'shared feature cache', '*features*.pt', '*features*.npy', 'locks and temporary dotfiles'],
    'stdout_links': 'copied as regular files',
}
for name in ('outputs', 'logs', 'results'):
    paths = [p for p in Path(name).rglob('*') if p.is_file() and p.name != 'sync_receipt.json']
    summary['directories'][name] = {'files': len(paths), 'bytes': sum(p.stat().st_size for p in paths)}
Path('results/sync_receipt.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
PY
