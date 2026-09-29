#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source configs/clusters/snellius.env
squeue -u "$USER" -o '%.18i %.12P %.28j %.9T %.10M %.30R'
python3 - "$V2R_REGISTRY" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if p.exists():
    for name,run in json.loads(p.read_text())['experiments'].items():
        print(name,run.get('status'),run.get('job_id'),run.get('output_dir',''))
PY
