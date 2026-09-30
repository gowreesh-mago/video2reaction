#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source configs/clusters/snellius.env
release=$(python3 scripts/freeze_release.py)
cd "$release"
python3 scripts/submit_tier0.py
