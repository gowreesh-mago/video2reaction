#!/usr/bin/env bash
# Shared-filesystem mmap/import reads were slow in the first Snellius smoke run.
# Reading these dependency files in parallel warms the node's filesystem cache.
# This reads the uv environment only; it does not copy or scan the dataset.
set -euo pipefail
cd "$(dirname "$0")/.."
packages=.venv/lib/python3.11/site-packages
test -d "$packages"
echo "Warming Python dependencies on $(hostname) at $(date -u +%FT%TZ)"
rg --files --hidden --no-ignore --null -g '*.py' -g '*.pyc' -g '*.so*' -g METADATA "$packages" \
  | xargs -0 -P 8 -n 16 cat > /dev/null
echo "Dependency warm-up finished at $(date -u +%FT%TZ)"
