#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo 'Install uv from https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 1; }
uv python install "$(cat .python-version)"
uv sync --frozen
uv run --frozen python scripts/check_env.py
