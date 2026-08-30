#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT="${OPENPI_ROOT:-/home/rllab2/jellyho/openpi}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_scoop_mix_c08_s02}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/rllab2/jellyho/checkpoints/pi05_scoop_mix_c08_s02/pi05_scoop_mix_c08_s02_run/29999}"
PORT="${PORT:-8000}"

cd "$OPENPI_ROOT"

exec uv run scripts/serve_policy.py \
  --port="$PORT" \
  policy:checkpoint \
  --policy.config="$POLICY_CONFIG" \
  --policy.dir="$CHECKPOINT_DIR"
