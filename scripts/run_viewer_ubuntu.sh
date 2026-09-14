#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

exec .venv/bin/python -m go2_mujoco_benchmark.viewer \
  --policy-dir "${POLICY_DIR:-policies/him_policy}" \
  --robot-scene third_party/unitree_go2/scene.xml \
  --track "${TRACK:-configs/tracks/flat_20m.yaml}" \
  --backend mjviser \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8080}" \
  --vx "${VX:-1.0}" --vy "${VY:-0.0}" --wz "${WZ:-0.0}"
