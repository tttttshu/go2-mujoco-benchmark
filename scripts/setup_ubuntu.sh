#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bootstrap="${PYTHON:-python3}"
cd "$project_root"

"$python_bootstrap" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python -m go2_mujoco_benchmark.doctor --allow-missing-policy
echo "UBUNTU_SETUP_OK"
