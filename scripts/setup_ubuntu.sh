#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${PYTHON:-}" ]]; then
  python_bootstrap="$PYTHON"
elif command -v python3.11 >/dev/null 2>&1; then
  python_bootstrap="python3.11"
elif command -v python3.12 >/dev/null 2>&1; then
  python_bootstrap="python3.12"
else
  echo "Python 3.11 or 3.12 x86-64 is required." >&2
  echo "Install a supported Python or set PYTHON=/path/to/python." >&2
  exit 1
fi
cd "$project_root"

"$python_bootstrap" -c 'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 13) and sys.maxsize > 2**32 else 1)' || {
  echo "Selected interpreter must be 64-bit Python 3.11 or 3.12: $python_bootstrap" >&2
  exit 1
}

if [[ ! -x .venv/bin/python ]]; then
  "$python_bootstrap" -m venv .venv
fi
.venv/bin/python -c 'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 13) and sys.maxsize > 2**32 else 1)' || {
  echo "Existing .venv is incompatible. Remove it and rerun setup_ubuntu.sh." >&2
  exit 1
}
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python -m go2_mujoco_benchmark.doctor --allow-missing-policy
echo "UBUNTU_SETUP_OK"
