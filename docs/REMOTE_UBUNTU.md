# Generic remote Ubuntu usage

The benchmark is not tied to a particular server, GPU, user account, or Conda
installation. It runs on Ubuntu 22.04/24.04 x86-64 with Python 3.11 or 3.12.
ONNX Runtime uses its CPU provider by default, so an NVIDIA GPU is optional.

## Install

Clone the repository on the Ubuntu machine and enter it:

```bash
git clone https://github.com/tttttshu/go2-mujoco-benchmark.git
cd go2-mujoco-benchmark
```

Copy the deployment bundle into the ignored policy directory:

```text
policies/him_policy/deploy.json
policies/him_policy/policy.onnx
```

Create the repository-local environment:

```bash
bash scripts/setup_ubuntu.sh
```

Ubuntu 24.04 provides Python 3.12 by default. Ubuntu 22.04 users must install
Python 3.11 or provide a compatible interpreter explicitly:

```bash
PYTHON=/path/to/python3.11 bash scripts/setup_ubuntu.sh
```

No Isaac Lab checkout, training repository, or external MuJoCo Menagerie clone
is required.

## Validate the installation

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m go2_mujoco_benchmark.doctor
.venv/bin/python -m go2_mujoco_benchmark.rollout \
  --track configs/tracks/flat_20m.yaml \
  --steps 500 --vx 1.0 --vy 0 --wz 0 \
  --output outputs/ubuntu_flat_vx1.json
```

A policy that falls, stalls, or tracks velocity poorly can still have a valid
runtime interface. Use the JSON report to distinguish controller performance
from missing files, invalid dimensions, non-finite state, or runtime errors.

## Browser viewer on the Ubuntu desktop

```bash
TRACK=configs/tracks/mixed_easy.yaml VX=0.5 bash scripts/run_viewer_ubuntu.sh
```

Open `http://127.0.0.1:8080` on the same machine.

## Browser viewer through SSH

Keep the server bound to loopback:

```bash
HOST=127.0.0.1 PORT=8080 \
  TRACK=configs/tracks/mixed_easy.yaml VX=0.5 \
  bash scripts/run_viewer_ubuntu.sh
```

On the client computer, create a private tunnel:

```bash
ssh -N -L 18080:127.0.0.1:8080 USER@HOST
```

Then open `http://127.0.0.1:18080` on the client. Do not expose the Viser port
directly to the public network.

## Headless depth-rendering fallback

The real-time depth panel uses MuJoCo offscreen rendering. On a headless Linux
host with working EGL, start with:

```bash
MUJOCO_GL=egl bash scripts/run_viewer_ubuntu.sh
```

If no usable OpenGL/EGL context is available, the viewer prints
`DEPTH_CAMERA_DISABLED` and continues without the depth image. It can also be
disabled explicitly:

```bash
.venv/bin/python -m go2_mujoco_benchmark.viewer --no-depth-camera
```

Depth rendering is visualization-only and never changes the policy observation.
