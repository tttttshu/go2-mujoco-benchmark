# Go2 MuJoCo Benchmark

Independent MuJoCo benchmark for policies exported by `go2_locomotion_lab`.
The training repository owns observation, action and controller semantics; this
repository owns track composition, simulation rollouts and evaluation reports.

The delivered runtime is self-contained and supports both Windows and Ubuntu.
The pinned Unitree Go2 MuJoCo model is included under `third_party/`; an Isaac
Lab installation is not required for policy inference.

## Quick start

Windows PowerShell with Python 3.11 or 3.12 installed:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
.\scripts\run_viewer_windows.ps1
```

Ubuntu with Python 3.11 or 3.12 installed:

```bash
bash scripts/setup_ubuntu.sh
bash scripts/run_viewer_ubuntu.sh
```

Before launching, put `policy.onnx` and `deploy.json` in
`policies/him_policy/`. See `docs/CROSS_PLATFORM.md` for all platform commands.

## Current milestone

The repository currently includes the versioned `TrackSpec`, deterministic
flat/slope/stair/maze composition, 1–9 difficulty levels, physical boundary walls, native `MjSpec` scene
assembly, a self-contained ONNX/PD runtime, headless metrics, and native or
browser-based real-time viewers. The same pinned dependencies and Go2 model are
accepted on Windows and Ubuntu.

Preview a track manifest without launching a simulator:

```bash
PYTHONPATH=src python -m go2_mujoco_benchmark.cli \
  --track configs/tracks/mixed_maze.yaml --difficulty-level 7 \
  --output outputs/mixed_manifest.json
```

Run tests:

```bash
PYTHONPATH=src python -m pytest -q
```

Compile the mixed course into the official Menagerie Go2 scene and run the
physics smoke test:

```bash
PYTHONPATH=src python -m go2_mujoco_benchmark.physics_smoke \
  --track configs/tracks/mixed.yaml \
  --model-revision 8161bba264d7fa7c99ca301e91e7fb44737676ad \
  --output outputs/physics_smoke.json
```

Run an exported policy bundle through the shared deployment runtime:

```bash
python -m go2_mujoco_benchmark.rollout \
  --track configs/tracks/mixed.yaml \
  --steps 150 --vx 0 --vy 0 --wz 0 \
  --output outputs/policy_interface.json
```

`scripts/make_synthetic_bundle.py` creates zero-action Competition or HIM
fixtures strictly for interface acceptance. A synthetic result is labelled
`evaluation_scope: interface_only` and must never be reported as trained-policy
performance.

Start the real-time browser viewer locally:

```bash
python -m go2_mujoco_benchmark.viewer \
  --track configs/tracks/mixed_maze.yaml --difficulty-level 5 \
  --backend mjviser --host 127.0.0.1 --port 8080 \
  --vx 1.0 --vy 0 --wz 0 \
  --camera-distance 1.25 --camera-fov 38
```

For a remote Ubuntu host, keep the service bound to loopback and forward it
from the client computer with:

```bash
ssh -N -L 18080:127.0.0.1:8080 USER@HOST
```

Then open `http://127.0.0.1:18080`. The browser starts with a close view of the
robot and follows its base while preserving interactive orbit and zoom. Use the
`Follow robot` checkbox to freeze the camera or `Reset camera` to restore the
default view. `--camera-distance`, `--camera-azimuth`, `--camera-elevation`, and
`--camera-fov` tune the initial framing. The first browser connection resets and
aligns the robot at the course start, so loading time cannot consume part of an
episode. The side panel includes target/actual motion telemetry, a stall warning,
`Reset & align`, and a 320×180 forward-facing depth image at 5 FPS. Pass
`--no-depth-camera` to disable offscreen depth rendering. Because generated courses are finite,
the viewer automatically resets the simulation when the robot falls or leaves
the track; pass `--no-auto-reset` when deliberate off-track inspection is needed.
Track YAML files can add physical side walls with `boundary_walls.enabled`,
`height`, and `thickness`. The flat 20 m validation course enables 0.45 m walls.
`configs/tracks/mixed_easy.yaml` is the first visual acceptance course: it uses
a wide lane and low training-range slopes and stairs before harder sweeps.

`segments` in a track YAML are compiled in the exact listed order, so terrain
types can be selected, repeated, and rearranged freely. `difficulty_level` is
an integer from 1 (easiest) to 9 (hardest), globally or on an individual
segment. Maze difficulty increases wall thickness and wall count while reducing
the alternating corridor width. Its final barrier always leaves exactly one
centered 1.2 m exit. See [docs/TRACK_FORMAT.md](docs/TRACK_FORMAT.md) and
`configs/tracks/mixed_maze.yaml` for a complete example.

The `native` backend opens a GLFW window and can be wrapped by termview. The
remote machine must have Rust/cargo and Xvfb before termview's private-display
mode can run.

## Supported platforms

The supported matrix is Windows 10/11 and Ubuntu 22.04/24.04 on x86-64 with
Python 3.11 or 3.12. GitHub Actions runs the same test suite on both operating
systems and both Python versions. A GPU is optional; policy inference defaults
to ONNX Runtime CPU. Validate any checkout with:

```bash
python -m pytest -q
python -m go2_mujoco_benchmark.doctor
```

See `docs/CROSS_PLATFORM.md` for local Windows and Ubuntu commands, and
`docs/REMOTE_UBUNTU.md` for a server-neutral SSH-tunnel workflow.

## Build delivery archives

Build both platform archives without the private policy files:

```bash
python -m go2_mujoco_benchmark.release
```

For a private, immediately runnable handoff, explicitly include
`policies/him_policy/policy.onnx` and `deploy.json`:

```bash
python -m go2_mujoco_benchmark.release --include-policy
```

The command writes a Windows ZIP, an Ubuntu `tar.gz`, and `SHA256SUMS.txt` under
`dist/`. Every archive also contains `RELEASE-MANIFEST.json` with the size and
SHA-256 of every delivered file.
