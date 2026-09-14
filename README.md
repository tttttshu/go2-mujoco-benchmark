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

Ubuntu with Python 3.11 installed:

```bash
bash scripts/setup_ubuntu.sh
bash scripts/run_viewer_ubuntu.sh
```

Before launching, put `policy.onnx` and `deploy.json` in
`policies/him_policy/`. See `docs/CROSS_PLATFORM.md` for all platform commands.

## Current milestone

The repository currently includes the versioned `TrackSpec`, deterministic
flat/slope/stair composition, physical boundary walls, native `MjSpec` scene
assembly, a self-contained ONNX/PD runtime, headless metrics, and native or
browser-based real-time viewers. The same pinned dependencies and Go2 model are
accepted on Windows and Ubuntu.

Preview a track manifest without launching a simulator:

```bash
PYTHONPATH=src python -m go2_mujoco_benchmark.cli \
  --track configs/tracks/mixed.yaml \
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

Start the real-time browser viewer on the remote host:

```bash
python -m go2_mujoco_benchmark.viewer \
  --track configs/tracks/flat_20m.yaml \
  --backend mjviser --host 127.0.0.1 --port 8080 \
  --vx 1.0 --vy 0 --wz 0 \
  --camera-distance 2.2 --camera-fov 45
```

Forward the loopback-only server with:

```bash
ssh -N -L 18080:127.0.0.1:8080 rtx5090-proxy
```

Then open `http://127.0.0.1:18080`. The browser starts with a close view of the
robot and follows its base while preserving interactive orbit and zoom. Use the
`Follow robot` checkbox to freeze the camera or `Reset camera` to restore the
default view. `--camera-distance`, `--camera-azimuth`, `--camera-elevation`, and
`--camera-fov` tune the initial framing. Because generated courses are finite,
the viewer automatically resets the simulation when the robot falls or leaves
the track; pass `--no-auto-reset` when deliberate off-track inspection is needed.
Track YAML files can add physical side walls with `boundary_walls.enabled`,
`height`, and `thickness`. The flat 20 m validation course enables 0.45 m walls.
`configs/tracks/mixed_easy.yaml` is the first visual acceptance course: it uses
a wide lane and low training-range slopes and stairs before harder sweeps.

The `native` backend opens a GLFW window and can be wrapped by termview. The
remote machine must have Rust/cargo and Xvfb before termview's private-display
mode can run.

## Verified platforms

Windows 11 with Python 3.12 passes all 25 tests, including release-archive
validation, plus the full environment doctor and real ONNX policy rollout.
Ubuntu with Python 3.11 passed the previous 24-test core suite, full doctor, and
the same rollout before the release-packaging test was added. The final 25-test
Ubuntu rerun is pending reconnection of the 5090 host. On Ubuntu 5090:

```bash
cd /home/hujunyi/work/go2_mujoco_benchmark
bash scripts/setup_ubuntu.sh
.venv/bin/python -m pytest -q
.venv/bin/python -m go2_mujoco_benchmark.doctor
```

See `docs/CROSS_PLATFORM.md` for local Windows and Ubuntu commands, and
`docs/REMOTE_5090_VALIDATION.md` for the private SSH-tunnel workflow.

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
