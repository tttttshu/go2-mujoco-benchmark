# Windows and Ubuntu delivery

The physics, policy runtime, terrain compiler, metrics, and browser viewer are
identical on both operating systems. Only environment creation and shell syntax
differ. Python 3.11 or 3.12 x64 is supported.

## Included and external inputs

- `third_party/unitree_go2/` contains the pinned MuJoCo Menagerie Go2 model and
  its license.
- `policies/him_policy/deploy.json` and `policy.onnx` are user policy inputs and
  are intentionally ignored by Git. `policy.pt` is optional and unused by
  MuJoCo inference.
- The runtime under `go2_mujoco_benchmark.deployment` is self-contained and
  does not import Isaac Lab or the training repository.

## Windows 10/11

Install 64-bit Python 3.11 or 3.12, then run in PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
.\scripts\run_viewer_windows.ps1
```

The second command opens `http://127.0.0.1:8080` in the default browser. Select
a different course without editing code:

```powershell
.\scripts\run_viewer_windows.ps1 -Track configs/tracks/mixed_easy.yaml -Vx 1.0
```

The browser viewer resets and aligns the Go2 when its first client connects,
uses a close 1.25 m chase view, reports commanded versus actual velocity, and
shows a robot-mounted 320×180 depth camera. The depth image is visualization
only and is not added to the policy observation. Use `--no-depth-camera` on the
Python viewer command if offscreen rendering is unavailable.

The repository may be located in a path containing Chinese characters. MuJoCo's
C-level XML loader cannot open such paths directly on Windows, so the benchmark
automatically mirrors the complete robot model into an ASCII-only cache below
`%LOCALAPPDATA%\go2_mujoco_benchmark\model_cache`.

Run acceptance checks:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m go2_mujoco_benchmark.doctor
```

## Ubuntu 22.04/24.04

```bash
bash scripts/setup_ubuntu.sh
bash scripts/run_viewer_ubuntu.sh
```

For a remote host, leave the viewer bound to loopback and forward it:

```bash
ssh -N -L 18080:127.0.0.1:8080 USER@HOST
```

Then open `http://127.0.0.1:18080` locally. Environment variables customize
the Ubuntu launcher, for example:

```bash
TRACK=configs/tracks/mixed_easy.yaml VX=1.0 bash scripts/run_viewer_ubuntu.sh
```

## Headless evaluation on either OS

Once the editable package is installed, all paths have repository-local
defaults:

```text
python -m go2_mujoco_benchmark.doctor
python -m go2_mujoco_benchmark.rollout --steps 500 --vx 1.0
python -m go2_mujoco_benchmark.viewer --backend mjviser
```

Reports store policy, deployment, course, and robot-model hashes so Windows and
Ubuntu results can be compared against the same inputs.

## Release archives

Create public/source-only archives with:

```text
python -m go2_mujoco_benchmark.release
```

Use `--include-policy` only for a private handoff that is allowed to contain the
trained ONNX file and deployment contract. The generated checksum file can be
verified with `Get-FileHash -Algorithm SHA256` on Windows or `sha256sum -c` on
Ubuntu.
