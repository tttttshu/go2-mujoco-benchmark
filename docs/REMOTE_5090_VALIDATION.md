# Remote 5090 validation

## Known server layout

- SSH alias: `rtx5090-proxy`
- validation checkout: `/home/hujunyi/work/go2_mujoco_benchmark`
- MuJoCo/test interpreter:
  `/home/hujunyi/.conda/envs/openpi-hujunyi/bin/python`
- Isaac Lab checkout: `/home/hujunyi/work/IsaacLab`
- Isaac Lab environment: `regrind`

The `openpi-hujunyi` environment has Python 3.11, MuJoCo 3.8.1, PyTorch
2.7.1, NumPy 2.2.6 and pytest. Do not modify this shared environment in place.
The setup script creates a repository-local `.venv` and installs the same
pinned runtime used by Windows.

The `regrind` environment contains Isaac Sim 5.1 and an Isaac Lab 2.3-era
checkout. That is newer than the training repository's documented Isaac Sim
4.5 stack, so it is a migration target, not a bit-for-bit reproduction of the
original training environment.

## Check capacity before a run

```bash
ssh rtx5090-proxy \
  "nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits"
```

Pick a device immediately before launching a GPU workload because availability
can change between checks. The TrackSpec compiler and its tests do not require
a GPU.

For Isaac Sim on this host, prefer its explicit device option such as
`--device cuda:5`. Do not set `CUDA_VISIBLE_DEVICES` for Isaac Sim validation:
in the current server setup it can interfere with Kit's physical GPU indexing.

## Milestone 1 acceptance

```bash
cd /home/hujunyi/work/go2_mujoco_benchmark
PYTHONPATH=src /home/hujunyi/.conda/envs/openpi-hujunyi/bin/python -m pytest -q
PYTHONPATH=src /home/hujunyi/.conda/envs/openpi-hujunyi/bin/python \
  -m go2_mujoco_benchmark.cli \
  --track configs/tracks/mixed.yaml \
  --output outputs/mixed_manifest.json
```

Required result:

- all tests pass;
- CLI prints `TRACK_OK`;
- mixed course length is 16 m;
- patch order is flat, flat, slope up, flat, stairs up, flat, stairs down,
  slope down, flat;
- final height is zero within floating-point tolerance;
- the training-profile example contains no out-of-distribution segment.

## Later milestone split

1. Export the policy and deployment contract from the original, compatible
   Isaac Lab training environment.
2. Copy only the immutable deployment bundle into this repository's validation
   run directory.
3. Run MuJoCo headless rollouts first. Rendering is optional and should not be
   part of pass/fail metrics.
4. Record the track file, seed, policy hash, deployment-contract hash, MuJoCo
   version, per-episode metrics and selected device in every report.
5. Use Isaac Sim 5.1 only for a separately labelled migration comparison until
   the task starts and observations have been matched against the 4.5 stack.

## Milestone 2 physics smoke test

The pinned reference Go2 model is delivered inside this repository at
`third_party/unitree_go2`. Its upstream revision is recorded in
`third_party/unitree_go2/REVISION`; no separate Menagerie checkout is needed.

```bash
cd /home/hujunyi/work/go2_mujoco_benchmark
PYTHONPATH=src .venv/bin/python -m go2_mujoco_benchmark.physics_smoke \
  --track configs/tracks/mixed.yaml \
  --model-revision 8161bba264d7fa7c99ca301e91e7fb44737676ad \
  --output outputs/physics_smoke.json
```

The pass criteria require finite state, actual robot-to-track contacts, no
fall-through, recovery to the home base height, an upright final orientation
and bounded joint velocity. This test exercises physics and collision only; it
does not claim policy transfer success.

## Milestone 3 policy-runtime acceptance

Copy the immutable deployment files into the ignored run directory before
validation:

```text
policies/him_policy/policy.onnx
policies/him_policy/deploy.json
```

The runtime consumes these files directly and does not import Isaac Lab or the
training repository.

Create the isolated runtime without changing either shared Conda environment:

```bash
cd /home/hujunyi/work/go2_mujoco_benchmark
PYTHON=/home/hujunyi/.conda/envs/openpi-hujunyi/bin/python \
  bash scripts/setup_ubuntu.sh
```

Run the real exported policy on the flat course:

```bash
cd /home/hujunyi/work/go2_mujoco_benchmark
.venv/bin/python -m go2_mujoco_benchmark.rollout \
  --track configs/tracks/flat_20m.yaml \
  --steps 500 --vx 1.0 --vy 0 --wz 0 \
  --label ubuntu_flat_vx1 \
  --output outputs/ubuntu_flat_vx1.json
```

The current policy may fall or move poorly; that is a controller-quality result,
not a runtime failure. Runtime acceptance requires successful ONNX loading,
matching observation/action dimensions, finite state and torque values, and a
complete JSON report. `scripts/make_synthetic_bundle.py` remains available for
interface-only testing without a trained policy.

## Live viewer

The working viewer path on this host is mjviser over a private SSH tunnel:

```bash
cd /home/hujunyi/work/go2_mujoco_benchmark
TRACK=configs/tracks/flat_20m.yaml VX=1.0 bash scripts/run_viewer_ubuntu.sh
```

From Windows, keep a second terminal running:

```powershell
ssh -N -L 18080:127.0.0.1:8080 rtx5090-proxy
```

Then open `http://127.0.0.1:18080`. Do not bind the viewer to a public server
interface.

termview can wrap the `--backend native` viewer and forward keyboard/mouse input
through an ANSI terminal. On this 5090 host it is not currently runnable because
both `cargo` and `Xvfb` are absent. After an administrator installs those system
dependencies, build termview under the user's home directory and launch:

```bash
termview --virtual-display 1280x720 --input -- \
  .venv/bin/python -m go2_mujoco_benchmark.viewer \
  --policy-dir policies/him_policy \
  --robot-scene third_party/unitree_go2/scene.xml \
  --track configs/tracks/flat_20m.yaml --backend native --vx 0.4
```
