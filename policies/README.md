# Local policy bundles

Put each immutable deployment bundle in its own directory:

```text
policies/<bundle_name>/
├── deploy.json       # required observation, action, robot, and control contract
├── policy.onnx       # required by the MuJoCo runtime
├── policy.pt         # optional export/checkpoint retained for traceability
└── bundle_manifest.json
```

Policy binaries and machine-local manifests are intentionally ignored by Git.
Do not evaluate an ONNX file without the `deploy.json` generated alongside it.
The benchmark includes its own deployment runtime, so inference does not need
an Isaac Lab checkout or the training repository on `PYTHONPATH`.
