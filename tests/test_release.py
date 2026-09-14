from __future__ import annotations

import json
import tarfile
import zipfile
from pathlib import Path

from go2_mujoco_benchmark.release import build_release


def test_release_archives_are_self_contained_and_exclude_policy_by_default(tmp_path: Path):
    zip_path, tar_path, checksums_path = build_release(output_dir=tmp_path)

    assert zip_path.is_file()
    assert tar_path.is_file()
    assert checksums_path.read_text(encoding="ascii").count("\n") == 2

    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        manifest_name = next(name for name in names if name.endswith("/RELEASE-MANIFEST.json"))
        manifest = json.loads(archive.read(manifest_name))
        assert manifest["version"] == "0.2.0"
        assert manifest["policy_included"] is False
        assert any(name.endswith("/third_party/unitree_go2/scene.xml") for name in names)
        assert not any(name.endswith("/policy.onnx") for name in names)
        assert not any(".venv/" in name or "/outputs/" in name for name in names)

    with tarfile.open(tar_path, "r:gz") as archive:
        names = archive.getnames()
        assert any(name.endswith("/scripts/setup_ubuntu.sh") for name in names)
        setup = next(member for member in archive.getmembers() if member.name.endswith("/scripts/setup_ubuntu.sh"))
        assert setup.mode == 0o755
