"""Build deterministic repository delivery archives."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
import tomllib
import zipfile
from pathlib import Path

from .paths import repository_root

ROOT_FILES = (
    ".gitattributes",
    ".gitignore",
    "README.md",
    "pyproject.toml",
    "requirements-lock.txt",
    "requirements-validation.txt",
)
CONTENT_DIRS = (".github", "configs", "docs", "scripts", "src", "tests", "third_party")
POLICY_FILES = ("deploy.json", "policy.onnx")
FIXED_ZIP_TIME = (2000, 1, 1, 0, 0, 0)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _version(root: Path) -> str:
    with (root / "pyproject.toml").open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def collect_release_files(root: Path, *, include_policy: bool = False) -> list[Path]:
    """Return sorted repository-relative files included in a release."""
    relative_files: list[Path] = []
    for name in ROOT_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Required release file is missing: {path}")
        relative_files.append(Path(name))
    for directory in CONTENT_DIRS:
        base = root / directory
        if not base.is_dir():
            raise FileNotFoundError(f"Required release directory is missing: {base}")
        relative_files.extend(path.relative_to(root) for path in base.rglob("*") if path.is_file())

    policy_readme = root / "policies" / "README.md"
    if not policy_readme.is_file():
        raise FileNotFoundError(f"Required release file is missing: {policy_readme}")
    relative_files.append(policy_readme.relative_to(root))
    if include_policy:
        policy_dir = root / "policies" / "him_policy"
        for name in POLICY_FILES:
            path = policy_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"Policy-inclusive release requires: {path}")
            relative_files.append(path.relative_to(root))
    return sorted(set(relative_files), key=lambda path: path.as_posix())


def _manifest(root: Path, files: list[Path], *, version: str, include_policy: bool) -> bytes:
    entries = []
    for relative in files:
        data = (root / relative).read_bytes()
        entries.append(
            {
                "path": relative.as_posix(),
                "bytes": len(data),
                "sha256": _sha256(data),
            }
        )
    payload = {
        "schema_version": 1,
        "name": "go2_mujoco_benchmark",
        "version": version,
        "policy_included": include_policy,
        "files": entries,
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _archive_name(prefix: str, relative: Path) -> str:
    return f"{prefix}/{relative.as_posix()}"


def _write_zip(
    output: Path,
    root: Path,
    files: list[Path],
    *,
    prefix: str,
    manifest: bytes,
) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        members = [(relative, (root / relative).read_bytes()) for relative in files]
        members.append((Path("RELEASE-MANIFEST.json"), manifest))
        for relative, data in members:
            info = zipfile.ZipInfo(_archive_name(prefix, relative), date_time=FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o755 if relative.suffix == ".sh" else 0o644
            info.external_attr = (mode & 0xFFFF) << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _write_tar_gz(
    output: Path,
    root: Path,
    files: list[Path],
    *,
    prefix: str,
    manifest: bytes,
) -> None:
    with output.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gzip_handle:
            with tarfile.open(fileobj=gzip_handle, mode="w", format=tarfile.PAX_FORMAT) as archive:
                members = [(relative, (root / relative).read_bytes()) for relative in files]
                members.append((Path("RELEASE-MANIFEST.json"), manifest))
                for relative, data in members:
                    info = tarfile.TarInfo(_archive_name(prefix, relative))
                    info.size = len(data)
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mode = 0o755 if relative.suffix == ".sh" else 0o644
                    archive.addfile(info, io.BytesIO(data))


def build_release(
    *,
    root: Path | None = None,
    output_dir: Path | None = None,
    include_policy: bool = False,
) -> tuple[Path, Path, Path]:
    """Build ZIP, tar.gz, and checksum files and return their paths."""
    root = (root or repository_root()).resolve()
    output_dir = (output_dir or root / "dist").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    version = _version(root)
    suffix = "-with-policy" if include_policy else ""
    prefix = f"go2_mujoco_benchmark-{version}{suffix}"
    files = collect_release_files(root, include_policy=include_policy)
    manifest = _manifest(root, files, version=version, include_policy=include_policy)
    zip_path = output_dir / f"{prefix}-windows.zip"
    tar_path = output_dir / f"{prefix}-ubuntu.tar.gz"
    checksums_path = output_dir / f"{prefix}-SHA256SUMS.txt"
    _write_zip(zip_path, root, files, prefix=prefix, manifest=manifest)
    _write_tar_gz(tar_path, root, files, prefix=prefix, manifest=manifest)
    checksums = [
        f"{_sha256(path.read_bytes())}  {path.name}"
        for path in (zip_path, tar_path)
    ]
    checksums_path.write_text("\n".join(checksums) + "\n", encoding="ascii", newline="\n")
    return zip_path, tar_path, checksums_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--include-policy",
        action="store_true",
        help="include policies/him_policy/deploy.json and policy.onnx for private delivery",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    outputs = build_release(output_dir=args.output_dir, include_policy=args.include_policy)
    for path in outputs:
        print(f"RELEASE_FILE path={path} bytes={path.stat().st_size}")
    print(f"RELEASE_OK policy_included={str(args.include_policy).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
