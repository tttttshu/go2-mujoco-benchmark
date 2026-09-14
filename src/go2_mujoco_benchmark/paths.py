"""Repository-local default paths shared by Windows and Ubuntu entry points."""

from __future__ import annotations

from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_robot_scene() -> Path:
    return repository_root() / "third_party" / "unitree_go2" / "scene.xml"


def default_policy_dir() -> Path:
    return repository_root() / "policies" / "him_policy"


def default_track() -> Path:
    return repository_root() / "configs" / "tracks" / "flat_20m.yaml"


def default_output() -> Path:
    return repository_root() / "outputs" / "latest_rollout.json"
