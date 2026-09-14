"""Command-line utilities for validating and compiling track specifications."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .composer import TrackComposer
from .track import TerrainLimits, TrackSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=Path, required=True)
    parser.add_argument("--deploy-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--difficulty-level", type=int, choices=range(1, 10), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = TrackSpec.load(args.track)
    if args.difficulty_level is not None:
        spec = replace(spec, difficulty_level=args.difficulty_level, schema_version=2)
    limits = TerrainLimits.from_deploy_json(args.deploy_json) if args.deploy_json else TerrainLimits()
    track = TrackComposer(limits).compile(spec)
    result = track.to_dict()
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"TRACK_OK name={track.name} length={track.total_length:.3f} output={args.output}")


if __name__ == "__main__":
    main()
