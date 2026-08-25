#!/usr/bin/env python3
"""Build a fused 3D point cloud from an RGB-D capture session.

The input directory must contain ``session.json`` or the legacy
``frames.jsonl`` manifest. If camera poses are absent, RGB-D odometry is run
before fusing the accepted keyframes with TSDF.

Examples:
    python3 tools/build_pointcloud.py data/zed_highres_20260825_120000
    python3 tools/build_pointcloud.py data/session --poses-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "renception" / "configs" / "fusion.yaml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from renception.reconstruction import reconstruct_dataset


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the reconstruction YAML configuration."""
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a YAML mapping: {path}")
    return config


def build_pointcloud(
    dataset_dir: str | Path,
    output_path: str | Path,
    config: dict[str, Any],
    intrinsics_path: str | Path | None = None,
    poses_only: bool = False,
) -> Any:
    """Build a cloud; the reconstruction implementation lives in renception."""
    return reconstruct_dataset(
        dataset_dir,
        config,
        output_path,
        intrinsics_path,
        poses_only,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Directory containing session.json or legacy frames.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output PLY (default: fused_cloud.ply in the dataset)",
    )
    parser.add_argument(
        "--intrinsics",
        type=Path,
        help="camera_intrinsics.json (default: dataset/intrinsics/...)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML containing fusion parameters (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--poses-only",
        action="store_true",
        help="Estimate and save the trajectory without TSDF fusion",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run offline RGB-D reconstruction from the command line."""
    args = build_argument_parser().parse_args(argv)

    if not args.dataset.is_dir():
        print(f"Dataset directory does not exist: {args.dataset}", file=sys.stderr)
        return 1
    if not args.config.is_file():
        print(f"Configuration file does not exist: {args.config}", file=sys.stderr)
        return 1
    if args.intrinsics is not None and not args.intrinsics.is_file():
        print(f"Intrinsics file does not exist: {args.intrinsics}", file=sys.stderr)
        return 1

    output_path = args.out or (args.dataset.resolve() / "fused_cloud.ply")
    try:
        config = load_config(args.config)
        cloud = build_pointcloud(
            args.dataset,
            output_path,
            config,
            args.intrinsics,
            poses_only=args.poses_only,
        )
    except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError) as error:
        print(f"Error building point cloud: {error}", file=sys.stderr)
        return 1

    if cloud is not None:
        print(f"Fused point cloud: {len(cloud.points)} points -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
