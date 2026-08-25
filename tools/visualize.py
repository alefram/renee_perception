#!/usr/bin/env python3
"""Open3D viewer for a fused session (or any point cloud / mesh file).

Usage:
    # a session dir built by build_pointcloud.py (looks for fused_cloud.ply /
    # tsdf_mesh.ply inside it, plus session.json for camera poses)
    python tools/visualize.py data/alex_dataset/extracted/session_144058

    # the session.json directly -- cloud/mesh paths + camera poses come from it
    python tools/visualize.py data/alex_dataset/extracted/session_144058/session.json

    # or point it at any .ply directly (no camera frames, nothing to read them from)
    python tools/visualize.py data/alex_dataset/extracted/session_144058/fused_cloud.ply

Flags:
    --mesh-only / --cloud-only   show just one of the two artefacts
    --no-cameras                 don't draw the estimated camera poses
    --every N                    draw every Nth camera frustum (default: 10)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import open3d as o3d

from renception.contracts import ScanSession  # noqa: E402


def _camera_frames(session_json: Path, every: int, size: float = 0.05) -> list:
    session = ScanSession.load(session_json)
    frames = []

    for keyframe in session.keyframes[::every]:
        if keyframe.T_world_cam is None:
            continue

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        frame.transform(np.asarray(keyframe.T_world_cam, dtype=float))
        frames.append(frame)

    return frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path",
        type=Path,
        help="session directory (from build_pointcloud.py) or a .ply file",
    )
    parser.add_argument(
        "--mesh-only", action="store_true", help="show only tsdf_mesh.ply"
    )
    parser.add_argument(
        "--cloud-only", action="store_true", help="show only fused_cloud.ply"
    )
    parser.add_argument(
        "--no-cameras",
        action="store_true",
        help="don't draw estimated camera poses",
    )
    parser.add_argument(
        "--every", type=int, default=10, help="draw every Nth camera frustum"
    )
    args = parser.parse_args()

    geometries = []
    session_json: Path | None = None

    path_is_directory = args.path.is_dir()
    path_is_session = path_is_directory or args.path.suffix.lower() == ".json"

    if path_is_session:
        session_json_candidate = (
            args.path / "session.json" if path_is_directory else args.path
        )
        root = args.path if path_is_directory else args.path.parent

        cloud_path = root / "fused_cloud.ply"
        mesh_path = root / "tsdf_mesh.ply"

        if session_json_candidate.exists():
            session_json = session_json_candidate
            session = ScanSession.load(session_json)
            cloud_path = root / session.cloud_path if session.cloud_path else None
            mesh_path = root / session.mesh_path if session.mesh_path else None

        if not args.mesh_only and cloud_path is not None and cloud_path.exists():
            geometries.append(("cloud", o3d.io.read_point_cloud(str(cloud_path))))
        if not args.cloud_only and mesh_path is not None and mesh_path.exists():
            geometries.append(("mesh", o3d.io.read_triangle_mesh(str(mesh_path))))

        can_draw_poses = session_json is not None and not args.no_cameras
        if not geometries and not can_draw_poses:
            raise FileNotFoundError(
                "no fused_cloud.ply, tsdf_mesh.ply, or drawable poses found "
                f"for {args.path}"
            )
    else:
        suffix = args.path.suffix.lower()
        geometry = (
            o3d.io.read_triangle_mesh(str(args.path))
            if suffix in (".obj", ".stl")
            else o3d.io.read_point_cloud(str(args.path))
        )
        geometries.append(("file", geometry))

    to_draw = [geometry for _, geometry in geometries]
    for kind, geometry in geometries:
        element_count = (
            len(geometry.triangles) if kind == "mesh" else len(geometry.points)
        )
        element_name = "triangles" if kind == "mesh" else "points"
        displayed_source = args.path if kind == "file" else geometry

        print(f"{kind}: {displayed_source} ({element_count} {element_name})")

    if not args.no_cameras and session_json is not None:
        frames = _camera_frames(session_json, max(args.every, 1))
        print(f"drawing {len(frames)} camera poses (every {args.every}th keyframe)")
        to_draw.extend(frames)

    if not to_draw:
        raise ValueError(f"nothing to visualize for {args.path}")

    o3d.visualization.draw_geometries(to_draw)


if __name__ == "__main__":
    main()
