#!/usr/bin/env python3
"""Open3D viewer for a fused session (or any point cloud / mesh file).

Usage:
    # a session dir built by build_pointcloud.py (looks for fused_cloud.ply /
    # tsdf_mesh.ply inside it, plus session.json for camera poses)
    python scripts/visualize.py data/alex_dataset/extracted/session_144058

    # the session.json directly -- cloud/mesh paths + camera poses come from it
    python scripts/visualize.py data/alex_dataset/extracted/session_144058/session.json

    # or point it at any .ply directly (no camera frames, nothing to read them from)
    python scripts/visualize.py data/alex_dataset/extracted/session_144058/fused_cloud.ply

Flags:
    --mesh-only / --cloud-only   show just one of the two artefacts
    --no-cameras                 don't draw the estimated camera poses
    --every N                    draw every Nth camera frustum (default: 10)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import open3d as o3d

from perception.contracts import ScanSession  # noqa: E402


def _camera_frames(session_json: Path, every: int, size: float = 0.05) -> list:
    session = ScanSession.load(session_json)
    frames = []
    for kf in session.keyframes[::every]:
        if kf.T_world_cam is None:
            continue
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        frame.transform(np.asarray(kf.T_world_cam, dtype=float))
        frames.append(frame)
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path,
                    help="session directory (from build_pointcloud.py) or a .ply file")
    ap.add_argument("--mesh-only", action="store_true", help="show only tsdf_mesh.ply")
    ap.add_argument("--cloud-only", action="store_true", help="show only fused_cloud.ply")
    ap.add_argument("--no-cameras", action="store_true", help="don't draw estimated camera poses")
    ap.add_argument("--every", type=int, default=10, help="draw every Nth camera frustum")
    args = ap.parse_args()

    geometries = []
    session_json = None

    if args.path.is_dir() or args.path.suffix.lower() == ".json":
        if args.path.is_dir():
            session_json = args.path / "session.json"
            root = args.path
        else:
            session_json = args.path
            root = args.path.parent

        cloud_path = mesh_path = None
        if session_json.exists():
            session = ScanSession.load(session_json)
            if session.cloud_path:
                cloud_path = root / session.cloud_path
            if session.mesh_path:
                mesh_path = root / session.mesh_path
        else:
            cloud_path, mesh_path = root / "fused_cloud.ply", root / "tsdf_mesh.ply"

        if not args.mesh_only and cloud_path is not None and cloud_path.exists():
            geometries.append(("cloud", o3d.io.read_point_cloud(str(cloud_path))))
        if not args.cloud_only and mesh_path is not None and mesh_path.exists():
            geometries.append(("mesh", o3d.io.read_triangle_mesh(str(mesh_path))))
        if not geometries:
            raise FileNotFoundError(
                f"no fused_cloud.ply or tsdf_mesh.ply found for {args.path} "
                "-- run scripts/build_pointcloud.py on this session first")
    else:
        suffix = args.path.suffix.lower()
        geom = (o3d.io.read_triangle_mesh(str(args.path)) if suffix in (".obj", ".stl")
               else o3d.io.read_point_cloud(str(args.path)))
        geometries.append(("file", geom))

    to_draw = [g for _, g in geometries]
    for kind, g in geometries:
        n = len(g.triangles) if kind == "mesh" else len(g.points)
        print(f"{kind}: {args.path if kind == 'file' else g} ({n} {'triangles' if kind == 'mesh' else 'points'})")

    if not args.no_cameras and session_json is not None and session_json.exists():
        frames = _camera_frames(session_json, max(args.every, 1))
        print(f"drawing {len(frames)} camera poses (every {args.every}th keyframe)")
        to_draw.extend(frames)

    o3d.visualization.draw_geometries(to_draw)


if __name__ == "__main__":
    main()
