#!/usr/bin/env python3
"""Build a fused 3D point cloud + mesh from an extracted RGB-D session.

Input: a directory produced by ``extract_video_frames.py`` (``frames.jsonl``
+ ``rgb/`` + ``depth_mm/``) plus the ``camera_intrinsics.json`` recorded
alongside the capture.

    1. session_from_extracted  -> ScanSession (keyframes, no poses yet)
    2. odometry.estimate_poses -> T_world_cam per keyframe (RGB-D odometry + pose graph)
    3. fusion.fuse_session     -> TSDF integration -> point cloud + mesh
    4. writes fused_cloud.ply, tsdf_mesh.ply, session.json into the session dir

Usage:
    python scripts/build_pointcloud.py \\
        data/alex_dataset/extracted/session_144058 \\
        --intrinsics data/alex_dataset/intrinsics/camera_intrinsics.json

    # sanity-check on a subset first (real sessions can be thousands of frames)
    python scripts/build_pointcloud.py <session> --intrinsics <intrinsics.json> --stride 10

    # room-scale capture: the default voxel (8mm, tuned for ~0.5m fixtures) will
    # OOM on a multi-metre TSDF volume -- use a coarser voxel and a tighter
    # depth range to keep the volume small
    python scripts/build_pointcloud.py <session> --intrinsics <intrinsics.json> \\
        --voxel 0.02 --depth-trunc 3.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from perception.config import load_config  # noqa: E402
from perception import io as sio  # noqa: E402
from perception import odometry, fusion  # noqa: E402


def build(session_dir: Path, intrinsics_path: Path, cfg: dict, fps: float,
         stride: int = 1, voxel: float | None = None, depth_trunc: float | None = None) -> Path:
    if voxel is not None:
        cfg["fusion"]["voxel"] = voxel
    if depth_trunc is not None:
        cfg["fusion"]["depth_trunc"] = depth_trunc
        cfg["odometry"]["depth_trunc"] = depth_trunc

    session = sio.session_from_extracted(session_dir, intrinsics_path, fps=fps)
    if stride > 1:
        session.keyframes = session.keyframes[::stride]
    root = sio.session_root(session)

    print(f"loaded {len(session.keyframes)} keyframes from {session_dir}"
         + (f" (every {stride}th frame)" if stride > 1 else ""))
    poses = odometry.estimate_poses(session, root, cfg)
    print(f"estimated {len(poses)} poses (RGB-D odometry + pose graph)")

    cloud, mesh = fusion.fuse_session(session, root, cfg, poses)
    session_json = session_dir / "session.json"
    sio.write_fused(root, session_json, session, cloud, mesh)

    print(f"fused -> {len(cloud.points)} pts, {len(mesh.triangles)} tris")
    print(f"wrote {session_json}, {root / session.cloud_path}, {root / session.mesh_path}")
    return session_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir", type=Path, help="extracted session directory (frames.jsonl + rgb/ + depth_mm/)")
    ap.add_argument("--intrinsics", type=Path, required=True, help="camera_intrinsics.json")
    ap.add_argument("--config", type=Path, default=None, help="fusion config YAML")
    ap.add_argument("--fps", type=float, default=30.0, help="capture fps, for keyframe timestamps")
    ap.add_argument("--stride", type=int, default=1,
                   help="use every Nth keyframe (for a quick sanity check on large sessions)")
    ap.add_argument("--voxel", type=float, default=None,
                   help="override fusion.voxel [m] (default 0.008 -- too fine for room-scale scans)")
    ap.add_argument("--depth-trunc", type=float, default=None,
                   help="override fusion/odometry depth_trunc [m] (ignore depth beyond this range)")
    args = ap.parse_args()
    build(args.session_dir, args.intrinsics, load_config(args.config), args.fps, args.stride,
         args.voxel, args.depth_trunc)


if __name__ == "__main__":
    main()
