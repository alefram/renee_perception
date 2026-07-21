#!/usr/bin/env python3
"""ScanSession I/O: build sessions from extracted captures, resolve assets,
and persist fused artefacts.

A :class:`ScanSession` stores image/cloud paths relative to its directory (so
a session is relocatable). These helpers turn a session + keyframe into the
Open3D objects the odometry/fusion stages need, and write results back into
the session.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d

from .contracts import Keyframe, ScanSession


def session_root(session: ScanSession, session_json: str | Path | None = None) -> Path:
    """Directory the session's relative paths resolve against."""
    if session_json is not None:
        return Path(session_json).resolve().parent
    return Path(session.session_dir)


def resolve(root: Path, rel_or_abs: str) -> Path:
    """Resolve a (possibly relative) path against the session root."""
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (root / p)


def load_rgb(root: Path, kf: Keyframe) -> o3d.geometry.Image:
    return o3d.io.read_image(str(resolve(root, kf.rgb_path)))


def load_depth(root: Path, kf: Keyframe) -> o3d.geometry.Image:
    """16-bit depth image (millimetres); scale is applied at integration time."""
    return o3d.io.read_image(str(resolve(root, kf.depth_path)))


def keyframe_intrinsic(kf: Keyframe, width: int, height: int) -> o3d.camera.PinholeCameraIntrinsic:
    K = np.asarray(kf.intrinsics, dtype=float)
    return o3d.camera.PinholeCameraIntrinsic(
        width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2])


def write_fused(root: Path, session_json: str | Path, session: ScanSession,
                cloud: o3d.geometry.PointCloud, mesh: o3d.geometry.TriangleMesh,
                cloud_rel: str = "fused_cloud.ply",
                mesh_rel: str = "tsdf_mesh.ply") -> None:
    """Persist fused cloud + mesh and point the session at them."""
    o3d.io.write_point_cloud(str(root / cloud_rel), cloud)
    o3d.io.write_triangle_mesh(str(root / mesh_rel), mesh)
    session.cloud_path = cloud_rel
    session.mesh_path = mesh_rel
    session.meta["fused_points"] = len(cloud.points)
    session.meta["fused_triangles"] = len(mesh.triangles)
    session.save(session_json)


def _load_intrinsics_matrix(intrinsics_path: str | Path) -> tuple[np.ndarray, int, int]:
    d = json.loads(Path(intrinsics_path).read_text(encoding="utf-8"))
    color = d["color_camera"]
    K = np.asarray(color["camera_matrix"], dtype=float)
    return K, int(color["width"]), int(color["height"])


def session_from_extracted(session_dir: str | Path, intrinsics_path: str | Path,
                           fps: float = 30.0) -> ScanSession:
    """Build a (pose-less) ScanSession from a directory produced by
    ``scripts/extract_video_frames.py`` (``frames.jsonl`` + ``rgb/`` +
    ``depth_mm/``).

    Frame indices become ``timestamp = frame_index / fps`` (the manifest has
    no absolute per-frame time). Poses are left unset (``T_world_cam=None``);
    run ``perception.odometry.estimate_poses`` before fusing.
    """
    session_dir = Path(session_dir)
    manifest = session_dir / "frames.jsonl"
    K, width, height = _load_intrinsics_matrix(intrinsics_path)

    keyframes: list[Keyframe] = []
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "depth_mm" not in rec:
                continue  # frame has no matching depth, cannot be fused
            keyframes.append(Keyframe(
                rgb_path=rec["rgb"],
                depth_path=rec["depth_mm"],
                intrinsics=K,
                station_id=0,
                timestamp=float(rec["frame_index"]) / fps,
            ))

    if not keyframes:
        raise ValueError(f"no rgb+depth_mm pairs found in {manifest}")

    return ScanSession(
        session_dir=str(session_dir),
        keyframes=keyframes,
        meta={"source": "extract_video_frames", "width": width, "height": height, "fps": fps},
    )
