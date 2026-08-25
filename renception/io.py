#!/usr/bin/env python3
"""Load RGB-D capture sessions and persist fused reconstruction artifacts.

All paths stored in :class:`ScanSession` are relative to ``session_dir`` when
possible, allowing a session directory to be moved without changing its
keyframes, point cloud, or mesh references.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from .contracts import Keyframe, ScanSession


DEFAULT_CAPTURE_FPS = 30.0
DEFAULT_CLOUD_PATH = "fused_cloud.ply"
DEFAULT_MESH_PATH = "tsdf_mesh.ply"

PathLike = str | Path
IntrinsicsData = tuple[np.ndarray, int, int]


def session_root(
    session: ScanSession,
    session_json: PathLike | None = None,
) -> Path:
    """Return the base directory used to resolve session asset paths.

    Args:
        session (ScanSession): Session with ``session_dir: str``.
        session_json (str | Path | None): Session JSON path or ``None``.

    Returns:
        Path: Parent of ``session_json`` when provided; otherwise
            ``Path(session.session_dir)``.
    """
    if session_json is not None:
        return Path(session_json).resolve().parent
    return Path(session.session_dir)


def resolve(root: Path, rel_or_abs: PathLike) -> Path:
    """Resolve an asset path against a session directory.

    Args:
        root (Path): Base session directory.
        rel_or_abs (str | Path): Relative session path or absolute filesystem
            path.

    Returns:
        Path: Absolute input path unchanged, or ``root / rel_or_abs``.
    """
    path = Path(rel_or_abs)
    return path if path.is_absolute() else root / path


def load_rgb(root: Path, keyframe: Keyframe) -> o3d.geometry.Image:
    """Load an RGB image referenced by a keyframe.

    Args:
        root (Path): Base directory for the keyframe asset paths.
        keyframe (Keyframe): Record containing ``rgb_path: str``.

    Returns:
        open3d.geometry.Image: Open3D image decoded from ``rgb_path``.
    """
    return o3d.io.read_image(str(resolve(root, keyframe.rgb_path)))


def load_depth(root: Path, keyframe: Keyframe) -> o3d.geometry.Image:
    """Load a millimetre depth image referenced by a keyframe.

    Args:
        root (Path): Base directory for the keyframe asset paths.
        keyframe (Keyframe): Record containing ``depth_path: str``.

    Returns:
        open3d.geometry.Image: Typically a 16-bit depth image in millimetres.
    """
    return o3d.io.read_image(str(resolve(root, keyframe.depth_path)))


def keyframe_intrinsic(
    keyframe: Keyframe,
    width: int,
    height: int,
) -> o3d.camera.PinholeCameraIntrinsic:
    """Build Open3D pinhole intrinsics for one keyframe image size.

    Args:
        keyframe (Keyframe): Record with ``intrinsics`` as a ``float`` matrix
            with shape ``(3, 3)``.
        width (int): Image width in pixels.
        height (int): Image height in pixels.

    Returns:
        open3d.camera.PinholeCameraIntrinsic: Camera model using ``fx``,
            ``fy``, ``cx``, and ``cy`` from the intrinsic matrix.

    Raises:
        ValueError: If ``keyframe.intrinsics`` does not have shape ``(3, 3)``.
    """
    matrix = np.asarray(keyframe.intrinsics, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(
            "keyframe intrinsics must have shape (3, 3), "
            f"got {matrix.shape}"
        )
    return o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        matrix[0, 0],
        matrix[1, 1],
        matrix[0, 2],
        matrix[1, 2],
    )


def write_fused(
    root: Path,
    session_json: PathLike,
    session: ScanSession,
    cloud: o3d.geometry.PointCloud,
    mesh: o3d.geometry.TriangleMesh,
    cloud_rel: PathLike = DEFAULT_CLOUD_PATH,
    mesh_rel: PathLike = DEFAULT_MESH_PATH,
) -> None:
    """Write fused geometry and update the corresponding session manifest.

    Args:
        root (Path): Base directory where cloud and mesh paths are resolved.
        session_json (str | Path): Destination JSON file for ``session``.
        session (ScanSession): Session whose artifact paths and metadata change.
        cloud (open3d.geometry.PointCloud): Fused point cloud to serialize.
        mesh (open3d.geometry.TriangleMesh): Fused mesh to serialize.
        cloud_rel (str | Path): Cloud path relative to ``root``.
        mesh_rel (str | Path): Mesh path relative to ``root``.

    Returns:
        None: The cloud, mesh, and updated session manifest are written to disk.

    Raises:
        OSError: If Open3D cannot write the cloud or mesh file.
    """
    cloud_path = resolve(root, cloud_rel)
    mesh_path = resolve(root, mesh_rel)
    if not o3d.io.write_point_cloud(str(cloud_path), cloud):
        raise OSError(f"could not write point cloud: {cloud_path}")
    if not o3d.io.write_triangle_mesh(str(mesh_path), mesh):
        raise OSError(f"could not write triangle mesh: {mesh_path}")

    session.cloud_path = str(cloud_rel)
    session.mesh_path = str(mesh_rel)
    session.meta["fused_points"] = len(cloud.points)
    session.meta["fused_triangles"] = len(mesh.triangles)
    session.save(session_json)


def _load_intrinsics_matrix(intrinsics_path: PathLike) -> IntrinsicsData:
    """Load pinhole intrinsics and image dimensions from calibration JSON.

    Args:
        intrinsics_path (str | Path): UTF-8 JSON file containing
            ``color_camera.camera_matrix`` with shape ``(3, 3)``, plus image
            ``width`` and ``height``.

    Returns:
        tuple[np.ndarray, int, int]: ``float64`` matrix with shape ``(3, 3)``,
            width in pixels, and height in pixels.

    Raises:
        ValueError: If the camera matrix does not have shape ``(3, 3)``.
    """
    data = json.loads(Path(intrinsics_path).read_text(encoding="utf-8"))
    color_camera = data["color_camera"]
    matrix = np.asarray(color_camera["camera_matrix"], dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(
            "color_camera.camera_matrix must have shape (3, 3), "
            f"got {matrix.shape}"
        )
    return matrix, int(color_camera["width"]), int(color_camera["height"])


def session_from_extracted(
    session_dir: PathLike,
    intrinsics_path: PathLike,
    fps: float = DEFAULT_CAPTURE_FPS,
) -> ScanSession:
    """Create a pose-less session from an extracted RGB-D manifest.

    Args:
        session_dir (str | Path): Directory containing ``frames.jsonl``,
            ``rgb/``, and ``depth_mm/``.
        intrinsics_path (str | Path): Calibration JSON for the color camera.
        fps (float): Frame rate in Hz used when a manifest record has no
            ``timestamp``.

    Returns:
        ScanSession: Session with keyframes whose ``T_world_cam`` is ``None``.

    Raises:
        ValueError: If no manifest record contains both RGB and depth paths.
    """
    root = Path(session_dir)
    manifest = root / "frames.jsonl"
    matrix, width, height = _load_intrinsics_matrix(intrinsics_path)
    keyframes: list[Keyframe] = []

    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        record: dict[str, Any] = json.loads(line)
        if "depth_mm" not in record:
            continue

        frame_index = int(record["frame_index"])
        keyframes.append(
            Keyframe(
                rgb_path=record["rgb"],
                depth_path=record["depth_mm"],
                intrinsics=matrix,
                station_id=frame_index,
                timestamp=float(
                    record.get("timestamp", frame_index / fps)
                ),
            )
        )

    if not keyframes:
        raise ValueError(f"no rgb+depth_mm pairs found in {manifest}")

    return ScanSession(
        session_dir=str(root),
        keyframes=keyframes,
        meta={
            "source": "extract_video_frames",
            "width": width,
            "height": height,
            "fps": fps,
        },
    )


def session_from_manifest(
    session_dir: PathLike,
    intrinsics_path: PathLike,
    fps: float = DEFAULT_CAPTURE_FPS,
    require_poses: bool = False,
) -> ScanSession:
    """Load a session from a legacy manifest and optional camera poses.

    Args:
        session_dir (str | Path): Directory containing ``frames.jsonl``.
        intrinsics_path (str | Path): Calibration JSON for the color camera.
        fps (float): Frame rate in Hz used for missing timestamps.
        require_poses (bool): Require ``T_world_camera`` for every keyframe.

    Returns:
        ScanSession: Session with pose-less keyframes when ``require_poses`` is
            ``False``; otherwise with finite ``float64`` poses of shape
            ``(4, 4)``.

    Raises:
        ValueError: If a required pose is missing, malformed, or non-finite.
    """
    session = session_from_extracted(session_dir, intrinsics_path, fps=fps)
    if not require_poses:
        return session

    manifest = Path(session_dir) / "frames.jsonl"
    poses: dict[int, np.ndarray] = {}
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        record: dict[str, Any] = json.loads(line)
        pose = np.asarray(record.get("T_world_camera"), dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(
                f"{manifest}:{line_number} T_world_camera must be a finite "
                f"4x4 matrix, got {pose.shape}"
            )
        poses[int(record["frame_index"])] = pose

    for keyframe in session.keyframes:
        try:
            keyframe.T_world_cam = poses[keyframe.station_id]
        except KeyError as error:
            raise ValueError(
                f"{manifest}: frame {keyframe.station_id} is missing "
                "T_world_camera"
            ) from error
    return session
