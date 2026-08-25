#!/usr/bin/env python3
"""Reconstruct a fused point cloud from an RGB-D capture session.

``session.json`` is the primary input format. Legacy ``frames.jsonl`` datasets
remain supported and are converted to a :class:`ScanSession` before pose
estimation and TSDF integration.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from . import io as session_io
from . import odometry
from .contracts import ScanSession


DEFAULT_CAPTURE_FPS = 30.0
DEFAULT_BLOCK_COUNT = 5000

Config = dict[str, Any]
PoseList = list[np.ndarray]


def _camera_file(
    root: Path,
    intrinsics_path: str | Path | None,
) -> Path:
    """Resolve the camera-intrinsics file used by legacy datasets.

    Args:
        root (Path): Absolute capture-session directory.
        intrinsics_path (str | Path | None): Explicit intrinsics path or
            ``None``.

    Returns:
        Path: Resolved explicit path or ``root/intrinsics/`` default path.
    """
    if intrinsics_path is not None:
        return Path(intrinsics_path).resolve()
    return root / "intrinsics" / "camera_intrinsics.json"


def _capture_fps(camera_file: Path) -> float:
    """Read the capture frequency from a camera-intrinsics file.

    Args:
        camera_file (Path): UTF-8 JSON file with
            ``stream_info.fps: int | float``.

    Returns:
        float: Capture frequency in frames per second.
    """
    data = json.loads(camera_file.read_text(encoding="utf-8"))
    try:
        return float(data["stream_info"]["fps"])
    except (KeyError, TypeError, ValueError):
        return DEFAULT_CAPTURE_FPS


def _manifest_has_poses(manifest: Path) -> bool:
    """Check whether every legacy manifest record has a camera pose.

    Args:
        manifest (Path): UTF-8 JSON Lines file. Each record may contain
            ``T_world_camera: list[list[float]] | null`` with shape ``(4, 4)``.

    Returns:
        bool: ``True`` when every record has a non-null pose.

    Raises:
        ValueError: If the manifest contains invalid JSON or mixes records
            with and without camera poses.
    """
    pose_flags: list[bool] = []
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSON in {manifest}:{line_number}"
            ) from error
        pose_flags.append(record.get("T_world_camera") is not None)

    if pose_flags and any(pose_flags) and not all(pose_flags):
        raise ValueError(
            f"{manifest} mixes frames with and without T_world_camera; "
            "provide poses for every frame or for none"
        )
    return bool(pose_flags) and all(pose_flags)


def _session_has_poses(session: ScanSession, session_file: Path) -> bool:
    """Check whether every keyframe in a session has a camera pose.

    Args:
        session (ScanSession): Session containing ``list[Keyframe]``.
        session_file (Path): Session JSON path used in error messages.

    Returns:
        bool: ``True`` when every ``T_world_cam`` is non-null.

    Raises:
        ValueError: If the session mixes keyframes with and without poses.
    """
    pose_flags = [
        keyframe.T_world_cam is not None
        for keyframe in session.keyframes
    ]
    if pose_flags and any(pose_flags) and not all(pose_flags):
        raise ValueError(
            f"{session_file} mixes keyframes with and without T_world_cam; "
            "provide poses for every keyframe or for none"
        )
    return bool(pose_flags) and all(pose_flags)


def _offline_config(config: Config) -> Config:
    """Complete the reconstruction configuration with offline defaults.

    Args:
        config (dict[str, Any]): Nested mapping with ``fusion``, optional
            ``odometry``, and optional ``cleanup`` mappings.

    Returns:
        dict[str, Any]: Deep-copied configuration with default values filled.
    """
    result = copy.deepcopy(config)
    fusion_config = result["fusion"]
    fusion_config.setdefault("device", "CPU:0")
    fusion_config.setdefault("block_count", DEFAULT_BLOCK_COUNT)

    defaults = {
        "depth_trunc": fusion_config["depth_trunc"],
        "depth_diff_max": odometry.DEFAULT_DEPTH_DIFF_MAX,
        "min_depth_valid_fraction": odometry.DEFAULT_MIN_DEPTH_FRACTION,
        "min_blur_variance": odometry.DEFAULT_MIN_BLUR_VARIANCE,
        "pose_graph_max_correspondence": (
            odometry.DEFAULT_POSE_GRAPH_DISTANCE
        ),
        "edge_prune_threshold": odometry.DEFAULT_EDGE_PRUNE_THRESHOLD,
    }
    odometry_config = result.setdefault("odometry", {})
    for name, value in defaults.items():
        odometry_config.setdefault(name, value)
    result.setdefault("cleanup", {})
    return result


def _load_dataset(
    root: Path,
    intrinsics_path: str | Path | None,
) -> tuple[ScanSession, bool]:
    """Load a capture session, preferring ``session.json``.

    Args:
        root (Path): Absolute capture-session directory.
        intrinsics_path (str | Path | None): Legacy camera-intrinsics path or
            ``None``.

    Returns:
        tuple[ScanSession, bool]: Loaded session and all-poses-present flag.

    Raises:
        FileNotFoundError: If neither supported manifest exists or legacy
            camera intrinsics cannot be found.
        ValueError: If the session is empty or its pose data is inconsistent.
    """
    session_file = root / "session.json"
    manifest = root / "frames.jsonl"

    if session_file.is_file():
        session = ScanSession.load(session_file)
        if session.keyframes:
            print(
                f"loaded {len(session.keyframes)} keyframes from "
                f"{session_file}"
            )
            return session, _session_has_poses(session, session_file)
        if not manifest.is_file():
            raise ValueError(f"session contains no keyframes: {session_file}")

    if not manifest.is_file():
        raise FileNotFoundError(
            f"no session.json or dataset manifest found in: {root}"
        )

    camera_file = _camera_file(root, intrinsics_path)
    if not camera_file.is_file():
        raise FileNotFoundError(f"camera intrinsics not found: {camera_file}")

    has_poses = _manifest_has_poses(manifest)
    session = session_io.session_from_manifest(
        root,
        camera_file,
        fps=_capture_fps(camera_file),
        require_poses=has_poses,
    )
    print(f"loaded {len(session.keyframes)} keyframes from {manifest}")
    return session, has_poses


def _poses_from_session(session: ScanSession) -> PoseList:
    """Read and validate the camera poses stored in a session.

    Args:
        session (ScanSession): Session whose keyframes contain
            ``T_world_cam`` matrices.

    Returns:
        list[np.ndarray]: Finite ``float64`` matrices with shape ``(4, 4)``.

    Raises:
        ValueError: If any pose is missing, non-finite, or not 4x4.
    """
    poses: PoseList = []
    for keyframe in session.keyframes:
        pose = np.asarray(keyframe.T_world_cam, dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(
                f"keyframe {keyframe.station_id} has an invalid T_world_cam"
            )
        poses.append(pose)
    return poses


def _fuse_legacy(
    root: Path,
    session: ScanSession,
    poses: PoseList,
    config: Config,
) -> o3d.geometry.PointCloud:
    """Fuse registered keyframes with Open3D's classic TSDF volume.

    Args:
        root (Path): Absolute base path for RGB-D image files.
        session (ScanSession): Session with ``N`` ordered keyframes.
        poses (list[np.ndarray]): ``N`` camera-to-world ``float`` matrices,
            each with shape ``(4, 4)`` and translation in meters.
        config (dict[str, Any]): Nested mapping containing ``fusion`` values.

    Returns:
        open3d.geometry.PointCloud: Fused point cloud with XYZ points in meters.

    Raises:
        ValueError: If pose and keyframe counts differ or fusion produces an
            empty point cloud.
    """
    if len(poses) != len(session.keyframes):
        raise ValueError(
            f"{len(poses)} poses for {len(session.keyframes)} keyframes"
        )

    fusion_config = config["fusion"]
    voxel_size = float(fusion_config["voxel"])
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=voxel_size * float(fusion_config["sdf_trunc_factor"]),
        color_type=(
            o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        ),
    )

    for keyframe, pose in zip(session.keyframes, poses):
        color = session_io.load_rgb(root, keyframe)
        depth = session_io.load_depth(root, keyframe)
        height, width = np.asarray(depth).shape[:2]
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=1000.0,
            depth_trunc=float(fusion_config["depth_trunc"]),
            convert_rgb_to_intensity=False,
        )
        intrinsic = session_io.keyframe_intrinsic(
            keyframe,
            width,
            height,
        )
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))

    cloud = volume.extract_point_cloud()
    if len(cloud.points) == 0:
        raise ValueError(
            "TSDF fusion produced an empty cloud; check depth and poses"
        )
    return cloud


def reconstruct_dataset(
    dataset_dir: str | Path,
    config: Config,
    output_path: str | Path | None = None,
    intrinsics_path: str | Path | None = None,
    poses_only: bool = False,
) -> o3d.geometry.PointCloud | None:
    """Estimate missing poses and reconstruct an RGB-D capture session.

    Args:
        dataset_dir (str | Path): Directory containing ``session.json`` or
            ``frames.jsonl``.
        config (dict[str, Any]): Nested ``fusion``, ``odometry``, and
            ``cleanup`` configuration mappings.
        output_path (str | Path | None): Output PLY path or ``None``.
        intrinsics_path (str | Path | None): Legacy intrinsics JSON path or
            ``None``.
        poses_only (bool): Skip TSDF fusion when ``True``.

    Returns:
        open3d.geometry.PointCloud | None: Fused XYZ point cloud, or ``None``
            when ``poses_only=True``.

    Raises:
        FileNotFoundError: If required session data cannot be found.
        ValueError: If the session, poses, or fused point cloud are invalid.
        OSError: If the resulting point cloud cannot be written.
    """
    root = Path(dataset_dir).resolve()
    session_file = root / "session.json"
    report_file = root / "registration_report.json"
    reconstruction_config = _offline_config(config)
    session, has_poses = _load_dataset(root, intrinsics_path)

    if has_poses:
        poses = _poses_from_session(session)
    else:
        print(
            f"estimating RGB-D odometry for "
            f"{len(session.keyframes)} frames..."
        )
        session, poses, report = odometry.estimate_poses_validated(
            session,
            root,
            reconstruction_config,
        )
        print(
            f"accepted {len(poses)}/{report['input_frames']} camera poses"
        )
        for keyframe, pose in zip(session.keyframes, poses):
            keyframe.T_world_cam = pose
        report_file.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"saved registration metrics: {report_file}")

    session.save(session_file)
    print(f"saved trajectory: {session_file}")
    if poses_only:
        return None

    print("fusing accepted keyframes into TSDF...")
    cloud = _fuse_legacy(root, session, poses, reconstruction_config)
    output = (
        Path(output_path)
        if output_path is not None
        else root / "fused_cloud.ply"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(output), cloud):
        raise OSError(f"could not write point cloud: {output}")

    try:
        session.cloud_path = output.resolve().relative_to(root).as_posix()
    except ValueError:
        session.cloud_path = str(output.resolve())
    session.save(session_file)
    return cloud
