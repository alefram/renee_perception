#!/usr/bin/env python3
"""Fuse registered RGB-D keyframes into a filtered cloud and TSDF mesh."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from . import cleanup
from . import io as session_io
from .contracts import ScanSession


Config = dict[str, Any]
PoseList = list[np.ndarray]


def fuse_session(
    session: ScanSession,
    root: Path,
    config: Config,
    poses: PoseList,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]:
    """Integrate registered RGB-D keyframes and filter the extracted geometry.

    Args:
        session (ScanSession): Session containing ``N`` ordered keyframes.
        root (Path): Absolute base path for RGB-D image files.
        config (dict[str, Any]): Mapping containing ``fusion`` and ``cleanup``
            parameter sections loaded from ``fusion.yaml``.
        poses (list[np.ndarray]): ``N`` camera-to-world ``float`` matrices,
            each with shape ``(4, 4)`` and translation in meters.

    Returns:
        tuple[open3d.geometry.PointCloud, open3d.geometry.TriangleMesh]: Filtered
            XYZ point cloud and cropped TSDF triangle mesh.

    Raises:
        ValueError: If pose and keyframe counts differ, an intrinsic matrix is
            invalid, or fusion/cleanup produces an empty point cloud.
    """
    if len(poses) != len(session.keyframes):
        raise ValueError(
            f"{len(poses)} poses for {len(session.keyframes)} keyframes"
        )

    fusion_config = config["fusion"]
    voxel_size = float(fusion_config["voxel"])
    depth_scale = float(fusion_config["depth_scale"])
    depth_max = float(fusion_config["depth_trunc"])
    truncation_factor = float(fusion_config["sdf_trunc_factor"])
    device = o3d.core.Device(str(fusion_config["device"]))

    volume = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(
            o3d.core.Dtype.Float32,
            o3d.core.Dtype.Float32,
            o3d.core.Dtype.Float32,
        ),
        attr_channels=((1,), (1,), (3,)),
        voxel_size=voxel_size,
        block_resolution=int(fusion_config["block_resolution"]),
        block_count=int(fusion_config["block_count"]),
        device=device,
    )

    for keyframe, world_from_camera in zip(session.keyframes, poses):
        color = o3d.t.geometry.Image.from_legacy(
            session_io.load_rgb(root, keyframe),
            device=device,
        )
        depth = o3d.t.geometry.Image.from_legacy(
            session_io.load_depth(root, keyframe),
            device=device,
        )
        intrinsic_matrix = np.asarray(keyframe.intrinsics, dtype=np.float64)
        if intrinsic_matrix.shape != (3, 3):
            raise ValueError(
                "keyframe intrinsics must have shape (3, 3), "
                f"got {intrinsic_matrix.shape}"
            )

        intrinsic = o3d.core.Tensor(intrinsic_matrix, device=device)
        extrinsic = o3d.core.Tensor(
            np.linalg.inv(world_from_camera),
            device=device,
        )
        block_coordinates = volume.compute_unique_block_coordinates(
            depth,
            intrinsic,
            extrinsic,
            depth_scale=depth_scale,
            depth_max=depth_max,
            trunc_voxel_multiplier=truncation_factor,
        )
        volume.integrate(
            block_coordinates,
            depth,
            color,
            intrinsic,
            extrinsic,
            depth_scale=depth_scale,
            depth_max=depth_max,
            trunc_voxel_multiplier=truncation_factor,
        )

    cloud = volume.extract_point_cloud().to_legacy()
    mesh = volume.extract_triangle_mesh().to_legacy()
    if len(cloud.points) == 0:
        raise ValueError(
            "TSDF fusion produced an empty cloud; check poses, depth, and "
            "intrinsics"
        )

    cloud = cleanup.filter_cloud(cloud, config)
    mesh = cleanup.crop_mesh_to_cloud(
        mesh,
        cloud,
        float(config["cleanup"]["crop_margin"]),
    )
    mesh = cleanup.smooth_mesh(mesh, config)
    mesh.compute_vertex_normals()
    return cloud, mesh
