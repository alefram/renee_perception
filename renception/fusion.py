#!/usr/bin/env python3
"""TSDF fusion of registered RGB-D keyframes into a clean cloud + mesh.

Each keyframe is integrated into a truncated signed-distance volume
(Open3D's tensor-based ``VoxelBlockGrid``, which runs on whatever
``fusion.device`` in the config points at -- "CPU:0" by default, "CUDA:0" for
GPU integration on a machine with a CUDA-enabled Open3D build) at its given
world pose; the zero level set is then extracted as a point cloud and a
triangle mesh (marching cubes). Poses are an explicit argument (not read off
``Keyframe.T_world_cam``): callers decide whether they come from
``renception.odometry.estimate_poses`` (real data) or are already known
(synthetic tests), which keeps fusion decoupled from how poses were obtained.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from . import cleanup
from . import io as sio
from .contracts import ScanSession


def fuse_session(session: ScanSession, root: Path, cfg: dict, poses: list[np.ndarray]
                 ) -> tuple[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]:
    """Integrate all keyframes into a TSDF and extract (cloud, mesh).

    ``poses[i]`` is ``T_world_cam`` for ``session.keyframes[i]``.
    """
    if len(poses) != len(session.keyframes):
        raise ValueError(f"{len(poses)} poses for {len(session.keyframes)} keyframes")

    f = cfg["fusion"]
    voxel = float(f["voxel"])
    depth_max = float(f["depth_trunc"])
    trunc_voxel_multiplier = float(f["sdf_trunc_factor"])
    device = o3d.core.Device(f.get("device", "CPU:0"))

    volume = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3d.core.Dtype.Float32, o3d.core.Dtype.Float32, o3d.core.Dtype.Float32),
        attr_channels=((1,), (1,), (3,)),
        voxel_size=voxel,
        block_resolution=16,
        block_count=int(f.get("block_count", 100000)),
        device=device)

    for kf, T_world_cam in zip(session.keyframes, poses):
        color = o3d.t.geometry.Image.from_legacy(sio.load_rgb(root, kf), device=device)
        depth = o3d.t.geometry.Image.from_legacy(sio.load_depth(root, kf), device=device)
        intrinsic = o3d.core.Tensor(np.asarray(kf.intrinsics, dtype=np.float64))
        extrinsic = o3d.core.Tensor(np.linalg.inv(T_world_cam))  # extrinsic = T_cam_world

        block_coords = volume.compute_unique_block_coordinates(
            depth, intrinsic, extrinsic, depth_scale=1000.0, depth_max=depth_max,
            trunc_voxel_multiplier=trunc_voxel_multiplier)
        volume.integrate(block_coords, depth, color, intrinsic, extrinsic,
                         depth_scale=1000.0, depth_max=depth_max,
                         trunc_voxel_multiplier=trunc_voxel_multiplier)

    cloud = volume.extract_point_cloud().to_legacy()
    mesh = volume.extract_triangle_mesh().to_legacy()
    if len(cloud.points) == 0:
        raise ValueError("TSDF fusion produced an empty cloud; check poses/depth/intrinsics")

    cloud = cleanup.filter_cloud(cloud, cfg)
    margin = float(cfg["cleanup"].get("crop_margin", 0.03))
    mesh = cleanup.crop_mesh_to_cloud(mesh, cloud, margin)
    mesh = cleanup.smooth_mesh(mesh, cfg)
    mesh.compute_vertex_normals()
    return cloud, mesh
