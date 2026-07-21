#!/usr/bin/env python3
"""Post-fusion point cloud / mesh cleanup.

Ported from the RENEE cleaning pipeline's ``cleaning/surface/meshing.py``
(same filters, same ``cleanup.*`` config keys) so a session fused by
``perception.fusion.fuse_session`` can be de-noised and, when the camera
orbited a single isolated object, have disconnected background left over
from TSDF integration discarded -- without switching mesh reconstruction
away from marching cubes.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d


def filter_cloud(pcd: o3d.geometry.PointCloud, cfg: dict) -> o3d.geometry.PointCloud:
    """Remove outliers and optionally voxel-downsample a fused point cloud.

    Passes (all skipped when the relevant param is 0 / false):
      1. Voxel downsample      -- normalises density.
      2. Statistical OR (SOR)  -- drops points far from their neighbours globally.
      3. Radius OR (ROR)       -- drops points with too few neighbours in a radius;
                                  catches floating blobs that SOR misses.
      4. Largest-cluster keep  -- DBSCAN clustering; only the biggest cluster
                                  (the scanned target) is kept, background/noise
                                  blobs discarded.
    """
    c = cfg["cleanup"]

    vox = float(c.get("downsample_voxel", 0.0))
    if vox > 0.0:
        before = len(pcd.points)
        pcd = pcd.voxel_down_sample(vox)
        print(f"  [cleanup] voxel {vox * 100:.1f}cm: {before} -> {len(pcd.points)} pts")

    nb = int(c.get("sor_nb_neighbors", 0))
    if nb > 0:
        std_ratio = float(c.get("sor_std_ratio", 2.0))
        before = len(pcd.points)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std_ratio)
        print(f"  [cleanup] SOR (nb={nb}, std={std_ratio}): {before} -> {len(pcd.points)} pts")

    ror_nb = int(c.get("ror_nb_neighbors", 0))
    if ror_nb > 0:
        ror_radius = float(c.get("ror_radius", 0.04))
        before = len(pcd.points)
        pcd, _ = pcd.remove_radius_outlier(nb_points=ror_nb, radius=ror_radius)
        print(f"  [cleanup] ROR (nb={ror_nb}, r={ror_radius * 100:.1f}cm): "
              f"{before} -> {len(pcd.points)} pts")

    if c.get("keep_largest_cluster", False):
        eps = float(c.get("cluster_eps", 0.05))
        min_pts = int(c.get("cluster_min_pts", 10))
        before = len(pcd.points)
        labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_pts))
        if labels.max() >= 0:
            largest = int(np.bincount(labels[labels >= 0]).argmax())
            pcd = pcd.select_by_index(np.where(labels == largest)[0])
            print(f"  [cleanup] DBSCAN largest cluster: {before} -> {len(pcd.points)} pts")

    if len(pcd.points) == 0:
        raise ValueError("cloud is empty after cleanup filters -- loosen cleanup params")
    return pcd


def crop_mesh_to_cloud(mesh: o3d.geometry.TriangleMesh, pcd: o3d.geometry.PointCloud,
                       margin: float) -> o3d.geometry.TriangleMesh:
    """Crop ``mesh`` to the (filtered) cloud's bounding box + margin.

    Keeps the TSDF marching-cubes mesh consistent with whatever ``filter_cloud``
    removed (background clusters, outliers, ...) -- the mesh has no filters of
    its own, so without this it would still show geometry the cloud dropped.
    """
    aabb = pcd.get_axis_aligned_bounding_box()
    grown = o3d.geometry.AxisAlignedBoundingBox(
        aabb.get_min_bound() - margin, aabb.get_max_bound() + margin)
    return mesh.crop(grown)


def smooth_mesh(mesh: o3d.geometry.TriangleMesh, cfg: dict) -> o3d.geometry.TriangleMesh:
    """Taubin smoothing: reduces surface noise without shrinking the mesh."""
    iters = int(cfg["cleanup"].get("smooth_iter", 0))
    if iters <= 0:
        return mesh
    return mesh.filter_smooth_taubin(number_of_iterations=iters, lambda_filter=0.5, mu=-0.53)
