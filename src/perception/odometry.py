#!/usr/bin/env python3
"""Pose estimation for raw (unposed) RGB-D captures.

Real captures (ZED video -> ``extract_video_frames.py``) carry no camera pose,
unlike a simulated Scan-0 orbit where poses are known exactly. This module
fills that gap: sequential RGB-D odometry between consecutive keyframes,
refined by a pose graph. A loop-closure edge back to the first keyframe is
added (when it registers well) so ``global_optimization`` has something to
redistribute drift against — a purely sequential chain has no redundant
constraint to optimise.

Frame 0 defines the world frame: ``T_world_cam[0] = identity``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from . import io as sio
from .contracts import Keyframe, ScanSession


def _rgbd_for_odometry(root: Path, kf: Keyframe, depth_trunc: float) -> o3d.geometry.RGBDImage:
    color = sio.load_rgb(root, kf)
    depth = sio.load_depth(root, kf)
    # RGB-D odometry operates on intensity, not color -> convert_rgb_to_intensity=True
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_scale=1000.0, depth_trunc=depth_trunc,
        convert_rgb_to_intensity=True)


def _pairwise_odometry(rgbd_source: o3d.geometry.RGBDImage, rgbd_target: o3d.geometry.RGBDImage,
                       intrinsic: o3d.camera.PinholeCameraIntrinsic, o: dict
                       ) -> tuple[bool, np.ndarray, np.ndarray]:
    option = o3d.pipelines.odometry.OdometryOption()
    option.depth_diff_max = o["depth_diff_max"]
    option.depth_max = o["depth_trunc"]
    success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
        rgbd_source, rgbd_target, intrinsic, np.eye(4),
        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
    return success, trans, info


def estimate_poses(session: ScanSession, root: Path, cfg: dict) -> list[np.ndarray]:
    """Estimate ``T_world_cam`` for every keyframe in ``session``.

    Sequential RGB-D odometry chains consecutive keyframes; the result feeds a
    pose graph (with an optional loop-closure edge) that is globally optimised
    to spread out accumulated drift. Returns one 4x4 pose per keyframe, same
    order as ``session.keyframes`` — does NOT mutate the session.
    """
    o = cfg["odometry"]
    kfs = session.keyframes
    n = len(kfs)
    if n == 0:
        return []
    if n == 1:
        return [np.eye(4)]

    depth0 = np.asarray(o3d.io.read_image(str(sio.resolve(root, kfs[0].depth_path))))
    h, w = depth0.shape[:2]
    intrinsic = sio.keyframe_intrinsic(kfs[0], w, h)

    rgbds = [_rgbd_for_odometry(root, kf, o["depth_trunc"]) for kf in kfs]

    pose_graph = o3d.pipelines.registration.PoseGraph()
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))

    accum = np.eye(4)  # T_{i<-0}: world (frame-0) coords -> frame i's local coords
    failed: list[int] = []
    for i in range(1, n):
        success, trans, info = _pairwise_odometry(rgbds[i - 1], rgbds[i], intrinsic, o)
        if not success:
            failed.append(i)
            trans, info = np.eye(4), np.eye(6)  # assume no motion rather than drop the frame
        accum = trans @ accum
        T_world_cam = np.linalg.inv(accum)
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(T_world_cam))
        pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
            i - 1, i, trans, info, uncertain=False))
    if failed:
        print(f"warning: odometry failed on {len(failed)}/{n - 1} pairs "
              f"(frames {failed[:5]}{'...' if len(failed) > 5 else ''}); held pose constant")

    if o.get("loop_closure", True) and n > 2:
        success, trans, info = _pairwise_odometry(rgbds[0], rgbds[-1], intrinsic, o)
        if success:
            pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                0, n - 1, trans, info, uncertain=True))

    opt_option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=o.get("pose_graph_max_correspondence", 0.05),
        edge_prune_threshold=o.get("edge_prune_threshold", 0.25),
        reference_node=0)
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        opt_option)

    return [np.asarray(node.pose, dtype=float).copy() for node in pose_graph.nodes]
