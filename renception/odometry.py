#!/usr/bin/env python3
"""Estimate camera poses from synchronized RGB-D keyframes.

Open3D computes the relative transformation between RGB-D frames and refines
the resulting trajectory with pose-graph optimization. Frame zero defines the
world frame, so its ``T_world_cam`` is the identity matrix.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d

from . import io as session_io
from .contracts import Keyframe, ScanSession


DEFAULT_DEPTH_DIFF_MAX = 0.07
DEFAULT_MIN_DEPTH_FRACTION = 0.20
DEFAULT_MIN_BLUR_VARIANCE = 40.0
DEFAULT_LOOP_MIN_SEPARATION = 30
DEFAULT_LOOP_CHECK_INTERVAL = 10
DEFAULT_LOOP_SEARCH_RADIUS = 0.25
DEFAULT_LOOP_MAX_EDGES = 20
DEFAULT_ICP_VOXEL_SIZE = 0.02
DEFAULT_ICP_MAX_CORRESPONDENCE_DISTANCE = 0.05
DEFAULT_ICP_MAX_ITERATIONS = 30
DEFAULT_ICP_MIN_FITNESS = 0.20
DEFAULT_ICP_MIN_POINTS = 100
DEFAULT_POSE_GRAPH_DISTANCE = 0.05
DEFAULT_EDGE_PRUNE_THRESHOLD = 0.25

SKIP_REASONS = (
    "insufficient_depth",
    "blurry",
)

Config = dict[str, Any]
PoseList = list[np.ndarray]
Report = dict[str, Any]


@dataclass(frozen=True)
class ValidationThresholds:
    """Thresholds used to accept or reject RGB-D keyframes."""

    min_depth_valid_fraction: float
    min_blur_variance: float

    @classmethod
    def from_config(cls, config: Config) -> ValidationThresholds:
        """Create validation thresholds from an odometry configuration.

        Args:
            config (dict[str, Any]): Mapping of threshold names to numeric
                ``int | float`` values.

        Returns:
            ValidationThresholds: Immutable threshold values stored as floats.
        """
        return cls(
            min_depth_valid_fraction=float(
                config.get(
                    "min_depth_valid_fraction",
                    DEFAULT_MIN_DEPTH_FRACTION,
                )
            ),
            min_blur_variance=float(
                config.get("min_blur_variance", DEFAULT_MIN_BLUR_VARIANCE)
            ),
        )


def _rgbd_for_odometry(
    root: Path,
    keyframe: Keyframe,
    depth_trunc: float,
) -> o3d.geometry.RGBDImage:
    """Load one keyframe as an intensity RGB-D image for Open3D.

    Args:
        root (Path): Absolute base path for RGB and depth files.
        keyframe (Keyframe): Record containing relative RGB/depth paths.
        depth_trunc (float): Maximum valid depth in meters.

    Returns:
        open3d.geometry.RGBDImage: Intensity color image plus metric depth image.
    """
    color = session_io.load_rgb(root, keyframe)
    depth = session_io.load_depth(root, keyframe)
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_scale=1000.0,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=True,
    )


def _camera_intrinsic(
    root: Path,
    keyframe: Keyframe,
) -> o3d.camera.PinholeCameraIntrinsic:
    """Create Open3D camera intrinsics from a session keyframe.

    Args:
        root (Path): Absolute base path for the depth file.
        keyframe (Keyframe): Record with a ``(3, 3)`` intrinsic matrix and
            depth-image path.

    Returns:
        open3d.camera.PinholeCameraIntrinsic: Intrinsic matrix and image size.
    """
    depth = np.asarray(
        o3d.io.read_image(str(session_io.resolve(root, keyframe.depth_path)))
    )
    height, width = depth.shape[:2]
    return session_io.keyframe_intrinsic(keyframe, width, height)


def _refine_with_icp(
    source: o3d.geometry.RGBDImage,
    target: o3d.geometry.RGBDImage,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    initial_transform: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Refine an RGB-D transformation with point-to-plane ICP.

    Args:
        source (open3d.geometry.RGBDImage): Source RGB-D frame.
        target (open3d.geometry.RGBDImage): Target RGB-D frame.
        intrinsic (open3d.camera.PinholeCameraIntrinsic): Shared camera model.
        initial_transform (np.ndarray): Source-to-target ``float`` matrix with
            shape ``(4, 4)``.
        config (dict[str, Any]): Mapping containing ICP voxel size,
            correspondence distance, iterations, and minimum fitness.

    Returns:
        tuple[np.ndarray, np.ndarray] | None: Refined source-to-target
            ``float64`` matrix with shape ``(4, 4)`` and information matrix
            with shape ``(6, 6)``; ``None`` when refinement is unreliable.
    """
    voxel_size = float(config.get("icp_voxel_size", DEFAULT_ICP_VOXEL_SIZE))
    max_distance = float(
        config.get(
            "icp_max_correspondence_distance",
            DEFAULT_ICP_MAX_CORRESPONDENCE_DISTANCE,
        )
    )
    max_iterations = int(
        config.get("icp_max_iterations", DEFAULT_ICP_MAX_ITERATIONS)
    )
    min_fitness = float(
        config.get("icp_min_fitness", DEFAULT_ICP_MIN_FITNESS)
    )

    source_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        source,
        intrinsic,
    )
    target_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        target,
        intrinsic,
    )
    if voxel_size > 0.0:
        source_cloud = source_cloud.voxel_down_sample(voxel_size)
        target_cloud = target_cloud.voxel_down_sample(voxel_size)

    if (
        len(source_cloud.points) < DEFAULT_ICP_MIN_POINTS
        or len(target_cloud.points) < DEFAULT_ICP_MIN_POINTS
    ):
        return None

    normal_radius = max(2.0 * voxel_size, max_distance)
    normal_search = o3d.geometry.KDTreeSearchParamHybrid(
        radius=normal_radius,
        max_nn=30,
    )
    source_cloud.estimate_normals(normal_search)
    target_cloud.estimate_normals(normal_search)

    result = o3d.pipelines.registration.registration_icp(
        source_cloud,
        target_cloud,
        max_distance,
        np.asarray(initial_transform, dtype=float),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iterations,
        ),
    )
    transform = np.asarray(result.transformation, dtype=float)
    if result.fitness < min_fitness or not np.isfinite(transform).all():
        return None

    information = (
        o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source_cloud,
            target_cloud,
            max_distance,
            transform,
        )
    )
    return transform, np.asarray(information, dtype=float)


def _pairwise_odometry(
    source: o3d.geometry.RGBDImage,
    target: o3d.geometry.RGBDImage,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    config: Config,
    initial_transform: np.ndarray | None = None,
) -> tuple[bool, np.ndarray, np.ndarray]:
    """Estimate and optionally refine a transformation between RGB-D frames.

    Args:
        source (open3d.geometry.RGBDImage): Source RGB-D frame.
        target (open3d.geometry.RGBDImage): Target RGB-D frame.
        intrinsic (open3d.camera.PinholeCameraIntrinsic): Shared camera model.
        config (dict[str, Any]): Mapping containing RGB-D odometry limits and
            optional ICP refinement parameters.
        initial_transform (np.ndarray | None): Initial source-to-target
            ``float`` matrix with shape ``(4, 4)``, or ``None`` for identity.

    Returns:
        tuple[bool, np.ndarray, np.ndarray]: Success flag, source-to-target
            ``float64`` transform with shape ``(4, 4)``, and ``float64``
            information matrix with shape ``(6, 6)``.
    """
    option = o3d.pipelines.odometry.OdometryOption()
    option.depth_diff_max = float(config["depth_diff_max"])
    option.depth_max = float(config["depth_trunc"])
    initial = (
        np.eye(4)
        if initial_transform is None
        else np.asarray(initial_transform, dtype=float)
    )
    success, transform, information = (
        o3d.pipelines.odometry.compute_rgbd_odometry(
            source,
            target,
            intrinsic,
            initial,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            option,
        )
    )
    if success and config.get("icp_refinement", True):
        refinement = _refine_with_icp(
            source,
            target,
            intrinsic,
            transform,
            config,
        )
        if refinement is not None:
            transform, information = refinement
    return success, transform, information


def _optimized_poses(
    pose_graph: o3d.pipelines.registration.PoseGraph,
    config: Config,
) -> PoseList:
    """Optimize a pose graph and extract its camera poses.

    Args:
        pose_graph (open3d.pipelines.registration.PoseGraph): Graph with ``N``
            pose nodes and registration edges.
        config (dict[str, Any]): Mapping containing pose-graph optimization
            thresholds.

    Returns:
        list[np.ndarray]: ``N`` independent ``float64`` matrices with shape
            ``(4, 4)``.
    """
    options = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=float(
            config.get(
                "pose_graph_max_correspondence",
                DEFAULT_POSE_GRAPH_DISTANCE,
            )
        ),
        edge_prune_threshold=float(
            config.get("edge_prune_threshold", DEFAULT_EDGE_PRUNE_THRESHOLD)
        ),
        reference_node=0,
    )
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        options,
    )
    return [
        np.asarray(node.pose, dtype=float).copy()
        for node in pose_graph.nodes
    ]


def estimate_poses(
    session: ScanSession,
    root: Path,
    config: Config,
) -> PoseList:
    """Estimate one ``T_world_cam`` per keyframe without rejecting frames.

    Args:
        session (ScanSession): Session containing ``N`` ordered keyframes.
        root (Path): Absolute base path for RGB-D image files.
        config (dict[str, Any]): Nested mapping containing ``odometry`` values.

    Returns:
        list[np.ndarray]: ``N`` camera-to-world ``float64`` matrices with shape
            ``(4, 4)`` and translation in meters.
    """
    keyframes = session.keyframes
    if not keyframes:
        return []
    if len(keyframes) == 1:
        return [np.eye(4)]

    odometry_config = config["odometry"]
    intrinsic = _camera_intrinsic(root, keyframes[0])
    rgbd_frames = [
        _rgbd_for_odometry(
            root,
            keyframe,
            float(odometry_config["depth_trunc"]),
        )
        for keyframe in keyframes
    ]

    pose_graph = o3d.pipelines.registration.PoseGraph()
    pose_graph.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(np.eye(4))
    )
    accumulated_transform = np.eye(4)
    failed_frames: list[int] = []

    for index in range(1, len(keyframes)):
        success, transform, information = _pairwise_odometry(
            rgbd_frames[index - 1],
            rgbd_frames[index],
            intrinsic,
            odometry_config,
        )
        if not success:
            failed_frames.append(index)
            transform = np.eye(4)
            information = np.eye(6)

        accumulated_transform = transform @ accumulated_transform
        world_from_camera = np.linalg.inv(accumulated_transform)
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(world_from_camera)
        )
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                index - 1,
                index,
                transform,
                information,
                uncertain=False,
            )
        )

    if failed_frames:
        suffix = "..." if len(failed_frames) > 5 else ""
        print(
            f"warning: odometry failed on {len(failed_frames)}/"
            f"{len(keyframes) - 1} pairs "
            f"(frames {failed_frames[:5]}{suffix}); held pose constant"
        )

    loop_edges = 0
    loop_min_separation = int(
        odometry_config.get(
            "loop_min_separation",
            DEFAULT_LOOP_MIN_SEPARATION,
        )
    )
    loop_interval = max(
        1,
        int(
            odometry_config.get(
                "loop_check_interval",
                DEFAULT_LOOP_CHECK_INTERVAL,
            )
        ),
    )
    loop_radius = float(
        odometry_config.get("loop_search_radius", DEFAULT_LOOP_SEARCH_RADIUS)
    )
    loop_max_edges = int(
        odometry_config.get("loop_max_edges", DEFAULT_LOOP_MAX_EDGES)
    )

    if (
        odometry_config.get("loop_closure", True)
        and len(keyframes) > loop_min_separation
    ):
        positions = np.asarray(
            [node.pose[:3, 3] for node in pose_graph.nodes]
        )
        for target_index in range(
            loop_min_separation,
            len(keyframes),
            loop_interval,
        ):
            if loop_edges >= loop_max_edges:
                break

            eligible = positions[: target_index - loop_min_separation + 1]
            distances = np.linalg.norm(
                eligible - positions[target_index],
                axis=1,
            )
            source_index = int(np.argmin(distances))
            if distances[source_index] > loop_radius:
                continue

            initial_transform = (
                np.linalg.inv(pose_graph.nodes[target_index].pose)
                @ pose_graph.nodes[source_index].pose
            )
            success, transform, information = _pairwise_odometry(
                rgbd_frames[source_index],
                rgbd_frames[target_index],
                intrinsic,
                odometry_config,
                initial_transform=initial_transform,
            )
            if success and np.isfinite(transform).all():
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_index,
                        target_index,
                        transform,
                        information,
                        uncertain=True,
                    )
                )
                loop_edges += 1

    if loop_edges:
        print(f"added {loop_edges} loop-closure edges")

    return _optimized_poses(pose_graph, odometry_config)


def _frame_quality(rgbd: o3d.geometry.RGBDImage) -> tuple[float, float]:
    """Measure the depth coverage and sharpness of an RGB-D frame.

    Args:
        rgbd (open3d.geometry.RGBDImage): Intensity and metric-depth images.

    Returns:
        tuple[float, float]: Valid-depth ratio within ``[0.0, 1.0]`` and
            Laplacian variance in intensity-squared units.
    """
    depth = np.asarray(rgbd.depth)
    depth_fraction = float(np.count_nonzero(depth) / depth.size)
    grayscale = np.asarray(rgbd.color)
    if grayscale.max(initial=0.0) <= 1.0:
        grayscale = grayscale * 255.0
    blur_variance = float(
        cv2.Laplacian(grayscale.astype(np.uint8), cv2.CV_64F).var()
    )
    return depth_fraction, blur_variance


def estimate_poses_validated(
    session: ScanSession,
    root: Path,
    config: Config,
) -> tuple[ScanSession, PoseList, Report]:
    """Filter unsuitable RGB-D keyframes and estimate their poses.

    Args:
        session (ScanSession): Session containing ``N`` ordered keyframes.
        root (Path): Absolute base path for RGB-D image files.
        config (dict[str, Any]): Nested mapping containing ``odometry`` values.

    Returns:
        tuple[ScanSession, list[np.ndarray], dict[str, Any]]: Filtered session,
            ``M`` camera-to-world ``float64`` matrices with shape ``(4, 4)``,
            and registration metrics for the ``M`` accepted keyframes.
    """
    candidates = session.keyframes
    if not candidates:
        raise ValueError("cannot estimate poses from an empty session")

    odometry_config = config["odometry"]
    thresholds = ValidationThresholds.from_config(odometry_config)
    depth_trunc = float(odometry_config["depth_trunc"])
    accepted: list[Keyframe] = []
    skipped = {reason: 0 for reason in SKIP_REASONS}
    skipped_frames: list[Report] = []

    for input_index, keyframe in enumerate(candidates):
        rgbd = _rgbd_for_odometry(root, keyframe, depth_trunc)
        depth_fraction, blur_variance = _frame_quality(rgbd)

        if depth_fraction < thresholds.min_depth_valid_fraction:
            reason = "insufficient_depth"
        elif blur_variance < thresholds.min_blur_variance:
            reason = "blurry"
        else:
            accepted.append(keyframe)
            continue

        skipped[reason] += 1
        skipped_frames.append(
            {
                "input_index": input_index,
                "frame_index": int(keyframe.station_id),
                "reason": reason,
                "depth_valid_fraction": depth_fraction,
                "blur_variance": blur_variance,
            }
        )

    if not accepted:
        raise ValueError("no keyframes passed the quality filters")

    filtered_session = ScanSession(
        session_dir=session.session_dir,
        keyframes=accepted,
        cloud_path=session.cloud_path,
        mesh_path=session.mesh_path,
        meta={
            **session.meta,
            "input_frames": len(candidates),
            "accepted_frames": len(accepted),
        },
    )
    poses = estimate_poses(filtered_session, root, config)
    report = {
        "input_frames": len(candidates),
        "accepted_frames": len(accepted),
        "accepted_frame_indices": [
            int(keyframe.station_id) for keyframe in accepted
        ],
        "skipped": skipped,
        "thresholds": asdict(thresholds),
        "skipped_frames": skipped_frames,
    }
    return filtered_session, poses, report
