#!/usr/bin/env python3
"""Shared pytest fixtures: a small synthetic RGB-D orbit around a textured
box, written to disk in the exact layout ``extract_video_frames.py`` produces
(``frames.jsonl`` + ``rgb/`` + ``depth_mm/``), so tests exercise the same
loader (``session_from_extracted``) real data goes through.

Depth + a deterministic surface texture are rendered with a raycasting scene
(headless, no GPU/display needed) rather than needing an OpenGL context.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from renception.config import load_config  # noqa: E402

WIDTH, HEIGHT = 160, 120
FX = FY = 150.0
CX, CY = WIDTH / 2.0, HEIGHT / 2.0


def _look_at(eye: np.ndarray, target: np.ndarray, world_up: np.ndarray = np.array([0.0, 0.0, 1.0])
            ) -> np.ndarray:
    """T_world_cam for a camera at `eye` looking at `target` (OpenCV/Open3D
    convention: camera x=right, y=down, z=forward)."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    T = np.eye(4)
    T[:3, 0] = right
    T[:3, 1] = down
    T[:3, 2] = forward
    T[:3, 3] = eye
    return T


def _texture(points: np.ndarray) -> np.ndarray:
    """Deterministic grayscale pattern painted on the surface — gives RGB-D
    odometry real intensity gradients to track (flat-shaded geometry has none)."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    v = 0.5 + 0.5 * np.sin(25 * x) * np.cos(25 * y) * np.sin(20 * z)
    return np.clip(v * 255.0, 0, 255)


def _render_view(scene: "o3d.t.geometry.RaycastingScene", T_world_cam: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    """(color_uint8 HxWx3, depth_mm uint16 HxW) for one camera pose."""
    us, vs = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
    # z-component left at 1 (not unit-normalized): t_hit along this ray IS the
    # pinhole z-depth, matching what a real depth camera reports.
    dirs_cam = np.stack([
        (us - CX) / FX,
        (vs - CY) / FY,
        np.ones_like(us, dtype=float),
    ], axis=-1).reshape(-1, 3)

    R = T_world_cam[:3, :3]
    origin = T_world_cam[:3, 3]
    dirs_world = dirs_cam @ R.T

    rays = np.concatenate(
        [np.broadcast_to(origin, dirs_world.shape), dirs_world], axis=-1
    ).astype(np.float32)
    ans = scene.cast_rays(o3d.core.Tensor(rays))
    t_hit = ans["t_hit"].numpy()
    valid = np.isfinite(t_hit)
    safe_t = np.where(valid, t_hit, 0.0)

    hit_points = origin[None, :] + dirs_world * safe_t.reshape(-1, 1)
    gray = np.where(valid, _texture(hit_points), 0).astype(np.uint8).reshape(HEIGHT, WIDTH)
    color = np.stack([gray, gray, gray], axis=-1)

    depth_mm = np.where(valid.reshape(HEIGHT, WIDTH), safe_t.reshape(HEIGHT, WIDTH) * 1000.0, 0)
    depth_mm = np.clip(depth_mm, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return color, depth_mm


@pytest.fixture(scope="session")
def synth_cfg() -> dict:
    return load_config()


@pytest.fixture(scope="session")
def ground_truth_mesh() -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_box(0.4, 0.3, 0.5)
    mesh.translate(-mesh.get_center())
    mesh.compute_vertex_normals()
    return mesh


@pytest.fixture(scope="session")
def synth_session(tmp_path_factory, ground_truth_mesh):
    """Returns (session_dir, intrinsics_path, gt_world_poses, gt_poses_rel_frame0).

    ``gt_world_poses`` are true T_world_cam (box centred at the world origin) —
    what fusion needs. ``gt_poses_rel_frame0`` are the same poses expressed
    with frame 0 as the reference (T0^-1 @ T_i) — the convention
    ``odometry.estimate_poses`` anchors its output to (node 0 = identity).
    """
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(ground_truth_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_mesh)

    # RGB-D odometry needs real overlap between consecutive keyframes: a wide
    # angular step (few views around a full circle) or a box too small in
    # frame (mostly invalid background depth) makes odometry converge to a
    # bogus transform instead of failing loudly. Dense views + a close/large
    # box avoid that.
    n_views = 24
    radius = 0.8
    gt_world_poses = []
    for i in range(n_views):
        angle = 2 * np.pi * i / n_views
        eye = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.15])
        gt_world_poses.append(_look_at(eye, np.zeros(3)))

    d = tmp_path_factory.mktemp("synth_session")
    rgb_dir, depth_dir = d / "rgb", d / "depth_mm"
    rgb_dir.mkdir()
    depth_dir.mkdir()

    manifest = d / "frames.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for i, T in enumerate(gt_world_poses):
            color, depth_mm = _render_view(scene, T)
            stem = f"frame_{i:06d}"
            cv2.imwrite(str(rgb_dir / f"{stem}.png"), color)
            cv2.imwrite(str(depth_dir / f"{stem}.png"), depth_mm)
            f.write(json.dumps({
                "frame_index": i,
                "rgb": f"rgb/{stem}.png",
                "depth_mm": f"depth_mm/{stem}.png",
            }) + "\n")

    intrinsics_path = d / "camera_intrinsics.json"
    intrinsics_path.write_text(json.dumps({
        "color_camera": {
            "camera_matrix": [[FX, 0, CX], [0, FY, CY], [0, 0, 1]],
            "width": WIDTH, "height": HEIGHT,
        }
    }))

    T0_inv = np.linalg.inv(gt_world_poses[0])
    gt_rel = [T0_inv @ T for T in gt_world_poses]

    return d, intrinsics_path, gt_world_poses, gt_rel
