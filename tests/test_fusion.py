"""TSDF fusion of a synthetic RGB-D session into a point cloud + mesh."""

from __future__ import annotations

import numpy as np

from perception import fusion
from perception import io as sio


def test_fuse_session_reconstructs_box(synth_session, synth_cfg, ground_truth_mesh):
    session_dir, intrinsics_path, gt_world_poses, _gt_rel = synth_session
    session = sio.session_from_extracted(session_dir, intrinsics_path, fps=30.0)
    root = sio.session_root(session)

    cloud, mesh = fusion.fuse_session(session, root, synth_cfg, gt_world_poses)

    assert len(cloud.points) > 500
    assert len(mesh.triangles) > 100
    assert np.allclose(cloud.get_max_bound(), ground_truth_mesh.get_max_bound(), atol=0.05)
    assert np.allclose(cloud.get_min_bound(), ground_truth_mesh.get_min_bound(), atol=0.05)


def test_build_pointcloud_script_writes_session(synth_session, synth_cfg, tmp_path):
    import shutil

    from scripts import build_pointcloud

    session_dir, intrinsics_path, _gt_world, _gt_rel = synth_session
    work = tmp_path / "sess"
    shutil.copytree(session_dir, work)

    session_json = build_pointcloud.build(work, intrinsics_path, synth_cfg, fps=30.0)

    assert session_json.exists()
    assert (work / "fused_cloud.ply").exists()
    assert (work / "tsdf_mesh.ply").exists()
