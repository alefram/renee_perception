"""RGB-D odometry + pose graph pose estimation, against known synthetic poses."""

from __future__ import annotations

import numpy as np

from perception import io as sio
from perception import odometry


def test_estimate_poses_matches_ground_truth(synth_session, synth_cfg):
    session_dir, intrinsics_path, _gt_world, gt_rel = synth_session
    session = sio.session_from_extracted(session_dir, intrinsics_path, fps=30.0)
    root = sio.session_root(session)

    poses = odometry.estimate_poses(session, root, synth_cfg)

    assert len(poses) == len(session.keyframes)
    assert np.allclose(poses[0], np.eye(4), atol=1e-6)  # frame 0 anchors the world frame

    for est, gt in zip(poses, gt_rel):
        translation_error = np.linalg.norm(est[:3, 3] - gt[:3, 3])
        assert translation_error < 0.08, f"translation error {translation_error:.3f} m too large"

        R_err = est[:3, :3].T @ gt[:3, :3]
        angle = np.degrees(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1)))
        assert angle < 10.0, f"rotation error {angle:.1f} deg too large"
