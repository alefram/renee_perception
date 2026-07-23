# renee_perception

RGB-D capture and 3D reconstruction package

## Pipeline

```bash
python scripts/record_data.py --mode video --output-name my_scan          # capture (ZED)
python scripts/extract_video_frames.py --video my_scan/rgb/rgb_video_*.mp4 \
    --depth-dir my_scan/depth --output my_scan/extracted/session_x        # pair rgb/depth frames
python scripts/build_pointcloud.py my_scan/extracted/session_x \
    --intrinsics my_scan/intrinsics/camera_intrinsics.json                # -> fused_cloud.ply + tsdf_mesh.ply
```

`build_pointcloud.py`:
1. `perception.io.session_from_extracted` — reads `frames.jsonl` + `rgb/` +
   `depth_mm/` into a (pose-less) `ScanSession`.
2. `perception.odometry.estimate_poses` — sequential RGB-D odometry between
   consecutive keyframes, refined by a pose graph (with a loop-closure edge
   back to frame 0 when it registers, so `global_optimization` has a
   redundant constraint to spread drift against). Real captures have no pose
   attached (unlike a simulated scan), so this step is what makes fusion
   possible on real data.
3. `perception.fusion.fuse_session` — TSDF integration of every keyframe at
   its estimated pose -> point cloud + mesh (marching cubes).

All fusion/odometry parameters live in `src/perception/configs/fusion.yaml` (no magic numbers
in code).

## Testing

```bash
python -m pytest
```

Runs against a small synthetic RGB-D orbit (textured box, rendered headless
via an Open3D raycasting scene) with known ground-truth poses — no hardware,
no network, no large datasets required.
