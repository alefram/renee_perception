# renee_perception

RGB-D capture and 3D reconstruction package for the RENEE project

## Pipeline


```bash
python tools/record_data.py --mode video --output-name my_scan          # capture (ZED)
# Or with an Intel RealSense. Depth is aligned to the colour image before saving.
python tools/record_data.py --camera realsense --mode video --output-name my_scan
# Select a device explicitly when more than one RealSense is connected:
python tools/record_data.py --camera realsense --serial-number 123456789 --mode preview
python tools/extract_video_frames.py --video my_scan/rgb/rgb_video_*.mp4 \
    --depth-dir my_scan/depth --depth-units auto --every 3 \
    --intrinsics my_scan/intrinsics/camera_intrinsics.json \
    --output my_scan/extracted/session_x
python tools/build_pointcloud.py my_scan/extracted/session_x --poses-only # inspect trajectory first
python tools/visualize.py my_scan/extracted/session_x/session.json
python tools/build_pointcloud.py my_scan/extracted/session_x              # -> fused_cloud.ply + session.json + registration_report.json
```


## Testing

```bash
python -m pytest
```