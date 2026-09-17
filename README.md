# renee_perception

RGB-D capture and 3D reconstruction package for the RENEE project

## Pipeline


```bash
python tools/record_data.py                                      # ZED preview
python tools/record_data.py --camera realsense      # complete RealSense dataset
# Select a device explicitly when more than one RealSense is connected.
python tools/record_data.py --camera realsense --serial-number 123456789
# Optional streams are enabled by default when the camera supports them.
python tools/record_data.py --camera realsense --no-infrared --no-imu --no-aligned-depth

# In the preview press c, then c again. One shot captures 30 samples by default.
# The prompt allows selecting a different number before taking the shot.

python tools/extract_video_frames.py --video my_scan/rgb/rgb_video_*.mp4 \
    --depth-dir my_scan/depth --depth-units auto --every 3 \
    --intrinsics my_scan/intrinsics/camera_intrinsics.json \
    --output my_scan/extracted/session_x
python tools/build_pointcloud.py my_scan/extracted/session_x --poses-only # inspect trajectory first
python tools/visualize.py my_scan/extracted/session_x/session.json
python tools/build_pointcloud.py my_scan/extracted/session_x              # -> fused_cloud.ply + session.json + registration_report.json
```

## Campetella continuous sweep

Start the direct recorder with the RealSense backend (and do not run the ROS
RealSense camera node at the same time, because only one process can own the
device):

```bash
python tools/record_data.py --camera realsense --fps 30
```

The RealSense exposure and gain are allowed to settle and are then locked by
default. Do not pass `--no-lock-capture-settings` for this dataset.

From the preview, use one of these continuous modes:

- Press `r` for video-rate RGB-D capture. This is preferred when the computer
  and storage can sustain it.
- Press `b` for a lower-bandwidth continuous sweep. The default is 2 saved
  RGB-D pairs/s; use a higher value when possible and press `q` only after the
  complete 180--360 degree lap.

At the end, the recorder prints the effective saved RGB-D rate and warns if it
was below 2 fps. Keep the camera 1--2 m from the machine, move roughly 10--15 cm
between saved frames, and include both beam ends and the column. Use the same
output session name for all parts that belong to one dataset.

For the two ArUco close-ups, return to the preview, press `c`, select one image
per shot, and capture each marker from 0.8--1.2 m. Also include a view containing
both markers during the sweep.

Run `capture_station.sh` in parallel for the ROS/Vogui logs, with a duration
that covers the entire sweep. It is responsible for `joint_states.csv`, `/tf`,
`/tf_static`, and the `vogui/csv` tree; check its warnings and confirm that
`joint_states.csv` contains data before leaving the machine.

## RealSense dataset

The supported robot workflow is the ROS `/capture_rgbd` action. The direct
Python recorder below remains an offline utility and must not run while the
ROS RealSense node owns the device.

A RealSense image shot stores synchronized records in `frames.jsonl` and:

- `rgb/`: colour PNGs.
- `depth_raw_16/`: native, unaligned Z16 PNGs in device units.
- `depth_aligned_16/`: optional Z16 PNGs aligned to colour.
- `depth_mm/`: aligned depth in millimetres for the reconstruction pipeline.
- `depth/`: aligned float32 depth arrays in metres.
- `infrared_left/` and `infrared_right/`: stereo Y8 images when supported.
- `imu.jsonl`: accelerometer (m/s²) and gyroscope (rad/s) samples from the direct recorder.
- `intrinsics/camera_intrinsics.json`: native intrinsics, distortion models,
  depth-to-colour/IR-to-colour extrinsics, device information and
  `depth_units` (`metres_per_unit`).

For the direct recorder, `frames.jsonl` keeps the hardware timestamps and paths
for every saved image stream. Datasets produced through the ROS
`/capture_rgbd` action additionally include the per-keyframe TF chain, wheel
odometry, AMCL covariance and rover IMU, with station validity under
`stations/`. The separate `capture_station.sh` logger writes its ROS/Vogui data
under `vogui/`. Empty optional directories can be ignored in simulation.

In simulation, `/capture_rgbd` also supports `wrist_camera:=stereolabs_zed2i`.
The left image is stored as the primary RGB stream, registered depth is used
for reconstruction, the right image is stored under `stereo_right/`, and ZED
IMU samples are written to `camera_imu.jsonl`. Dataset metadata records the
camera model separately from whether the source is simulation or hardware.


## Testing

```bash
python -m pytest
```
