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
## Camera link service (Jetson, ZED 2i)

`tools/camera_link_service.py` keeps the ZED open on the Jetson and serves RGB or RGB-D
frames to the PC on request, over TCP (protocol:
`renee_action_servers/docs/camera_link_protocol.md`; PC side: the `capture_camera_frames`
action). Nothing is written to disk. It targets the Jetson's **Python 3.6** (stdlib + numpy +
pyzed only), unlike the rest of this repo.

```bash
python3 tools/camera_link_service.py                      # ZED, listens on 127.0.0.1:7788
python3 tools/camera_link_service.py --host 0.0.0.0       # reachable without an SSH tunnel
python3 tools/camera_link_service.py --source synthetic   # no camera: protocol/network tests
python3 test/test_camera_link_service.py -v               # tests (synthetic camera, stdlib only)
ssh -N -L 7788:127.0.0.1:7788 jetson                      # on the PC, to reach it
```

- Run it with `tools/camera_link_ctl.sh start|stop|restart|status` (no systemd, no sudo: nohup +
  pidfile, state and log in `~/.camera_link/`; pass service flags with `CAMERA_LINK_ARGS`). It prints one
  `camera-link: ...` status line; exit 0 = ok/running, 1 = failure or (for `status`) not running.
  `stop`/`restart` wait until the process has exited, so the ZED is released before returning.
- The ZED can be held by one process only: stop the ROS1 `zed_wrapper` (started by
  `jetson-ros.service` / `bringup.sh`, now disabled on this Jetson) before starting the service.
- The ZED 2i must be linked at **USB 3 (5000M)**. If it enumerates on USB 2.0 (480M) the SDK fails with
  `CAMERA NOT DETECTED`; the service stays up, keeps retrying with back-off (each failed open resets the
  camera's USB), says why in `status`/the log, and answers capture requests with that error. Check with
  `lsusb -t` (the video interface should be under `Bus 02 ... 5000M`).
- Idle it grabs without depth at `--idle-period` (default 0.2 s) and stops grabbing while a capture is
  being encoded/sent; depth is computed only for `rgbd` requests (`RuntimeParameters.enable_depth`
  per grab, SDK 3.7; measured on the 5W Nano: grab 67 ms without depth vs 205 ms with QUALITY depth).
  Why not grab continuously: pyzed's `grab()` holds the Python GIL while it waits for a frame, which
  starved the socket and zlib threads (ping RTT ~100 ms, every step rounded up to a 66 ms frame period).
- One client at a time: a new connection replaces (and aborts the capture of) the previous one.
- Returned frames are always captured after the request arrived; timestamps are the ZED image
  time in the Jetson's wall clock. The PC measures the clock offset itself on every goal.
- `deploy/camera-link.service` is **not used by default** (the service is managed with
  `camera_link_ctl.sh`); it is kept only as an optional systemd alternative.
