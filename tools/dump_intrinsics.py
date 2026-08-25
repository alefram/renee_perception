#!/usr/bin/env python3
"""Print the calibration of a connected RealSense camera.

Run this with the SAME stream configuration you used for the capture you already sent
us (same resolution, same streams). Intrinsics belong to the camera and the stream
profile, not to the individual recording, so re-running the configuration now gives
exactly the values that were in effect during that capture.

    python dump_intrinsics.py                      # 640x480, the config we assume you used
    python dump_intrinsics.py --width 1280 --height 720
    python dump_intrinsics.py > intrinsics.txt     # then send us the file

Requires: pyrealsense2 (pip install pyrealsense2), and the camera plugged in.
"""

from __future__ import annotations

import argparse
import json

import pyrealsense2 as rs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    pipeline, config = rs.pipeline(), rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipeline.start(config)

    try:
        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()

        out: dict = {
            "device": device.get_info(rs.camera_info.name),
            "serial_number": device.get_info(rs.camera_info.serial_number),
            "firmware_version": device.get_info(rs.camera_info.firmware_version),
            "depth_scale_metres_per_unit": depth_sensor.get_depth_scale(),
            "requested": {"width": args.width, "height": args.height, "fps": args.fps},
            "streams": {},
        }

        for name, stream in (("color", rs.stream.color), ("depth", rs.stream.depth)):
            vsp = profile.get_stream(stream).as_video_stream_profile()
            i = vsp.get_intrinsics()
            out["streams"][name] = {
                "width": i.width, "height": i.height,
                "fx": i.fx, "fy": i.fy, "cx": i.ppx, "cy": i.ppy,
                "distortion_model": str(i.model), "distortion_coeffs": list(i.coeffs),
            }

        # depth -> color extrinsics: needed to know how the two sensors relate
        d_prof = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        c_prof = profile.get_stream(rs.stream.color).as_video_stream_profile()
        e = d_prof.get_extrinsics_to(c_prof)
        out["extrinsics_depth_to_color"] = {"rotation": list(e.rotation),
                                            "translation": list(e.translation)}

        if args.json:
            print(json.dumps(out, indent=2))
        else:
            print(f"device            : {out['device']}")
            print(f"serial number     : {out['serial_number']}")
            print(f"firmware          : {out['firmware_version']}")
            print(f"depth scale       : {out['depth_scale_metres_per_unit']} m per unit")
            print()
            for name, s in out["streams"].items():
                print(f"{name}: {s['width']}x{s['height']}  "
                      f"fx={s['fx']:.4f} fy={s['fy']:.4f} cx={s['cx']:.4f} cy={s['cy']:.4f}")
                print(f"       distortion {s['distortion_model']} {s['distortion_coeffs']}")
            print()
            print(f"depth->color translation (m): {out['extrinsics_depth_to_color']['translation']}")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()