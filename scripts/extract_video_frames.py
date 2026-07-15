#!/usr/bin/env python3
"""
Extract image frames from a recorded video (e.g. rgb_video_*.mp4 produced by
zed_data_capturing.py --mode video), optionally paired with the raw per-frame
depth data (.npy) and/or the depth colormap video recorded alongside it.

Frame indices line up 1:1 across rgb_video_*.mp4, depth_video_*.mp4, and
depth_raw_frame_<index>_<timestamp>.npy, since record_video() in
zed_data_capturing.py writes all three for every successfully grabbed frame.

Output layout (written to --output):
    rgb/                  frame_<index>.png
    depth_npy/             frame_<index>.npy   (float32 meters, only with --depth-dir)
    depth_mm/              frame_<index>.png   (16-bit mm, only with --depth-dir)
    depth_visualization/   frame_<index>.png   (only with --depth-video)
    frames.jsonl            per-frame manifest of the files written

Examples:
    # Just pull RGB frames out of a video
    python3 extract_video_frames.py --video rgb_video_20260715_144058.mp4 --output extracted

    # Also copy/convert the matching raw depth for each extracted RGB frame
    python3 extract_video_frames.py --video rgb_video_20260715_144058.mp4 \
        --depth-dir depth --output extracted

    # Also pull matching frames from the depth colormap video
    python3 extract_video_frames.py --video rgb_video_20260715_144058.mp4 \
        --depth-dir depth --depth-video depth/depth_video_20260715_144058.mp4 \
        --output extracted
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

DEPTH_RAW_PATTERN = re.compile(r"depth_raw_frame_(\d+)_")


def index_depth_raw_files(depth_dir: Path) -> dict[int, Path]:
    """Map frame index -> path for depth_raw_frame_<index>_<timestamp>.npy files"""
    index = {}
    for path in depth_dir.glob("depth_raw_frame_*.npy"):
        match = DEPTH_RAW_PATTERN.search(path.name)
        if match:
            index[int(match.group(1))] = path
    return index


def depth_to_uint16_mm(depth_m: np.ndarray) -> np.ndarray:
    """Convert a float32 depth map in meters to a 16-bit millimeter map (0 = invalid)"""
    valid = np.isfinite(depth_m) & (depth_m > 0)
    depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
    depth_mm[valid] = np.clip(np.round(depth_m[valid] * 1000.0), 1, np.iinfo(np.uint16).max).astype(np.uint16)
    return depth_mm


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract image frames from a recorded video, optionally paired with raw depth"
    )
    parser.add_argument("--video", type=Path, required=True, help="Video to extract RGB frames from")
    parser.add_argument("--output", type=Path, default=Path("extracted_frames"), help="Output directory (default: extracted_frames)")
    parser.add_argument("--every", type=int, default=1, help="Extract every Nth frame (default: 1, every frame)")
    parser.add_argument("--depth-dir", type=Path, help="Directory of depth_raw_frame_<index>_<timestamp>.npy files to pair by frame index")
    parser.add_argument("--depth-video", type=Path, help="Depth colormap video recorded alongside --video, to extract matching frames from")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Video does not exist: {args.video}", file=sys.stderr)
        return 1

    rgb_dir = args.output / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    depth_index: dict[int, Path] = {}
    if args.depth_dir:
        if not args.depth_dir.exists():
            print(f"Depth directory does not exist: {args.depth_dir}", file=sys.stderr)
            return 1
        depth_index = index_depth_raw_files(args.depth_dir)
        print(f"Found {len(depth_index)} raw depth frames in {args.depth_dir}")
        depth_npy_dir = args.output / "depth_npy"
        depth_mm_dir = args.output / "depth_mm"
        depth_npy_dir.mkdir(parents=True, exist_ok=True)
        depth_mm_dir.mkdir(parents=True, exist_ok=True)

    depth_video_cap = None
    if args.depth_video:
        if not args.depth_video.exists():
            print(f"Depth video does not exist: {args.depth_video}", file=sys.stderr)
            return 1
        depth_video_cap = cv2.VideoCapture(str(args.depth_video))
        depth_vis_dir = args.output / "depth_visualization"
        depth_vis_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"Could not open video: {args.video}", file=sys.stderr)
        return 1

    manifest_path = args.output / "frames.jsonl"

    frame_index = 0
    extracted_index = 0

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        while True:
            ok, frame = cap.read()

            depth_vis_frame = None
            if depth_video_cap is not None:
                depth_ok, depth_vis_frame = depth_video_cap.read()
                if not depth_ok:
                    depth_vis_frame = None

            if not ok:
                break

            if frame_index % args.every != 0:
                frame_index += 1
                continue

            stem = f"frame_{frame_index:06d}"
            record = {"frame_index": frame_index}

            rgb_path = rgb_dir / f"{stem}.png"
            cv2.imwrite(str(rgb_path), frame)
            record["rgb"] = str(rgb_path.relative_to(args.output))

            if frame_index in depth_index:
                src = depth_index[frame_index]

                dst_npy = depth_npy_dir / f"{stem}.npy"
                shutil.copy2(src, dst_npy)
                record["depth_npy"] = str(dst_npy.relative_to(args.output))

                depth_m = np.load(src)
                dst_mm = depth_mm_dir / f"{stem}.png"
                cv2.imwrite(str(dst_mm), depth_to_uint16_mm(depth_m))
                record["depth_mm"] = str(dst_mm.relative_to(args.output))
            elif args.depth_dir:
                print(f"Warning: no raw depth found for frame {frame_index}")

            if depth_vis_frame is not None:
                dst_vis = depth_vis_dir / f"{stem}.png"
                cv2.imwrite(str(dst_vis), depth_vis_frame)
                record["depth_visualization"] = str(dst_vis.relative_to(args.output))

            manifest_file.write(json.dumps(record) + "\n")

            extracted_index += 1
            frame_index += 1

            if extracted_index % 50 == 0:
                print(f"\rExtracted {extracted_index} frames", end="", flush=True)

    cap.release()
    if depth_video_cap is not None:
        depth_video_cap.release()

    print(f"\nDone. Extracted {extracted_index} frames to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
