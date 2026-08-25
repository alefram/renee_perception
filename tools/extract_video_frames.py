#!/usr/bin/env python3
"""Extract image frames from a recorded video and optional depth sources.

The RGB video may be paired with raw per-frame depth arrays and a depth
colormap video, as produced by tools/record_data.py in video mode.

Frame indices line up 1:1 across rgb_video_*.mp4, depth_video_*.mp4, and
depth_raw_frame_<index>_<timestamp>.npy, since record_video() in
tools/record_data.py writes all three for every successfully grabbed frame.

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
from typing import Sequence

import cv2
import numpy as np


DEPTH_RAW_PATTERN = re.compile(r"depth_raw_frame_(\d+)_")
DEFAULT_OUTPUT_DIR = Path("extracted_frames")
DEFAULT_FRAME_INTERVAL = 1
PROGRESS_INTERVAL = 50
DEPTH_UNITS = ("auto", "m", "mm")


def index_depth_raw_files(depth_dir: Path) -> dict[int, Path]:
    """Map frame indexes to raw depth file paths."""
    index = {}
    for path in depth_dir.glob("depth_raw_frame_*.npy"):
        match = DEPTH_RAW_PATTERN.search(path.name)
        if match:
            index[int(match.group(1))] = path
    return index


def depth_to_uint16_mm(
    depth: np.ndarray,
    units: str = "auto",
) -> tuple[np.ndarray, str]:
    """Return (depth_mm, detected_units) for legacy-mm or modern-metre arrays."""
    detected = units
    if units == "auto":
        if np.issubdtype(depth.dtype, np.integer):
            detected = "mm"
        elif np.issubdtype(depth.dtype, np.floating):
            detected = "m"
        else:
            raise ValueError(f"cannot infer depth units from dtype {depth.dtype}")
    if detected not in {"m", "mm"}:
        raise ValueError(f"unsupported depth units: {detected}")

    depth_values = depth.astype(np.float64)
    if detected == "m":
        depth_values *= 1000.0
    valid = np.isfinite(depth_values) & (depth_values > 0)
    depth_mm = np.zeros(depth.shape, dtype=np.uint16)
    depth_mm[valid] = np.clip(
        np.round(depth_values[valid]),
        1,
        np.iinfo(np.uint16).max,
    ).astype(np.uint16)
    return depth_mm, detected


def positive_int(value: str) -> int:
    """Parse a positive integer for argparse."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Video to extract RGB frames from",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--every",
        type=positive_int,
        default=DEFAULT_FRAME_INTERVAL,
        help="Extract every Nth frame (default: 1, every frame)",
    )
    parser.add_argument(
        "--depth-dir",
        type=Path,
        help="Directory of depth_raw_frame_<index>_<timestamp>.npy files",
    )
    parser.add_argument(
        "--depth-units",
        choices=DEPTH_UNITS,
        default="auto",
        help="Raw .npy units (default: infer integers=mm, floats=m)",
    )
    parser.add_argument(
        "--depth-video",
        type=Path,
        help="Depth colormap video recorded alongside --video",
    )
    parser.add_argument(
        "--intrinsics",
        type=Path,
        help="camera_intrinsics.json to copy into output/intrinsics/",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)

    if args.output.exists() and any(args.output.iterdir()):
        print(f"Output directory is not empty: {args.output}", file=sys.stderr)
        return 1
    if not args.video.exists():
        print(f"Video does not exist: {args.video}", file=sys.stderr)
        return 1

    rgb_dir = args.output / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    if args.intrinsics is not None:
        if not args.intrinsics.is_file():
            print(f"Intrinsics file does not exist: {args.intrinsics}", file=sys.stderr)
            return 1
        intrinsics_dir = args.output / "intrinsics"
        intrinsics_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.intrinsics, intrinsics_dir / "camera_intrinsics.json")

    depth_index: dict[int, Path] = {}
    if args.depth_dir is not None:
        if not args.depth_dir.exists():
            print(f"Depth directory does not exist: {args.depth_dir}", file=sys.stderr)
            return 1
        depth_index = index_depth_raw_files(args.depth_dir)
        print(f"Found {len(depth_index)} raw depth frames in {args.depth_dir}")
        depth_npy_dir = args.output / "depth_npy"
        depth_mm_dir = args.output / "depth_mm"
        depth_npy_dir.mkdir(parents=True, exist_ok=True)
        depth_mm_dir.mkdir(parents=True, exist_ok=True)

    depth_video_capture = None
    if args.depth_video is not None:
        if not args.depth_video.exists():
            print(f"Depth video does not exist: {args.depth_video}", file=sys.stderr)
            return 1
        depth_video_capture = cv2.VideoCapture(str(args.depth_video))
        depth_vis_dir = args.output / "depth_visualization"
        depth_vis_dir.mkdir(parents=True, exist_ok=True)

    video_capture = cv2.VideoCapture(str(args.video))
    if not video_capture.isOpened():
        print(f"Could not open video: {args.video}", file=sys.stderr)
        return 1
    source_fps = float(video_capture.get(cv2.CAP_PROP_FPS)) or 30.0

    manifest_path = args.output / "frames.jsonl"

    frame_index = 0
    extracted_index = 0

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        while True:
            ok, frame = video_capture.read()

            depth_vis_frame = None
            if depth_video_capture is not None:
                depth_ok, depth_vis_frame = depth_video_capture.read()
                if not depth_ok:
                    depth_vis_frame = None

            if not ok:
                break

            if frame_index % args.every != 0:
                frame_index += 1
                continue

            stem = f"frame_{frame_index:06d}"
            record = {
                "frame_index": frame_index,
                "timestamp": frame_index / source_fps,
            }

            rgb_path = rgb_dir / f"{stem}.png"
            cv2.imwrite(str(rgb_path), frame)
            record["rgb"] = str(rgb_path.relative_to(args.output))

            if frame_index in depth_index:
                source_depth_path = depth_index[frame_index]

                depth_npy_path = depth_npy_dir / f"{stem}.npy"
                shutil.copy2(source_depth_path, depth_npy_path)
                record["depth_npy"] = str(depth_npy_path.relative_to(args.output))

                depth_raw = np.load(source_depth_path)
                depth_mm_path = depth_mm_dir / f"{stem}.png"
                depth_mm, detected_units = depth_to_uint16_mm(
                    depth_raw,
                    args.depth_units,
                )
                if not cv2.imwrite(str(depth_mm_path), depth_mm):
                    raise OSError(f"Could not write depth image: {depth_mm_path}")
                record["depth_mm"] = str(depth_mm_path.relative_to(args.output))
                record["depth_source_units"] = detected_units
            elif args.depth_dir is not None:
                print(f"Warning: no raw depth found for frame {frame_index}")

            if depth_vis_frame is not None:
                depth_visualization_path = depth_vis_dir / f"{stem}.png"
                cv2.imwrite(str(depth_visualization_path), depth_vis_frame)
                record["depth_visualization"] = str(
                    depth_visualization_path.relative_to(args.output)
                )

            manifest_file.write(json.dumps(record) + "\n")

            extracted_index += 1
            frame_index += 1

            if extracted_index % PROGRESS_INTERVAL == 0:
                print(f"\rExtracted {extracted_index} frames", end="", flush=True)

    video_capture.release()
    if depth_video_capture is not None:
        depth_video_capture.release()

    metadata = {
        "source_video": str(args.video.resolve()),
        "source_fps": source_fps,
        "every": args.every,
        "extracted_frames": extracted_index,
        "depth_units": args.depth_units,
    }
    (args.output / "extraction_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"\nDone. Extracted {extracted_index} frames to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
