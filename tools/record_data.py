#!/usr/bin/env python3
"""Capture aligned RGB and depth data from a ZED 2i or RealSense camera.

Preview controls:
    ``c`` captures one RGB-D image pair.
    ``r`` records RGB video and per-frame depth data.
    ``b`` captures RGB-D image pairs at a selected frequency.
    ``q`` exits the current mode.

Output layout:
    rgb/          RGB images or the RGB video.
    depth_mm/     16-bit depth PNGs in millimetres.
    depth/        Raw float32 depth arrays in metres.
    intrinsics/   Camera calibration data.
    session.json  Keyframes captured with ``c`` or burst mode.

Examples:
    python3 tools/record_data.py
    python3 tools/record_data.py --camera realsense
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from renception.contracts import Keyframe, ScanSession


DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30
DEFAULT_DEPTH_MODE = "NEURAL"
DEFAULT_BURST_FPS = 1.0

DEPTH_MODES = ("NEURAL", "ULTRA", "QUALITY", "PERFORMANCE")
VIDEO_CODEC = "mp4v"
PREVIEW_COLOUR = (0, 255, 0)
TEXT_ORIGIN = (20, 40)
TEXT_LINE_HEIGHT = 40
FONT = cv2.FONT_HERSHEY_SIMPLEX

ZED_FALLBACK_CONFIGS = (
    (1280, 720, 30),
    (1280, 720, 15),
    (672, 376, 30),
)
REALSENSE_FALLBACK_CONFIGS = (
    (1280, 720, 30),
    (1280, 720, 15),
    (640, 480, 30),
)

Frame = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class CapturePaths:
    """Filesystem layout for one capture session."""

    base: Path
    rgb: Path
    depth: Path
    depth_mm: Path
    intrinsics: Path
    session_file: Path

    @classmethod
    def create(cls, camera_type: str, output_name: str | None) -> CapturePaths:
        data_root = REPO_ROOT / "data"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if output_name:
            requested_path = Path(output_name)
            base = requested_path if requested_path.is_absolute() else data_root / requested_path
        else:
            base = data_root / f"{camera_type}_highres_{timestamp}"

        paths = cls(
            base=base,
            rgb=base / "rgb",
            depth=base / "depth",
            depth_mm=base / "depth_mm",
            intrinsics=base / "intrinsics",
            session_file=base / "session.json",
        )
        for directory in (
            paths.base,
            paths.rgb,
            paths.depth,
            paths.depth_mm,
            paths.intrinsics,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return paths


def positive_int(value: str) -> int:
    """Parse a positive integer for argparse."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


class RGBDHighResCapture:
    """Interactive aligned RGB-D capture application."""

    def __init__(
        self,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        fps: int = DEFAULT_FPS,
        depth_mode: str = DEFAULT_DEPTH_MODE,
        camera_type: str = "zed",
    ) -> None:
        self.depth_mode_name = depth_mode
        self.camera_type = camera_type
        self.camera_label = "ZED" if camera_type == "zed" else "RealSense"
        self.paths: CapturePaths | None = None
        self.session: ScanSession | None = None

        if camera_type == "zed":
            from renception.drivers.zed import ZedCamera

            self.camera: Any = ZedCamera()
        else:
            from renception.drivers.realsense import RealSenseCamera

            self.camera = RealSenseCamera()

        print("Configuring stream:")
        print(f"  Requested: {width}x{height} at {fps} FPS")
        if camera_type == "zed":
            from renception.drivers.zed import closest_resolution_name

            print(f"  Using ZED preset: {closest_resolution_name(width, height)}")

        if not self._open_camera(width, height, fps):
            print("Trying fallback resolutions...")
            self._try_fallback_resolutions()

        print(
            "✓ Camera opened successfully at "
            f"{self.camera.width}x{self.camera.height} @ {self.camera.fps}fps"
        )
        matrix, distortion = self.camera.get_intrinsics()
        self.color_camera_matrix = matrix
        self.color_dist_coeffs = distortion
        self.depth_camera_matrix = matrix
        self.depth_dist_coeffs = distortion
        print("Camera intrinsics loaded!")

    def _open_camera(self, width: int, height: int, fps: int) -> bool:
        """Open the camera with the requested stream configuration."""
        if self.camera_type == "zed":
            from renception.drivers.zed import closest_resolution_name

            resolution = closest_resolution_name(width, height)
            ok, status = self.camera.open(
                resolution,
                fps,
                self.depth_mode_name,
            )
        else:
            ok, status = self.camera.open(width, height, fps)

        if ok:
            print(f"✓ {self.camera_label} camera initialized successfully!")
            return True

        print(f"✗ Failed to start camera: {status}")
        return False

    def _try_fallback_resolutions(self) -> None:
        """Open the first supported fallback resolution."""
        configurations = (
            REALSENSE_FALLBACK_CONFIGS
            if self.camera_type == "realsense"
            else ZED_FALLBACK_CONFIGS
        )
        for width, height, fps in configurations:
            print(f"Trying: {width}x{height} at {fps}fps")
            if self._open_camera(width, height, fps):
                print("✓ Successfully initialized with fallback resolution")
                return

        raise RuntimeError("Could not initialize camera with any supported resolution")

    def _prepare_session(self) -> None:
        """Create the output directories, intrinsics and session manifest."""
        output_name = input(
            "Enter output session name (or press Enter for an automatic name): "
        ).strip()
        self.paths = CapturePaths.create(self.camera_type, output_name or None)
        print(f"Created directories in: {self.paths.base}")

        payload = {
            "color_camera": {
                "camera_matrix": self.color_camera_matrix.tolist(),
                "distortion_coefficients": self.color_dist_coeffs.tolist(),
                "width": self.camera.width,
                "height": self.camera.height,
            },
            "depth_camera": {
                "camera_matrix": self.depth_camera_matrix.tolist(),
                "distortion_coefficients": self.depth_dist_coeffs.tolist(),
                "width": self.camera.width,
                "height": self.camera.height,
            },
            "stream_info": {
                "resolution": f"{self.camera.width}x{self.camera.height}",
                "fps": self.camera.fps,
                "camera_type": self.camera_type,
                "depth_mode": (
                    self.depth_mode_name if self.camera_type == "zed" else None
                ),
                "note": "Depth is aligned to the color image and shares its intrinsics",
            },
        }
        intrinsics_file = self.paths.intrinsics / "camera_intrinsics.json"
        with intrinsics_file.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=4)
        print(f"Intrinsics saved to: {intrinsics_file}")

        if self.paths.session_file.exists():
            self.session = ScanSession.load(self.paths.session_file)
            print(f"Session manifest loaded: {self.paths.session_file}")
            return

        try:
            session_dir = self.paths.base.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            session_dir = str(self.paths.base)

        self.session = ScanSession(
            session_dir=session_dir,
            keyframes=[],
            meta={
                "camera_type": self.camera_type,
                "resolution": f"{self.camera.width}x{self.camera.height}",
                "fps": self.camera.fps,
                "pose_note": (
                    "T_world_cam is null for raw captures; estimate poses with "
                    "RGB-D odometry."
                ),
            },
        )
        self.session.save(self.paths.session_file)
        print(f"Session manifest created: {self.paths.session_file}")

    def _grab_frame(self) -> Frame | None:
        """Return the latest aligned BGR and metric-depth frame."""
        ok, _ = self.camera.grab()
        if not ok:
            return None

        color, depth_m = self.camera.retrieve_rgb_depth()
        if self.camera_type == "zed":
            color = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
        return color, depth_m

    @staticmethod
    def _depth_to_uint16_mm(depth_m: np.ndarray) -> np.ndarray:
        """Convert metric float depth to uint16 millimetres; zero is invalid."""
        valid = np.isfinite(depth_m) & (depth_m > 0)
        depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
        depth_mm[valid] = np.clip(
            np.round(depth_m[valid] * 1000.0),
            1,
            np.iinfo(np.uint16).max,
        ).astype(np.uint16)
        return depth_mm

    def _show_frame(
        self,
        mode: str,
        color_image: np.ndarray,
        depth_m: np.ndarray,
        *,
        status_text: str | None = None,
        show_distance: bool = True,
    ) -> int:
        """Display an RGB-D frame and return the pressed key."""
        rgb_image = color_image.copy()
        depth_mm = self._depth_to_uint16_mm(depth_m)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_mm, alpha=0.03),
            cv2.COLORMAP_JET,
        )

        if show_distance:
            height, width = depth_m.shape
            center = (width // 2, height // 2)
            distance_m = depth_m[center[1], center[0]]
            depth_text = (
                f"Centre: {distance_m:.2f} m"
                if np.isfinite(distance_m) and distance_m > 0
                else "Centre: invalid depth"
            )
            text_y = TEXT_ORIGIN[1]
            if status_text:
                cv2.putText(
                    rgb_image,
                    status_text,
                    (TEXT_ORIGIN[0], text_y),
                    FONT,
                    1,
                    PREVIEW_COLOUR,
                    2,
                    cv2.LINE_AA,
                )
                text_y += TEXT_LINE_HEIGHT

            cv2.circle(rgb_image, center, 6, PREVIEW_COLOUR, -1)
            cv2.putText(
                rgb_image,
                depth_text,
                (TEXT_ORIGIN[0], text_y),
                FONT,
                1,
                PREVIEW_COLOUR,
                2,
                cv2.LINE_AA,
            )

        resolution = f"{self.camera.width}x{self.camera.height}"
        suffix = f" {mode}" if mode else ""

        cv2.imshow(f"RGB{suffix} ({resolution})", rgb_image)
        cv2.imshow(f"Depth{suffix} ({resolution})", depth_colormap)
        return cv2.waitKey(1) & 0xFF

    def _save_image_pair(
        self,
        color_image: np.ndarray,
        depth_m: np.ndarray,
    ) -> bool:
        """Save an aligned RGB-D pair and append it to session.json."""
        if self.paths is None or self.session is None:
            raise RuntimeError("A capture session has not been started")

        paths = self.paths
        session = self.session
        capture_timestamp = time.time()
        filename_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        rgb_path = paths.rgb / f"rgb_{filename_timestamp}.png"
        depth_path = paths.depth_mm / f"depth_{filename_timestamp}.png"
        raw_depth_path = paths.depth / f"depth_raw_{filename_timestamp}.npy"
        depth_mm = self._depth_to_uint16_mm(depth_m)

        if not cv2.imwrite(str(rgb_path), color_image):
            raise OSError(f"Could not write image: {rgb_path}")
        if not cv2.imwrite(str(depth_path), depth_mm):
            raise OSError(f"Could not write image: {depth_path}")
        np.save(raw_depth_path, depth_m)

        keyframe = Keyframe(
            rgb_path=rgb_path.relative_to(paths.base).as_posix(),
            depth_path=depth_path.relative_to(paths.base).as_posix(),
            intrinsics=self.color_camera_matrix.copy(),
            station_id=len(session.keyframes),
            timestamp=capture_timestamp,
            T_world_cam=None,
        )
        session.keyframes.append(keyframe)
        session.save(paths.session_file)

        print(f"✓ Captured high-res image pair: {filename_timestamp}")
        print(f"  RGB: {rgb_path} ({color_image.shape[1]}x{color_image.shape[0]})")
        print(f"  Depth: {depth_path} ({depth_mm.shape[1]}x{depth_mm.shape[0]})")
        print(f"  Session: {paths.session_file} (keyframe {keyframe.station_id})")
        return True

    def capture_single_image(self) -> bool:
        """Prepare a session, then capture and save one RGB-D image pair."""
        try:
            self._prepare_session()
            frame = self._grab_frame()
            if frame is None:
                print("Failed to capture frames")
                return False
            return self._save_image_pair(*frame)
        except (OSError, RuntimeError, ValueError, cv2.error) as error:
            print(f"Error capturing image: {error}")
            return False

    def capture_burst(self) -> None:
        """Configure and run burst capture from the live preview."""
        cv2.destroyAllWindows()
        self._prepare_session()
        try:
            raw_fps = input(
                "Enter burst frequency in images per second (default: 1): "
            ).strip()
            raw_duration = input(
                "Enter burst duration in seconds "
                "(or press Enter for manual stop): "
            ).strip()
            burst_fps = float(raw_fps) if raw_fps else DEFAULT_BURST_FPS
            duration_seconds = float(raw_duration) if raw_duration else None
            if burst_fps <= 0 or (
                duration_seconds is not None and duration_seconds <= 0
            ):
                raise ValueError("frequency and duration must be greater than zero")
        except ValueError as error:
            print(f"Invalid burst configuration: {error}")
            return

        interval_seconds = 1.0 / burst_fps
        started_at = time.monotonic()
        next_capture_at = started_at
        capture_count = 0

        print("Starting burst capture...")
        print(f"  Capture rate: {burst_fps:g} image pairs/second")
        if duration_seconds is None:
            print("  Press q or Ctrl+C to stop")
        else:
            print(f"  Duration: {duration_seconds:g} seconds")

        try:
            while (
                duration_seconds is None
                or time.monotonic() - started_at < duration_seconds
            ):
                frame = self._grab_frame()
                if frame is None:
                    continue

                color_image, depth_m = frame
                status = f"BURST: {burst_fps:g} img/s | Saved: {capture_count}"
                if self._show_frame(
                    "Burst",
                    color_image,
                    depth_m,
                    status_text=status,
                ) == ord("q"):
                    print("Burst capture stopped with q")
                    break

                now = time.monotonic()
                if now < next_capture_at:
                    continue

                self._save_image_pair(color_image, depth_m)
                capture_count += 1
                next_capture_at += interval_seconds
                if next_capture_at < time.monotonic():
                    next_capture_at = time.monotonic() + interval_seconds
        except KeyboardInterrupt:
            print("\nBurst capture interrupted by user")
        finally:
            cv2.destroyAllWindows()

        print(f"Burst capture complete: {capture_count} RGB-D image pairs saved")
        print("Resuming live preview...")

    def record_video(self) -> None:
        """Configure and record RGB video plus per-frame depth data."""
        cv2.destroyAllWindows()
        self._prepare_session()
        try:
            raw_duration = input(
                "Enter recording duration in seconds "
                "(or press Enter for manual stop): "
            ).strip()
            duration_seconds = float(raw_duration) if raw_duration else None
            if duration_seconds is not None and duration_seconds <= 0:
                raise ValueError("duration must be greater than zero")
        except ValueError as error:
            print(f"Invalid recording duration: {error}")
            return

        if self.paths is None:
            raise RuntimeError("A capture session has not been started")
        paths = self.paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rgb_video_path = paths.rgb / f"rgb_video_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        rgb_writer = cv2.VideoWriter(
            str(rgb_video_path),
            fourcc,
            self.camera.fps,
            (self.camera.width, self.camera.height),
        )
        if not rgb_writer.isOpened():
            rgb_writer.release()
            raise RuntimeError(f"Could not open video writer: {rgb_video_path}")

        print("Recording high-resolution video...")
        print(f"  Resolution: {self.camera.width}x{self.camera.height}")
        print("  Raw depth data will be saved for each frame")
        if duration_seconds is None:
            print("  Press q to stop")
        else:
            print(f"  Recording for {duration_seconds:g} seconds")

        frame_count = 0
        started_at = time.monotonic()
        try:
            while (
                duration_seconds is None
                or time.monotonic() - started_at < duration_seconds
            ):
                frame = self._grab_frame()
                if frame is None:
                    continue

                color_image, depth_m = frame
                frame_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                stem = f"frame_{frame_count:06d}"
                raw_depth_path = (
                    paths.depth
                    / f"depth_raw_{stem}_{frame_timestamp}.npy"
                )
                depth_mm_path = paths.depth_mm / f"{stem}.png"

                np.save(raw_depth_path, depth_m)
                depth_mm = self._depth_to_uint16_mm(depth_m)
                if not cv2.imwrite(str(depth_mm_path), depth_mm):
                    raise OSError(f"Could not write image: {depth_mm_path}")
                rgb_writer.write(color_image)
                frame_count += 1

                if self._show_frame(
                    "",
                    color_image,
                    depth_m,
                    show_distance=False,
                ) == ord("q"):
                    break

                if frame_count % 30 == 0:
                    print(f"Recorded {frame_count} frames (with raw depth data)...")
        except KeyboardInterrupt:
            print("\nRecording interrupted by user")
        finally:
            rgb_writer.release()
            cv2.destroyAllWindows()

        elapsed_seconds = time.monotonic() - started_at
        print("✓ High-resolution video saved:")
        print(f"  RGB: {rgb_video_path}")
        print(f"  Raw depth frames: {frame_count} .npy files in {paths.depth}")
        print(f"  Per-frame depth PNGs: {paths.depth_mm}")
        print(f"  Frames recorded: {frame_count}")
        print(f"  Duration: {elapsed_seconds:.2f} seconds")
        print("Resuming live preview...")

    def live_preview(self) -> None:
        """Run the interactive RGB-D preview."""
        print("High-resolution live preview")
        print(
            "Controls: 'c' = capture image, 'r' = record video, "
            "'b' = burst capture, 'q' = quit"
        )
        print(
            f"Streaming at {self.camera.width}x{self.camera.height} "
            "for both RGB and depth"
        )

        try:
            while True:
                frame = self._grab_frame()
                if frame is None:
                    continue

                color_image, depth_m = frame
                key = self._show_frame("Live", color_image, depth_m)
                if key == ord("q"):
                    break
                if key == ord("c"):
                    self.capture_single_image()
                elif key == ord("r"):
                    self.record_video()
                elif key == ord("b"):
                    self.capture_burst()
        except KeyboardInterrupt:
            print("\nLive preview interrupted")
        finally:
            cv2.destroyAllWindows()

    def cleanup(self) -> None:
        """Close the camera and all OpenCV windows."""
        self.camera.close()
        cv2.destroyAllWindows()
        print("Camera resources cleaned up")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera",
        choices=("zed", "realsense"),
        default="zed",
        help="Camera backend (default: zed)",
    )
    parser.add_argument(
        "--width",
        type=positive_int,
        default=DEFAULT_WIDTH,
        help=f"Requested RGB-D width (default: {DEFAULT_WIDTH})",
    )
    parser.add_argument(
        "--height",
        type=positive_int,
        default=DEFAULT_HEIGHT,
        help=f"Requested RGB-D height (default: {DEFAULT_HEIGHT})",
    )
    parser.add_argument(
        "--fps",
        type=positive_int,
        default=DEFAULT_FPS,
        help=f"Camera stream frequency (default: {DEFAULT_FPS})",
    )
    parser.add_argument(
        "--depth-mode",
        choices=DEPTH_MODES,
        default=DEFAULT_DEPTH_MODE,
        help="ZED depth computation mode; ignored for RealSense",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    capture: RGBDHighResCapture | None = None

    try:
        capture = RGBDHighResCapture(
            width=args.width,
            height=args.height,
            fps=args.fps,
            depth_mode=args.depth_mode,
            camera_type=args.camera,
        )
        capture.live_preview()
    except (RuntimeError, OSError, ValueError, cv2.error) as error:
        print(f"Error: {error}")
        print(
            f"Make sure the selected {args.camera} camera is connected "
            "and its Python SDK is installed"
        )
        return 1
    finally:
        if capture is not None:
            capture.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
