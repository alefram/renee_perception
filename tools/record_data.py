#!/usr/bin/env python3
"""Capture aligned RGB and depth data from a ZED 3i or RealSense camera.

Preview controls:
    ``c`` starts an image session. Inside it, ``c`` captures one shot and
    ``q`` returns to the main preview.
    ``r`` records RGB video and per-frame depth data.
    ``b`` captures a continuous RGB-D sweep at a selected frequency (2 fps by
    default).
    ``q`` exits the current mode.

Output layout:
    rgb/          RGB images or the RGB video.
    depth_raw_16/ Native RealSense Z16 depth PNGs (unaligned SDK units).
    depth_aligned_16/ Optional Z16 depth aligned to RGB.
    depth_mm/     Aligned 16-bit depth PNGs in millimetres.
    depth/        Aligned float32 depth arrays in metres.
    infrared_left/, infrared_right/  Optional stereo IR images.
    intrinsics/   Native calibration, extrinsics and depth units.
    frames.jsonl  Paths and timestamps for each saved RGB-D sample.
    imu.jsonl     Accelerometer and gyroscope samples.
    session.json  Reconstruction keyframes for every saved RGB-D sample.

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
DEFAULT_BURST_FPS = 2.0
MIN_RECOMMENDED_SWEEP_FPS = 2.0
DEFAULT_IMAGES_PER_SHOT = 30

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


@dataclass
class Frame:
    """One camera sample plus optional RealSense calibration-dataset data."""

    color_image: np.ndarray
    depth_m: np.ndarray
    depth_raw_16: np.ndarray | None = None
    depth_aligned_16: np.ndarray | None = None
    infrared_left: np.ndarray | None = None
    infrared_right: np.ndarray | None = None
    timestamps: dict[str, Any] | None = None
    imu_samples: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class CapturePaths:
    """Filesystem layout for one capture session."""

    base: Path
    rgb: Path
    depth: Path
    depth_mm: Path
    depth_raw_16: Path
    depth_aligned_16: Path
    infrared_left: Path
    infrared_right: Path
    intrinsics: Path
    session_file: Path
    frames_file: Path
    imu_file: Path

    @classmethod
    def create(cls, camera_type: str, output_name: str | None) -> CapturePaths:
        data_root = REPO_ROOT / "data"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if output_name:
            requested_path = Path(output_name)
            base = (
                requested_path
                if requested_path.is_absolute()
                else data_root / requested_path
            )
        else:
            base = data_root / f"{camera_type}_highres_{timestamp}"

        paths = cls(
            base=base,
            rgb=base / "rgb",
            depth=base / "depth",
            depth_mm=base / "depth_mm",
            depth_raw_16=base / "depth_raw_16",
            depth_aligned_16=base / "depth_aligned_16",
            infrared_left=base / "infrared_left",
            infrared_right=base / "infrared_right",
            intrinsics=base / "intrinsics",
            session_file=base / "session.json",
            frames_file=base / "frames.jsonl",
            imu_file=base / "imu.jsonl",
        )
        for directory in (
            paths.base,
            paths.rgb,
            paths.depth,
            paths.depth_mm,
            paths.depth_raw_16,
            paths.depth_aligned_16,
            paths.infrared_left,
            paths.infrared_right,
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
        serial_number: str | None = None,
        save_aligned_depth: bool = True,
        enable_infrared: bool = True,
        enable_imu: bool = True,
        lock_capture_settings: bool = True,
        exposure: float | None = None,
        gain: float | None = None,
    ) -> None:
        self.depth_mode_name = depth_mode
        self.camera_type = camera_type
        self.camera_label = "ZED" if camera_type == "zed" else "RealSense"
        self.save_aligned_depth = save_aligned_depth
        self.paths: CapturePaths | None = None
        self.session: ScanSession | None = None

        if camera_type == "zed":
            from renception.drivers.zed import ZedCamera

            self.camera: Any = ZedCamera()
        else:
            from renception.drivers.realsense import RealSenseCamera

            self.camera = RealSenseCamera(
                serial_number,
                enable_infrared=enable_infrared,
                enable_imu=enable_imu,
                lock_capture_settings=lock_capture_settings,
                exposure=exposure,
                gain=gain,
            )

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
        self.calibration = (
            self.camera.get_calibration() if camera_type == "realsense" else None
        )
        for warning in getattr(self.camera, "feature_warnings", []):
            print(f"Warning: {warning}")
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

        if self.calibration is not None:
            color_calibration = self.calibration["streams"]["color"]
            depth_calibration = self.calibration["streams"]["depth"]
            payload = {
                **self.calibration,
                # Legacy aliases retained for current reconstruction tools.
                "color_camera": color_calibration,
                "depth_camera": depth_calibration,
                "stream_info": {
                    "resolution": f"{self.camera.width}x{self.camera.height}",
                    "fps": self.camera.fps,
                    "camera_type": self.camera_type,
                    "native_depth_format": "z16",
                    "native_depth_units": "device_units",
                    "aligned_depth_saved": self.save_aligned_depth,
                    "reconstruction_depth": (
                        "depth_mm is aligned to color and uses color intrinsics"
                    ),
                },
            }
        else:
            payload = {
                "schema_version": 1,
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
                    "depth_mode": self.depth_mode_name,
                    "note": (
                        "Depth is aligned to the color image and shares its "
                        "intrinsics"
                    ),
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
                "frames_manifest": "frames.jsonl",
                "imu_manifest": (
                    "imu.jsonl"
                    if getattr(self.camera, "imu_enabled", False)
                    else None
                ),
                "pose_note": (
                    "T_world_cam is null for raw captures; estimate poses with "
                    "RGB-D odometry."
                ),
            },
        )
        self.session.save(self.paths.session_file)
        print(f"Session manifest created: {self.paths.session_file}")

    def _grab_frame(self) -> Frame | None:
        """Return the latest images, timestamps and optional sensor samples."""
        ok, _ = self.camera.grab()
        if not ok:
            return None

        if self.camera_type == "realsense":
            capture = self.camera.retrieve_capture()
            return Frame(
                color_image=capture.color_bgr,
                depth_m=capture.depth_aligned_m,
                depth_raw_16=capture.depth_raw,
                depth_aligned_16=capture.depth_aligned_raw,
                infrared_left=capture.infrared_left,
                infrared_right=capture.infrared_right,
                timestamps=capture.timestamps,
                imu_samples=capture.imu_samples,
            )

        color, depth_m = self.camera.retrieve_rgb_depth()
        return Frame(
            color_image=cv2.cvtColor(color, cv2.COLOR_BGRA2BGR),
            depth_m=depth_m,
            timestamps={
                "host_unix_ns": time.time_ns(),
                "host_monotonic_ns": time.monotonic_ns(),
            },
        )

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

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, separators=(",", ":")))
            file.write("\n")

    @staticmethod
    def _report_capture_rate(captured_frames: int, elapsed_seconds: float) -> None:
        """Report whether a continuous capture met the sweep's minimum rate."""
        achieved_fps = (
            captured_frames / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
        )
        print(f"  Effective saved RGB-D rate: {achieved_fps:.2f} fps")
        if achieved_fps < MIN_RECOMMENDED_SWEEP_FPS:
            print(
                "WARNING: saved RGB-D rate is below the 2 fps sweep minimum. "
                "Repeat the sweep more slowly or reduce the stream resolution."
            )

    def _save_image_pair(self, frame: Frame) -> bool:
        """Save one calibration sample and append both dataset manifests."""
        if self.paths is None or self.session is None:
            raise RuntimeError("A capture session has not been started")

        paths = self.paths
        session = self.session
        frame_index = len(session.keyframes)
        timestamps = frame.timestamps or {}
        host_unix_ns = timestamps.get("host_unix_ns") or time.time_ns()
        capture_timestamp = host_unix_ns / 1_000_000_000.0
        filename_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        stem = f"frame_{frame_index:06d}"

        rgb_path = paths.rgb / f"{stem}_{filename_timestamp}.png"
        depth_path = paths.depth_mm / f"{stem}.png"
        depth_m_path = paths.depth / f"{stem}.npy"
        depth_raw_16_path = paths.depth_raw_16 / f"{stem}.png"
        depth_aligned_16_path = paths.depth_aligned_16 / f"{stem}.png"
        ir_left_path = paths.infrared_left / f"{stem}.png"
        ir_right_path = paths.infrared_right / f"{stem}.png"
        depth_mm = self._depth_to_uint16_mm(frame.depth_m)

        if not cv2.imwrite(str(rgb_path), frame.color_image):
            raise OSError(f"Could not write image: {rgb_path}")
        if not cv2.imwrite(str(depth_path), depth_mm):
            raise OSError(f"Could not write image: {depth_path}")
        np.save(depth_m_path, frame.depth_m)

        native_depth_relative = None
        if frame.depth_raw_16 is not None:
            if frame.depth_raw_16.dtype != np.uint16:
                raise ValueError("Native RealSense depth must have dtype uint16")
            if not cv2.imwrite(str(depth_raw_16_path), frame.depth_raw_16):
                raise OSError(f"Could not write image: {depth_raw_16_path}")
            native_depth_relative = depth_raw_16_path.relative_to(
                paths.base
            ).as_posix()

        aligned_depth_relative = None
        if self.save_aligned_depth and frame.depth_aligned_16 is not None:
            if not cv2.imwrite(str(depth_aligned_16_path), frame.depth_aligned_16):
                raise OSError(f"Could not write image: {depth_aligned_16_path}")
            aligned_depth_relative = (
                depth_aligned_16_path.relative_to(paths.base).as_posix()
            )

        ir_left_relative = None
        if frame.infrared_left is not None:
            if not cv2.imwrite(str(ir_left_path), frame.infrared_left):
                raise OSError(f"Could not write image: {ir_left_path}")
            ir_left_relative = ir_left_path.relative_to(paths.base).as_posix()

        ir_right_relative = None
        if frame.infrared_right is not None:
            if not cv2.imwrite(str(ir_right_path), frame.infrared_right):
                raise OSError(f"Could not write image: {ir_right_path}")
            ir_right_relative = ir_right_path.relative_to(paths.base).as_posix()

        keyframe = Keyframe(
            rgb_path=rgb_path.relative_to(paths.base).as_posix(),
            depth_path=depth_path.relative_to(paths.base).as_posix(),
            intrinsics=self.color_camera_matrix.copy(),
            station_id=frame_index,
            timestamp=capture_timestamp,
            T_world_cam=None,
        )
        session.keyframes.append(keyframe)
        session.save(paths.session_file)

        imu_samples = frame.imu_samples or []
        for sample in imu_samples:
            self._append_jsonl(
                paths.imu_file,
                {"capture_index": frame_index, **sample},
            )
        frame_record = {
            "index": frame_index,
            "rgb": keyframe.rgb_path,
            "depth_m": depth_m_path.relative_to(paths.base).as_posix(),
            "depth_mm": keyframe.depth_path,
            "depth_raw_16": native_depth_relative,
            "depth_aligned_16": aligned_depth_relative,
            "infrared_left": ir_left_relative,
            "infrared_right": ir_right_relative,
            "timestamps": {**timestamps, "host_unix_ns": host_unix_ns},
            "imu_sample_count": len(imu_samples),
        }
        self._append_jsonl(paths.frames_file, frame_record)

        print(f"✓ Captured high-res image pair: {filename_timestamp}")
        print(
            f"  RGB: {rgb_path} "
            f"({frame.color_image.shape[1]}x{frame.color_image.shape[0]})"
        )
        print(f"  Depth: {depth_path} ({depth_mm.shape[1]}x{depth_mm.shape[0]})")
        if native_depth_relative:
            print(f"  Native Z16 depth: {depth_raw_16_path}")
        if ir_left_relative and ir_right_relative:
            print(f"  Stereo IR: {ir_left_path}, {ir_right_path}")
        print(f"  Session: {paths.session_file} (keyframe {keyframe.station_id})")
        return True

    def _capture_shot(
        self,
        first_frame: Frame,
        images_per_shot: int,
    ) -> int:
        """Save the requested number of image pairs for one shot."""
        saved_count = 0

        for image_index in range(images_per_shot):
            frame = first_frame if image_index == 0 else self._grab_frame()
            if frame is None:
                print(
                    f"Failed to capture image {image_index + 1}/"
                    f"{images_per_shot}"
                )
                continue

            if self._save_image_pair(frame):
                saved_count += 1

        print(f"Shot complete: {saved_count}/{images_per_shot} image pairs saved")
        return saved_count

    def capture_image_session(self) -> None:
        """Capture configurable shots into one output session."""
        cv2.destroyAllWindows()
        try:
            self._prepare_session()
            raw_count = input(
                f"Enter images per shot (default: {DEFAULT_IMAGES_PER_SHOT}): "
            ).strip()
            images_per_shot = (
                int(raw_count) if raw_count else DEFAULT_IMAGES_PER_SHOT
            )
            if images_per_shot <= 0:
                raise ValueError("images per shot must be greater than zero")
        except (OSError, RuntimeError, ValueError, cv2.error) as error:
            print(f"Invalid image session configuration: {error}")
            return

        if self.paths is None:
            raise RuntimeError("A capture session has not been started")

        total_saved = 0
        print("Image capture session started")
        print(f"  Output: {self.paths.base}")
        print(f"  Images per shot: {images_per_shot}")
        print("  Press c to capture a shot, or q to finish")

        try:
            while True:
                frame = self._grab_frame()
                if frame is None:
                    continue

                status = (
                    f"IMAGE SESSION: c = shot ({images_per_shot}) | "
                    f"Saved: {total_saved} | q = finish"
                )
                key = self._show_frame(
                    "Images",
                    frame.color_image,
                    frame.depth_m,
                    status_text=status,
                )
                if key == ord("q"):
                    break
                if key == ord("c"):
                    total_saved += self._capture_shot(frame, images_per_shot)
        except (OSError, RuntimeError, ValueError, cv2.error) as error:
            print(f"Error capturing images: {error}")
        except KeyboardInterrupt:
            print("\nImage capture session interrupted")
        finally:
            cv2.destroyAllWindows()

        print(f"Image capture session complete: {total_saved} image pairs saved")
        print("Resuming live preview...")

    def capture_burst(self) -> None:
        """Configure and run burst capture from the live preview."""
        cv2.destroyAllWindows()
        self._prepare_session()
        try:
            raw_fps = input(
                "Enter burst frequency in images per second "
                f"(default: {DEFAULT_BURST_FPS:g}): "
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

                status = f"BURST: {burst_fps:g} img/s | Saved: {capture_count}"
                if self._show_frame(
                    "Burst",
                    frame.color_image,
                    frame.depth_m,
                    status_text=status,
                ) == ord("q"):
                    print("Burst capture stopped with q")
                    break

                now = time.monotonic()
                if now < next_capture_at:
                    continue

                self._save_image_pair(frame)
                capture_count += 1
                next_capture_at += interval_seconds
                if next_capture_at < time.monotonic():
                    next_capture_at = time.monotonic() + interval_seconds
        except KeyboardInterrupt:
            print("\nBurst capture interrupted by user")
        finally:
            cv2.destroyAllWindows()

        elapsed_seconds = time.monotonic() - started_at
        print(f"Burst capture complete: {capture_count} RGB-D image pairs saved")
        self._report_capture_rate(capture_count, elapsed_seconds)
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

                self._save_image_pair(frame)
                rgb_writer.write(frame.color_image)
                frame_count += 1

                if self._show_frame(
                    "",
                    frame.color_image,
                    frame.depth_m,
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
        self._report_capture_rate(frame_count, elapsed_seconds)
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

                key = self._show_frame(
                    "Live", frame.color_image, frame.depth_m
                )
                if key == ord("q"):
                    break
                if key == ord("c"):
                    self.capture_image_session()
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
    parser.add_argument(
        "--serial-number",
        help="RealSense serial number to use when several cameras are connected",
    )
    parser.add_argument(
        "--no-aligned-depth",
        action="store_true",
        help=(
            "Do not save the optional depth_aligned_16 PNGs. The aligned "
            "depth_mm reconstruction images are still saved"
        ),
    )
    parser.add_argument(
        "--no-infrared",
        action="store_true",
        help="Do not enable or save RealSense left/right infrared streams",
    )
    parser.add_argument(
        "--no-imu",
        action="store_true",
        help="Do not enable or save RealSense accelerometer/gyroscope samples",
    )
    parser.add_argument(
        "--no-lock-capture-settings",
        action="store_true",
        help=(
            "Do not force the RealSense High Accuracy preset, IR emitter on, "
            "and fixed exposure/gain. Ignored for ZED."
        ),
    )
    parser.add_argument(
        "--exposure",
        type=float,
        default=None,
        help="Fixed depth/IR sensor exposure to lock (skip auto-settle); RealSense only",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=None,
        help="Fixed depth/IR sensor gain to lock (skip auto-settle); RealSense only",
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
            serial_number=args.serial_number,
            save_aligned_depth=not args.no_aligned_depth,
            enable_infrared=not args.no_infrared,
            enable_imu=not args.no_imu,
            lock_capture_settings=not args.no_lock_capture_settings,
            exposure=args.exposure,
            gain=args.gain,
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
