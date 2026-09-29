#!/usr/bin/env python3
"""Capture aligned RGB and depth data from a ZED 3i or RealSense camera.

Preview controls:
    ``c`` starts an image session. Inside it, ``c`` captures one shot and
    ``q`` returns to the main preview.
    ``r`` records RGB video and per-frame depth data.
    ``b`` captures a continuous RGB-D sweep at a selected frequency (2 fps by
    default).
    ``q`` exits the current mode.

Headless (no ``DISPLAY``, no TTY, e.g. over SSH): ``--headless`` captures
``--images-per-shot`` image pairs and exits; ``--headless --video`` records
the RGB video plus per-frame depth for ``--duration`` seconds, or until
Ctrl+C / SIGTERM without it.

Video mode saves every frame (RGB, depth) and, when the recording ends,
builds ``rgb/rgb_video_<time>.mp4`` and ``depth/depth_video_<time>.mp4``
(depth colormap) from the saved frames at the rate they were actually
captured (from frames.jsonl), so the videos play back in real time even when
saving limits the capture to a few frames per second (e.g. on the Jetson).
``--make-video SESSION`` rebuilds them for an existing session.

Output layout:
    rgb/          RGB images, and in video mode rgb_video_<time>.mp4.
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
    python3 tools/record_data.py --headless --video --duration 60 --output sweep_1
    python3 tools/record_data.py --make-video data/zed_highres_20260928_112333
"""

import argparse
import json
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from renception.contracts import Keyframe, ScanSession


DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30
DEFAULT_DEPTH_MODE = "QUALITY"
DEFAULT_BURST_FPS = 2.0
MIN_RECOMMENDED_SWEEP_FPS = 2.0
DEFAULT_IMAGES_PER_SHOT = 30

# Torn-frame check (ZED only). When the USB link drops data (e.g. an
# under-powered Jetson/hub), the ZED image arrives as horizontal bands that
# are shifted/out of place. Each band edge is a row boundary where nearly the
# whole row jumps in brightness. Calibrated on 2026-09-28: 0 false positives
# on 585 clean ZED frames (incl. full-width rails), ~50% of torn frames caught.
TEAR_CHECK_WIDTH = 672
TEAR_ROW_GAP = 2
TEAR_PIXEL_JUMP = 30
TEAR_ROW_FRACTION = 0.7
TORN_MIN_SEAMS = 3

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
    depth_raw_16: Optional[np.ndarray] = None
    depth_aligned_16: Optional[np.ndarray] = None
    infrared_left: Optional[np.ndarray] = None
    infrared_right: Optional[np.ndarray] = None
    timestamps: Optional[Dict[str, Any]] = None
    imu_samples: Optional[List[Dict[str, Any]]] = None


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
    def create(cls, camera_type: str, output_name: Optional[str]) -> "CapturePaths":
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


def count_tear_seams(color_bgr: np.ndarray) -> int:
    """Count full-width seams (row boundaries where most of the row jumps).

    Resized to TEAR_CHECK_WIDTH first, so the thresholds hold for any stream
    resolution. Frames with TORN_MIN_SEAMS or more seams are likely torn.
    """
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    height = int(round(gray.shape[0] * TEAR_CHECK_WIDTH / gray.shape[1]))
    gray = cv2.resize(
        gray, (TEAR_CHECK_WIDTH, height), interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    jump = np.abs(gray[TEAR_ROW_GAP:] - gray[:-TEAR_ROW_GAP]) > TEAR_PIXEL_JUMP
    seam_rows = (jump.mean(axis=1) > TEAR_ROW_FRACTION).astype(np.int8)
    return int(np.sum(np.diff(seam_rows, prepend=0) == 1))


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
        serial_number: Optional[str] = None,
        save_aligned_depth: bool = True,
        enable_infrared: bool = True,
        enable_imu: bool = True,
        lock_capture_settings: bool = True,
        exposure: Optional[float] = None,
        gain: Optional[float] = None,
    ) -> None:
        self.depth_mode_name = depth_mode
        self.camera_type = camera_type
        self.camera_label = "ZED" if camera_type == "zed" else "RealSense"
        self.save_aligned_depth = save_aligned_depth
        self.paths: Optional[CapturePaths] = None
        self.session: Optional[ScanSession] = None
        self.torn_frames = 0

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

    def _prepare_session(
        self, output_name: Optional[str] = None, interactive: bool = True
    ) -> None:
        """Create the output directories, intrinsics and session manifest.

        ``output_name`` is used as-is when given; otherwise it is read
        interactively from stdin, or (``interactive=False``, headless mode)
        an automatic timestamped name is used.
        """
        if output_name is None and interactive:
            output_name = (
                input(
                    "Enter output session name (or press Enter for an automatic name): "
                ).strip()
                or None
            )
        self.paths = CapturePaths.create(self.camera_type, output_name)
        self.torn_frames = 0
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

    def _grab_frame(self) -> Optional[Frame]:
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
                "host_unix_ns": int(time.time() * 1e9),
                "host_monotonic_ns": int(time.monotonic() * 1e9),
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

    @staticmethod
    def _centre_depth_text(depth_m: np.ndarray) -> str:
        """Describe the depth at the image centre pixel."""
        height, width = depth_m.shape
        distance_m = depth_m[height // 2, width // 2]
        return (
            f"Centre: {distance_m:.2f} m"
            if np.isfinite(distance_m) and distance_m > 0
            else "Centre: invalid depth"
        )

    def _show_frame(
        self,
        mode: str,
        color_image: np.ndarray,
        depth_m: np.ndarray,
        *,
        status_text: Optional[str] = None,
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
            depth_text = self._centre_depth_text(depth_m)
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
    def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
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

    def _report_torn_frames(self) -> None:
        """Warn when the session has frames flagged as torn (ZED only)."""
        if self.torn_frames == 0 or self.session is None:
            return
        print(
            f"WARNING: {self.torn_frames}/{len(self.session.keyframes)} frames "
            "look torn (horizontal bands, see 'torn' in frames.jsonl). The "
            "check misses milder tears, so expect more. This comes from USB "
            "data loss: check the Jetson's 5 V supply and the camera's USB "
            "connection/hub."
        )

    def _save_image_pair(self, frame: Frame) -> bool:
        """Save one calibration sample and append both dataset manifests."""
        if self.paths is None or self.session is None:
            raise RuntimeError("A capture session has not been started")

        paths = self.paths
        session = self.session
        frame_index = len(session.keyframes)
        timestamps = frame.timestamps or {}
        host_unix_ns = timestamps.get("host_unix_ns") or int(time.time() * 1e9)
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
        tear_seams = None
        if self.camera_type == "zed":
            tear_seams = count_tear_seams(frame.color_image)
            frame_record["tear_seams"] = tear_seams
            frame_record["torn"] = tear_seams >= TORN_MIN_SEAMS
            if frame_record["torn"]:
                self.torn_frames += 1
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
        if tear_seams is not None and tear_seams >= TORN_MIN_SEAMS:
            print(f"  WARNING: frame looks torn ({tear_seams} seams)")
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
        self._report_torn_frames()
        print("Resuming live preview...")

    def capture_headless(
        self, output: Optional[str], images_per_shot: int, timeout: float
    ) -> int:
        """Capture ``images_per_shot`` image pairs with no interactive I/O.

        Meant to be triggered over a non-interactive SSH session (no
        ``DISPLAY``, no TTY for ``input()``), e.g. from an action server
        that shells out to this script on the Jetson.
        """
        self._prepare_session(output, interactive=False)
        saved_count = 0
        deadline = time.monotonic() + timeout
        for image_index in range(images_per_shot):
            frame = None
            while frame is None:
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"Timed out waiting for frame {image_index + 1}/{images_per_shot}"
                    )
                frame = self._grab_frame()
            if self._save_image_pair(frame):
                saved_count += 1
            print(f"  {self._centre_depth_text(frame.depth_m)}", flush=True)
            deadline = time.monotonic() + timeout

        self._report_torn_frames()
        print(f"HEADLESS_CAPTURE_DONE: {self.paths.base} images={saved_count}")
        return saved_count

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
        self._report_torn_frames()
        print("Resuming live preview...")

    def record_video(self) -> None:
        """Configure (interactively) and record RGB video plus per-frame depth data."""
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
        self._record_video_loop(duration_seconds, headless=False)
        print("Resuming live preview...")

    def record_video_headless(
        self, output: Optional[str], duration_seconds: Optional[float], timeout: float
    ) -> int:
        """Record RGB video plus per-frame depth with no interactive I/O.

        Like capture_headless() (no ``DISPLAY``, no TTY): records for
        ``duration_seconds``, or until Ctrl+C / SIGTERM when None. Aborts if
        no frame arrives for ``timeout`` seconds.

        Output:
            int: the number of frames recorded.
        """
        self._prepare_session(output, interactive=False)
        # SIGTERM (e.g. the process is stopped remotely) ends the recording
        # like Ctrl+C, so the video file is closed properly.
        previous = signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
        try:
            frame_count = self._record_video_loop(
                duration_seconds, headless=True, timeout=timeout
            )
        finally:
            signal.signal(signal.SIGTERM, previous)
        print(f"HEADLESS_VIDEO_DONE: {self.paths.base} frames={frame_count}", flush=True)
        return frame_count

    def _record_video_loop(
        self,
        duration_seconds: Optional[float],
        headless: bool,
        timeout: Optional[float] = None,
    ) -> int:
        """Record into the current session until the duration, ``q`` or Ctrl+C.

        With ``headless`` there is no preview window (and no ``q``); a
        ``timeout`` (s) aborts when no frame arrives for that long.

        Output:
            int: the number of frames recorded.
        """
        if self.paths is None:
            raise RuntimeError("A capture session has not been started")
        paths = self.paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print("Recording high-resolution video...")
        print(f"  Resolution: {self.camera.width}x{self.camera.height}")
        print("  Raw depth data will be saved for each frame")
        if duration_seconds is None:
            print("  Press Ctrl+C to stop" if headless else "  Press q to stop")
        else:
            print(f"  Recording for {duration_seconds:g} seconds")

        frame_count = 0
        started_at = time.monotonic()
        last_frame_at = started_at
        try:
            while (
                duration_seconds is None
                or time.monotonic() - started_at < duration_seconds
            ):
                frame = self._grab_frame()
                if frame is None:
                    if timeout is not None and time.monotonic() - last_frame_at > timeout:
                        raise RuntimeError(
                            f"No frame for {timeout:g} s after {frame_count} frames"
                        )
                    continue
                last_frame_at = time.monotonic()

                self._save_image_pair(frame)
                frame_count += 1

                if not headless and self._show_frame(
                    "",
                    frame.color_image,
                    frame.depth_m,
                    show_distance=False,
                ) == ord("q"):
                    break

                if frame_count % 30 == 0:
                    print(
                        f"Recorded {frame_count} frames (with raw depth data)...",
                        flush=True,
                    )
        except KeyboardInterrupt:
            print("\nRecording interrupted by user")
        finally:
            if not headless:
                cv2.destroyAllWindows()

        elapsed_seconds = time.monotonic() - started_at
        rgb_video_path, depth_video_path, video_fps = write_session_videos(
            paths.base, timestamp
        )
        print("✓ High-resolution video saved:")
        print(f"  RGB video: {rgb_video_path} ({video_fps:.2f} fps, the capture rate)")
        print(f"  Depth video: {depth_video_path}")
        print(f"  Raw depth frames: {frame_count} .npy files in {paths.depth}")
        print(f"  Per-frame depth PNGs: {paths.depth_mm}")
        print(f"  Frames recorded: {frame_count}")
        print(f"  Duration: {elapsed_seconds:.2f} seconds")
        self._report_capture_rate(frame_count, elapsed_seconds)
        self._report_torn_frames()
        return frame_count

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

    def cleanup(self, headless: bool = False) -> None:
        """Close the camera and, unless running headless, all OpenCV windows.

        Skipping ``destroyAllWindows()`` in headless mode avoids touching a
        display backend on a machine with no ``DISPLAY`` (e.g. over SSH).
        """
        self.camera.close()
        if not headless:
            cv2.destroyAllWindows()
        print("Camera resources cleaned up")


def write_session_videos(
    base: Path, timestamp: Optional[str] = None
) -> "tuple[Optional[Path], Optional[Path], float]":
    """Build the RGB and depth videos of a session from its saved frames.

    The frame rate is the one the frames were captured at (their
    frames.jsonl timestamps), so the videos play back in real time.

    Input:
        base: the session folder (with frames.jsonl).
        timestamp: the videos' name suffix; default: the first frame's time.

    Output:
        (rgb video path, depth video path, fps); (None, None, 0.0) without
        frames.
    """
    base = Path(base)
    frames_file = base / "frames.jsonl"
    records = []
    if frames_file.exists():
        with frames_file.open() as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    records = [r for r in records if r.get("rgb")]
    if not records:
        print(f"No frames in {frames_file}: no video written")
        return None, None, 0.0

    times_ns = [
        (r.get("timestamps") or {}).get("host_unix_ns") for r in records
    ]
    fps = float(DEFAULT_FPS)
    if len(records) > 1 and all(times_ns) and times_ns[-1] > times_ns[0]:
        fps = (len(records) - 1) / ((times_ns[-1] - times_ns[0]) / 1e9)
    if timestamp is None:
        start = datetime.fromtimestamp(times_ns[0] / 1e9) if times_ns[0] else datetime.now()
        timestamp = start.strftime("%Y%m%d_%H%M%S")

    first = cv2.imread(str(base / records[0]["rgb"]))
    if first is None:
        raise OSError(f"Could not read image: {base / records[0]['rgb']}")
    size = (first.shape[1], first.shape[0])
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    rgb_path = base / "rgb" / f"rgb_video_{timestamp}.mp4"
    depth_path = base / "depth" / f"depth_video_{timestamp}.mp4"
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_writer = cv2.VideoWriter(str(rgb_path), fourcc, fps, size)
    depth_writer = cv2.VideoWriter(str(depth_path), fourcc, fps, size)
    if not rgb_writer.isOpened() or not depth_writer.isOpened():
        rgb_writer.release()
        depth_writer.release()
        raise RuntimeError(f"Could not open video writers in {base}")
    written_depth = 0
    try:
        for record in records:
            rgb = cv2.imread(str(base / record["rgb"]))
            if rgb is None:
                raise OSError(f"Could not read image: {base / record['rgb']}")
            if (rgb.shape[1], rgb.shape[0]) != size:
                rgb = cv2.resize(rgb, size)
            rgb_writer.write(rgb)
            depth_mm = (
                cv2.imread(str(base / record["depth_mm"]), cv2.IMREAD_UNCHANGED)
                if record.get("depth_mm")
                else None
            )
            if depth_mm is None:
                continue
            # Same colormap as the live preview.
            colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_mm, alpha=0.03), cv2.COLORMAP_JET
            )
            if (colormap.shape[1], colormap.shape[0]) != size:
                colormap = cv2.resize(colormap, size, interpolation=cv2.INTER_NEAREST)
            depth_writer.write(colormap)
            written_depth += 1
    finally:
        rgb_writer.release()
        depth_writer.release()
    if written_depth == 0:
        depth_path.unlink(missing_ok=True)
        depth_path = None
    return rgb_path, depth_path, fps


def _raise_keyboard_interrupt(signum, frame) -> None:
    """Signal handler: stop a headless recording like Ctrl+C."""
    raise KeyboardInterrupt


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
    parser.add_argument(
        "--headless",
        action="store_true",
        help=(
            "Skip the interactive preview; capture --images-per-shot images "
            "once and exit. For non-interactive SSH-triggered capture "
            "(no DISPLAY, no TTY)."
        ),
    )
    parser.add_argument(
        "--make-video",
        metavar="SESSION",
        help=(
            "Rebuild the RGB and depth videos of an existing session folder "
            "(at its capture rate) and exit; no camera needed"
        ),
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help=(
            "--headless mode: record RGB video plus per-frame depth instead "
            "of --images-per-shot images"
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help=(
            "--headless --video: recording length in seconds (default: until "
            "Ctrl+C or SIGTERM)"
        ),
    )
    parser.add_argument(
        "--output",
        help=(
            "Output directory name/path for --headless mode (default: an "
            "automatic timestamped name, same as pressing Enter interactively)"
        ),
    )
    parser.add_argument(
        "--images-per-shot",
        type=positive_int,
        default=DEFAULT_IMAGES_PER_SHOT,
        help=(
            "--headless mode: number of image pairs to capture "
            f"(default: {DEFAULT_IMAGES_PER_SHOT})"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="--headless mode: seconds to wait for each frame before aborting (default: 30.0)",
    )
    return parser


def validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject option combinations that would be silently ignored."""
    if (args.video or args.duration is not None) and not args.headless:
        parser.error("--video and --duration need --headless (press r in the preview instead)")
    if args.duration is not None and not args.video:
        parser.error("--duration needs --video")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be greater than zero")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    validate_arguments(parser, args)
    if args.make_video:
        rgb_video, depth_video, fps = write_session_videos(Path(args.make_video))
        if rgb_video is None:
            return 1
        print(f"RGB video: {rgb_video} ({fps:.2f} fps)")
        print(f"Depth video: {depth_video}")
        return 0
    capture: Optional[RGBDHighResCapture] = None

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
        if args.headless and args.video:
            capture.record_video_headless(args.output, args.duration, args.timeout)
        elif args.headless:
            capture.capture_headless(args.output, args.images_per_shot, args.timeout)
        else:
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
            capture.cleanup(headless=args.headless)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
