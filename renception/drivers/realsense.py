#!/usr/bin/env python3
"""Intel RealSense RGB-D, infrared and IMU camera driver.

Video frames are acquired as one SDK frameset. Native Z16 depth is retained
for calibration datasets and a second copy is aligned to the colour optical
frame for the reconstruction pipeline. When requested and supported by the
device, stereo infrared and motion streams are captured as well.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from queue import Empty, Full, Queue
import time
from typing import Any

import numpy as np


SUCCESS_STATUS = "SUCCESS"


@dataclass
class RealSenseCapture:
    """Plain-numpy representation of one synchronized RealSense frameset."""

    color_bgr: np.ndarray
    depth_raw: np.ndarray
    depth_aligned_raw: np.ndarray
    depth_aligned_m: np.ndarray
    infrared_left: np.ndarray | None = None
    infrared_right: np.ndarray | None = None
    timestamps: dict[str, Any] = field(default_factory=dict)
    imu_samples: list[dict[str, Any]] = field(default_factory=list)


class RealSenseCamera:
    """Camera-agnostic wrapper around ``pyrealsense2``."""

    def __init__(
        self,
        serial_number: str | None = None,
        *,
        enable_infrared: bool = True,
        enable_imu: bool = True,
        lock_capture_settings: bool = True,
        exposure: float | None = None,
        gain: float | None = None,
        settle_seconds: float = 1.5,
    ) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "RealSense support requires pyrealsense2. Install the Intel "
                "RealSense SDK Python bindings and try again."
            ) from exc

        self.rs = rs
        self.serial_number = serial_number
        self.request_infrared = enable_infrared
        self.request_imu = enable_imu
        self.lock_capture_settings = lock_capture_settings
        self.requested_exposure = exposure
        self.requested_gain = gain
        self.settle_seconds = settle_seconds
        self.infrared_enabled = False
        self.imu_enabled = False
        self.capture_settings: dict[str, Any] = {}
        self.feature_warnings: list[str] = []
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        self.profile = None
        self.width: int | None = None
        self.height: int | None = None
        self.fps: int | None = None
        self.depth_scale: float | None = None
        self._frames = None
        self._aligned_frames = None
        self._video_queue: Queue[Any] = Queue(maxsize=2)
        self._motion_samples: Queue[dict[str, Any]] = Queue()
        self._last_host_unix_ns: int | None = None
        self._last_host_monotonic_ns: int | None = None

    def _select_device(self):
        devices = list(self.rs.context().query_devices())
        if self.serial_number is None:
            if not devices:
                raise RuntimeError("No RealSense camera is connected")
            return devices[0]
        for device in devices:
            serial = device.get_info(self.rs.camera_info.serial_number)
            if serial == self.serial_number:
                return device
        raise RuntimeError(
            f"RealSense serial number not found: {self.serial_number}"
        )

    def _available_profiles(self, device) -> list[Any]:
        return [
            profile
            for sensor in device.query_sensors()
            for profile in sensor.get_stream_profiles()
        ]

    def _has_video_profile(
        self,
        profiles: list[Any],
        stream,
        index: int,
        width: int,
        height: int,
        fps: int,
    ) -> bool:
        for profile in profiles:
            if profile.stream_type() != stream or profile.stream_index() != index:
                continue
            try:
                video = profile.as_video_stream_profile()
                if (
                    video.width() == width
                    and video.height() == height
                    and video.fps() == fps
                    and profile.format() == self.rs.format.y8
                ):
                    return True
            except RuntimeError:
                continue
        return False

    @staticmethod
    def _motion_profile_fps(profiles: list[Any], stream) -> int | None:
        rates = [p.fps() for p in profiles if p.stream_type() == stream]
        return max(rates) if rates else None

    def _configure_optional_streams(
        self,
        config,
        device,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        profiles = self._available_profiles(device)
        left = self._has_video_profile(
            profiles, self.rs.stream.infrared, 1, width, height, fps
        )
        right = self._has_video_profile(
            profiles, self.rs.stream.infrared, 2, width, height, fps
        )
        self.infrared_enabled = self.request_infrared and left and right
        if self.infrared_enabled:
            config.enable_stream(
                self.rs.stream.infrared, 1, width, height, self.rs.format.y8, fps
            )
            config.enable_stream(
                self.rs.stream.infrared, 2, width, height, self.rs.format.y8, fps
            )
        elif self.request_infrared:
            self.feature_warnings.append(
                f"Stereo IR is unavailable at {width}x{height} @ {fps} FPS"
            )

        accel_fps = self._motion_profile_fps(profiles, self.rs.stream.accel)
        gyro_fps = self._motion_profile_fps(profiles, self.rs.stream.gyro)
        self.imu_enabled = (
            self.request_imu and accel_fps is not None and gyro_fps is not None
        )
        if self.imu_enabled:
            config.enable_stream(
                self.rs.stream.accel, self.rs.format.motion_xyz32f, accel_fps
            )
            config.enable_stream(
                self.rs.stream.gyro, self.rs.format.motion_xyz32f, gyro_fps
            )
        elif self.request_imu:
            self.feature_warnings.append(
                "This RealSense device has no complete accel+gyro IMU"
            )

    def _on_frame(self, frame) -> None:
        """Queue framesets and retain every asynchronous SDK IMU sample."""
        try:
            if frame.is_frameset():
                try:
                    self._video_queue.put_nowait(frame.as_frameset())
                except Full:
                    try:
                        self._video_queue.get_nowait()
                    except Empty:
                        pass
                    self._video_queue.put_nowait(frame.as_frameset())
                return
            if not frame.is_motion_frame():
                return
            motion = frame.as_motion_frame()
            vector = motion.get_motion_data()
            profile = motion.get_profile()
            stream_name = str(profile.stream_type()).split(".")[-1]
            self._motion_samples.put_nowait(
                {
                    "stream": stream_name,
                    "units": "m/s^2" if stream_name == "accel" else "rad/s",
                    "timestamp_ms": float(motion.get_timestamp()),
                    "timestamp_domain": str(motion.get_frame_timestamp_domain()),
                    "frame_number": int(motion.get_frame_number()),
                    "x": float(vector.x),
                    "y": float(vector.y),
                    "z": float(vector.z),
                }
            )
        except (RuntimeError, AttributeError):
            # Optional metadata must never stop the SDK callback thread.
            return

    def open(self, width: int, height: int, fps: int) -> tuple[bool, str]:
        """Start native colour/depth and any supported requested streams."""
        self.feature_warnings.clear()
        self._video_queue = Queue(maxsize=2)
        self._motion_samples = Queue()
        config = self.rs.config()
        try:
            device = self._select_device()
            serial = device.get_info(self.rs.camera_info.serial_number)
            config.enable_device(serial)
            config.enable_stream(
                self.rs.stream.color, width, height, self.rs.format.bgr8, fps
            )
            config.enable_stream(
                self.rs.stream.depth, width, height, self.rs.format.z16, fps
            )
            self._configure_optional_streams(config, device, width, height, fps)
            self.profile = self.pipeline.start(config, self._on_frame)
            color_profile = self.profile.get_stream(
                self.rs.stream.color
            ).as_video_stream_profile()
            self.width = color_profile.width()
            self.height = color_profile.height()
            self.fps = color_profile.fps()
            self.depth_scale = (
                self.profile.get_device().first_depth_sensor().get_depth_scale()
            )
            if self.lock_capture_settings:
                self._apply_capture_settings(device)
            return True, SUCCESS_STATUS
        except RuntimeError as exc:
            self.profile = None
            return False, str(exc)

    @staticmethod
    def _find_sensor(device, stream_type):
        for sensor in device.query_sensors():
            for profile in sensor.get_stream_profiles():
                if profile.stream_type() == stream_type:
                    return sensor
        return None

    def _lock_exposure_and_gain(
        self, sensor, label: str, forced_exposure: float | None, forced_gain: float | None
    ) -> dict[str, Any]:
        """Disable auto-exposure and pin exposure/gain to a fixed value.

        Without an explicit override, lets auto-exposure settle for
        ``settle_seconds`` first and locks whatever value it converged to —
        so every station in the scan uses the same exposure/gain instead of
        each one re-adjusting to local lighting.
        """
        if sensor is None or not sensor.supports(self.rs.option.enable_auto_exposure):
            self.feature_warnings.append(f"{label} sensor has no auto-exposure control to lock")
            return {}

        if forced_exposure is None or forced_gain is None:
            time.sleep(self.settle_seconds)

        result: dict[str, Any] = {}
        if sensor.supports(self.rs.option.exposure):
            exposure = (
                forced_exposure
                if forced_exposure is not None
                else sensor.get_option(self.rs.option.exposure)
            )
            sensor.set_option(self.rs.option.enable_auto_exposure, 0.0)
            sensor.set_option(self.rs.option.exposure, exposure)
            result["exposure"] = exposure
        if sensor.supports(self.rs.option.gain):
            gain = (
                forced_gain if forced_gain is not None else sensor.get_option(self.rs.option.gain)
            )
            sensor.set_option(self.rs.option.gain, gain)
            result["gain"] = gain
        return result

    def _apply_capture_settings(self, device) -> None:
        """High Accuracy preset, IR projector on, exposure/gain fixed (raw, no post-filters)."""
        depth_sensor = device.first_depth_sensor()
        color_sensor = self._find_sensor(device, self.rs.stream.color)

        preset_applied = False
        if depth_sensor.supports(self.rs.option.visual_preset):
            depth_sensor.set_option(
                self.rs.option.visual_preset,
                float(self.rs.rs400_visual_preset.high_accuracy),
            )
            preset_applied = True
        else:
            self.feature_warnings.append("Depth sensor has no visual_preset option")

        emitter_enabled = False
        if depth_sensor.supports(self.rs.option.emitter_enabled):
            depth_sensor.set_option(self.rs.option.emitter_enabled, 1.0)
            emitter_enabled = True
        else:
            self.feature_warnings.append("Depth sensor has no emitter_enabled option")

        depth_lock = self._lock_exposure_and_gain(
            depth_sensor, "Depth/IR", self.requested_exposure, self.requested_gain
        )
        color_lock = self._lock_exposure_and_gain(color_sensor, "Color", None, None)

        self.capture_settings = {
            "visual_preset": "high_accuracy" if preset_applied else None,
            "emitter_enabled": emitter_enabled,
            "post_processing_filters": "none (raw depth)",
            "depth_ir_exposure_gain": depth_lock,
            "color_exposure_gain": color_lock,
        }

    def is_opened(self) -> bool:
        return self.profile is not None

    @staticmethod
    def _intrinsics_dict(profile) -> dict[str, Any]:
        video = profile.as_video_stream_profile()
        intrinsics = video.get_intrinsics()
        # Effective FOV at the streamed resolution, derived from the
        # device-reported focal lengths (matches librealsense's own
        # rs2_fov()): 2*atan(dimension / (2*f)).
        fov_h_deg = float(
            np.degrees(2.0 * np.arctan2(intrinsics.width, 2.0 * intrinsics.fx))
        )
        fov_v_deg = float(
            np.degrees(2.0 * np.arctan2(intrinsics.height, 2.0 * intrinsics.fy))
        )
        return {
            "width": int(intrinsics.width),
            "height": int(intrinsics.height),
            "fx": float(intrinsics.fx),
            "fy": float(intrinsics.fy),
            "cx": float(intrinsics.ppx),
            "cy": float(intrinsics.ppy),
            "camera_matrix": [
                [float(intrinsics.fx), 0.0, float(intrinsics.ppx)],
                [0.0, float(intrinsics.fy), float(intrinsics.ppy)],
                [0.0, 0.0, 1.0],
            ],
            "distortion_model": str(intrinsics.model),
            "distortion_coefficients": [
                float(value) for value in intrinsics.coeffs
            ],
            "fps": int(video.fps()),
            "fov_deg": {"horizontal": fov_h_deg, "vertical": fov_v_deg},
        }

    @staticmethod
    def _extrinsics_dict(source, target) -> dict[str, Any]:
        extrinsics = source.get_extrinsics_to(target)
        rotation_column_major = [float(value) for value in extrinsics.rotation]
        rotation_matrix = np.asarray(rotation_column_major).reshape(3, 3).T
        translation = np.asarray(extrinsics.translation, dtype=float)
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = translation
        return {
            "rotation_column_major": rotation_column_major,
            "translation_m": translation.tolist(),
            "matrix_4x4": transform.tolist(),
        }

    def get_calibration(self) -> dict[str, Any]:
        """Return complete native-stream calibration and device metadata."""
        if self.profile is None or self.depth_scale is None:
            raise RuntimeError("RealSense camera is not open")
        color = self.profile.get_stream(self.rs.stream.color)
        depth = self.profile.get_stream(self.rs.stream.depth)
        device = self.profile.get_device()
        streams: dict[str, Any] = {
            "color": self._intrinsics_dict(color),
            "depth": self._intrinsics_dict(depth),
        }
        extrinsics: dict[str, Any] = {
            "depth_to_color": self._extrinsics_dict(depth, color),
        }
        if self.infrared_enabled:
            ir_left = self.profile.get_stream(self.rs.stream.infrared, 1)
            ir_right = self.profile.get_stream(self.rs.stream.infrared, 2)
            streams["infrared_left"] = self._intrinsics_dict(ir_left)
            streams["infrared_right"] = self._intrinsics_dict(ir_right)
            extrinsics["infrared_left_to_color"] = self._extrinsics_dict(
                ir_left, color
            )
            extrinsics["infrared_right_to_color"] = self._extrinsics_dict(
                ir_right, color
            )

        return {
            "schema_version": 2,
            "device": {
                "model": device.get_info(self.rs.camera_info.name),
                "serial_number": device.get_info(self.rs.camera_info.serial_number),
                "firmware_version": device.get_info(
                    self.rs.camera_info.firmware_version
                ),
            },
            "depth_units_m": float(self.depth_scale),
            "depth_units": {
                "metres_per_unit": float(self.depth_scale),
                "raw_format": "z16",
                "raw_dtype": "uint16",
            },
            "streams": streams,
            "extrinsics": extrinsics,
            "capture_options": {
                "depth_aligned_to_color": True,
                "infrared_enabled": self.infrared_enabled,
                "imu_enabled": self.imu_enabled,
                "warnings": self.feature_warnings.copy(),
                **self.capture_settings,
            },
        }

    def get_intrinsics(self) -> tuple[np.ndarray, np.ndarray]:
        """Return colour intrinsics used by aligned reconstruction images."""
        calibration = self.get_calibration()["streams"]["color"]
        return (
            np.asarray(calibration["camera_matrix"], dtype=np.float64),
            np.asarray(calibration["distortion_coefficients"], dtype=np.float64),
        )

    def grab(self) -> tuple[bool, str]:
        if self.profile is None:
            return False, "RealSense camera is not open"
        try:
            self._frames = self._video_queue.get(timeout=5.0)
            self._aligned_frames = self.align.process(self._frames)
            self._last_host_unix_ns = time.time_ns()
            self._last_host_monotonic_ns = time.monotonic_ns()
            return True, SUCCESS_STATUS
        except Empty:
            return False, "Timed out waiting for a RealSense video frameset"
        except RuntimeError as exc:
            return False, str(exc)

    def _frame_timestamp(self, frame) -> dict[str, Any]:
        result: dict[str, Any] = {
            "timestamp_ms": float(frame.get_timestamp()),
            "timestamp_domain": str(frame.get_frame_timestamp_domain()),
            "frame_number": int(frame.get_frame_number()),
        }
        metadata_key = self.rs.frame_metadata_value.sensor_timestamp
        if frame.supports_frame_metadata(metadata_key):
            result["sensor_timestamp_us"] = int(
                frame.get_frame_metadata(metadata_key)
            )
        return result

    def retrieve_capture(self) -> RealSenseCapture:
        """Return native/aligned images, per-stream timestamps and pending IMU."""
        if self._frames is None or self._aligned_frames is None:
            raise RuntimeError("Call grab() successfully before retrieving frames")
        if self.depth_scale is None:
            raise RuntimeError("RealSense depth scale is unavailable")

        color = self._frames.get_color_frame()
        depth = self._frames.get_depth_frame()
        aligned_depth = self._aligned_frames.get_depth_frame()
        if not color or not depth or not aligned_depth:
            raise RuntimeError("RealSense returned an incomplete colour/depth frameset")

        ir_left = (
            self._frames.get_infrared_frame(1) if self.infrared_enabled else None
        )
        ir_right = (
            self._frames.get_infrared_frame(2) if self.infrared_enabled else None
        )
        motion_samples = []
        while True:
            try:
                motion_samples.append(self._motion_samples.get_nowait())
            except Empty:
                break
        timestamps = {
            "host_unix_ns": self._last_host_unix_ns,
            "host_monotonic_ns": self._last_host_monotonic_ns,
            "color": self._frame_timestamp(color),
            "depth": self._frame_timestamp(depth),
            "aligned_depth": self._frame_timestamp(aligned_depth),
        }
        if ir_left:
            timestamps["infrared_left"] = self._frame_timestamp(ir_left)
        if ir_right:
            timestamps["infrared_right"] = self._frame_timestamp(ir_right)

        depth_aligned_raw = np.asanyarray(aligned_depth.get_data()).copy()
        return RealSenseCapture(
            color_bgr=np.asanyarray(color.get_data()).copy(),
            depth_raw=np.asanyarray(depth.get_data()).copy(),
            depth_aligned_raw=depth_aligned_raw,
            depth_aligned_m=depth_aligned_raw.astype(np.float32) * self.depth_scale,
            infrared_left=(
                np.asanyarray(ir_left.get_data()).copy() if ir_left else None
            ),
            infrared_right=(
                np.asanyarray(ir_right.get_data()).copy() if ir_right else None
            ),
            timestamps=timestamps,
            imu_samples=motion_samples,
        )

    def retrieve_rgb_depth(self) -> tuple[np.ndarray, np.ndarray]:
        capture = self.retrieve_capture()
        return capture.color_bgr, capture.depth_aligned_m

    def list_devices(self) -> list[dict[str, str]]:
        return [
            {
                "model": device.get_info(self.rs.camera_info.name),
                "serial_number": device.get_info(self.rs.camera_info.serial_number),
                "state": "available",
            }
            for device in self.rs.context().query_devices()
        ]

    def get_active_info(self) -> dict[str, str | int | bool | None]:
        if self.profile is None:
            raise RuntimeError("RealSense camera is not open")
        device = self.profile.get_device()
        return {
            "model": device.get_info(self.rs.camera_info.name),
            "serial_number": device.get_info(self.rs.camera_info.serial_number),
            "firmware_version": device.get_info(self.rs.camera_info.firmware_version),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "infrared_enabled": self.infrared_enabled,
            "imu_enabled": self.imu_enabled,
        }

    def close(self) -> None:
        if self.profile is not None:
            self.pipeline.stop()
            self.profile = None
        self._frames = None
        self._aligned_frames = None
        self.width = None
        self.height = None
        self.fps = None
        self.depth_scale = None
