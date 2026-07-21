#!/usr/bin/env python3
"""ZED SDK camera driver.

Every ``pyzed.sl`` symbol used by the capture scripts (``scripts/record_data.py``,
``scripts/live_pointcloud_zed.py``) is confined to this module, behind
:class:`ZedCamera` and the helpers below. The scripts themselves only deal in
plain numpy arrays, strings and booleans -- swapping in another camera later
means writing a new driver module with the same shape, not touching the
scripts.
"""

from __future__ import annotations

import numpy as np
import pyzed.sl as sl

RESOLUTIONS = {
    "HD2K": (sl.RESOLUTION.HD2K, 2208, 1242),
    "HD1080": (sl.RESOLUTION.HD1080, 1920, 1080),
    "HD720": (sl.RESOLUTION.HD720, 1280, 720),
    "VGA": (sl.RESOLUTION.VGA, 672, 376),
}

DEPTH_MODES = {
    "NEURAL": sl.DEPTH_MODE.NEURAL,
    "ULTRA": sl.DEPTH_MODE.ULTRA,
    "QUALITY": sl.DEPTH_MODE.QUALITY,
    "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
}


def closest_resolution_name(width: int, height: int) -> str:
    """Pick the ZED resolution preset whose pixel count is closest to the requested size"""
    target = width * height
    return min(RESOLUTIONS, key=lambda name: abs(RESOLUTIONS[name][1] * RESOLUTIONS[name][2] - target))


def unpack_xyzrgba(xyzrgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ZED XYZRGBA measure: Nx4 float32, column 3 packs 4 BGRA uint8 bytes into
    one float32. Returns (points Nx3, colors Nx3 in [0, 1])."""
    xyz = xyzrgba[:, :3].astype(np.float64)
    packed = xyzrgba[:, 3].copy().view(np.uint32)
    b = (packed & 0xFF).astype(np.uint8)
    g = ((packed >> 8) & 0xFF).astype(np.uint8)
    r = ((packed >> 16) & 0xFF).astype(np.uint8)
    colors = np.stack([r, g, b], axis=-1).astype(np.float64) / 255.0
    return xyz, colors


def _camera_params_to_matrix(params) -> tuple[np.ndarray, np.ndarray]:
    """Build a 3x3 camera matrix and distortion vector from ZED calibration parameters"""
    camera_matrix = np.array([
        [params.fx, 0, params.cx],
        [0, params.fy, params.cy],
        [0, 0, 1]
    ])
    dist_coeffs = np.array(params.disto)
    return camera_matrix, dist_coeffs


class ZedCamera:
    """Thin wrapper around ``sl.Camera``. Returns plain numpy/str/bool data;
    never hands back an ``sl.*`` object. Does not print -- callers own their
    own logging/UX."""

    def __init__(self):
        self.zed = sl.Camera()
        self.runtime_params = sl.RuntimeParameters()
        self.runtime_params.enable_depth = True
        self.width: int | None = None
        self.height: int | None = None
        self.fps: int | None = None
        self._image_mat = sl.Mat()
        self._depth_mat = sl.Mat()
        self._xyzrgba_mat = sl.Mat()
        self._pose = sl.Pose()

    def open(self, resolution_name: str, fps: int, depth_mode: str = "NEURAL",
             depth_min: float = 0.3, depth_max: float = 20.0,
             coordinate_system: str = "RIGHT_HANDED_Y_UP") -> tuple[bool, str]:
        """Attempt to open the camera with the given resolution/fps. Returns
        (ok, status_str); sets self.width/height/fps on success.

        coordinate_system is an ``sl.COORDINATE_SYSTEM`` member name -- e.g.
        "IMAGE" (OpenCV/pinhole convention: X right, Y down, Z forward) is
        what the rest of this repo's z-forward pipeline expects for XYZ/pose
        data; "RIGHT_HANDED_Y_UP" (OpenGL-style) is the ZED SDK's default.
        """
        init_params = sl.InitParameters()
        init_params.camera_resolution = RESOLUTIONS[resolution_name][0]
        init_params.camera_fps = fps
        init_params.coordinate_units = sl.UNIT.METER
        init_params.coordinate_system = getattr(sl.COORDINATE_SYSTEM, coordinate_system)
        init_params.depth_mode = DEPTH_MODES[depth_mode]
        init_params.depth_minimum_distance = depth_min
        init_params.depth_maximum_distance = depth_max

        status = self.zed.open(init_params)
        ok = status == sl.ERROR_CODE.SUCCESS
        if ok:
            _, self.width, self.height = RESOLUTIONS[resolution_name]
            self.fps = fps
        return ok, str(status)

    def is_opened(self) -> bool:
        return self.zed.is_opened()

    def get_intrinsics(self) -> tuple[np.ndarray, np.ndarray]:
        """Left/color camera (camera_matrix, dist_coeffs). Depth is computed
        on the rectified left image, so it shares these intrinsics."""
        camera_info = self.zed.get_camera_information()
        calibration = camera_info.camera_configuration.calibration_parameters
        return _camera_params_to_matrix(calibration.left_cam)

    def grab(self) -> tuple[bool, str]:
        """Pull the next frame from the camera. Must be called before any
        retrieve_*()/get_pose() call. Returns (ok, status_str)."""
        status = self.zed.grab(self.runtime_params)
        return status == sl.ERROR_CODE.SUCCESS, str(status)

    def retrieve_rgb_depth(self) -> tuple[np.ndarray, np.ndarray]:
        """(color_bgra, depth_m) from the last successful grab()."""
        self.zed.retrieve_image(self._image_mat, sl.VIEW.LEFT)
        self.zed.retrieve_measure(self._depth_mat, sl.MEASURE.DEPTH)
        color_bgra = self._image_mat.get_data()
        depth_m = np.asarray(self._depth_mat.get_data(), dtype=np.float32).copy()
        return color_bgra, depth_m

    def retrieve_xyzrgba(self) -> np.ndarray:
        """Raw XYZRGBA measure (Nx4) from the last successful grab(); decode
        with unpack_xyzrgba()."""
        self.zed.retrieve_measure(self._xyzrgba_mat, sl.MEASURE.XYZRGBA)
        return self._xyzrgba_mat.get_data().reshape(-1, 4)

    def enable_positional_tracking(self) -> tuple[bool, str]:
        status = self.zed.enable_positional_tracking(sl.PositionalTrackingParameters())
        return status == sl.ERROR_CODE.SUCCESS, str(status)

    def disable_positional_tracking(self) -> None:
        self.zed.disable_positional_tracking()

    def get_pose(self) -> tuple[np.ndarray, str, bool]:
        """(T_world_cam, tracking_status_str, tracking_ok). A stale/bad pose
        (tracking_ok False) should not be trusted -- caller should skip the
        frame rather than stack it on the wrong spot."""
        tracking_state = self.zed.get_position(self._pose, sl.REFERENCE_FRAME.WORLD)
        T = np.eye(4)
        T[:3, :3] = np.array(self._pose.get_rotation_matrix(sl.Rotation()).r)
        T[:3, 3] = np.array(self._pose.get_translation(sl.Translation()).get())
        return T, str(tracking_state), tracking_state == sl.POSITIONAL_TRACKING_STATE.OK

    def list_devices(self) -> list[dict]:
        return [
            {"model": d.camera_model, "serial_number": d.serial_number, "state": d.camera_state}
            for d in sl.Camera.get_device_list()
        ]

    def get_active_info(self) -> dict:
        camera_info = self.zed.get_camera_information()
        configuration = camera_info.camera_configuration
        return {
            "model": camera_info.camera_model,
            "serial_number": camera_info.serial_number,
            "firmware_version": configuration.firmware_version,
            "width": configuration.resolution.width,
            "height": configuration.resolution.height,
            "fps": configuration.fps,
        }

    def close(self) -> None:
        self.zed.close()
