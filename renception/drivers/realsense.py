#!/usr/bin/env python3
"""Intel RealSense RGB-D camera driver.

The public API mirrors :mod:`renception.drivers.zed` closely so capture tools
can store data independently from the camera manufacturer.  Depth frames are
aligned to the colour optical frame before being returned in metres.

Example:
    camera = RealSenseCamera()
    opened, message = camera.open(1280, 720, 30)
    if opened:
        captured, _ = camera.grab()
        if captured:
            color_bgr, depth_m = camera.retrieve_rgb_depth()
        camera.close()
"""

from __future__ import annotations

import numpy as np


SUCCESS_STATUS = "SUCCESS"


class RealSenseCamera:
    """Control one Intel RealSense camera through a small, camera-agnostic API.

    Input:
        serial_number: Optional RealSense serial number. When omitted, the SDK
            selects the default available camera.

    RGB frames are returned in BGR format for OpenCV. Depth frames are aligned
    to RGB and converted to metres. Typical use is ``open()`` once, then
    ``grab()`` and ``retrieve_rgb_depth()`` for every frame, followed by
    ``close()``.

    Methods:
        open: Start the RGB-D stream.
        grab: Acquire and align the next frame.
        retrieve_rgb_depth: Return the acquired BGR and depth arrays.
        get_intrinsics: Return calibration for the aligned images.
        list_devices and get_active_info: Return camera metadata.
        is_opened and close: Inspect and release the stream state.
    """

    def __init__(self, serial_number: str | None = None) -> None:
        """Create the driver without starting a camera stream.

        Args:
            serial_number: Optional serial number of the RealSense to use.

        Returns:
            None.

        Raises:
            RuntimeError: If the ``pyrealsense2`` SDK binding is not installed.
        """
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "RealSense support requires pyrealsense2. Install the Intel "
                "RealSense SDK Python bindings and try again."
            ) from exc

        self.rs = rs
        self.serial_number = serial_number
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        self.profile = None
        self.width: int | None = None
        self.height: int | None = None
        self.fps: int | None = None
        self.depth_scale: float | None = None
        self._frames = None

    def open(self, width: int, height: int, fps: int) -> tuple[bool, str]:
        """Start aligned colour and depth streams at the requested settings.

        Args:
            width: Requested image width in pixels.
            height: Requested image height in pixels.
            fps: Requested capture frequency in frames per second.

        Returns:
            A pair ``(opened, status)``. ``opened`` is ``True`` on success;
            otherwise it is ``False`` and ``status`` explains the SDK error.

        The SDK may select a supported equivalent configuration. On success,
        :attr:`width`, :attr:`height`, :attr:`fps`, and :attr:`depth_scale` are
        populated.
        """
        config = self.rs.config()
        if self.serial_number:
            config.enable_device(self.serial_number)
        config.enable_stream(
            self.rs.stream.color,
            width,
            height,
            self.rs.format.bgr8,
            fps,
        )
        config.enable_stream(
            self.rs.stream.depth,
            width,
            height,
            self.rs.format.z16,
            fps,
        )
        try:
            self.profile = self.pipeline.start(config)
            color_profile = self.profile.get_stream(
                self.rs.stream.color
            ).as_video_stream_profile()
            self.width = color_profile.width()
            self.height = color_profile.height()
            self.fps = color_profile.fps()
            device = self.profile.get_device()
            self.depth_scale = device.first_depth_sensor().get_depth_scale()
            return True, SUCCESS_STATUS
        except RuntimeError as exc:
            self.profile = None
            return False, str(exc)

    def is_opened(self) -> bool:
        """Report whether this driver's RealSense stream is open.

        Args:
            None.

        Returns:
            ``True`` after a successful :meth:`open` and before :meth:`close`;
            otherwise ``False``.
        """
        return self.profile is not None

    def get_intrinsics(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the 3x3 colour camera matrix and its distortion coefficients.

        Args:
            None. The camera must already be open.

        Returns:
            A pair ``(camera_matrix, distortion_coefficients)``. The matrix is
            3x3 ``float64`` and the coefficients are a ``float64`` array.

        Raises:
            RuntimeError: If the camera stream is not open.

        Because depth is aligned to the colour image, these intrinsics apply to
        both arrays returned by :meth:`retrieve_rgb_depth`.
        """
        if self.profile is None:
            raise RuntimeError("RealSense camera is not open")
        profile = self.profile.get_stream(
            self.rs.stream.color
        ).as_video_stream_profile()
        intrinsics = profile.get_intrinsics()
        camera_matrix = np.array(
            [
                [intrinsics.fx, 0.0, intrinsics.ppx],
                [0.0, intrinsics.fy, intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return camera_matrix, np.asarray(intrinsics.coeffs, dtype=np.float64)

    def grab(self) -> tuple[bool, str]:
        """Wait for the next RGB-D frame and align depth with colour.

        Args:
            None. Call :meth:`open` before this method.

        Returns:
            A pair ``(captured, status)``. ``captured`` is ``True`` when a new
            frame is ready for :meth:`retrieve_rgb_depth`; otherwise ``status``
            contains the reason it could not be acquired.

        Call this once before every :meth:`retrieve_rgb_depth`. It returns a
        success flag and an SDK status message instead of raising for ordinary
        acquisition failures.
        """
        if self.profile is None:
            return False, "RealSense camera is not open"
        try:
            self._frames = self.align.process(self.pipeline.wait_for_frames())
            return True, SUCCESS_STATUS
        except RuntimeError as exc:
            return False, str(exc)

    def retrieve_rgb_depth(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the arrays produced by the most recent successful :meth:`grab`.

        Args:
            None. :meth:`grab` must have completed successfully first.

        Returns:
            A pair ``(color_bgr, depth_m)``: an OpenCV-ready BGR image and an
            aligned ``float32`` depth image in metres. Invalid depth pixels
            retain the SDK value, normally zero.

        Raises:
            RuntimeError: If no frame was captured or depth scale is unavailable.
        """
        if self._frames is None:
            raise RuntimeError("Call grab() successfully before retrieving frames")
        if self.depth_scale is None:
            raise RuntimeError("RealSense depth scale is unavailable")
        color = self._frames.get_color_frame()
        depth = self._frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError("RealSense returned an incomplete colour/depth frame")
        color_bgr = np.asanyarray(color.get_data()).copy()
        depth_units = np.asanyarray(depth.get_data())
        depth_m = depth_units.astype(np.float32) * self.depth_scale
        return color_bgr, depth_m

    def list_devices(self) -> list[dict[str, str]]:
        """List connected RealSense devices without opening a stream.

        Args:
            None.

        Returns:
            A list of dictionaries with ``model``, ``serial_number``, and
            ``state`` keys. The list is empty when no camera is connected.

        Each item contains the model, serial number, and an ``"available"``
        state, making the result suitable for a simple device-selection UI.
        """
        return [
            {
                "model": device.get_info(self.rs.camera_info.name),
                "serial_number": device.get_info(self.rs.camera_info.serial_number),
                "state": "available",
            }
            for device in self.rs.context().query_devices()
        ]

    def get_active_info(self) -> dict[str, str | int | None]:
        """Return model, serial, firmware, and active stream settings.

        Args:
            None. The camera must already be open.

        Returns:
            A dictionary containing the camera model, serial number, firmware
            version, and active ``width``, ``height``, and ``fps`` values.

        Raises:
            RuntimeError: If the camera stream is not open.

        Use this after :meth:`open` to record which physical camera and stream
        configuration produced a capture.
        """
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
        }

    def close(self) -> None:
        """Stop the stream and clear cached frames and stream metadata.

        Args:
            None.

        Returns:
            None.

        It is safe to call after an unsuccessful :meth:`open`; no action is
        taken if the pipeline was never started.
        """
        if self.profile is not None:
            self.pipeline.stop()
            self.profile = None
        self._frames = None
        self.width = None
        self.height = None
        self.fps = None
        self.depth_scale = None
