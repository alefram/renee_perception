#!/usr/bin/env python3

"""
Capture RGB and depth data from a ZED 2i camera using the ZED SDK (pyzed).

Depth is computed by the SDK directly on the rectified left/color image, so
RGB and depth are always pixel-aligned at the same resolution with no
separate alignment step.

Modes (--mode):
    preview  Live RGB/depth view. 'c' capture image, 'r' record video, 'q' quit.
    image    Capture a single RGB/depth pair.
    video    Record RGB/depth video for --duration seconds (or until 'q').
    info     Print available ZED devices and the active device's configuration.

Output layout (written to zed_highres_<timestamp>/ by default, or
--output-name <name>/ if given):
    rgb/         rgb_<timestamp>.png, rgb_video_<timestamp>.mp4
    depth/       depth_<timestamp>.png (16-bit mm), depth_raw_<timestamp>.npy
                 (float32 meters), depth_video_<timestamp>.mp4 (colormap)
    intrinsics/  camera_intrinsics.json

Examples:
    python3 zed_data_capturing.py --mode image
    python3 zed_data_capturing.py --mode video --duration 10
    python3 zed_data_capturing.py --preset high --depth-mode ULTRA
    python3 zed_data_capturing.py --mode video --output-name kitchen_scan_01
"""

import pyzed.sl as sl
import numpy as np
import cv2
import os
from datetime import datetime
import json
import argparse

# ZED resolution presets: name -> (sl.RESOLUTION, width, height)
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


def closest_resolution_name(width, height):
    """Pick the ZED resolution preset whose pixel count is closest to the requested size"""
    target = width * height
    return min(RESOLUTIONS, key=lambda name: abs(RESOLUTIONS[name][1] * RESOLUTIONS[name][2] - target))


class ZED2iHighResCapture:
    def __init__(self, width=1280, height=720, fps=30, depth_mode="NEURAL", output_name=None):
        """
        Initialize ZED 2i camera capture. RGB and depth share the same resolution
        because ZED depth is computed directly on the rectified left (color) image,
        so no separate depth/color alignment step is needed.

        Args:
            width (int): Requested width, mapped to the closest ZED resolution preset (default: 1280)
            height (int): Requested height, mapped to the closest ZED resolution preset (default: 720)
            fps (int): Frames per second (default: 30)
            depth_mode (str): One of NEURAL, ULTRA, QUALITY, PERFORMANCE (default: NEURAL)
            output_name (str): Name for the output directory (default: zed_highres_<timestamp>)
        """
        self.requested_width = width
        self.requested_height = height
        self.depth_mode_name = depth_mode

        self.zed = sl.Camera()
        self.runtime_params = sl.RuntimeParameters()
        self.runtime_params.enable_depth = True

        resolution_name = closest_resolution_name(width, height)
        print(f"Configuring stream:")
        print(f"  Requested: {width}x{height} at {fps} FPS")
        print(f"  Using ZED preset: {resolution_name}")

        # Create directories for saving data
        self.create_directories(output_name)

        if not self.open_camera(resolution_name, fps):
            print("Trying fallback resolutions...")
            self.try_fallback_resolutions()

        # Get camera intrinsics
        self.get_camera_intrinsics()

        # Mats reused across grabs
        self.image_mat = sl.Mat()
        self.depth_mat = sl.Mat()

    def open_camera(self, resolution_name, fps):
        """Attempt to open the ZED camera with the given resolution/fps"""
        init_params = sl.InitParameters()
        init_params.camera_resolution = RESOLUTIONS[resolution_name][0]
        init_params.camera_fps = fps
        init_params.coordinate_units = sl.UNIT.METER
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init_params.depth_mode = DEPTH_MODES[self.depth_mode_name]
        init_params.depth_minimum_distance = 0.3
        init_params.depth_maximum_distance = 20.0

        status = self.zed.open(init_params)
        if status == sl.ERROR_CODE.SUCCESS:
            _, self.width, self.height = RESOLUTIONS[resolution_name]
            self.fps = fps
            print("✓ ZED 2i camera initialized successfully!")
            return True

        print(f"✗ Failed to start camera: {status}")
        return False

    def try_fallback_resolutions(self):
        """Try different resolution/fps combinations if the initial setup fails"""
        fallback_configs = [
            ("HD720", 30),
            ("HD720", 15),
            ("VGA", 60),
            ("VGA", 30),
        ]

        for resolution_name, fps in fallback_configs:
            print(f"Trying: {resolution_name} at {fps}fps")
            if self.open_camera(resolution_name, fps):
                print(f"✓ Successfully initialized with fallback resolution")
                return

        raise RuntimeError("Could not initialize camera with any supported resolution")

    def create_directories(self, output_name=None):
        """Create directories for saving RGB and depth data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = output_name if output_name else f"zed_highres_{timestamp}"

        self.rgb_dir = os.path.join(self.base_dir, "rgb")
        self.depth_dir = os.path.join(self.base_dir, "depth")
        self.intrinsics_dir = os.path.join(self.base_dir, "intrinsics")

        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.intrinsics_dir, exist_ok=True)

        print(f"Created directories in: {self.base_dir}")

    @staticmethod
    def _camera_params_to_matrix(params):
        """Build a 3x3 camera matrix and distortion vector from ZED calibration parameters"""
        camera_matrix = np.array([
            [params.fx, 0, params.cx],
            [0, params.fy, params.cy],
            [0, 0, 1]
        ])
        dist_coeffs = np.array(params.disto)
        return camera_matrix, dist_coeffs

    def get_camera_intrinsics(self):
        """Get camera intrinsic parameters"""
        camera_info = self.zed.get_camera_information()
        calibration = camera_info.camera_configuration.calibration_parameters

        # Left (color) camera intrinsics
        self.color_camera_matrix, self.color_dist_coeffs = self._camera_params_to_matrix(calibration.left_cam)

        # Depth is computed on the rectified left image, so it shares the color camera's intrinsics
        self.depth_camera_matrix, self.depth_dist_coeffs = self._camera_params_to_matrix(calibration.left_cam)

        # Save intrinsics to file
        self.save_intrinsics()

        print("Camera intrinsics loaded and saved!")
        print(f"Left camera focal length: fx={calibration.left_cam.fx:.2f}, fy={calibration.left_cam.fy:.2f}")

    def save_intrinsics(self):
        """Save camera intrinsic parameters to JSON file"""
        intrinsics_data = {
            "color_camera": {
                "camera_matrix": self.color_camera_matrix.tolist(),
                "distortion_coefficients": self.color_dist_coeffs.tolist(),
                "width": self.width,
                "height": self.height
            },
            "depth_camera": {
                "camera_matrix": self.depth_camera_matrix.tolist(),
                "distortion_coefficients": self.depth_dist_coeffs.tolist(),
                "width": self.width,
                "height": self.height
            },
            "stream_info": {
                "resolution": f"{self.width}x{self.height}",
                "fps": self.fps,
                "depth_mode": self.depth_mode_name,
                "note": "Depth is computed on the rectified left image, so it shares the color camera's intrinsics"
            }
        }

        intrinsics_file = os.path.join(self.intrinsics_dir, "camera_intrinsics.json")
        with open(intrinsics_file, 'w') as f:
            json.dump(intrinsics_data, f, indent=4)

        print(f"Intrinsics saved to: {intrinsics_file}")

    def _grab_frame(self):
        """Grab a frame and return (color_bgr, depth_m) or (None, None) on failure"""
        status = self.zed.grab(self.runtime_params)
        if status != sl.ERROR_CODE.SUCCESS:
            return None, None

        self.zed.retrieve_image(self.image_mat, sl.VIEW.LEFT)
        self.zed.retrieve_measure(self.depth_mat, sl.MEASURE.DEPTH)

        color_bgra = self.image_mat.get_data()
        color_bgr = cv2.cvtColor(color_bgra, cv2.COLOR_BGRA2BGR)
        depth_m = np.asarray(self.depth_mat.get_data(), dtype=np.float32).copy()

        return color_bgr, depth_m

    @staticmethod
    def _depth_to_uint16_mm(depth_m):
        """Convert a float32 depth map in meters to a 16-bit millimeter map (0 = invalid)"""
        valid = np.isfinite(depth_m) & (depth_m > 0)
        depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
        depth_mm[valid] = np.clip(np.round(depth_m[valid] * 1000.0), 1, np.iinfo(np.uint16).max).astype(np.uint16)
        return depth_mm

    def capture_single_image(self):
        """Capture a single RGB and depth image pair at high resolution"""
        try:
            color_image, depth_m = self._grab_frame()

            if color_image is None:
                print("Failed to capture frames")
                return False

            print(f"Captured image sizes - Color: {color_image.shape}, Depth: {depth_m.shape}")

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # microseconds to milliseconds

            # Save RGB image (high resolution)
            rgb_filename = os.path.join(self.rgb_dir, f"rgb_{timestamp}.png")
            cv2.imwrite(rgb_filename, color_image)

            # Save depth image (16-bit, millimeters, aligned to color resolution)
            depth_image = self._depth_to_uint16_mm(depth_m)
            depth_filename = os.path.join(self.depth_dir, f"depth_{timestamp}.png")
            cv2.imwrite(depth_filename, depth_image)

            # Save raw depth data (float32, meters) as numpy array for precise measurements
            depth_raw_filename = os.path.join(self.depth_dir, f"depth_raw_{timestamp}.npy")
            np.save(depth_raw_filename, depth_m)

            print(f"✓ Captured high-res image pair: {timestamp}")
            print(f"  RGB: {rgb_filename} ({color_image.shape[1]}x{color_image.shape[0]})")
            print(f"  Depth: {depth_filename} ({depth_image.shape[1]}x{depth_image.shape[0]})")
            print(f"  Both streams at matching {self.width}x{self.height} resolution")
            return True

        except Exception as e:
            print(f"Error capturing image: {e}")
            return False

    def record_video(self, duration_seconds=None):
        """
        Record high-resolution video with both RGB and depth data
        Also saves raw depth data for each frame

        Args:
            duration_seconds (int): Recording duration. If None, record until 'q' is pressed
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup video writers with appropriate codecs for high resolution
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Better codec for high resolution

        rgb_video_path = os.path.join(self.rgb_dir, f"rgb_video_{timestamp}.mp4")
        depth_video_path = os.path.join(self.depth_dir, f"depth_video_{timestamp}.mp4")

        # Use same resolution for both RGB and depth video
        rgb_writer = cv2.VideoWriter(rgb_video_path, fourcc, self.fps,
                                   (self.width, self.height))

        depth_writer = cv2.VideoWriter(depth_video_path, fourcc, self.fps,
                                     (self.width, self.height))

        print(f"Recording high-resolution video...")
        print(f"  Resolution: {self.width}x{self.height} (both RGB and depth)")
        print(f"  Raw depth data will be saved for each frame")
        print(f"  {'Press q to stop' if duration_seconds is None else f'Recording for {duration_seconds} seconds'}")

        frame_count = 0
        max_frames = duration_seconds * self.fps if duration_seconds else float('inf')

        try:
            while frame_count < max_frames:
                color_image, depth_m = self._grab_frame()

                if color_image is None:
                    continue

                # Generate frame-specific timestamp
                frame_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

                # Save raw depth data as numpy array for precise measurements
                depth_raw_filename = os.path.join(self.depth_dir, f"depth_raw_frame_{frame_count:06d}_{frame_timestamp}.npy")
                np.save(depth_raw_filename, depth_m)

                # Convert depth to 8-bit for video (normalize)
                depth_image = self._depth_to_uint16_mm(depth_m)
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )

                # Write frames
                rgb_writer.write(color_image)
                depth_writer.write(depth_colormap)

                # Display frames (no scaling needed since both streams are same resolution)
                cv2.imshow(f'RGB ({self.width}x{self.height})', color_image)
                cv2.imshow(f'Depth ({self.width}x{self.height})', depth_colormap)

                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                frame_count += 1

                # Print progress every 30 frames
                if frame_count % 30 == 0:
                    print(f"Recorded {frame_count} frames (with raw depth data)...")

        except KeyboardInterrupt:
            print("\nRecording interrupted by user")

        finally:
            # Release video writers
            rgb_writer.release()
            depth_writer.release()
            cv2.destroyAllWindows()

            print(f"✓ High-resolution video saved:")
            print(f"  RGB: {rgb_video_path}")
            print(f"  Depth: {depth_video_path}")
            print(f"  Raw depth frames: {frame_count} .npy files in {self.depth_dir}")
            print(f"  Frames recorded: {frame_count}")
            print(f"  Duration: {frame_count/self.fps:.2f} seconds")

    def live_preview(self):
        """Show live preview of high-resolution RGB and depth streams"""
        print("High-resolution live preview")
        print("Controls: 'c' = capture image, 'r' = record video, 'q' = quit")
        print(f"Streaming at {self.width}x{self.height} for both RGB and depth")

        try:
            while True:
                color_image, depth_m = self._grab_frame()

                if color_image is None:
                    continue

                # Apply colormap to depth image for visualization
                depth_image = self._depth_to_uint16_mm(depth_m)
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )

                # Display images (both at same resolution)
                cv2.imshow(f'RGB Live ({self.width}x{self.height})', color_image)
                cv2.imshow(f'Depth Live ({self.width}x{self.height})', depth_colormap)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.capture_single_image()
                elif key == ord('r'):
                    cv2.destroyAllWindows()
                    duration = input("Enter recording duration in seconds (or press Enter for manual stop): ")
                    duration = int(duration) if duration.strip() else None
                    self.record_video(duration)
                    print("Resuming live preview...")

        except KeyboardInterrupt:
            print("\nLive preview interrupted")

        finally:
            cv2.destroyAllWindows()

    def get_device_info(self):
        """Print detailed device information"""
        try:
            devices = sl.Camera.get_device_list()

            if not devices:
                print("No ZED devices found")
                return

            print(f"\n=== Available Devices ===")
            for device in devices:
                print(f"Model: {device.camera_model}")
                print(f"Serial: {device.serial_number}")
                print(f"State: {device.camera_state}")

            if self.zed.is_opened():
                camera_info = self.zed.get_camera_information()
                configuration = camera_info.camera_configuration
                print(f"\n=== Active Device Configuration ===")
                print(f"Model: {camera_info.camera_model}")
                print(f"Serial: {camera_info.serial_number}")
                print(f"Firmware: {configuration.firmware_version}")
                print(f"Resolution: {configuration.resolution.width}x{configuration.resolution.height} @ {configuration.fps}fps")
                print(f"Depth mode: {self.depth_mode_name}")

        except Exception as e:
            print(f"Error getting device info: {e}")

    def cleanup(self):
        """Clean up resources"""
        self.zed.close()
        cv2.destroyAllWindows()
        print("Camera resources cleaned up")

def main():
    parser = argparse.ArgumentParser(description="ZED 2i Camera Capture - Matching RGB/Depth Resolution")
    parser.add_argument("--mode", choices=["preview", "image", "video", "info"], default="preview",
                      help="Capture mode: preview (live view), image (single capture), video (record), info (device info)")
    parser.add_argument("--duration", type=int, help="Video recording duration in seconds")
    parser.add_argument("--output-name", type=str, default=None,
                      help="Name for the output directory (default: zed_highres_<timestamp>)")

    # Resolution options (same for both RGB and depth)
    parser.add_argument("--width", type=int, default=1280, help="Requested image width for both RGB and depth (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Requested image height for both RGB and depth (default: 720)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--depth-mode", choices=["NEURAL", "ULTRA", "QUALITY", "PERFORMANCE"], default="NEURAL",
                      help="ZED depth computation mode (default: NEURAL)")

    # Preset options
    parser.add_argument("--preset", choices=["high", "medium", "low"],
                      help="Resolution preset (overrides width/height)")

    args = parser.parse_args()

    # Apply presets (mapped to native ZED resolutions)
    if args.preset == "high":
        args.width, args.height = 1920, 1080
        args.fps = 30
    elif args.preset == "medium":
        args.width, args.height = 1280, 720
        args.fps = 30
    elif args.preset == "low":
        args.width, args.height = 672, 376
        args.fps = 30

    try:
        # Initialize camera
        camera = ZED2iHighResCapture(args.width, args.height, args.fps, args.depth_mode, args.output_name)

        if args.mode == "info":
            camera.get_device_info()
        elif args.mode == "preview":
            camera.live_preview()
        elif args.mode == "image":
            camera.capture_single_image()
        elif args.mode == "video":
            camera.record_video(args.duration)

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the ZED 2i camera is connected and pyzed is installed")

    finally:
        try:
            camera.cleanup()
        except:
            pass

if __name__ == "__main__":
    main()
