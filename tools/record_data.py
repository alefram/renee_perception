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
    rgb/          rgb_<timestamp>.png (image mode); frame_<i>.png + rgb_video_<ts>.mp4 (video mode)
    depth_mm/     frame_<i>.png (16-bit mm, video mode only)
    depth/        depth_<timestamp>.png (16-bit mm), depth_raw_<timestamp|frame>.npy
                  (float32 meters), depth_video_<timestamp>.mp4 (colormap, video mode)
    intrinsics/   camera_intrinsics.json
    frames.jsonl  video mode only: per-frame {frame_index, rgb, depth_mm} manifest --
                  this + rgb/ + depth_mm/ is exactly what
                  renception.io.session_from_extracted reads, so a video-mode
                  session is immediately usable with tools/build_pointcloud.py,
                  no tools/extract_video_frames.py step needed.

Examples:
    python3 tools/record_data.py --mode image
    python3 tools/record_data.py --mode video --duration 10
    python3 tools/record_data.py --preset high --depth-mode ULTRA
    python3 tools/record_data.py --mode video --output-name kitchen_scan_01

    # then, directly (no tools/extract_video_frames.py needed for video-mode captures):
    python3 tools/build_pointcloud.py kitchen_scan_01 --intrinsics kitchen_scan_01/intrinsics/camera_intrinsics.json
"""

import numpy as np
import cv2
import os
from datetime import datetime
import json
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from renception.drivers.zed import ZedCamera, closest_resolution_name  # noqa: E402


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
        self.camera = ZedCamera()

        resolution_name = closest_resolution_name(width, height)
        print(f"Configuring stream:")
        print(f"  Requested: {width}x{height} at {fps} FPS")
        print(f"  Using ZED preset: {resolution_name}")

        if not self.open_camera(resolution_name, fps):
            print("Trying fallback resolutions...")
            self.try_fallback_resolutions()
        else:
            print(f"✓ Camera opened successfully at {self.camera.width}x{self.camera.height} @ {self.camera.fps}fps")
            self.create_directories(output_name)

        # Get camera intrinsics
        self.get_camera_intrinsics()

    def open_camera(self, resolution_name, fps):
        """Attempt to open the ZED camera with the given resolution/fps"""
        ok, status = self.camera.open(resolution_name, fps, self.depth_mode_name)
        if ok:
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
        self.depth_mm_dir = os.path.join(self.base_dir, "depth_mm")
        self.intrinsics_dir = os.path.join(self.base_dir, "intrinsics")

        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.depth_mm_dir, exist_ok=True)
        os.makedirs(self.intrinsics_dir, exist_ok=True)

        print(f"Created directories in: {self.base_dir}")

    def get_camera_intrinsics(self):
        """Get camera intrinsic parameters"""
        # Left (color) camera intrinsics
        self.color_camera_matrix, self.color_dist_coeffs = self.camera.get_intrinsics()

        # Depth is computed on the rectified left image, so it shares the color camera's intrinsics
        self.depth_camera_matrix, self.depth_dist_coeffs = self.color_camera_matrix, self.color_dist_coeffs

        # Save intrinsics to file
        self.save_intrinsics()

        print("Camera intrinsics loaded and saved!")
        fx, fy = self.color_camera_matrix[0, 0], self.color_camera_matrix[1, 1]
        print(f"Left camera focal length: fx={fx:.2f}, fy={fy:.2f}")

    def save_intrinsics(self):
        """Save camera intrinsic parameters to JSON file"""
        intrinsics_data = {
            "color_camera": {
                "camera_matrix": self.color_camera_matrix.tolist(),
                "distortion_coefficients": self.color_dist_coeffs.tolist(),
                "width": self.camera.width,
                "height": self.camera.height
            },
            "depth_camera": {
                "camera_matrix": self.depth_camera_matrix.tolist(),
                "distortion_coefficients": self.depth_dist_coeffs.tolist(),
                "width": self.camera.width,
                "height": self.camera.height
            },
            "stream_info": {
                "resolution": f"{self.camera.width}x{self.camera.height}",
                "fps": self.camera.fps,
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
        ok, _ = self.camera.grab()
        if not ok:
            return None, None

        color_bgra, depth_m = self.camera.retrieve_rgb_depth()
        color_bgr = cv2.cvtColor(color_bgra, cv2.COLOR_BGRA2BGR)
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
            print(f"  Both streams at matching {self.camera.width}x{self.camera.height} resolution")
            return True

        except Exception as e:
            print(f"Error capturing image: {e}")
            return False

    def record_video(self, duration_seconds=None):
        """
        Record high-resolution video with both RGB and depth data.

        Besides the preview video/raw-depth files, writes per-frame
        rgb/frame_<i>.png + depth_mm/frame_<i>.png + frames.jsonl directly in
        self.base_dir -- the same layout extract_video_frames.py produces from
        a video, so this session is immediately usable with
        tools/build_pointcloud.py (no extraction step).

        Args:
            duration_seconds (int): Recording duration. If None, record until 'q' is pressed
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup video writers with appropriate codecs for high resolution
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Better codec for high resolution

        rgb_video_path = os.path.join(self.rgb_dir, f"rgb_video_{timestamp}.mp4")
        depth_video_path = os.path.join(self.depth_dir, f"depth_video_{timestamp}.mp4")

        # Use same resolution for both RGB and depth video
        rgb_writer = cv2.VideoWriter(rgb_video_path, fourcc, self.camera.fps,
                                   (self.camera.width, self.camera.height))

        depth_writer = cv2.VideoWriter(depth_video_path, fourcc, self.camera.fps,
                                     (self.camera.width, self.camera.height))

        manifest_path = os.path.join(self.base_dir, "frames.jsonl")
        manifest_file = open(manifest_path, "w", encoding="utf-8")

        print(f"Recording high-resolution video...")
        print(f"  Resolution: {self.camera.width}x{self.camera.height} (both RGB and depth)")
        print(f"  Raw depth data will be saved for each frame")
        print(f"  {'Press q to stop' if duration_seconds is None else f'Recording for {duration_seconds} seconds'}")

        frame_count = 0
        max_frames = duration_seconds * self.camera.fps if duration_seconds else float('inf')

        try:
            while frame_count < max_frames:
                color_image, depth_m = self._grab_frame()

                if color_image is None:
                    continue

                # Generate frame-specific timestamp
                frame_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                stem = f"frame_{frame_count:06d}"

                # Save raw depth data as numpy array for precise measurements
                depth_raw_filename = os.path.join(self.depth_dir, f"depth_raw_frame_{frame_count:06d}_{frame_timestamp}.npy")
                np.save(depth_raw_filename, depth_m)

                # 16-bit mm depth (what build_pointcloud.py's TSDF fusion reads)
                depth_image = self._depth_to_uint16_mm(depth_m)

                # Per-frame rgb + depth_mm PNGs + manifest entry -> ready for
                # renception.io.session_from_extracted, no extraction step needed
                rgb_png_path = os.path.join(self.rgb_dir, f"{stem}.png")
                depth_mm_path = os.path.join(self.depth_mm_dir, f"{stem}.png")
                cv2.imwrite(rgb_png_path, color_image)
                cv2.imwrite(depth_mm_path, depth_image)
                manifest_file.write(json.dumps({
                    "frame_index": frame_count,
                    "rgb": f"rgb/{stem}.png",
                    "depth_mm": f"depth_mm/{stem}.png",
                }) + "\n")

                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )

                # Write preview videos
                rgb_writer.write(color_image)
                depth_writer.write(depth_colormap)

                # Display frames (no scaling needed since both streams are same resolution)
                cv2.imshow(f'RGB ({self.camera.width}x{self.camera.height})', color_image)
                cv2.imshow(f'Depth ({self.camera.width}x{self.camera.height})', depth_colormap)

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
            manifest_file.close()
            cv2.destroyAllWindows()

            print(f"✓ High-resolution video saved:")
            print(f"  RGB: {rgb_video_path}")
            print(f"  Depth: {depth_video_path}")
            print(f"  Raw depth frames: {frame_count} .npy files in {self.depth_dir}")
            print(f"  Per-frame rgb/depth_mm PNGs + {manifest_path} ready for build_pointcloud.py")
            print(f"  Frames recorded: {frame_count}")
            print(f"  Duration: {frame_count/self.camera.fps:.2f} seconds")

    def live_preview(self):
        """Show live preview of high-resolution RGB and depth streams"""
        print("High-resolution live preview")
        print("Controls: 'c' = capture image, 'r' = record video, 'q' = quit")
        print(f"Streaming at {self.camera.width}x{self.camera.height} for both RGB and depth")

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
                cv2.imshow(f'RGB Live ({self.camera.width}x{self.camera.height})', color_image)
                cv2.imshow(f'Depth Live ({self.camera.width}x{self.camera.height})', depth_colormap)

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
            devices = self.camera.list_devices()

            if not devices:
                print("No ZED devices found")
                return

            print(f"\n=== Available Devices ===")
            for device in devices:
                print(f"Model: {device['model']}")
                print(f"Serial: {device['serial_number']}")
                print(f"State: {device['state']}")

            if self.camera.is_opened():
                info = self.camera.get_active_info()
                print(f"\n=== Active Device Configuration ===")
                print(f"Model: {info['model']}")
                print(f"Serial: {info['serial_number']}")
                print(f"Firmware: {info['firmware_version']}")
                print(f"Resolution: {info['width']}x{info['height']} @ {info['fps']}fps")
                print(f"Depth mode: {self.depth_mode_name}")

        except Exception as e:
            print(f"Error getting device info: {e}")

    def cleanup(self):
        """Clean up resources"""
        self.camera.close()
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
