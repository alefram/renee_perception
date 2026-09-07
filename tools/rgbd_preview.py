#!/usr/bin/env python3
"""Large RGB/depth preview with center distance and camera orientation."""
import math
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener


def quaternion_to_rpy(q):
    sinr = 2.0 * (q.w * q.x + q.y * q.z)
    cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny, cosy)
    return tuple(math.degrees(v) for v in (roll, pitch, yaw))


class RgbdPreview(Node):
    def __init__(self):
        super().__init__('rgbd_preview')
        self.rgb_topic = self.declare_parameter('rgb_topic', '/robot/arm_rgbd_camera/color/image_raw').value
        self.depth_topic = self.declare_parameter('depth_topic', '/robot/arm_rgbd_camera/aligned_depth_to_color/image_raw').value
        self.camera_frame = self.declare_parameter('camera_frame', 'robot_arm_rgbd_camera_color_optical_frame').value
        self.reference_frame = self.declare_parameter('reference_frame', 'robot_base_link').value
        self.max_depth_m = float(self.declare_parameter('max_depth_m', 3.0).value)
        self.window_width = int(self.declare_parameter('window_width', 1600).value)
        self.window_height = int(self.declare_parameter('window_height', 700).value)
        self.bridge = CvBridge()
        self.rgb = None
        self.depth = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(Image, self.rgb_topic, self.on_rgb, qos_profile_sensor_data)
        self.create_subscription(Image, self.depth_topic, self.on_depth, qos_profile_sensor_data)
        self.create_timer(0.05, self.render)
        cv2.namedWindow('RealSense RGB-D preview', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('RealSense RGB-D preview', self.window_width, self.window_height)

    def on_rgb(self, message):
        try:
            self.rgb = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warning(f'RGB conversion failed: {exc}')

    def on_depth(self, message):
        try:
            depth = np.asarray(self.bridge.imgmsg_to_cv2(message, desired_encoding='passthrough'))
            self.depth = depth.astype(np.float32) * (0.001 if message.encoding in ('16UC1', 'mono16') else 1.0)
        except Exception as exc:
            self.get_logger().warning(f'Depth conversion failed: {exc}')

    def distance(self):
        h, w = self.depth.shape[:2]
        cx, cy = w // 2, h // 2
        patch = self.depth[max(0, cy - 2):cy + 3, max(0, cx - 2):cx + 3]
        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        return float(np.median(valid)) if valid.size else float('nan')

    def orientation(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.reference_frame, self.camera_frame, rclpy.time.Time())
            r, p, y = quaternion_to_rpy(tf.transform.rotation)
            return f'Roll {r:+6.1f} deg  Pitch {p:+6.1f} deg  Yaw {y:+6.1f} deg'
        except Exception:
            return 'Roll/Pitch/Yaw: TF unavailable'

    def render(self):
        if self.rgb is None or self.depth is None:
            return
        rgb = self.rgb.copy()
        depth = self.depth
        if depth.shape[:2] != rgb.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        valid = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth_u8 = np.clip(valid / self.max_depth_m * 255.0, 0, 255).astype(np.uint8)
        depth_view = cv2.applyColorMap(255 - depth_u8, cv2.COLORMAP_TURBO)
        cx, cy = rgb.shape[1] // 2, rgb.shape[0] // 2
        for image in (rgb, depth_view):
            cv2.drawMarker(image, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
        distance = self.distance()
        distance_text = f'Center depth: {distance:.3f} m' if math.isfinite(distance) else 'Center depth: invalid'
        orientation_text = self.orientation()
        cv2.putText(rgb, distance_text, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(rgb, orientation_text, (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(depth_view, distance_text, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        target = (max(1, self.window_width // 2), max(1, self.window_height - 70))
        combined = np.hstack((cv2.resize(rgb, target, interpolation=cv2.INTER_AREA), cv2.resize(depth_view, target, interpolation=cv2.INTER_NEAREST)))
        cv2.imshow('RealSense RGB-D preview', combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main():
    rclpy.init()
    node = RgbdPreview()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
