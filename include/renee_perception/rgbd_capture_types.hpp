#ifndef RENEE_PERCEPTION__RGBD_CAPTURE_TYPES_HPP_
#define RENEE_PERCEPTION__RGBD_CAPTURE_TYPES_HPP_

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <builtin_interfaces/msg/time.hpp>

#include <string>
#include <vector>

namespace renee_perception
{

struct RgbdCaptureRequest
{
  std::string waypoint_id;
  std::string session_dir;
};

struct RgbdCaptureData
{
  std::string waypoint_id;
  std::string session_dir;
  std::string camera_model;
  sensor_msgs::msg::Image rgb_image;
  sensor_msgs::msg::Image depth_image;
  sensor_msgs::msg::Image aligned_depth_image;
  sensor_msgs::msg::Image stereo_right_image;
  sensor_msgs::msg::Image infrared_left_image;
  sensor_msgs::msg::Image infrared_right_image;
  sensor_msgs::msg::CameraInfo color_camera_info;
  sensor_msgs::msg::CameraInfo depth_camera_info;
  sensor_msgs::msg::CameraInfo stereo_right_camera_info;
  sensor_msgs::msg::CameraInfo infrared_left_camera_info;
  sensor_msgs::msg::CameraInfo infrared_right_camera_info;
  std::string color_metadata_json;
  std::string depth_metadata_json;
  std::string infrared_left_metadata_json;
  std::string infrared_right_metadata_json;
  std::string depth_to_color_extrinsics_json;
  std::string capture_settings_json;
  double depth_units_m{0.001};
  bool simulated{false};
  std::vector<sensor_msgs::msg::Imu> camera_imu_samples;
  std::vector<nav_msgs::msg::Odometry> wheel_odometry_samples;
  std::vector<geometry_msgs::msg::PoseWithCovarianceStamped> amcl_pose_samples;
  std::vector<sensor_msgs::msg::Imu> rover_imu_samples;
  nav_msgs::msg::Odometry wheel_odometry;
  geometry_msgs::msg::PoseWithCovarianceStamped amcl_pose;
  sensor_msgs::msg::Imu rover_imu;
  bool has_wheel_odometry{false};
  bool has_amcl_pose{false};
  bool has_rover_imu{false};
  geometry_msgs::msg::TransformStamped map_to_odom;
  geometry_msgs::msg::TransformStamped odom_to_base;
  geometry_msgs::msg::TransformStamped base_to_camera;
  geometry_msgs::msg::PoseStamped robot_pose;
  geometry_msgs::msg::PoseStamped camera_pose;
};

struct RgbdCaptureRecord
{
  std::string waypoint_id;
  builtin_interfaces::msg::Time timestamp;
  std::string rgb_path;
  std::string depth_path;
  std::string metadata_path;
  geometry_msgs::msg::PoseStamped robot_pose;
  geometry_msgs::msg::PoseStamped camera_pose;
};

struct RgbdStationSummary
{
  std::string waypoint_id;
  std::string session_dir;
  std::string camera_model;
  std::size_t requested_frames{0};
  std::size_t captured_frames{0};
  bool valid{false};
  bool simulated{false};
  std::string message;
};

}  // namespace renee_perception

#endif  // RENEE_PERCEPTION__RGBD_CAPTURE_TYPES_HPP_
