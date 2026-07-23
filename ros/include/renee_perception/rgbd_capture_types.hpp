#ifndef RENEE_PERCEPTION__RGBD_CAPTURE_TYPES_HPP_
#define RENEE_PERCEPTION__RGBD_CAPTURE_TYPES_HPP_

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <builtin_interfaces/msg/time.hpp>

#include <string>

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
  sensor_msgs::msg::Image rgb_image;
  sensor_msgs::msg::Image depth_image;
  sensor_msgs::msg::CameraInfo camera_info;
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

}  // namespace renee_perception

#endif  // RENEE_PERCEPTION__RGBD_CAPTURE_TYPES_HPP_
