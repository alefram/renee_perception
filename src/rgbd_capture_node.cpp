#include "renee_perception/rgbd_capture_node.hpp"

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/time.h>

#include <cmath>
#include <functional>
#include <stdexcept>
#include <utility>

namespace renee_perception
{
namespace
{

geometry_msgs::msg::PoseStamped transformToPoseStamped(
  const geometry_msgs::msg::TransformStamped & transform)
{
  geometry_msgs::msg::PoseStamped pose;
  pose.header = transform.header;
  pose.pose.position.x = transform.transform.translation.x;
  pose.pose.position.y = transform.transform.translation.y;
  pose.pose.position.z = transform.transform.translation.z;
  pose.pose.orientation = transform.transform.rotation;
  return pose;
}

bool hasNonZeroStamp(const builtin_interfaces::msg::Time & stamp)
{
  return stamp.sec != 0 || stamp.nanosec != 0;
}

double stampDeltaMs(
  const builtin_interfaces::msg::Time & lhs,
  const builtin_interfaces::msg::Time & rhs)
{
  const rclcpp::Time lhs_time(lhs);
  const rclcpp::Time rhs_time(rhs);
  return std::abs((lhs_time - rhs_time).nanoseconds()) / 1.0e6;
}

}  // namespace

RgbdCaptureNode::RgbdCaptureNode(const rclcpp::NodeOptions & options)
: Node("rgbd_capture_node", options)
{
  rgb_topic_ = this->declare_parameter<std::string>("rgb_topic", "/camera/color/image_raw");
  depth_topic_ = this->declare_parameter<std::string>("depth_topic", "/camera/depth/image_raw");
  camera_info_topic_ = this->declare_parameter<std::string>(
    "camera_info_topic", "/camera/color/camera_info");
  default_session_dir_ = this->declare_parameter<std::string>(
    "session_dir", "/tmp/renee_scan_session");
  world_frame_ = this->declare_parameter<std::string>("world_frame", "world");
  robot_frame_ = this->declare_parameter<std::string>("robot_frame", "robot_base_link");
  camera_frame_ = this->declare_parameter<std::string>("camera_frame", "camera_optical_frame");
  sync_tolerance_ms_ = this->declare_parameter<double>("sync_tolerance_ms", 50.0);
  tf_timeout_sec_ = this->declare_parameter<double>("tf_timeout_sec", 0.5);

  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  rgb_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    rgb_topic_, rclcpp::SensorDataQoS(),
    std::bind(&RgbdCaptureNode::onRgbImage, this, std::placeholders::_1));
  depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    depth_topic_, rclcpp::SensorDataQoS(),
    std::bind(&RgbdCaptureNode::onDepthImage, this, std::placeholders::_1));
  camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
    camera_info_topic_, rclcpp::SensorDataQoS(),
    std::bind(&RgbdCaptureNode::onCameraInfo, this, std::placeholders::_1));

  capture_service_ = this->create_service<CaptureRGBD>(
    "~/capture_rgbd",
    std::bind(
      &RgbdCaptureNode::handleCaptureRequest,
      this,
      std::placeholders::_1,
      std::placeholders::_2));

  RCLCPP_INFO(this->get_logger(), "RGB-D capture node initialized.");
  RCLCPP_INFO(this->get_logger(), "RGB topic: %s", rgb_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "Depth topic: %s", depth_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "Camera info topic: %s", camera_info_topic_.c_str());
}

void RgbdCaptureNode::handleCaptureRequest(
  const std::shared_ptr<CaptureRGBD::Request> request,
  std::shared_ptr<CaptureRGBD::Response> response)
{
  sensor_msgs::msg::Image::ConstSharedPtr rgb;
  sensor_msgs::msg::Image::ConstSharedPtr depth;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info;

  {
    std::lock_guard<std::mutex> lock(latest_data_mutex_);
    rgb = latest_rgb_;
    depth = latest_depth_;
    camera_info = latest_camera_info_;
  }

  if (!rgb || !depth || !camera_info) {
    response->success = false;
    response->message = "Cannot capture RGB-D frame: missing RGB, depth, or camera_info";
    return;
  }

  if (hasNonZeroStamp(rgb->header.stamp) && hasNonZeroStamp(depth->header.stamp)) {
    const double rgb_depth_delta_ms = stampDeltaMs(rgb->header.stamp, depth->header.stamp);
    if (rgb_depth_delta_ms > sync_tolerance_ms_) {
      response->success = false;
      response->message = "Cannot capture RGB-D frame: RGB/depth timestamps differ by " +
        std::to_string(rgb_depth_delta_ms) + " ms";
      return;
    }
  }

  try {
    const auto robot_transform = tf_buffer_->lookupTransform(
      world_frame_,
      robot_frame_,
      tf2::TimePointZero,
      tf2::durationFromSec(tf_timeout_sec_));
    const auto camera_transform = tf_buffer_->lookupTransform(
      world_frame_,
      camera_frame_,
      tf2::TimePointZero,
      tf2::durationFromSec(tf_timeout_sec_));

    RgbdCaptureData data;
    data.waypoint_id = request->waypoint_id;
    data.session_dir = request->session_dir.empty() ? default_session_dir_ : request->session_dir;
    data.rgb_image = *rgb;
    data.depth_image = *depth;
    data.camera_info = *camera_info;
    data.robot_pose = transformToPoseStamped(robot_transform);
    data.camera_pose = transformToPoseStamped(camera_transform);
    data.robot_pose.header.stamp = rgb->header.stamp;
    data.camera_pose.header.stamp = rgb->header.stamp;

    const auto record = dataset_writer_.writeCapture(data);
    response->success = true;
    response->message = "Captured RGB-D frame";
    response->frame.waypoint_id = record.waypoint_id;
    response->frame.timestamp = record.timestamp;
    response->frame.rgb_path = record.rgb_path;
    response->frame.depth_path = record.depth_path;
    response->frame.metadata_path = record.metadata_path;
    response->frame.robot_pose = record.robot_pose;
    response->frame.camera_pose = record.camera_pose;
  } catch (const std::exception & exception) {
    response->success = false;
    response->message = exception.what();
  }
}

void RgbdCaptureNode::onRgbImage(sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(latest_data_mutex_);
  latest_rgb_ = std::move(msg);
}

void RgbdCaptureNode::onDepthImage(sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(latest_data_mutex_);
  latest_depth_ = std::move(msg);
}

void RgbdCaptureNode::onCameraInfo(sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(latest_data_mutex_);
  latest_camera_info_ = std::move(msg);
}

}  // namespace renee_perception
