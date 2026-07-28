#ifndef RENEE_PERCEPTION__RGBD_CAPTURE_NODE_HPP_
#define RENEE_PERCEPTION__RGBD_CAPTURE_NODE_HPP_

#include "renee_perception/rgbd_dataset_writer.hpp"
#include "renee_perception/srv/capture_rgbd.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <memory>
#include <mutex>
#include <string>

namespace renee_perception
{

class RgbdCaptureNode : public rclcpp::Node
{
public:
  explicit RgbdCaptureNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  using CaptureRGBD = renee_perception::srv::CaptureRGBD;

  void handleCaptureRequest(
    const std::shared_ptr<CaptureRGBD::Request> request,
    std::shared_ptr<CaptureRGBD::Response> response);

  void onRgbImage(sensor_msgs::msg::Image::ConstSharedPtr msg);
  void onDepthImage(sensor_msgs::msg::Image::ConstSharedPtr msg);
  void onCameraInfo(sensor_msgs::msg::CameraInfo::ConstSharedPtr msg);

  std::string rgb_topic_;
  std::string depth_topic_;
  std::string camera_info_topic_;
  std::string default_session_dir_;
  std::string world_frame_;
  std::string robot_frame_;
  std::string camera_frame_;
  double sync_tolerance_ms_;
  double tf_timeout_sec_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  rclcpp::Service<CaptureRGBD>::SharedPtr capture_service_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  RgbdDatasetWriter dataset_writer_;

  std::mutex latest_data_mutex_;
  sensor_msgs::msg::Image::ConstSharedPtr latest_rgb_;
  sensor_msgs::msg::Image::ConstSharedPtr latest_depth_;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr latest_camera_info_;
};

}  // namespace renee_perception

#endif  // RENEE_PERCEPTION__RGBD_CAPTURE_NODE_HPP_
