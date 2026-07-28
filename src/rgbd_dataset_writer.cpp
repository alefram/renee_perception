#include "renee_perception/rgbd_dataset_writer.hpp"

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <string>

namespace renee_perception
{
namespace
{

namespace fs = std::filesystem;

std::string quote(const std::string & value)
{
  std::ostringstream stream;
  stream << '"';
  for (const char character : value) {
    if (character == '"' || character == '\\') {
      stream << '\\';
    }
    stream << character;
  }
  stream << '"';
  return stream.str();
}

std::string timeStem(const builtin_interfaces::msg::Time & stamp)
{
  std::ostringstream stream;
  stream << stamp.sec << "_" << std::setw(9) << std::setfill('0') << stamp.nanosec;
  return stream.str();
}

std::size_t nextFrameIndex(const fs::path & manifest_path)
{
  std::ifstream input(manifest_path);
  if (!input.is_open()) {
    return 0;
  }

  std::size_t count = 0;
  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty()) {
      ++count;
    }
  }
  return count;
}

cv::Mat toBgr8(const sensor_msgs::msg::Image & image)
{
  return cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8)->image;
}

cv::Mat toDepthMm(const sensor_msgs::msg::Image & image)
{
  if (image.encoding == sensor_msgs::image_encodings::TYPE_16UC1 ||
    image.encoding == sensor_msgs::image_encodings::MONO16)
  {
    return cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::TYPE_16UC1)->image;
  }

  if (image.encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
    const auto depth_m = cv_bridge::toCvCopy(
      image, sensor_msgs::image_encodings::TYPE_32FC1)->image;
    cv::Mat depth_mm(depth_m.rows, depth_m.cols, CV_16UC1, cv::Scalar(0));

    for (int row = 0; row < depth_m.rows; ++row) {
      for (int col = 0; col < depth_m.cols; ++col) {
        const float meters = depth_m.at<float>(row, col);
        if (std::isfinite(meters) && meters > 0.0F) {
          const double millimeters = std::round(static_cast<double>(meters) * 1000.0);
          depth_mm.at<std::uint16_t>(row, col) = static_cast<std::uint16_t>(
            std::clamp(
              millimeters,
              1.0,
              static_cast<double>(std::numeric_limits<std::uint16_t>::max())));
        }
      }
    }
    return depth_mm;
  }

  throw std::invalid_argument("Unsupported depth encoding: " + image.encoding);
}

std::string poseMatrixJson(const geometry_msgs::msg::PoseStamped & pose)
{
  const auto & q = pose.pose.orientation;
  const auto & t = pose.pose.position;

  const double xx = q.x * q.x;
  const double yy = q.y * q.y;
  const double zz = q.z * q.z;
  const double xy = q.x * q.y;
  const double xz = q.x * q.z;
  const double yz = q.y * q.z;
  const double wx = q.w * q.x;
  const double wy = q.w * q.y;
  const double wz = q.w * q.z;

  const double r00 = 1.0 - 2.0 * (yy + zz);
  const double r01 = 2.0 * (xy - wz);
  const double r02 = 2.0 * (xz + wy);
  const double r10 = 2.0 * (xy + wz);
  const double r11 = 1.0 - 2.0 * (xx + zz);
  const double r12 = 2.0 * (yz - wx);
  const double r20 = 2.0 * (xz - wy);
  const double r21 = 2.0 * (yz + wx);
  const double r22 = 1.0 - 2.0 * (xx + yy);

  std::ostringstream stream;
  stream << std::setprecision(12)
         << "[[" << r00 << "," << r01 << "," << r02 << "," << t.x << "],"
         << "[" << r10 << "," << r11 << "," << r12 << "," << t.y << "],"
         << "[" << r20 << "," << r21 << "," << r22 << "," << t.z << "],"
         << "[0,0,0,1]]";
  return stream.str();
}

void writeIntrinsics(const fs::path & intrinsics_path, const sensor_msgs::msg::CameraInfo & info)
{
  std::ofstream output(intrinsics_path);
  if (!output.is_open()) {
    throw std::runtime_error("Failed to open intrinsics file: " + intrinsics_path.string());
  }

  output << std::setprecision(12)
         << "{\n"
         << "  \"color_camera\": {\n"
         << "    \"camera_matrix\": [["
         << info.k[0] << ", " << info.k[1] << ", " << info.k[2] << "], ["
         << info.k[3] << ", " << info.k[4] << ", " << info.k[5] << "], ["
         << info.k[6] << ", " << info.k[7] << ", " << info.k[8] << "]],\n"
         << "    \"distortion_coefficients\": [";
  for (std::size_t i = 0; i < info.d.size(); ++i) {
    output << info.d[i];
    if (i + 1 < info.d.size()) {
      output << ", ";
    }
  }
  output << "],\n"
         << "    \"width\": " << info.width << ",\n"
         << "    \"height\": " << info.height << "\n"
         << "  },\n"
         << "  \"depth_camera\": {\n"
         << "    \"camera_matrix\": [["
         << info.k[0] << ", " << info.k[1] << ", " << info.k[2] << "], ["
         << info.k[3] << ", " << info.k[4] << ", " << info.k[5] << "], ["
         << info.k[6] << ", " << info.k[7] << ", " << info.k[8] << "]],\n"
         << "    \"distortion_coefficients\": [";
  for (std::size_t i = 0; i < info.d.size(); ++i) {
    output << info.d[i];
    if (i + 1 < info.d.size()) {
      output << ", ";
    }
  }
  output << "],\n"
         << "    \"width\": " << info.width << ",\n"
         << "    \"height\": " << info.height << "\n"
         << "  }\n"
         << "}\n";
}

}  // namespace

RgbdCaptureRecord RgbdDatasetWriter::writeCapture(const RgbdCaptureData & data) const
{
  if (data.session_dir.empty()) {
    throw std::invalid_argument("session_dir cannot be empty");
  }

  const fs::path session_dir(data.session_dir);
  const fs::path rgb_dir = session_dir / "rgb";
  const fs::path depth_dir = session_dir / "depth_mm";
  const fs::path intrinsics_dir = session_dir / "intrinsics";
  fs::create_directories(rgb_dir);
  fs::create_directories(depth_dir);
  fs::create_directories(intrinsics_dir);

  const auto stamp = data.rgb_image.header.stamp;
  const fs::path manifest_path = session_dir / "frames.jsonl";
  const std::size_t frame_index = nextFrameIndex(manifest_path);
  const std::string stem = "frame_" + std::to_string(frame_index) + "_" + timeStem(stamp);

  const fs::path rgb_path = rgb_dir / (stem + ".png");
  const fs::path depth_path = depth_dir / (stem + ".png");
  const fs::path intrinsics_path = intrinsics_dir / "camera_intrinsics.json";

  if (!cv::imwrite(rgb_path.string(), toBgr8(data.rgb_image))) {
    throw std::runtime_error("Failed to write RGB image: " + rgb_path.string());
  }
  if (!cv::imwrite(depth_path.string(), toDepthMm(data.depth_image))) {
    throw std::runtime_error("Failed to write depth image: " + depth_path.string());
  }

  writeIntrinsics(intrinsics_path, data.camera_info);

  std::ofstream manifest(manifest_path, std::ios::app);
  if (!manifest.is_open()) {
    throw std::runtime_error("Failed to open metadata file: " + manifest_path.string());
  }

  const std::string rgb_rel = "rgb/" + rgb_path.filename().string();
  const std::string depth_rel = "depth_mm/" + depth_path.filename().string();
  manifest << std::setprecision(12)
           << "{"
           << "\"frame_index\":" << frame_index << ","
           << "\"waypoint_id\":" << quote(data.waypoint_id) << ","
           << "\"timestamp\":{\"sec\":" << stamp.sec << ",\"nanosec\":" << stamp.nanosec << "},"
           << "\"rgb\":" << quote(rgb_rel) << ","
           << "\"depth_mm\":" << quote(depth_rel) << ","
           << "\"robot_pose_frame\":" << quote(data.robot_pose.header.frame_id) << ","
           << "\"camera_pose_frame\":" << quote(data.camera_pose.header.frame_id) << ","
           << "\"T_world_robot\":" << poseMatrixJson(data.robot_pose) << ","
           << "\"T_world_camera\":" << poseMatrixJson(data.camera_pose)
           << "}\n";

  RgbdCaptureRecord record;
  record.waypoint_id = data.waypoint_id;
  record.timestamp = stamp;
  record.rgb_path = rgb_path.string();
  record.depth_path = depth_path.string();
  record.metadata_path = manifest_path.string();
  record.robot_pose = data.robot_pose;
  record.camera_pose = data.camera_pose;
  return record;
}

}  // namespace renee_perception
