#include "renee_perception/rgbd_dataset_writer.hpp"

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace renee_perception
{
namespace
{
namespace fs = std::filesystem;

std::string quote(const std::string & value)
{
  std::ostringstream out;
  out << '"';
  for (const char c : value) {
    if (c == '"' || c == '\\') {out << '\\';}
    if (c == '\n') {out << "\\n";} else {out << c;}
  }
  out << '"';
  return out.str();
}

std::string timeJson(const builtin_interfaces::msg::Time & stamp)
{
  std::ostringstream out;
  out << "{\"sec\":" << stamp.sec << ",\"nanosec\":" << stamp.nanosec << "}";
  return out.str();
}

std::string timeStem(const builtin_interfaces::msg::Time & stamp)
{
  std::ostringstream out;
  out << stamp.sec << "_" << std::setw(9) << std::setfill('0') << stamp.nanosec;
  return out.str();
}

std::size_t nextFrameIndex(const fs::path & manifest_path)
{
  std::ifstream input(manifest_path);
  std::size_t count = 0;
  std::string line;
  while (std::getline(input, line)) {if (!line.empty()) {++count;}}
  return count;
}

cv::Mat toDepth16(const sensor_msgs::msg::Image & image)
{
  std::fprintf(
    stderr,
    "[rgbd_writer][TRACE] toDepth16: encoding='%s' width=%u height=%u step=%u data.size()=%zu is_bigendian=%d\n",
    image.encoding.c_str(), image.width, image.height, image.step, image.data.size(),
    static_cast<int>(image.is_bigendian));
  std::fflush(stderr);
  if (image.encoding == sensor_msgs::image_encodings::TYPE_16UC1 ||
    image.encoding == sensor_msgs::image_encodings::MONO16)
  {
    std::fprintf(stderr, "[rgbd_writer][TRACE] toDepth16: calling cv_bridge::toCvCopy (16UC1 branch)\n");
    std::fflush(stderr);
    const auto result = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::TYPE_16UC1)->image;
    std::fprintf(stderr, "[rgbd_writer][TRACE] toDepth16: cv_bridge::toCvCopy returned rows=%d cols=%d\n",
      result.rows, result.cols);
    std::fflush(stderr);
    return result;
  }
  if (image.encoding != sensor_msgs::image_encodings::TYPE_32FC1) {
    throw std::invalid_argument("Unsupported depth encoding: " + image.encoding);
  }
  std::fprintf(stderr, "[rgbd_writer][TRACE] toDepth16: calling cv_bridge::toCvCopy (32FC1 branch)\n");
  std::fflush(stderr);
  const auto metres = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::TYPE_32FC1)->image;
  std::fprintf(stderr, "[rgbd_writer][TRACE] toDepth16: 32FC1 toCvCopy returned rows=%d cols=%d, converting to mm\n",
    metres.rows, metres.cols);
  std::fflush(stderr);
  cv::Mat millimetres(metres.rows, metres.cols, CV_16UC1, cv::Scalar(0));
  for (int row = 0; row < metres.rows; ++row) {
    for (int col = 0; col < metres.cols; ++col) {
      const float value = metres.at<float>(row, col);
      if (std::isfinite(value) && value > 0.0F) {
        millimetres.at<std::uint16_t>(row, col) = static_cast<std::uint16_t>(
          std::clamp(std::round(static_cast<double>(value) * 1000.0), 1.0, 65535.0));
      }
    }
  }
  std::fprintf(stderr, "[rgbd_writer][TRACE] toDepth16: 32FC1->mm conversion done\n");
  std::fflush(stderr);
  return millimetres;
}

cv::Mat toDepthMillimetres(const sensor_msgs::msg::Image & image, double depth_units_m)
{
  if (image.encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
    return toDepth16(image);
  }
  const cv::Mat raw = toDepth16(image);
  if (std::abs(depth_units_m - 0.001) < 1.0e-12) {return raw;}
  cv::Mat millimetres(raw.rows, raw.cols, CV_16UC1, cv::Scalar(0));
  const double scale = depth_units_m * 1000.0;
  for (int row = 0; row < raw.rows; ++row) {
    for (int col = 0; col < raw.cols; ++col) {
      const auto value = raw.at<std::uint16_t>(row, col);
      if (value != 0) {
        millimetres.at<std::uint16_t>(row, col) = static_cast<std::uint16_t>(
          std::clamp(std::round(value * scale), 1.0, 65535.0));
      }
    }
  }
  return millimetres;
}

bool hasImage(const sensor_msgs::msg::Image & image) {return !image.data.empty();}

void writeImage(const fs::path & path, const cv::Mat & image)
{
  if (!cv::imwrite(path.string(), image)) {
    throw std::runtime_error("Failed to write image: " + path.string());
  }
}

std::string matrixJson(const geometry_msgs::msg::Transform & transform)
{
  const auto & q = transform.rotation;
  const auto & t = transform.translation;
  const double xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
  const double xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
  const double wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
  std::ostringstream out;
  out << std::setprecision(12)
      << "[[" << 1.0 - 2.0 * (yy + zz) << "," << 2.0 * (xy - wz) << "," <<
    2.0 * (xz + wy) << "," << t.x << "],[" << 2.0 * (xy + wz) << "," <<
    1.0 - 2.0 * (xx + zz) << "," << 2.0 * (yz - wx) << "," << t.y << "],[" <<
    2.0 * (xz - wy) << "," << 2.0 * (yz + wx) << "," <<
    1.0 - 2.0 * (xx + yy) << "," << t.z << "],[0,0,0,1]]";
  return out.str();
}

std::string poseMatrixJson(const geometry_msgs::msg::PoseStamped & pose)
{
  geometry_msgs::msg::Transform transform;
  transform.translation.x = pose.pose.position.x;
  transform.translation.y = pose.pose.position.y;
  transform.translation.z = pose.pose.position.z;
  transform.rotation = pose.pose.orientation;
  return matrixJson(transform);
}

std::string transformJson(const geometry_msgs::msg::TransformStamped & transform)
{
  std::ostringstream out;
  out << "{\"parent\":" << quote(transform.header.frame_id) << ",\"child\":" <<
    quote(transform.child_frame_id) << ",\"timestamp\":" << timeJson(transform.header.stamp) <<
    ",\"matrix\":" << matrixJson(transform.transform) << "}";
  return out.str();
}

std::string cameraInfoJson(const sensor_msgs::msg::CameraInfo & info)
{
  std::ostringstream out;
  const double hfov = info.k[0] > 0.0 ? 2.0 * std::atan2(info.width, 2.0 * info.k[0]) : 0.0;
  const double vfov = info.k[4] > 0.0 ? 2.0 * std::atan2(info.height, 2.0 * info.k[4]) : 0.0;
  out << std::setprecision(12) << "{\"frame_id\":" << quote(info.header.frame_id) <<
    ",\"width\":" << info.width << ",\"height\":" << info.height <<
    ",\"fx\":" << info.k[0] << ",\"fy\":" << info.k[4] <<
    ",\"cx\":" << info.k[2] << ",\"cy\":" << info.k[5] <<
    ",\"distortion_model\":" << quote(info.distortion_model) << ",\"distortion\":[";
  for (std::size_t i = 0; i < info.d.size(); ++i) {if (i) {out << ',';} out << info.d[i];}
  out << "],\"camera_matrix\":[[" << info.k[0] << ',' << info.k[1] << ',' << info.k[2] <<
    "],[" << info.k[3] << ',' << info.k[4] << ',' << info.k[5] << "],[" << info.k[6] <<
    ',' << info.k[7] << ',' << info.k[8] << "]],\"rectification_matrix\":[[" <<
    info.r[0] << ',' << info.r[1] << ',' << info.r[2] << "],[" << info.r[3] << ',' <<
    info.r[4] << ',' << info.r[5] << "],[" << info.r[6] << ',' << info.r[7] << ',' <<
    info.r[8] << "]],\"projection_matrix\":[[" << info.p[0] << ',' << info.p[1] << ',' <<
    info.p[2] << ',' << info.p[3] << "],[" << info.p[4] << ',' << info.p[5] << ',' <<
    info.p[6] << ',' << info.p[7] << "],[" << info.p[8] << ',' << info.p[9] << ',' <<
    info.p[10] << ',' << info.p[11] << "]],\"fov_rad\":{\"horizontal\":" << hfov <<
    ",\"vertical\":" << vfov << "}}";
  return out.str();
}

std::string vector3Json(const geometry_msgs::msg::Vector3 & value)
{
  std::ostringstream out;
  out << std::setprecision(12) << "[" << value.x << ',' << value.y << ',' << value.z << ']';
  return out.str();
}

std::string imuJson(const sensor_msgs::msg::Imu & imu)
{
  std::ostringstream out;
  out << "{\"timestamp\":" << timeJson(imu.header.stamp) << ",\"frame_id\":" <<
    quote(imu.header.frame_id) << ",\"angular_velocity\":" << vector3Json(imu.angular_velocity) <<
    ",\"linear_acceleration\":" << vector3Json(imu.linear_acceleration) << "}";
  return out.str();
}

std::string odomJson(const nav_msgs::msg::Odometry & odom)
{
  std::ostringstream out;
  out << std::setprecision(12) << "{\"timestamp\":" << timeJson(odom.header.stamp) <<
    ",\"frame_id\":" << quote(odom.header.frame_id) << ",\"child_frame_id\":" <<
    quote(odom.child_frame_id) << ",\"pose\":[" << odom.pose.pose.position.x << ',' <<
    odom.pose.pose.position.y << ',' << odom.pose.pose.position.z << ',' <<
    odom.pose.pose.orientation.x << ',' << odom.pose.pose.orientation.y << ',' <<
    odom.pose.pose.orientation.z << ',' << odom.pose.pose.orientation.w << "],\"twist_linear\":" <<
    vector3Json(odom.twist.twist.linear) << ",\"twist_angular\":" <<
    vector3Json(odom.twist.twist.angular) << ",\"pose_covariance\":[";
  for (std::size_t i = 0; i < odom.pose.covariance.size(); ++i) {if (i) {out << ',';} out << odom.pose.covariance[i];}
  out << "]}";
  return out.str();
}

std::string amclJson(const geometry_msgs::msg::PoseWithCovarianceStamped & pose)
{
  std::ostringstream out;
  out << std::setprecision(12) << "{\"timestamp\":" << timeJson(pose.header.stamp) <<
    ",\"frame_id\":" << quote(pose.header.frame_id) << ",\"pose\":[" <<
    pose.pose.pose.position.x << ',' << pose.pose.pose.position.y << ',' << pose.pose.pose.position.z <<
    ',' << pose.pose.pose.orientation.x << ',' << pose.pose.pose.orientation.y << ',' <<
    pose.pose.pose.orientation.z << ',' << pose.pose.pose.orientation.w << "],\"covariance\":[";
  for (std::size_t i = 0; i < pose.pose.covariance.size(); ++i) {if (i) {out << ',';} out << pose.pose.covariance[i];}
  out << "]}";
  return out.str();
}

void appendLine(const fs::path & path, const std::string & value)
{
  std::ofstream output(path, std::ios::app);
  if (!output) {throw std::runtime_error("Failed to open metadata file: " + path.string());}
  output << value << '\n';
}

std::string safeStem(const std::string & value)
{
  std::string result = value;
  for (char & c : result) {
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_') {c = '_';}
  }
  return result.empty() ? "station" : result;
}

void updateSession(
  const fs::path & root, const sensor_msgs::msg::CameraInfo & info,
  const std::string & camera_model)
{
  std::ifstream frames(root / "frames.jsonl");
  std::vector<std::string> keyframes;
  std::string line;
  const std::string marker = "\"session_keyframe\":";
  while (std::getline(frames, line)) {
    const auto begin = line.find(marker);
    if (begin != std::string::npos && line.size() > begin + marker.size() + 1) {
      keyframes.push_back(line.substr(begin + marker.size(), line.size() - begin - marker.size() - 1));
    }
  }
  std::ofstream session(root / "session.json");
  session << "{\n  \"session_dir\": \".\",\n  \"keyframes\": [\n";
  for (std::size_t i = 0; i < keyframes.size(); ++i) {
    session << "    " << keyframes[i] << (i + 1 < keyframes.size() ? "," : "") << '\n';
  }
  session << "  ],\n  \"cloud_path\": null,\n  \"mesh_path\": null,\n  \"meta\": " <<
    "{\"schema_version\":3,\"camera_model\":" << quote(camera_model) <<
    ",\"color_calibration\":" << cameraInfoJson(info) << "}\n}\n";
}

}  // namespace

RgbdCaptureRecord RgbdDatasetWriter::writeCapture(const RgbdCaptureData & data) const
{
  std::fprintf(stderr, "[rgbd_writer][TRACE] writeCapture: enter, session_dir='%s'\n", data.session_dir.c_str());
  std::fflush(stderr);
  if (data.session_dir.empty() || data.rgb_image.data.empty() || data.depth_image.data.empty()) {
    throw std::invalid_argument("session_dir, RGB and native depth are required");
  }
  const fs::path root(data.session_dir);
  for (const char * directory : {"rgb", "depth_mm", "depth_raw_16", "depth_aligned_16",
      "stereo_right", "infrared_left", "infrared_right", "intrinsics"})
  {
    fs::create_directories(root / directory);
  }
  const auto stamp = data.rgb_image.header.stamp;
  const fs::path frames_path = root / "frames.jsonl";
  const std::size_t index = nextFrameIndex(frames_path);
  const std::string stem = "frame_" + std::to_string(index) + "_" + timeStem(stamp);
  const fs::path rgb_path = root / "rgb" / (stem + ".png");
  const fs::path raw_path = root / "depth_raw_16" / (stem + ".png");
  const fs::path aligned_path = root / "depth_aligned_16" / (stem + ".png");
  const fs::path depth_mm_path = root / "depth_mm" / (stem + ".png");
  const fs::path stereo_right_path = root / "stereo_right" / (stem + ".png");
  const fs::path ir_left_path = root / "infrared_left" / (stem + ".png");
  const fs::path ir_right_path = root / "infrared_right" / (stem + ".png");

  std::fprintf(
    stderr, "[rgbd_writer][TRACE] writeCapture: rgb encoding='%s' %ux%u step=%u data.size()=%zu\n",
    data.rgb_image.encoding.c_str(), data.rgb_image.width, data.rgb_image.height, data.rgb_image.step,
    data.rgb_image.data.size());
  std::fflush(stderr);
  std::fprintf(stderr, "[rgbd_writer][TRACE] writeCapture: cv_bridge BGR8 conversion for rgb\n");
  std::fflush(stderr);
  writeImage(rgb_path, cv_bridge::toCvCopy(data.rgb_image, sensor_msgs::image_encodings::BGR8)->image);
  std::fprintf(stderr, "[rgbd_writer][TRACE] writeCapture: rgb written to %s\n", rgb_path.string().c_str());
  std::fflush(stderr);

  std::fprintf(stderr, "[rgbd_writer][TRACE] writeCapture: writing raw depth (toDepth16 on native depth_image)\n");
  std::fflush(stderr);
  writeImage(raw_path, toDepth16(data.depth_image));
  std::fprintf(stderr, "[rgbd_writer][TRACE] writeCapture: raw depth written to %s\n", raw_path.string().c_str());
  std::fflush(stderr);

  const auto & reconstruction_depth = hasImage(data.aligned_depth_image) ? data.aligned_depth_image : data.depth_image;
  std::fprintf(
    stderr, "[rgbd_writer][TRACE] writeCapture: writing depth_mm (source=%s, depth_units_m=%.6f)\n",
    hasImage(data.aligned_depth_image) ? "aligned" : "native", data.depth_units_m);
  std::fflush(stderr);
  writeImage(depth_mm_path, toDepthMillimetres(reconstruction_depth, data.depth_units_m));
  std::fprintf(stderr, "[rgbd_writer][TRACE] writeCapture: depth_mm written to %s\n", depth_mm_path.string().c_str());
  std::fflush(stderr);

  std::string aligned_relative;
  if (hasImage(data.aligned_depth_image)) {
    std::fprintf(stderr, "[rgbd_writer][TRACE] writeCapture: writing aligned depth (toDepth16 on aligned_depth_image)\n");
    std::fflush(stderr);
    writeImage(aligned_path, toDepth16(data.aligned_depth_image));
    aligned_relative = "depth_aligned_16/" + aligned_path.filename().string();
    std::fprintf(stderr, "[rgbd_writer][TRACE] writeCapture: aligned depth written to %s\n", aligned_path.string().c_str());
    std::fflush(stderr);
  }
  if (hasImage(data.stereo_right_image)) {
    writeImage(
      stereo_right_path,
      cv_bridge::toCvCopy(
        data.stereo_right_image, sensor_msgs::image_encodings::BGR8)->image);
  }
  if (hasImage(data.infrared_left_image)) {
    std::fprintf(
      stderr, "[rgbd_writer][TRACE] writeCapture: writing infrared_left (encoding='%s' %ux%u)\n",
      data.infrared_left_image.encoding.c_str(), data.infrared_left_image.width, data.infrared_left_image.height);
    std::fflush(stderr);
    writeImage(ir_left_path, cv_bridge::toCvCopy(data.infrared_left_image, sensor_msgs::image_encodings::MONO8)->image);
    std::fprintf(stderr, "[rgbd_writer][TRACE] writeCapture: infrared_left written to %s\n", ir_left_path.string().c_str());
    std::fflush(stderr);
  }
  if (hasImage(data.infrared_right_image)) {
    std::fprintf(
      stderr, "[rgbd_writer][TRACE] writeCapture: writing infrared_right (encoding='%s' %ux%u)\n",
      data.infrared_right_image.encoding.c_str(), data.infrared_right_image.width, data.infrared_right_image.height);
    std::fflush(stderr);
    writeImage(ir_right_path, cv_bridge::toCvCopy(data.infrared_right_image, sensor_msgs::image_encodings::MONO8)->image);
    std::fprintf(stderr, "[rgbd_writer][TRACE] writeCapture: infrared_right written to %s\n", ir_right_path.string().c_str());
    std::fflush(stderr);
  }

  std::fprintf(stderr, "[rgbd_writer][TRACE] writeCapture: writing intrinsics/calibration json\n");
  std::fflush(stderr);
  const fs::path calibration_path = root / "intrinsics" / "camera_intrinsics.json";
  std::ofstream calibration(calibration_path);
  calibration << "{\n  \"schema_version\": 3,\n  \"source\": " << quote(data.simulated ? "simulation" : "hardware") <<
    ",\n  \"camera_model\": " << quote(data.camera_model) <<
    ",\n  \"color_camera\": " << cameraInfoJson(data.color_camera_info) <<
    ",\n  \"depth_camera\": " << cameraInfoJson(data.depth_camera_info) <<
    ",\n  \"stereo_right\": " << (hasImage(data.stereo_right_image) ?
      cameraInfoJson(data.stereo_right_camera_info) : "null") <<
    ",\n  \"infrared_left\": " << cameraInfoJson(data.infrared_left_camera_info) <<
    ",\n  \"infrared_right\": " << cameraInfoJson(data.infrared_right_camera_info) <<
    ",\n  \"depth_to_color\": " << (data.depth_to_color_extrinsics_json.empty() ? "null" : data.depth_to_color_extrinsics_json) <<
    ",\n  \"depth_encoding\": " << quote(data.depth_image.encoding) <<
    ",\n  \"depth_units_m\": " << data.depth_units_m <<
    ",\n  \"capture_settings\": " << (data.capture_settings_json.empty() ? "{}" : data.capture_settings_json) << "\n}\n";
  std::fprintf(stderr, "[rgbd_writer][TRACE] writeCapture: calibration json written, building frames.jsonl entry\n");
  std::fflush(stderr);

  const std::string rgb_relative = "rgb/" + rgb_path.filename().string();
  const std::string raw_relative = "depth_raw_16/" + raw_path.filename().string();
  const std::string mm_relative = "depth_mm/" + depth_mm_path.filename().string();
  const std::string stereo_right_relative =
    "stereo_right/" + stereo_right_path.filename().string();
  const double timestamp_seconds = stamp.sec + stamp.nanosec / 1.0e9;
  std::ostringstream keyframe;
  keyframe << std::setprecision(12) << "{\"rgb_path\":" << quote(rgb_relative) <<
    ",\"depth_path\":" << quote(mm_relative) << ",\"intrinsics\":[[" << data.color_camera_info.k[0] <<
    ",0," << data.color_camera_info.k[2] << "],[0," << data.color_camera_info.k[4] << ',' <<
    data.color_camera_info.k[5] << "],[0,0,1]],\"station_id\":" << index << ",\"timestamp\":" <<
    timestamp_seconds << ",\"T_world_cam\":" << poseMatrixJson(data.camera_pose) << "}";

  std::ostringstream frame;
  frame << std::setprecision(12) << "{\"index\":" << index << ",\"waypoint_id\":" << quote(data.waypoint_id) <<
    ",\"source\":" << quote(data.simulated ? "simulation" : "hardware") <<
    ",\"camera_model\":" << quote(data.camera_model) << ",\"timestamp\":" << timeJson(stamp) <<
    ",\"rgb\":" << quote(rgb_relative) << ",\"depth_raw_16\":" << quote(raw_relative) <<
    ",\"depth_aligned_16\":" << (aligned_relative.empty() ? "null" : quote(aligned_relative)) <<
    ",\"depth_mm\":" << quote(mm_relative) << ",\"stereo_right\":" <<
    (hasImage(data.stereo_right_image) ? quote(stereo_right_relative) : "null") <<
    ",\"infrared_left\":" <<
    (hasImage(data.infrared_left_image) ? quote("infrared_left/" + ir_left_path.filename().string()) : "null") <<
    ",\"infrared_right\":" << (hasImage(data.infrared_right_image) ? quote("infrared_right/" + ir_right_path.filename().string()) : "null") <<
    ",\"hardware_metadata\":{\"color\":" << (data.color_metadata_json.empty() ? "null" : data.color_metadata_json) <<
    ",\"depth\":" << (data.depth_metadata_json.empty() ? "null" : data.depth_metadata_json) <<
    ",\"infrared_left\":" << (data.infrared_left_metadata_json.empty() ? "null" : data.infrared_left_metadata_json) <<
    ",\"infrared_right\":" << (data.infrared_right_metadata_json.empty() ? "null" : data.infrared_right_metadata_json) <<
    "},\"tf\":{\"map_to_odom\":" << transformJson(data.map_to_odom) << ",\"odom_to_base\":" << transformJson(data.odom_to_base) <<
    ",\"base_to_camera\":" << transformJson(data.base_to_camera) << "},\"wheel_odometry\":" <<
    (data.has_wheel_odometry ? odomJson(data.wheel_odometry) : "null") << ",\"amcl_pose\":" <<
    (data.has_amcl_pose ? amclJson(data.amcl_pose) : "null") << ",\"pose_source\":" << quote(data.has_amcl_pose ? "amcl" : "tf_ground_truth") <<
    ",\"rover_imu\":" << (data.has_rover_imu ? imuJson(data.rover_imu) : "null") <<
    ",\"T_world_robot\":" << poseMatrixJson(data.robot_pose) << ",\"T_world_camera\":" << poseMatrixJson(data.camera_pose) <<
    ",\"session_keyframe\":" << keyframe.str() << "}";
  std::fprintf(stderr, "[rgbd_writer][TRACE] writeCapture: appending frames.jsonl (index=%zu)\n", index);
  std::fflush(stderr);
  appendLine(frames_path, frame.str());
  std::fprintf(
    stderr,
    "[rgbd_writer][TRACE] writeCapture: appending sample logs (imu=%zu wheel_odom=%zu amcl=%zu rover_imu=%zu)\n",
    data.camera_imu_samples.size(), data.wheel_odometry_samples.size(), data.amcl_pose_samples.size(),
    data.rover_imu_samples.size());
  std::fflush(stderr);
  for (const auto & sample : data.camera_imu_samples) {
    appendLine(root / "camera_imu.jsonl", "{\"capture_index\":" + std::to_string(index) + ",\"sample\":" + imuJson(sample) + "}");
  }
  for (const auto & sample : data.wheel_odometry_samples) {
    appendLine(root / "wheel_odom.jsonl", odomJson(sample));
  }
  for (const auto & sample : data.amcl_pose_samples) {
    appendLine(root / "amcl_pose.jsonl", amclJson(sample));
  }
  for (const auto & sample : data.rover_imu_samples) {
    appendLine(root / "rover_imu.jsonl", imuJson(sample));
  }
  std::fprintf(stderr, "[rgbd_writer][TRACE] writeCapture: calling updateSession\n");
  std::fflush(stderr);
  updateSession(root, data.color_camera_info, data.camera_model);
  std::fprintf(stderr, "[rgbd_writer][TRACE] writeCapture: exit ok\n");
  std::fflush(stderr);

  RgbdCaptureRecord record;
  record.waypoint_id = data.waypoint_id;
  record.timestamp = stamp;
  record.rgb_path = rgb_path.string();
  record.depth_path = depth_mm_path.string();
  record.metadata_path = frames_path.string();
  record.robot_pose = data.robot_pose;
  record.camera_pose = data.camera_pose;
  return record;
}

std::string RgbdDatasetWriter::writeStationSummary(const RgbdStationSummary & summary) const
{
  const fs::path directory = fs::path(summary.session_dir) / "stations";
  fs::create_directories(directory);
  const fs::path path = directory / (safeStem(summary.waypoint_id) + ".json");
  std::ofstream output(path);
  if (!output) {throw std::runtime_error("Failed to write station summary: " + path.string());}
  output << "{\n  \"schema_version\": 3,\n  \"waypoint_id\": " << quote(summary.waypoint_id) <<
    ",\n  \"source\": " << quote(summary.simulated ? "simulation" : "hardware") <<
    ",\n  \"camera_model\": " << quote(summary.camera_model) <<
    ",\n  \"requested_frames\": " << summary.requested_frames <<
    ",\n  \"captured_frames\": " << summary.captured_frames <<
    ",\n  \"valid\": " << (summary.valid ? "true" : "false") <<
    ",\n  \"message\": " << quote(summary.message) << "\n}\n";
  return path.string();
}

}  // namespace renee_perception
