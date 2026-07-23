# Metadata for one RGB-D capture saved by renee_perception.
#
# Image payloads are stored on disk; this message carries paths and poses so
# scan execution code can log or publish what was captured.

string waypoint_id
builtin_interfaces/Time timestamp

string rgb_path
string depth_path
string metadata_path

geometry_msgs/PoseStamped robot_pose
geometry_msgs/PoseStamped camera_pose
