#pragma once

#include <string>
#include <vector>

namespace cake_slam {

struct ImuConfig
{
  bool enable = true;
  std::string topic;
  double acc_n = 0.02;
  double acc_w = 0.04;
  double gyr_n = 0.01;
  double gyr_w = 0.001;
  double g_norm = 9.81;
  double gyr_cov = 1.0;
  double acc_cov = 1.0;
  int imu_int_frame = 30;
  bool gravity_est = true;
  bool bias_est = true;
};

struct LidarConfig
{
  std::string topic;
  int type = 1;
  int scan_line = 6;
  int scan_rate = 10;
  int point_filter_num = 3;
  double blind = 0.01;
  bool feature_extract = false;
  double filter_size_surf = 0.5;
};

struct MapConfig
{
  double voxel_size = 0.5;
  int max_layer = 1;
  double min_eigen_value = 0.01;
  double sigma_num = 3.0;
  double beam_err = 0.02;
  double dept_err = 0.05;
  std::vector<int> layer_init_num = {5, 5, 5, 5, 5};
  int max_points_num = 50;
  int min_iterations = 5;
  bool sliding_enable = false;
  int half_map_size = 100;
  double sliding_thresh = 8.0;
};

struct VisionConfig
{
  std::string image_topic;
  int max_cnt = 150;
  int min_dist = 30;
  double f_threshold = 1.0;
  int show_track = 0;
  int flow_back = 0;
  int multiple_thread = 0;
  double max_solver_time = 0.04;
  int max_num_iterations = 8;
  double keyframe_parallax = 10.0;
  int estimate_extrinsic = 0;
  std::string cam0_calib;
  int image_height = 480;
  int image_width = 640;
};

struct ExtrinsicConfig
{
  std::vector<double> lidar_T;
  std::vector<double> lidar_R;
  std::vector<double> body_T_cam0;
};

struct FrameConfig
{
  std::string world = "world";
  std::string body = "body";
  std::string camera = "camera";
};

struct OutputConfig
{
  std::string path;
};

struct TimeOffsetConfig
{
  double td = 0.0;
  int estimate_td = 0;
};

struct Config
{
  ImuConfig imu;
  LidarConfig lidar;
  MapConfig map;
  VisionConfig vision;
  ExtrinsicConfig extrinsic;
  FrameConfig frame;
  OutputConfig output;
  TimeOffsetConfig time_offset;
};

bool LoadConfig(const std::string &path, Config &config);

} // namespace cake_slam
