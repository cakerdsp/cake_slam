#pragma once

#include <string>
#include <vector>

namespace cake_slam {

/**
 * @brief Main scheduler and synchronization options.
 *
 * These fields are loaded from the unified OpenCV YAML file. Times are in
 * seconds unless noted otherwise.
 */
struct CommonConfig
{
  bool lidar_enable = true;          ///< Subscribe and process LiDAR input.
  bool image_enable = true;          ///< Subscribe and process mono camera input.
  bool ros_driver_bug_fix = false;   ///< Optional whole-second IMU stamp correction.
  double imu_time_offset = 0.0;      ///< Applied as imu_stamp -= imu_time_offset [s].
  double image_time_offset = 0.0;    ///< Applied as image_stamp += image_time_offset [s].
  int max_buffer_size = 200000;      ///< Per-buffer hard cap [messages].
  bool gravity_align_enable = false; ///< Rotate initial gravity to the world z axis.
  bool imu_propagation_enable = true;///< Publish high-rate IMU propagated odometry.
};

/**
 * @brief IMU noise and online-state options shared by LIO and VIO.
 */
struct ImuConfig
{
  bool enable = true;
  std::string topic;
  double acc_n = 0.02;     ///< Accelerometer white-noise sigma [m/s^2/sqrt(Hz)].
  double acc_w = 0.04;     ///< Accelerometer random-walk sigma.
  double gyr_n = 0.01;     ///< Gyroscope white-noise sigma [rad/s/sqrt(Hz)].
  double gyr_w = 0.001;    ///< Gyroscope random-walk sigma.
  double g_norm = 9.81;    ///< Gravity magnitude [m/s^2].
  double gyr_cov = 1.0;    ///< LIO gyro covariance scale.
  double acc_cov = 1.0;    ///< LIO accelerometer covariance scale.
  int imu_int_frame = 30;  ///< Frames used for IMU initialization.
  bool gravity_est = true; ///< Estimate gravity direction online.
  bool bias_est = true;    ///< Estimate IMU biases online.
};

/**
 * @brief LiDAR preprocessing and scan model options.
 */
struct LidarConfig
{
  std::string topic;
  int type = 1;                 ///< LID_TYPE enum value.
  int scan_line = 6;            ///< Number of scan lines.
  int scan_rate = 10;           ///< Scan rate [Hz].
  int point_filter_num = 3;     ///< Preprocess decimation stride.
  double blind = 0.01;          ///< Near-range rejection distance [m].
  bool feature_extract = false; ///< Whether to extract edge/surface features.
  double filter_size_surf = 0.5;///< Voxel leaf size for LIO surf cloud [m].
};

/**
 * @brief Sparse voxel-map configuration for FAST-LIVO2-style LIO.
 */
struct MapConfig
{
  double voxel_size = 0.5;                     ///< Base voxel edge length [m].
  int max_layer = 1;                           ///< Octree depth.
  double min_eigen_value = 0.01;               ///< Plane reliability threshold.
  double sigma_num = 3.0;                      ///< Point-to-plane sigma gate.
  double beam_err = 0.02;                      ///< Beam/angular noise model term.
  double dept_err = 0.05;                      ///< Range/depth noise model term [m].
  std::vector<int> layer_init_num = {5, 5, 5, 5, 5};
  int max_points_num = 50;                     ///< Max retained points per voxel.
  int min_iterations = 5;                      ///< Minimum LIO iterations.
  bool sliding_enable = false;                 ///< Enable local-map sliding.
  int half_map_size = 100;                     ///< Local map half extent [m].
  double sliding_thresh = 8.0;                 ///< Map slide trigger distance [m].
};

/**
 * @brief Visual frontend/backend options.
 */
struct VisionConfig
{
  std::string image_topic;
  int max_cnt = 150;                 ///< Target tracked feature count.
  int min_dist = 30;                 ///< Minimum feature spacing [pixel].
  int show_track = 0;                ///< Draw/debug tracking image.
  int flow_back = 0;                 ///< Enable forward-backward LK check.
  int multiple_thread = 0;           ///< Enable legacy VINS processing thread.
  double max_solver_time = 0.04;     ///< Ceres time budget [s].
  int max_num_iterations = 8;        ///< Ceres iteration budget.
  double keyframe_parallax = 10.0;   ///< Keyframe parallax threshold [pixel].
  int estimate_extrinsic = 0;        ///< 0 fixed, 1 refine, 2 initialize online.
  std::string cam0_calib;            ///< Camera model YAML path.
  std::string fisheye_mask;          ///< Optional valid-domain mask path.
  int image_height = 480;            ///< Image rows [pixel].
  int image_width = 640;             ///< Image columns [pixel].

  bool lidar_depth_enable = true;          ///< Use LiDAR projections as visual seeds.
  bool optimize_lidar_inv_depth = true;    ///< Optimize LiDAR-seeded inverse depth.
  int max_lidar_features = 250;            ///< Per-frame LiDAR visual seed cap.
  double min_lidar_depth = 0.2;            ///< Minimum camera-frame depth [m].
  double max_lidar_depth = 80.0;           ///< Maximum camera-frame depth [m].
  int z_buffer_cell_size = 8;              ///< Z-buffer grid size [pixel].
  double z_buffer_depth_tolerance = 0.3;   ///< Same-cell occlusion tolerance [m].
  double shi_tomasi_min_score = 1e-6;      ///< Local texture threshold.
  double lidar_mask_radius = 20.0;         ///< LiDAR seed spacing radius [pixel].
  double lio_prior_reproj_threshold = 3.0; ///< LIO prior reprojection gate [pixel].
  double lidar_depth_std = 0.10;           ///< Fallback LiDAR depth sigma [m].
  double min_inv_depth_var = 1e-6;         ///< Inverse-depth prior variance floor [1/m^2].
  double min_lio_pose_prior_var = 1e-6;    ///< LIO pose covariance floor [rad^2 or m^2].
};

/**
 * @brief Sensor extrinsics.
 *
 * Conventions:
 * - lidar_R/lidar_T: P_imu = R_I_L * P_lidar + t_I_L.
 * - camera_R/camera_T: P_cam = R_C_L * P_lidar + t_C_L.
 * - body_T_cam0: T_I_C, i.e. P_imu = R_I_C * P_cam + t_I_C.
 */
struct ExtrinsicConfig
{
  std::vector<double> lidar_T;      ///< t_I_L [m], length 3.
  std::vector<double> lidar_R;      ///< R_I_L row-major, length 9.
  std::vector<double> camera_T;     ///< t_C_L [m], length 3.
  std::vector<double> camera_R;     ///< R_C_L row-major, length 9.
  std::vector<double> body_T_cam0;  ///< T_I_C row-major 4x4, length 16.
};

/**
 * @brief ROS frame names used by publishers and TF.
 */
struct FrameConfig
{
  std::string world = "world";
  std::string body = "body";
  std::string lidar = "lidar";
  std::string camera = "camera";
};

struct OutputConfig
{
  std::string path;
};

struct TimeOffsetConfig
{
  double td = 0.0;      ///< Camera-to-IMU delay used by VIO [s].
  int estimate_td = 0;  ///< Enable online time-delay estimation.
};

/**
 * @brief Unified runtime configuration loaded by LoadConfig().
 */
struct Config
{
  CommonConfig common;
  ImuConfig imu;
  LidarConfig lidar;
  MapConfig map;
  VisionConfig vision;
  ExtrinsicConfig extrinsic;
  FrameConfig frame;
  OutputConfig output;
  TimeOffsetConfig time_offset;
};

/**
 * @brief Load an OpenCV YAML configuration file into Config.
 * @param path Absolute or relative YAML path.
 * @param config Output structure; missing keys keep their defaults.
 * @return true if the file could be opened and parsed.
 */
bool LoadConfig(const std::string &path, Config &config);

} // namespace cake_slam
