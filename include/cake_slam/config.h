#pragma once

// 模块功能：统一配置数据结构定义，承载系统运行所需的各类参数项，
// 供配置加载器与各处理模块读取并共享。

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
  bool hilti_en = false;             ///< Legacy FAST-LIVO2 HILTI22 image decimation: keep every 4th frame.
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
  double acc_scale = 1.0;  ///< Multiplier from message acceleration units to m/s^2.
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
  int confidence_threshold = 0;  ///< Optional raw point confidence gate.
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
  bool update_after_vio = false;               ///< LIVO模式下将非首帧scan插图延后到VIO/local BA之后。
  std::string update_after_vio_fallback = "lio"; ///< VIO不可用或被门限拒绝时的回退策略：lio或skip。
  double max_vio_map_update_pos_delta = 0.30;  ///< VIO插图位姿相对LIO位姿的最大平移差[m]。
  double max_vio_map_update_rot_delta_deg = 5.0; ///< VIO插图位姿相对LIO位姿的最大旋转差[deg]。
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
  int use_fast_fisheye_undistort = 0;///< Use OpenCV fisheye undistort for KANNALA_BRANDT cameras.
  int image_process_interval = 1;    ///< Process every Nth image before VIO frontend.
  int multiple_thread = 0;           ///< Enable legacy VINS processing thread.
  double max_solver_time = 0.04;     ///< Ceres time budget [s].
  int max_num_iterations = 8;        ///< Ceres iteration budget.
  double keyframe_parallax = 10.0;   ///< Keyframe parallax threshold [pixel].
  int estimate_extrinsic = 0;        ///< 0 fixed, 1 refine, 2 initialize online.
  std::string cam0_calib;            ///< Camera model YAML path.
  std::string fisheye_mask;          ///< Optional valid-domain mask path.
  int image_height = 480;            ///< Image rows [pixel].
  int image_width = 640;             ///< Image columns [pixel].

  bool lidar_depth_enable = true;          ///< Enable LiDAR/local-map depth assistance for VIO.
  bool lidar_prior_feature_enable = true;  ///< Insert LiDAR/local-map depth points as new visual tracks.
  bool only_lidar_depth_features = false;  ///< Track/optimize only features with LiDAR depth priors.
  bool optimize_lidar_inv_depth = true;    ///< Optimize inverse depth for tracks with LiDAR/local-map priors.
  int lidar_depth_source = 1;              ///< 0 local voxel map, 1 current scan, 2 disabled.
  bool visual_feature_depth_prior_enable = false; ///< Attach map/scan depth priors to stable Shi-Tomasi tracks.
  bool voxel_raycast_enable = true;        ///< Raycast LocalVoxelMap planes when projected depth lookup misses.
  int max_lidar_features = 250;            ///< Per-frame LiDAR visual seed cap.
  double min_lidar_depth = 0.2;            ///< Minimum camera-frame depth [m].
  double max_lidar_depth = 80.0;           ///< Maximum camera-frame depth [m].
  int z_buffer_cell_size = 8;              ///< Z-buffer grid size [pixel].
  double z_buffer_depth_tolerance = 0.3;   ///< Same-cell occlusion tolerance [m].
  double shi_tomasi_min_score = 1e-6;      ///< Local texture threshold.
  double lidar_mask_radius = 20.0;         ///< LiDAR seed spacing radius [pixel].
  int lidar_depth_grid_rows = 12;          ///< Uniform candidate selection grid rows.
  int lidar_depth_grid_cols = 16;          ///< Uniform candidate selection grid cols.
  double max_lidar_depth_ratio = 0.6;      ///< Upper ratio of depth-prior tracks among all tracked features.
  double depth_search_radius = 8.0;        ///< Pixel radius for projected sparse depth lookup.
  int visual_depth_min_track_cnt = 2;      ///< Minimum optical-flow track age before depth attachment.
  int max_depth_update_features = 250;     ///< Per-frame cap for existing visual track depth updates.
  int max_raycast_features = 120;          ///< Per-frame cap for visual-track LocalVoxelMap raycast fallback.
  int max_raycast_steps = 160;             ///< Per-ray voxel stepping cap.
  double raycast_step = 0.5;               ///< Ray marching step [m], clamped by voxel size when possible.
  double raycast_min_cos = 0.15;           ///< Minimum ray/plane-normal cosine for stable intersections.
  double raycast_plane_radius_scale = 3.0; ///< Accept intersections within scale * plane radius.
  double depth_prior_update_var_ratio = 0.7; ///< Existing prior is updated only if new variance is smaller by this ratio.
  double lio_prior_reproj_threshold = 3.0; ///< LIO prior reprojection gate [pixel].
  double lidar_depth_std = 0.10;           ///< Fallback LiDAR depth sigma [m].
  double min_inv_depth_var = 1e-6;         ///< Inverse-depth prior variance floor [1/m^2].
  double min_lio_pose_prior_var = 1e-6;    ///< LIO pose covariance floor [rad^2 or m^2].
  bool lio_full_state_prior_enable = false;///< Add LIO velocity/bias prior factors after initialization.
  bool vio_debug_factor_costs = false;     ///< Print expensive per-factor VIO cost diagnostics.
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
  bool record_trajectory = false;
  std::string lio_trajectory_file = "lio_tum.txt";
  std::string vio_trajectory_file = "vio_tum.txt";
};

/**
 * @brief Optional ROS topics for additional RViz diagnostics.
 */
struct VisualizationConfig
{
  std::string lio_odom_topic = "/aft_mapped_to_init";
  std::string lio_path_topic = "/path";
  std::string mavros_pose_topic = "/mavros/vision_pose/pose";
  std::string vio_odom_topic = "/cake_slam/vio/odom";
  std::string vio_path_topic = "/cake_slam/vio/path";
  std::string vio_pose_topic = "/cake_slam/vio/pose";
  std::string vio_landmarks_topic = "/cake_slam/vio/landmarks";
  std::string vio_window_path_topic = "/cake_slam/vio/window_path";
  std::string vio_window_poses_topic = "/cake_slam/vio/window_poses";
  bool publish_lio_colored_cloud = true;       ///< Publish current scan colored with the LIO pose.
  std::string lio_colored_cloud_topic = "/cloud_colored";
  bool publish_vio_colored_cloud = false;      ///< Publish current scan colored with the optimized VIO pose.
  std::string vio_colored_cloud_topic = "/cake_slam/vio/colored_cloud";
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
  VisualizationConfig visualization;
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
