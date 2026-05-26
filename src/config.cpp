// 模块功能：配置文件读取与参数填充实现，
// 负责从统一 YAML 中提取各模块运行参数。

#include "cake_slam/config.h"

#include <opencv2/core.hpp>

namespace cake_slam {

namespace {

cv::FileNode nodeForKey(const cv::FileStorage &fs, const std::string &key)
{
  cv::FileNode direct = fs[key];
  if (!direct.empty()) {
    return direct;
  }

  cv::FileNode node = fs.root();
  size_t begin = 0;
  while (begin < key.size()) {
    const size_t end = key.find('.', begin);
    const std::string part = key.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
    node = node[part];
    if (node.empty()) {
      return node;
    }
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }
  return node;
}

template <typename T>
// 如果配置文件中存在指定键，则把值读入 out；
// 若键不存在，则保留调用者传入对象的默认值。
void readIfPresent(const cv::FileStorage &fs, const std::string &key, T &out)
{
  cv::FileNode node = nodeForKey(fs, key);
  if (!node.empty()) {
    node >> out;
  }
}

void readVectorIfPresent(const cv::FileStorage &fs, const std::string &key, std::vector<double> &out)
{
  cv::FileNode node = nodeForKey(fs, key);
  if (node.empty()) {
    return;
  }
  out.clear();
  if (node.isSeq()) {
    for (auto it = node.begin(); it != node.end(); ++it) {
      out.push_back((double)*it);
    }
  } else if (node.isMap()) {
    cv::Mat mat;
    node >> mat;
    out.assign((double *)mat.datastart, (double *)mat.dataend);
  }
}

} // namespace

bool LoadConfig(const std::string &path, Config &config)
{
  // 整体流程：
  // 1. 打开配置文件；
  // 2. 按模块逐项读取；
  // 3. 对数组字段做显式遍历和拷贝；
  // 4. 缺失字段保留默认值。
  cv::FileStorage fs(path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    return false;
  }

  // -------------------- 主程序参数 --------------------
  readIfPresent(fs, "common.lidar_enable", config.common.lidar_enable);
  readIfPresent(fs, "common.image_enable", config.common.image_enable);
  readIfPresent(fs, "common.lidar_en", config.common.lidar_enable);
  readIfPresent(fs, "common.img_en", config.common.image_enable);
  readIfPresent(fs, "common.ros_driver_bug_fix", config.common.ros_driver_bug_fix);
  readIfPresent(fs, "common.imu_time_offset", config.common.imu_time_offset);
  readIfPresent(fs, "common.image_time_offset", config.common.image_time_offset);
  readIfPresent(fs, "time_offset.img_time_offset", config.common.image_time_offset);
  readIfPresent(fs, "common.hilti_en", config.common.hilti_en);
  readIfPresent(fs, "common.max_buffer_size", config.common.max_buffer_size);
  readIfPresent(fs, "common.gravity_align_enable", config.common.gravity_align_enable);
  readIfPresent(fs, "uav.gravity_align_en", config.common.gravity_align_enable);
  readIfPresent(fs, "common.imu_propagation_enable", config.common.imu_propagation_enable);
  readIfPresent(fs, "uav.imu_rate_odom", config.common.imu_propagation_enable);

  // -------------------- IMU 参数 --------------------
  readIfPresent(fs, "imu.enable", config.imu.enable);
  readIfPresent(fs, "imu.topic", config.imu.topic);
  readIfPresent(fs, "imu.acc_n", config.imu.acc_n);
  readIfPresent(fs, "imu.acc_w", config.imu.acc_w);
  readIfPresent(fs, "imu.acc_scale", config.imu.acc_scale);
  readIfPresent(fs, "imu.gyr_n", config.imu.gyr_n);
  readIfPresent(fs, "imu.gyr_w", config.imu.gyr_w);
  readIfPresent(fs, "imu.g_norm", config.imu.g_norm);
  readIfPresent(fs, "imu.gyr_cov", config.imu.gyr_cov);
  readIfPresent(fs, "imu.acc_cov", config.imu.acc_cov);
  readIfPresent(fs, "imu.imu_int_frame", config.imu.imu_int_frame);
  readIfPresent(fs, "imu.gravity_est", config.imu.gravity_est);
  readIfPresent(fs, "imu.bias_est", config.imu.bias_est);

  // -------------------- LiDAR 参数 --------------------
  readIfPresent(fs, "lidar.topic", config.lidar.topic);
  readIfPresent(fs, "lidar.type", config.lidar.type);
  readIfPresent(fs, "lidar.scan_line", config.lidar.scan_line);
  readIfPresent(fs, "lidar.scan_rate", config.lidar.scan_rate);
  readIfPresent(fs, "lidar.point_filter_num", config.lidar.point_filter_num);
  readIfPresent(fs, "lidar.blind", config.lidar.blind);
  readIfPresent(fs, "lidar.feature_extract", config.lidar.feature_extract);
  readIfPresent(fs, "lidar.filter_size_surf", config.lidar.filter_size_surf);

  // -------------------- 地图参数 --------------------
  readIfPresent(fs, "map.voxel_size", config.map.voxel_size);
  readIfPresent(fs, "map.max_layer", config.map.max_layer);
  readIfPresent(fs, "map.min_eigen_value", config.map.min_eigen_value);
  readIfPresent(fs, "map.sigma_num", config.map.sigma_num);
  readIfPresent(fs, "map.beam_err", config.map.beam_err);
  readIfPresent(fs, "map.dept_err", config.map.dept_err);
  // layer_init_num 是数组，需要逐元素拷贝。
  if (!nodeForKey(fs, "map.layer_init_num").empty()) {
    cv::FileNode node = nodeForKey(fs, "map.layer_init_num");
    config.map.layer_init_num.clear();
    for (auto it = node.begin(); it != node.end(); ++it) {
      config.map.layer_init_num.push_back((int)*it);
    }
  }
  readIfPresent(fs, "map.max_points_num", config.map.max_points_num);
  readIfPresent(fs, "map.min_iterations", config.map.min_iterations);
  readIfPresent(fs, "map.sliding_enable", config.map.sliding_enable);
  readIfPresent(fs, "map.half_map_size", config.map.half_map_size);
  readIfPresent(fs, "map.sliding_thresh", config.map.sliding_thresh);

  // -------------------- 视觉参数 --------------------
  readIfPresent(fs, "vision.image_topic", config.vision.image_topic);
  readIfPresent(fs, "vision.max_cnt", config.vision.max_cnt);
  readIfPresent(fs, "vision.min_dist", config.vision.min_dist);
  readIfPresent(fs, "vision.show_track", config.vision.show_track);
  readIfPresent(fs, "vision.flow_back", config.vision.flow_back);
  readIfPresent(fs, "vision.use_fast_fisheye_undistort", config.vision.use_fast_fisheye_undistort);
  readIfPresent(fs, "vision.image_process_interval", config.vision.image_process_interval);
  readIfPresent(fs, "vision.multiple_thread", config.vision.multiple_thread);
  readIfPresent(fs, "vision.max_solver_time", config.vision.max_solver_time);
  readIfPresent(fs, "vision.max_num_iterations", config.vision.max_num_iterations);
  readIfPresent(fs, "vision.keyframe_parallax", config.vision.keyframe_parallax);
  readIfPresent(fs, "vision.estimate_extrinsic", config.vision.estimate_extrinsic);
  readIfPresent(fs, "vision.cam0_calib", config.vision.cam0_calib);
  readIfPresent(fs, "vision.fisheye_mask", config.vision.fisheye_mask);
  readIfPresent(fs, "vision.image_height", config.vision.image_height);
  readIfPresent(fs, "vision.image_width", config.vision.image_width);
  readIfPresent(fs, "vision.lidar_depth_enable", config.vision.lidar_depth_enable);
  readIfPresent(fs, "vision.lidar_prior_feature_enable", config.vision.lidar_prior_feature_enable);
  readIfPresent(fs, "vision.lidar_prior_enable", config.vision.lidar_prior_feature_enable);
  readIfPresent(fs, "vision.radar_prior_feature_enable", config.vision.lidar_prior_feature_enable);
  readIfPresent(fs, "vision.optimize_lidar_inv_depth", config.vision.optimize_lidar_inv_depth);
  readIfPresent(fs, "vision.max_lidar_features", config.vision.max_lidar_features);
  readIfPresent(fs, "vision.min_lidar_depth", config.vision.min_lidar_depth);
  readIfPresent(fs, "vision.max_lidar_depth", config.vision.max_lidar_depth);
  readIfPresent(fs, "vision.z_buffer_cell_size", config.vision.z_buffer_cell_size);
  readIfPresent(fs, "vision.z_buffer_depth_tolerance", config.vision.z_buffer_depth_tolerance);
  readIfPresent(fs, "vision.shi_tomasi_min_score", config.vision.shi_tomasi_min_score);
  readIfPresent(fs, "vision.lidar_mask_radius", config.vision.lidar_mask_radius);
  readIfPresent(fs, "vision.lio_prior_reproj_threshold", config.vision.lio_prior_reproj_threshold);
  readIfPresent(fs, "vision.lidar_depth_std", config.vision.lidar_depth_std);
  readIfPresent(fs, "vision.min_inv_depth_var", config.vision.min_inv_depth_var);
  readIfPresent(fs, "vision.min_lio_pose_prior_var", config.vision.min_lio_pose_prior_var);
  readIfPresent(fs, "vision.lio_full_state_prior_enable", config.vision.lio_full_state_prior_enable);
  readIfPresent(fs, "vision.vio_debug_factor_costs", config.vision.vio_debug_factor_costs);
  readIfPresent(fs, "vision.vio_state_feedback_enable", config.vision.vio_state_feedback_enable);
  readIfPresent(fs, "vision.min_vio_feedback_visual_residuals", config.vision.min_vio_feedback_visual_residuals);
  readIfPresent(fs, "vision.max_vio_feedback_pos_delta", config.vision.max_vio_feedback_pos_delta);
  readIfPresent(fs, "vision.max_vio_feedback_z_delta", config.vision.max_vio_feedback_z_delta);
  readIfPresent(fs, "vision.max_vio_feedback_rot_delta_deg", config.vision.max_vio_feedback_rot_delta_deg);
  if (config.vision.image_process_interval < 1) {
    config.vision.image_process_interval = 1;
  }

  // -------------------- 外参 --------------------
  // 这里把 OpenCV FileNode 序列平铺拷贝到 std::vector<double> 中。
  readVectorIfPresent(fs, "extrinsic.lidar_T", config.extrinsic.lidar_T);
  readVectorIfPresent(fs, "extrinsic.lidar_R", config.extrinsic.lidar_R);
  readVectorIfPresent(fs, "extrin_calib.extrinsic_T", config.extrinsic.lidar_T);
  readVectorIfPresent(fs, "extrin_calib.extrinsic_R", config.extrinsic.lidar_R);
  readVectorIfPresent(fs, "extrinsic.camera_T", config.extrinsic.camera_T);
  readVectorIfPresent(fs, "extrinsic.camera_R", config.extrinsic.camera_R);
  readVectorIfPresent(fs, "extrinsic.Pcl", config.extrinsic.camera_T);
  readVectorIfPresent(fs, "extrinsic.Rcl", config.extrinsic.camera_R);
  readVectorIfPresent(fs, "extrin_calib.Pcl", config.extrinsic.camera_T);
  readVectorIfPresent(fs, "extrin_calib.Rcl", config.extrinsic.camera_R);
  // body_T_cam0 在配置中按矩阵保存，这里按底层内存顺序拷贝为长度 16 的数组。
  readVectorIfPresent(fs, "extrinsic.body_T_cam0", config.extrinsic.body_T_cam0);

  // -------------------- 坐标系、输出与时间偏移 --------------------
  readIfPresent(fs, "frame.world", config.frame.world);
  readIfPresent(fs, "frame.body", config.frame.body);
  readIfPresent(fs, "frame.lidar", config.frame.lidar);
  readIfPresent(fs, "frame.camera", config.frame.camera);

  readIfPresent(fs, "output.path", config.output.path);

  readIfPresent(fs, "visualization.vio_landmarks_topic", config.visualization.vio_landmarks_topic);
  readIfPresent(fs, "visualization.vio_window_path_topic", config.visualization.vio_window_path_topic);
  readIfPresent(fs, "visualization.vio_window_poses_topic", config.visualization.vio_window_poses_topic);

  readIfPresent(fs, "time_offset.td", config.time_offset.td);
  readIfPresent(fs, "time_offset.estimate_td", config.time_offset.estimate_td);

  fs.release();
  return true;
}

} // namespace cake_slam
