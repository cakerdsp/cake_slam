#include "cake_slam/config.h"

#include <opencv2/core.hpp>

namespace cake_slam {

namespace {

template <typename T>
// 如果配置文件中存在指定键，则把值读入 out；
// 若键不存在，则保留调用者传入对象的默认值。
void readIfPresent(const cv::FileStorage &fs, const std::string &key, T &out)
{
  if (!fs[key].empty()) {
    fs[key] >> out;
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

  // -------------------- IMU 参数 --------------------
  readIfPresent(fs, "imu.enable", config.imu.enable);
  readIfPresent(fs, "imu.topic", config.imu.topic);
  readIfPresent(fs, "imu.acc_n", config.imu.acc_n);
  readIfPresent(fs, "imu.acc_w", config.imu.acc_w);
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
  if (!fs["map.layer_init_num"].empty()) {
    cv::FileNode node = fs["map.layer_init_num"];
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
  readIfPresent(fs, "vision.f_threshold", config.vision.f_threshold);
  readIfPresent(fs, "vision.show_track", config.vision.show_track);
  readIfPresent(fs, "vision.flow_back", config.vision.flow_back);
  readIfPresent(fs, "vision.multiple_thread", config.vision.multiple_thread);
  readIfPresent(fs, "vision.max_solver_time", config.vision.max_solver_time);
  readIfPresent(fs, "vision.max_num_iterations", config.vision.max_num_iterations);
  readIfPresent(fs, "vision.keyframe_parallax", config.vision.keyframe_parallax);
  readIfPresent(fs, "vision.estimate_extrinsic", config.vision.estimate_extrinsic);
  readIfPresent(fs, "vision.cam0_calib", config.vision.cam0_calib);
  readIfPresent(fs, "vision.image_height", config.vision.image_height);
  readIfPresent(fs, "vision.image_width", config.vision.image_width);

  // -------------------- 外参 --------------------
  // 这里把 OpenCV FileNode 序列平铺拷贝到 std::vector<double> 中。
  if (!fs["extrinsic.lidar_T"].empty()) {
    cv::FileNode node = fs["extrinsic.lidar_T"];
    config.extrinsic.lidar_T.clear();
    for (auto it = node.begin(); it != node.end(); ++it) {
      config.extrinsic.lidar_T.push_back((double)*it);
    }
  }
  if (!fs["extrinsic.lidar_R"].empty()) {
    cv::FileNode node = fs["extrinsic.lidar_R"];
    config.extrinsic.lidar_R.clear();
    for (auto it = node.begin(); it != node.end(); ++it) {
      config.extrinsic.lidar_R.push_back((double)*it);
    }
  }
  // body_T_cam0 在配置中按矩阵保存，这里按底层内存顺序拷贝为长度 16 的数组。
  if (!fs["extrinsic.body_T_cam0"].empty()) {
    cv::Mat mat;
    fs["extrinsic.body_T_cam0"] >> mat;
    config.extrinsic.body_T_cam0.assign((double *)mat.datastart, (double *)mat.dataend);
  }

  // -------------------- 坐标系、输出与时间偏移 --------------------
  readIfPresent(fs, "frame.world", config.frame.world);
  readIfPresent(fs, "frame.body", config.frame.body);
  readIfPresent(fs, "frame.camera", config.frame.camera);

  readIfPresent(fs, "output.path", config.output.path);

  readIfPresent(fs, "time_offset.td", config.time_offset.td);
  readIfPresent(fs, "time_offset.estimate_td", config.time_offset.estimate_td);

  fs.release();
  return true;
}

} // namespace cake_slam
