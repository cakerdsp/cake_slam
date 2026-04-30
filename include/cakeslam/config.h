#pragma once

#include <string>
#include <vector>

namespace cake_slam {

// 主程序运行模式与同步相关配置。
// 这些字段对应 FAST-LIVO2 主程序中的 common/time_offset 类参数，但统一由本项目 yaml 读取。
struct CommonConfig
{
  // 是否接入 LiDAR。关闭后主程序不会订阅或处理点云。
  bool lidar_enable = true;
  // 是否接入单目图像。当前 cake_slam 主程序只支持单目入口。
  bool image_enable = true;
  // 主循环定时器频率，单位 Hz。
  int process_rate_hz = 500;
  // 兼容部分驱动 IMU 秒级跳变的修正开关，默认关闭。
  bool ros_driver_bug_fix = false;
  // IMU 时间戳修正量，订阅后先执行 stamp - imu_time_offset。
  double imu_time_offset = 0.0;
  // 图像时间戳修正量，订阅后执行 stamp + image_time_offset。
  double image_time_offset = 0.0;
  // 各输入缓存最大长度，避免长期不同步时内存无限增长。
  int max_buffer_size = 200000;
  // 是否在 IMU 初始化后把估计重力对齐到世界 z 轴方向。
  bool gravity_align_enable = false;
  // 是否发布 IMU 高频传播里程计。
  bool imu_propagation_enable = true;
};

// IMU 相关配置。这里的字段同时服务于 LIO 和视觉惯导部分。
struct ImuConfig
{
  // 是否启用 IMU 融合。关闭后，相关模块会退化为不依赖 IMU 的流程。
  bool enable = true;
  // IMU 话题名，仅上层 ROS 节点订阅时使用。
  std::string topic;
  // 加速度白噪声标准差。
  double acc_n = 0.02;
  // 加速度随机游走标准差。
  double acc_w = 0.04;
  // 角速度白噪声标准差。
  double gyr_n = 0.01;
  // 角速度随机游走标准差。
  double gyr_w = 0.001;
  // 重力模长先验，单位 m/s^2。
  double g_norm = 9.81;
  // LIO 内部使用的陀螺仪协方差缩放。
  double gyr_cov = 1.0;
  // LIO 内部使用的加速度计协方差缩放。
  double acc_cov = 1.0;
  // IMU 初始化所需的帧数/积分段数。
  int imu_int_frame = 30;
  // 是否允许系统在线估计重力方向。
  bool gravity_est = true;
  // 是否允许系统在线估计 IMU 偏置。
  bool bias_est = true;
};

// LiDAR 预处理与 LIO 前端配置。
struct LidarConfig
{
  // LiDAR 点云话题名。
  std::string topic;
  // LiDAR 型号枚举值，对应 common_lib.h 中的 LID_TYPE。
  int type = 1;
  // 激光线数，例如 16/32/64/128 等。
  int scan_line = 6;
  // 雷达转速或扫描频率，单位 Hz。
  int scan_rate = 10;
  // 点云预处理时的点过滤步长。
  int point_filter_num = 3;
  // 盲区阈值，小于该距离的点通常会被剔除。
  double blind = 0.01;
  // 是否在预处理阶段提取边/面特征。
  bool feature_extract = false;
  // 下采样体素尺寸，单位米。
  double filter_size_surf = 0.5;
};

// 稀疏体素地图配置。
struct MapConfig
{
  // 体素基础边长，单位米。
  double voxel_size = 0.5;
  // 八叉树最大层数，层数越高可表达越细的几何结构。
  int max_layer = 1;
  // 判定局部面是否可靠的最小特征值阈值。
  double min_eigen_value = 0.01;
  // 点到面的马氏距离/门限中的 sigma 系数。
  double sigma_num = 3.0;
  // 激光束方向误差模型参数。
  double beam_err = 0.02;
  // 深度方向误差模型参数。
  double dept_err = 0.05;
  // 各层体素初始化平面所需的最小点数。
  std::vector<int> layer_init_num = {5, 5, 5, 5, 5};
  // 每个体素最多保留的点数。
  int max_points_num = 50;
  // 每帧状态估计的最小迭代次数。
  int min_iterations = 5;
  // 是否启用局部地图滑窗。
  bool sliding_enable = false;
  // 局部地图半边长，单位米。
  int half_map_size = 100;
  // 触发地图滑动的位移阈值，单位米。
  double sliding_thresh = 8.0;
};

// 视觉前端/后端公共配置。
struct VisionConfig
{
  // 图像话题名。
  std::string image_topic;
  // 每帧希望保留的最大特征点数量。
  int max_cnt = 150;
  // 新特征之间的最小像素间距。
  int min_dist = 30;
  // RANSAC 基础矩阵约束阈值。
  double f_threshold = 1.0;
  // 是否发布/显示跟踪结果图像。
  int show_track = 0;
  // 是否启用光流反向校验。
  int flow_back = 0;
  // 是否启用多线程处理。
  int multiple_thread = 0;
  // 单次优化允许的最长求解时间，单位秒。
  double max_solver_time = 0.04;
  // 单次优化允许的最大迭代次数。
  int max_num_iterations = 8;
  // 判断关键帧所需的视差阈值。
  double keyframe_parallax = 10.0;
  // 外参处理模式：固定/给定初值优化/完全估计等。
  int estimate_extrinsic = 0;
  // 相机内参文件路径。
  std::string cam0_calib;
  // 图像高度，单位像素。
  int image_height = 480;
  // 图像宽度，单位像素。
  int image_width = 640;
};

// 各传感器之间外参配置。
// 数组长度由对应模块约定：例如 lidar_R 为 9 个元素的行优先旋转矩阵。
struct ExtrinsicConfig
{
  // LiDAR 到 IMU/body 的平移外参 t_I_L，长度应为 3。
  std::vector<double> lidar_T;
  // LiDAR 到 IMU/body 的旋转外参 R_I_L，长度应为 9。
  std::vector<double> lidar_R;
  // LiDAR 到 cam0 的平移外参 t_C_L，长度应为 3。
  std::vector<double> camera_T;
  // LiDAR 到 cam0 的旋转外参 R_C_L，长度应为 9。
  std::vector<double> camera_R;
  // cam0 到 IMU/body 的 4x4 齐次变换矩阵 T_I_C，长度应为 16。
  std::vector<double> body_T_cam0;
};

// ROS 坐标系命名配置。
struct FrameConfig
{
  // 世界坐标系名称。
  std::string world = "world";
  // 机体/IMU 主体坐标系名称。
  std::string body = "body";
  // 相机坐标系名称。
  std::string camera = "camera";
};

// 输出路径配置。
struct OutputConfig
{
  // 轨迹、日志或标定结果保存目录。
  std::string path;
};

// 多传感器时间偏移配置。
struct TimeOffsetConfig
{
  // 相机与 IMU 等传感器之间的时间延迟初值，单位秒。
  double td = 0.0;
  // 是否在线估计时间延迟。
  int estimate_td = 0;
};

// 全量配置容器。
// LoadConfig 会按命名空间分别填充这些子结构。
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

// 从 OpenCV FileStorage 支持的配置文件中加载参数。
// 输入要求：
// 1. path 必须可读；
// 2. 键名需与本结构中的注释保持一致；
// 3. 缺失字段将保留结构体默认值。
bool LoadConfig(const std::string &path, Config &config);

} // namespace cake_slam
