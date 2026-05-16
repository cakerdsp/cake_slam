/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

// 模块功能：IMU 处理与点云去畸变接口定义，提供传播、初始化与偏置估计等能力，
// 支撑 LIO 主流程对 IMU 与 LiDAR 数据的紧耦合融合。

#ifndef IMU_PROCESSING_H
#define IMU_PROCESSING_H

#include <Eigen/Eigen>
#include <fstream>
#include "cake_slam/common_lib.h"
#include <condition_variable>
#include <nav_msgs/msg/odometry.hpp>
#include <cake_slam/utils/so3_math.h>

// 比较两个点的相对时间，用于按扫描内时间顺序排序点云。
// 这里默认逐点时间被编码在 curvature 或兼容字段中。
const bool time_list(PointType &x,
                     PointType &y); //{return (x.curvature < y.curvature);};

/// IMU 传播与 LiDAR 去畸变核心模块。
/// 主要职责：
/// 1. 利用 IMU 对当前扫描进行前向传播；
/// 2. 在初始化阶段估计重力/偏置；
/// 3. 根据逐点时间把整帧点云去畸变到扫描结束时刻。
class ImuProcess
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();

  // 重置内部状态，使对象回到“尚未初始化”的状态。
  void Reset();
  // 设置 LiDAR 到 body(IMU) 的平移和旋转外参。
  void set_extrinsic(const V3D &transl, const M3D &rot);
  // 仅设置平移外参，旋转保持已有值。
  void set_extrinsic(const V3D &transl);
  // 通过 4x4 齐次变换矩阵设置外参。
  void set_extrinsic(const MD(4, 4) & T);
  // 设置陀螺仪测量噪声缩放。
  void set_gyr_cov_scale(const V3D &scaler);
  // 设置加速度计测量噪声缩放。
  void set_acc_cov_scale(const V3D &scaler);
  // 设置陀螺仪偏置随机游走协方差。
  void set_gyr_bias_cov(const V3D &b_g);
  // 设置加速度计偏置随机游走协方差。
  void set_acc_bias_cov(const V3D &b_a);
  // 设置曝光时间倒数的协方差。
  void set_inv_expo_cov(const double &inv_expo);
  // 设置 IMU 初始化所需的数据帧数。
  void set_imu_init_frame_num(const int &num);
  // 显式关闭 IMU 融合。
  void disable_imu();
  // 禁止在线估计重力。
  void disable_gravity_est();
  // 禁止在线估计偏置。
  void disable_bias_est();
  // 禁止在线估计曝光相关状态。
  void disable_exposure_est();
  // LIO 主入口：处理一组 LiDAR 测量，更新状态并输出去畸变点云。
  // 输入要求：
  // 1. lidar_meas 中必须包含与当前扫描对应的 IMU 数据；
  // 2. stat 为上一时刻状态，函数返回后将被原位更新；
  // 3. cur_pcl_un_ 指向的点云会被写入去畸变结果。
  void Process2(FusionMeasureGroup &lidar_meas, StatesGroup &stat, PointCloudXYZI::Ptr cur_pcl_un_);
  // 仅执行点云去畸变，不额外封装上层处理流程。
  void UndistortPcl(FusionMeasureGroup &lidar_meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out);

  // IMU 调试输出文件流。
  ofstream fout_imu;
  // 当前初始化阶段估计得到的平均加速度模长。
  double IMU_mean_acc_norm;
  // 当前去除偏置后的角速度。
  V3D unbiased_gyr;

  // 加速度噪声协方差缩放。
  V3D cov_acc;
  // 角速度噪声协方差缩放。
  V3D cov_gyr;
  // 陀螺仪偏置随机游走协方差。
  V3D cov_bias_gyr;
  // 加速度计偏置随机游走协方差。
  V3D cov_bias_acc;
  // 曝光时间倒数协方差。
  double cov_inv_expo;
  // 第一帧 LiDAR 的时间戳，用于建立时序参考。
  double first_lidar_time;
  // 是否已经建立了 IMU 与 LiDAR 的时间基准。
  bool imu_time_init = false;
  // 是否仍需要执行 IMU 初始化。
  bool imu_need_init = true;
  // 常用单位阵缓存。
  M3D Eye3d;
  // 常用零向量缓存。
  V3D Zero3d;
  // 当前 LiDAR 类型。
  int lidar_type;

private:
  // 利用静止段或初始段数据估计重力方向、偏置等初值。
  void IMU_init(const FusionMeasure &meas, StatesGroup &state, int &N);
  // 在未启用 IMU 时，仅基于当前姿态假设做直通式点云输出。
  void Forward_without_imu(FusionMeasureGroup &meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out);
  // 待处理点云缓存。
  PointCloudXYZI pcl_wait_proc;
  // 上一个 IMU 样本，用于积分。
  ImuSample last_imu;
  // 指向当前输出去畸变点云的共享指针。
  PointCloudXYZI::Ptr cur_pcl_un_;
  // 扫描内离散姿态序列，供逐点去畸变插值使用。
  vector<Pose6D> IMUpose;
  // LiDAR 到 IMU 的旋转外参。
  M3D Lid_rot_to_IMU;
  // LiDAR 到 IMU 的平移外参。
  V3D Lid_offset_to_IMU;
  // 初始化阶段累计的平均加速度。
  V3D mean_acc;
  // 初始化阶段累计的平均角速度。
  V3D mean_gyr;
  // 上一次角速度。
  V3D angvel_last;
  // 上一次特定坐标系下的加速度。
  V3D acc_s_last;
  // 上次传播终点时间。
  double last_prop_end_time;
  // 上一帧扫描时间。
  double time_last_scan;
  // 初始化迭代计数与最大初始化帧数。
  int init_iter_num = 1, MAX_INI_COUNT = 20;
  // 是否正在处理第一帧。
  bool b_first_frame = true;
  // 功能开关：是否启用 IMU。
  bool imu_en = true;
  // 功能开关：是否估计重力。
  bool gravity_est_en = true;
  // 功能开关：是否估计 ba/bg。
  bool ba_bg_est_en = true;
  // 功能开关：是否估计曝光时间相关量。
  bool exposure_estimate_en = true;
};
typedef std::shared_ptr<ImuProcess> ImuProcessPtr;
#endif
