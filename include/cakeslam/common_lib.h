/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef COMMON_LIB_H
#define COMMON_LIB_H

#include <cake_slam/utils/so3_math.h>
#include <cake_slam/utils/types.h>
#include <cake_slam/utils/color.h>
#include <cake_slam/utils/utils.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <cake_slam/imu_sample.h>
#include <sophus/se3.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Transform.hpp>
#include <tf2/LinearMath/Quaternion.hpp>

using namespace std;
// using namespace Eigen;   // avoid cmake error: reference to ‘Matrix’ is ambiguous
using namespace Sophus;

#define print_line std::cout << __FILE__ << ", " << __LINE__ << std::endl;
#define G_m_s2 (9.81)   // Gravaty const in GuangDong/China
#define DIM_STATE (19)  // Dimension of states (Let Dim(SO(3)) = 3)
#define INIT_COV (0.01)
#define SIZE_LARGE (500)
#define SIZE_SMALL (100)
#define VEC_FROM_ARRAY(v) v[0], v[1], v[2]
#define MAT_FROM_ARRAY(v) v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]
#define DEBUG_FILE_DIR(name) (string(string(ROOT_DIR) + "Log/" + name))

enum LID_TYPE
{
  AVIA = 1,
  VELO16 = 2,
  OUST64 = 3,
  L515 = 4,
  XT32 = 5,
  PANDAR128 = 6
};
enum SLAM_MODE
{
  ONLY_LO = 0,
  ONLY_LIO = 1,
  LIVO = 2
};
enum EKF_STATE
{
  WAIT = 0,
  VIO = 1,
  LIO = 2,
  LO = 3
};

// 一组对齐到同一视觉时间的输入数据。
// 在视觉部分中通常表示“一张图像 + 覆盖该时间段的 IMU 序列”。
struct MeasureGroup
{
  // 当前视觉测量对应的时间戳。
  double vio_time;
  // 如果同一组数据还关联了 LIO 更新时间，则记录该时间戳。
  double lio_time;
  // 覆盖该时间区间的 IMU 数据，要求按时间升序排列。
  deque<ImuSample> imu;
  // 当前图像帧。
  cv::Mat img;
  MeasureGroup()
  {
    vio_time = 0.0;
    lio_time = 0.0;
  };
};

// 一组 LiDAR 扫描及其配套缓存。
// 该结构是 LIO 主循环的核心输入/中间载体。
struct LidarMeasureGroup
{
  // 当前点云帧起始时间。
  double lidar_frame_beg_time;
  // 当前点云帧结束时间。
  double lidar_frame_end_time;
  // 最近一次 LIO 成功更新时间。
  double last_lio_update_time;
  // 原始 LiDAR 点云。
  PointCloudXYZI::Ptr lidar;
  // 当前处理中的点云缓存。
  PointCloudXYZI::Ptr pcl_proc_cur;
  // 下一帧预取/处理中点云缓存。
  PointCloudXYZI::Ptr pcl_proc_next;
  // 与该 LiDAR 时间段对应的 IMU/图像测量序列。
  deque<struct MeasureGroup> measures;
  // 当前测量由哪种模式驱动更新。
  EKF_STATE lio_vio_flg;
  // 当前扫描编号或序列号。
  int lidar_scan_index_now;

  LidarMeasureGroup()
  {
    lidar_frame_beg_time = -0.0;
    lidar_frame_end_time = 0.0;
    last_lio_update_time = -1.0;
    lio_vio_flg = WAIT;
    this->lidar.reset(new PointCloudXYZI());
    this->pcl_proc_cur.reset(new PointCloudXYZI());
    this->pcl_proc_next.reset(new PointCloudXYZI());
    this->measures.clear();
    lidar_scan_index_now = 0;
    last_lio_update_time = -1.0;
  };
};

// 带有协方差信息的点结构。
// 主要用于体素地图配准时，统一保存点在不同坐标系中的位置及其误差传播结果。
typedef struct pointWithVar
{
  Eigen::Vector3d point_b;     // 点在 LiDAR 本体坐标系下的位置。
  Eigen::Vector3d point_i;     // 点在 IMU/body 坐标系下的位置。
  Eigen::Vector3d point_w;     // 点在世界坐标系下的位置。
  Eigen::Matrix3d var_nostate; // 不考虑状态不确定性时，仅由量测噪声传播得到的协方差。
  Eigen::Matrix3d body_var;    // 点在机体系下的协方差。
  Eigen::Matrix3d var;         // 完整点协方差，通常包含状态传播影响。
  Eigen::Matrix3d point_crossmat; // 点坐标的叉乘矩阵，用于线性化旋转扰动项。
  Eigen::Vector3d normal;      // 关联局部平面的法向量。
  pointWithVar()
  {
    var_nostate = Eigen::Matrix3d::Zero();
    var = Eigen::Matrix3d::Zero();
    body_var = Eigen::Matrix3d::Zero();
    point_crossmat = Eigen::Matrix3d::Zero();
    point_b = Eigen::Vector3d::Zero();
    point_i = Eigen::Vector3d::Zero();
    point_w = Eigen::Vector3d::Zero();
    normal = Eigen::Vector3d::Zero();
  };
} pointWithVar;


// LIO 内部的完整状态向量。
// 该状态既保存名义值，也保存 19 维误差状态协方差。
struct StatesGroup
{
  StatesGroup()
  {
    this->rot_end = M3D::Identity();
    this->pos_end = V3D::Zero();
    this->vel_end = V3D::Zero();
    this->bias_g = V3D::Zero();
    this->bias_a = V3D::Zero();
    this->gravity = V3D::Zero();
    this->inv_expo_time = 1.0;
    this->cov = MD(DIM_STATE, DIM_STATE)::Identity() * INIT_COV;
    this->cov(6, 6) = 0.00001;
    this->cov.block<9, 9>(10, 10) = MD(9, 9)::Identity() * 0.00001;
  };

  StatesGroup(const StatesGroup &b)
  {
    this->rot_end = b.rot_end;
    this->pos_end = b.pos_end;
    this->vel_end = b.vel_end;
    this->bias_g = b.bias_g;
    this->bias_a = b.bias_a;
    this->gravity = b.gravity;
    this->inv_expo_time = b.inv_expo_time;
    this->cov = b.cov;
  };

  StatesGroup &operator=(const StatesGroup &b)
  {
    this->rot_end = b.rot_end;
    this->pos_end = b.pos_end;
    this->vel_end = b.vel_end;
    this->bias_g = b.bias_g;
    this->bias_a = b.bias_a;
    this->gravity = b.gravity;
    this->inv_expo_time = b.inv_expo_time;
    this->cov = b.cov;
    return *this;
  };

  StatesGroup operator+(const Matrix<double, DIM_STATE, 1> &state_add)
  {
    StatesGroup a;
    a.rot_end = this->rot_end * Exp(state_add(0, 0), state_add(1, 0), state_add(2, 0));
    a.pos_end = this->pos_end + state_add.block<3, 1>(3, 0);
    a.inv_expo_time = this->inv_expo_time + state_add(6, 0);
    a.vel_end = this->vel_end + state_add.block<3, 1>(7, 0);
    a.bias_g = this->bias_g + state_add.block<3, 1>(10, 0);
    a.bias_a = this->bias_a + state_add.block<3, 1>(13, 0);
    a.gravity = this->gravity + state_add.block<3, 1>(16, 0);

    a.cov = this->cov;
    return a;
  };

  StatesGroup &operator+=(const Matrix<double, DIM_STATE, 1> &state_add)
  {
    this->rot_end = this->rot_end * Exp(state_add(0, 0), state_add(1, 0), state_add(2, 0));
    this->pos_end += state_add.block<3, 1>(3, 0);
    this->inv_expo_time += state_add(6, 0);
    this->vel_end += state_add.block<3, 1>(7, 0);
    this->bias_g += state_add.block<3, 1>(10, 0);
    this->bias_a += state_add.block<3, 1>(13, 0);
    this->gravity += state_add.block<3, 1>(16, 0);
    return *this;
  };

  Matrix<double, DIM_STATE, 1> operator-(const StatesGroup &b)
  {
    Matrix<double, DIM_STATE, 1> a;
    M3D rotd(b.rot_end.transpose() * this->rot_end);
    a.block<3, 1>(0, 0) = Log(rotd);
    a.block<3, 1>(3, 0) = this->pos_end - b.pos_end;
    a(6, 0) = this->inv_expo_time - b.inv_expo_time;
    a.block<3, 1>(7, 0) = this->vel_end - b.vel_end;
    a.block<3, 1>(10, 0) = this->bias_g - b.bias_g;
    a.block<3, 1>(13, 0) = this->bias_a - b.bias_a;
    a.block<3, 1>(16, 0) = this->gravity - b.gravity;
    return a;
  };

  void resetpose()
  {
    this->rot_end = M3D::Identity();
    this->pos_end = V3D::Zero();
    this->vel_end = V3D::Zero();
  }

  M3D rot_end;                              // 当前扫描结束时刻的姿态估计。
  V3D pos_end;                              // 当前扫描结束时刻的位置估计，世界系。
  V3D vel_end;                              // 当前扫描结束时刻的速度估计，世界系。
  double inv_expo_time;                     // 曝光时间倒数的估计值，主要为视觉/时延模型预留。
  V3D bias_g;                               // 陀螺仪零偏估计。
  V3D bias_a;                               // 加速度计零偏估计。
  V3D gravity;                              // 重力向量估计。
  Matrix<double, DIM_STATE, DIM_STATE> cov; // 19 维误差状态协方差矩阵。
};

// 组装一条 Pose6D 记录。
// 主要用于保存 IMU 传播轨迹，便于点云去畸变时对逐点姿态进行插值。
template <typename T>
auto set_pose6d(const double t, const Matrix<T, 3, 1> &a, const Matrix<T, 3, 1> &g, const Matrix<T, 3, 1> &v, const Matrix<T, 3, 1> &p,
                const Matrix<T, 3, 3> &R)
{
  Pose6D rot_kp;
  rot_kp.offset_time = t;
  for (int i = 0; i < 3; i++)
  {
    rot_kp.acc[i] = a(i);
    rot_kp.gyr[i] = g(i);
    rot_kp.vel[i] = v(i);
    rot_kp.pos[i] = p(i);
    for (int j = 0; j < 3; j++)
      rot_kp.rot[i * 3 + j] = R(i, j);
  }
  // Map<M3D>(rot_kp.rot, 3,3) = R;
  return move(rot_kp);
}

#endif
