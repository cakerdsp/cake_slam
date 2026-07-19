/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef COMMON_LIB_MULTI_CAM_H
#define COMMON_LIB_MULTI_CAM_H

#include <utils/so3_math.h>
#include <utils/types.h>
#include <utils/color.h>
#include <utils/utils.h>
#include <opencv2/opencv.hpp>
#include <cstdint>
#include <stdexcept>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Quaternion.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <vikit/sophus_compat.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Quaternion.h>

using namespace std;
// Avoid `using namespace Eigen`; Matrix is otherwise ambiguous in this project.
using namespace Sophus;

#define print_line std::cout << __FILE__ << ", " << __LINE__ << std::endl;
#define G_m_s2 (9.81)   // Gravaty const in GuangDong/China
#define BASE_STATE_DIM (18)  // State dimension excluding per-camera exposure/extrinsic states.
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

struct MultiCameraFrame
{
  uint64_t stamp_ns = 0;
  double timestamp = 0.0;  //!< Anchor capture time used by LIO/VIO synchronization.
  int frame_id = -1;
  std::vector<cv::Mat> images;
  std::vector<uint64_t> image_stamp_ns;
  std::vector<double> raw_timestamps;
  std::vector<double> corrected_timestamps;
  std::vector<double> capture_timestamps;
  std::vector<double> td_used;
  std::vector<int> time_offset_group;
};

struct MeasureGroup
{
  double vio_time;
  double lio_time;
  double img_time;
  deque<sensor_msgs::Imu::ConstPtr> imu;
  cv::Mat img;
  MultiCameraFrame multi_cam_frame;
  bool has_multi_cam_frame;
  MeasureGroup()
  {
    vio_time = 0.0;
    lio_time = 0.0;
    img_time = 0.0;
    has_multi_cam_frame = false;
  };
};

struct LidarMeasureGroup
{
  double lidar_frame_beg_time;
  double lidar_frame_end_time;
  double last_lio_update_time;
  PointCloudXYZI::Ptr lidar;
  PointCloudXYZI::Ptr pcl_proc_cur;
  PointCloudXYZI::Ptr pcl_proc_next;
  deque<struct MeasureGroup> measures;
  EKF_STATE lio_vio_flg;
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

typedef struct pointWithVar
{
  Eigen::Vector3d point_b;     // point in the lidar body frame
  Eigen::Vector3d point_i;     // point in the imu body frame
  Eigen::Vector3d point_w;     // point in the world frame
  Eigen::Matrix3d var_nostate; // the var removed the state covarience
  Eigen::Matrix3d body_var;
  Eigen::Matrix3d var;
  Eigen::Matrix3d point_crossmat;
  Eigen::Vector3d normal;
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

struct StatesGroup
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit StatesGroup(int camera_count = 1)
  {
    this->rot_end = M3D::Identity();
    this->pos_end = V3D::Zero();
    this->vel_end = V3D::Zero();
    this->bias_g = V3D::Zero();
    this->bias_a = V3D::Zero();
    this->gravity = V3D::Zero();
    configureCameras(camera_count, 1.0, 1, 0.0);
  };

  StatesGroup(const StatesGroup &b)
  {
    this->rot_end = b.rot_end;
    this->pos_end = b.pos_end;
    this->vel_end = b.vel_end;
    this->bias_g = b.bias_g;
    this->bias_a = b.bias_a;
    this->gravity = b.gravity;
    this->num_cameras = b.num_cameras;
    this->num_time_offset_groups = b.num_time_offset_groups;
    this->inv_expo_time = b.inv_expo_time;
    this->time_offset = b.time_offset;
    this->time_offset_prior = b.time_offset_prior;
    this->Rcl = b.Rcl;
    this->Pcl = b.Pcl;
    this->Rcl_prior = b.Rcl_prior;
    this->Pcl_prior = b.Pcl_prior;
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
    this->num_cameras = b.num_cameras;
    this->num_time_offset_groups = b.num_time_offset_groups;
    this->inv_expo_time = b.inv_expo_time;
    this->time_offset = b.time_offset;
    this->time_offset_prior = b.time_offset_prior;
    this->Rcl = b.Rcl;
    this->Pcl = b.Pcl;
    this->Rcl_prior = b.Rcl_prior;
    this->Pcl_prior = b.Pcl_prior;
    this->cov = b.cov;
    return *this;
  };

  int stateDim() const { return BASE_STATE_DIM + num_cameras + num_time_offset_groups + 6 * num_cameras; }
  int exposureIndex(int camera_id) const { return 6 + camera_id; }
  int timeOffsetBaseIndex() const { return 6 + num_cameras; }
  int timeOffsetIndex(int group_id) const { return timeOffsetBaseIndex() + group_id; }
  int velocityIndex() const { return 6 + num_cameras + num_time_offset_groups; }
  int gyroBiasIndex() const { return 9 + num_cameras + num_time_offset_groups; }
  int accelBiasIndex() const { return 12 + num_cameras + num_time_offset_groups; }
  int gravityIndex() const { return 15 + num_cameras + num_time_offset_groups; }
  int extrinsicBaseIndex() const { return BASE_STATE_DIM + num_cameras + num_time_offset_groups; }
  int extrinsicIndex(int camera_id) const { return extrinsicBaseIndex() + 6 * camera_id; }
  int extrinsicRotIndex(int camera_id) const { return extrinsicIndex(camera_id); }
  int extrinsicTransIndex(int camera_id) const { return extrinsicIndex(camera_id) + 3; }

  void configureCameras(int camera_count, double inv_expo_init = 1.0,
                        int time_offset_group_count = 1, double time_offset_init = 0.0)
  {
    if (camera_count < 1) throw std::invalid_argument("StatesGroup requires at least one camera");
    if (time_offset_group_count < 1) throw std::invalid_argument("StatesGroup requires at least one time-offset group");
    num_cameras = camera_count;
    num_time_offset_groups = time_offset_group_count;
    inv_expo_time = Eigen::VectorXd::Constant(num_cameras, inv_expo_init);
    time_offset = Eigen::VectorXd::Constant(num_time_offset_groups, time_offset_init);
    time_offset_prior = time_offset;
    Rcl.assign(num_cameras, M3D::Identity());
    Pcl.assign(num_cameras, V3D::Zero());
    Rcl_prior = Rcl;
    Pcl_prior = Pcl;
    cov = Eigen::MatrixXd::Identity(stateDim(), stateDim()) * INIT_COV;
    cov.block(6, 6, num_cameras, num_cameras) = Eigen::MatrixXd::Identity(num_cameras, num_cameras) * 0.00001;
    cov.block(timeOffsetBaseIndex(), timeOffsetBaseIndex(), num_time_offset_groups, num_time_offset_groups) =
        Eigen::MatrixXd::Identity(num_time_offset_groups, num_time_offset_groups) * 1.0e-6;
    cov.block(gyroBiasIndex(), gyroBiasIndex(), 9, 9) = Eigen::MatrixXd::Identity(9, 9) * 0.00001;
    for (int camera_id = 0; camera_id < num_cameras; ++camera_id)
    {
      cov.block<3, 3>(extrinsicRotIndex(camera_id), extrinsicRotIndex(camera_id)).setIdentity();
      cov.block<3, 3>(extrinsicRotIndex(camera_id), extrinsicRotIndex(camera_id)) *= 1.0e-6;
      cov.block<3, 3>(extrinsicTransIndex(camera_id), extrinsicTransIndex(camera_id)).setIdentity();
      cov.block<3, 3>(extrinsicTransIndex(camera_id), extrinsicTransIndex(camera_id)) *= 1.0e-6;
    }
  }

  void setCameraExtrinsic(int camera_id, const M3D &R_cl, const V3D &P_cl, bool reset_prior = true)
  {
    if (camera_id < 0 || camera_id >= num_cameras) throw std::out_of_range("invalid camera extrinsic index");
    if (!R_cl.allFinite() || R_cl.determinant() <= 0.0)
      throw std::invalid_argument("camera extrinsic rotation must be finite with positive determinant");
    Eigen::Quaterniond q_cl(R_cl);
    if (!q_cl.coeffs().allFinite() || q_cl.norm() <= 1.0e-12)
      throw std::invalid_argument("camera extrinsic rotation cannot be projected to SO(3)");
    const M3D R_cl_so3 = q_cl.normalized().toRotationMatrix();
    Rcl[camera_id] = R_cl_so3;
    Pcl[camera_id] = P_cl;
    if (reset_prior)
    {
      Rcl_prior[camera_id] = R_cl_so3;
      Pcl_prior[camera_id] = P_cl;
    }
  }

  void setTimeOffset(int group_id, double td)
  {
    if (group_id < 0 || group_id >= num_time_offset_groups) throw std::out_of_range("invalid time-offset group index");
    time_offset[group_id] = td;
    time_offset_prior[group_id] = td;
  }

  void setTimeOffsetCovariance(int group_id, double var)
  {
    if (group_id < 0 || group_id >= num_time_offset_groups) throw std::out_of_range("invalid time-offset covariance index");
    if (cov.rows() != stateDim() || cov.cols() != stateDim())
      cov = Eigen::MatrixXd::Identity(stateDim(), stateDim()) * INIT_COV;
    cov(timeOffsetIndex(group_id), timeOffsetIndex(group_id)) = var;
  }

  void setCameraExtrinsicCovariance(int camera_id, double rot_var, double trans_var)
  {
    if (camera_id < 0 || camera_id >= num_cameras) throw std::out_of_range("invalid camera extrinsic covariance index");
    if (cov.rows() != stateDim() || cov.cols() != stateDim())
      cov = Eigen::MatrixXd::Identity(stateDim(), stateDim()) * INIT_COV;
    cov.block<3, 3>(extrinsicRotIndex(camera_id), extrinsicRotIndex(camera_id)).setIdentity();
    cov.block<3, 3>(extrinsicRotIndex(camera_id), extrinsicRotIndex(camera_id)) *= rot_var;
    cov.block<3, 3>(extrinsicTransIndex(camera_id), extrinsicTransIndex(camera_id)).setIdentity();
    cov.block<3, 3>(extrinsicTransIndex(camera_id), extrinsicTransIndex(camera_id)) *= trans_var;
  }

  void validateIncrement(const Eigen::VectorXd &state_add) const
  {
    if (state_add.size() != stateDim()) throw std::invalid_argument("state increment dimension does not match StatesGroup");
  }

  StatesGroup operator+(const Eigen::VectorXd &state_add) const
  {
    validateIncrement(state_add);
    StatesGroup a(*this);
    a.rot_end = this->rot_end * Exp(state_add(0, 0), state_add(1, 0), state_add(2, 0));
    a.pos_end = this->pos_end + state_add.block<3, 1>(3, 0);
    a.inv_expo_time = this->inv_expo_time + state_add.segment(6, num_cameras);
    a.time_offset = this->time_offset + state_add.segment(timeOffsetBaseIndex(), num_time_offset_groups);
    a.vel_end = this->vel_end + state_add.segment<3>(velocityIndex());
    a.bias_g = this->bias_g + state_add.segment<3>(gyroBiasIndex());
    a.bias_a = this->bias_a + state_add.segment<3>(accelBiasIndex());
    a.gravity = this->gravity + state_add.segment<3>(gravityIndex());
    for (int camera_id = 0; camera_id < num_cameras; ++camera_id)
    {
      const int ridx = extrinsicRotIndex(camera_id);
      const int tidx = extrinsicTransIndex(camera_id);
      a.Rcl[camera_id] = this->Rcl[camera_id] * Exp(state_add(ridx, 0), state_add(ridx + 1, 0), state_add(ridx + 2, 0));
      a.Pcl[camera_id] = this->Pcl[camera_id] + state_add.segment<3>(tidx);
    }

    return a;
  };

  StatesGroup &operator+=(const Eigen::VectorXd &state_add)
  {
    validateIncrement(state_add);
    this->rot_end = this->rot_end * Exp(state_add(0, 0), state_add(1, 0), state_add(2, 0));
    this->pos_end += state_add.block<3, 1>(3, 0);
    this->inv_expo_time += state_add.segment(6, num_cameras);
    this->time_offset += state_add.segment(timeOffsetBaseIndex(), num_time_offset_groups);
    this->vel_end += state_add.segment<3>(velocityIndex());
    this->bias_g += state_add.segment<3>(gyroBiasIndex());
    this->bias_a += state_add.segment<3>(accelBiasIndex());
    this->gravity += state_add.segment<3>(gravityIndex());
    for (int camera_id = 0; camera_id < num_cameras; ++camera_id)
    {
      const int ridx = extrinsicRotIndex(camera_id);
      const int tidx = extrinsicTransIndex(camera_id);
      this->Rcl[camera_id] = this->Rcl[camera_id] * Exp(state_add(ridx, 0), state_add(ridx + 1, 0), state_add(ridx + 2, 0));
      this->Pcl[camera_id] += state_add.segment<3>(tidx);
    }
    return *this;
  };

  Eigen::VectorXd operator-(const StatesGroup &b) const
  {
    if (num_cameras != b.num_cameras) throw std::invalid_argument("StatesGroup camera counts do not match");
    if (num_time_offset_groups != b.num_time_offset_groups)
      throw std::invalid_argument("StatesGroup time-offset group counts do not match");
    Eigen::VectorXd a(stateDim());
    M3D rotd(b.rot_end.transpose() * this->rot_end);
    a.block<3, 1>(0, 0) = Log(rotd);
    a.block<3, 1>(3, 0) = this->pos_end - b.pos_end;
    a.segment(6, num_cameras) = this->inv_expo_time - b.inv_expo_time;
    a.segment(timeOffsetBaseIndex(), num_time_offset_groups) = this->time_offset - b.time_offset;
    a.segment<3>(velocityIndex()) = this->vel_end - b.vel_end;
    a.segment<3>(gyroBiasIndex()) = this->bias_g - b.bias_g;
    a.segment<3>(accelBiasIndex()) = this->bias_a - b.bias_a;
    a.segment<3>(gravityIndex()) = this->gravity - b.gravity;
    for (int camera_id = 0; camera_id < num_cameras; ++camera_id)
    {
      const int ridx = extrinsicRotIndex(camera_id);
      const int tidx = extrinsicTransIndex(camera_id);
      const M3D dR_cl = b.Rcl[camera_id].transpose() * this->Rcl[camera_id];
      a.segment<3>(ridx) = Log(dR_cl);
      a.segment<3>(tidx) = this->Pcl[camera_id] - b.Pcl[camera_id];
    }
    return a;
  };

  void resetpose()
  {
    this->rot_end = M3D::Identity();
    this->pos_end = V3D::Zero();
    this->vel_end = V3D::Zero();
  }

  M3D rot_end;                              // the estimated attitude (rotation matrix) at the end lidar point
  V3D pos_end;                              // the estimated position at the end lidar point (world frame)
  V3D vel_end;                              // the estimated velocity at the end lidar point (world frame)
  int num_cameras = 1;
  int num_time_offset_groups = 1;
  Eigen::VectorXd inv_expo_time;             // Per-camera estimated inverse exposure time.
  Eigen::VectorXd time_offset;               // Per-group camera time offset w.r.t. LiDAR/IMU clock.
  Eigen::VectorXd time_offset_prior;         // Initial/manual prior for each time-offset group.
  std::vector<M3D> Rcl;                      // Per-camera camera<-lidar rotation.
  std::vector<V3D> Pcl;                      // Per-camera camera<-lidar translation.
  std::vector<M3D> Rcl_prior;                // Initial/prior camera<-lidar rotation.
  std::vector<V3D> Pcl_prior;                // Initial/prior camera<-lidar translation.
  V3D bias_g;                               // gyroscope bias
  V3D bias_a;                               // accelerator bias
  V3D gravity;                              // the estimated gravity acceleration
  Eigen::MatrixXd cov;                       // Dynamic covariance.
};

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
