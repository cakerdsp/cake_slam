/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

// 模块功能：LiDAR 点云预处理与特征提取接口定义，
// 包含去畸变、滤波与几何特征分类所需的数据结构与参数。

#ifndef PREPROCESS_H_
#define PREPROCESS_H_

#include "cake_slam/common_lib.h"
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <pcl_conversions/pcl_conversions.h>

using namespace std;

#define IS_VALID(a) ((abs(a) > 1e8) ? true : false)

enum LiDARFeature
{
  Nor,
  Poss_Plane,
  Real_Plane,
  Edge_Jump,
  Edge_Plane,
  Wire,
  ZeroPoint
};
enum Surround
{
  Prev,
  Next
};
enum E_jump
{
  Nr_nor,
  Nr_zero,
  Nr_180,
  Nr_inf,
  Nr_blind
};

// 用于特征提取时保存原始点的几何上下文信息。
// 这些字段不是最终输出，而是帮助判断“当前点属于平面/边缘/无效点”的中间量。
struct orgtype
{
  // 当前点距离传感器原点的量测距离。
  double range;
  // 与邻域点之间的距离度量。
  double dista;
  // 与前后邻域方向相关的角度信息。
  double angle[2];
  // 邻域线段/平面相交性指标。
  double intersect;
  // 前向与后向跳变类型。
  E_jump edj[2];
  // 当前点最终判定出来的特征类别。
  LiDARFeature ftype;
  orgtype()
  {
    range = 0;
    edj[Prev] = Nr_nor;
    edj[Next] = Nr_nor;
    ftype = Nor;
    intersect = 2;
  }
};

/*** Velodyne ***/
namespace velodyne_ros
{
struct EIGEN_ALIGN16 Point
{
  PCL_ADD_POINT4D;
  float intensity;
  float time;
  std::uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
} // namespace velodyne_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, time, time)(std::uint16_t, ring, ring))
/****************/

/*** Ouster ***/
namespace ouster_ros
{
struct EIGEN_ALIGN16 Point
{
  PCL_ADD_POINT4D;
  float intensity;
  std::uint32_t t;
  std::uint16_t reflectivity;
  uint8_t ring;
  std::uint16_t ambient;
  std::uint32_t range;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
} // namespace ouster_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point, (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
                                  (std::uint32_t, t, t)(std::uint16_t, reflectivity,
                                                        reflectivity)(std::uint8_t, ring, ring)(std::uint16_t, ambient, ambient)(std::uint32_t, range, range))
/****************/

/*** Hesai_XT32 ***/
namespace xt32_ros
{
struct EIGEN_ALIGN16 Point
{
  PCL_ADD_POINT4D;
  float intensity;
  double timestamp;
  std::uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
} // namespace xt32_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(xt32_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(double, timestamp, timestamp)(std::uint16_t, ring, ring))
/*****************/

/*** Hesai_Pandar128 ***/
namespace Pandar128_ros
{
struct EIGEN_ALIGN16 Point
{
  PCL_ADD_POINT4D;
  float timestamp;
  uint8_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
} // namespace Pandar128_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(Pandar128_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(float, timestamp, timestamp))
/*****************/

/*** Odin1 raw cloud ***/
namespace odin1_ros
{
struct EIGEN_ALIGN16 Point
{
  float x;
  float y;
  float z;
  std::uint8_t intensity;
  std::uint16_t confidence;
  float offset_time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
} // namespace odin1_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(odin1_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)
                                  (std::uint8_t, intensity, intensity)
                                  (std::uint16_t, confidence, confidence)
                                  (float, offset_time, offset_time))
/*****************/

// 预处理器：负责解析各雷达格式点云，并输出统一特征点云。
class Preprocess
{
public:
  //   EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Preprocess();
  ~Preprocess();

  // 处理 Livox 自定义点云消息并输出项目统一点云格式。
  // 输入要求：消息中必须带有逐点时间和线束/标签信息。
  void process(const livox_ros_driver2::msg::CustomMsg::SharedPtr &msg, PointCloudXYZI::Ptr &pcl_out);
  // 处理标准 PointCloud2 消息，适用于 Velodyne/Ouster/Hesai 等驱动。
  void process(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg, PointCloudXYZI::Ptr &pcl_out);
  // 设置预处理参数。
  // feat_en：是否提取特征；
  // lid_type：雷达型号；
  // bld：盲区阈值；
  // pfilt_num：点过滤步长。
  void set(bool feat_en, int lid_type, double bld, int pfilt_num);
  void set_odin_confidence_threshold(int confidence_threshold);

  // 完整点云、角点、平面点输出缓存。
  PointCloudXYZI pl_full, pl_corn, pl_surf;
  // 各线束点云缓存，最多支持 128 线雷达。
  PointCloudXYZI pl_buff[128];
  // 各线束点的几何属性缓存，和 pl_buff 按索引一一对应。
  vector<orgtype> typess[128];
  // 当前雷达类型、滤波步长、线数和扫描频率。
  int lidar_type, point_filter_num, N_SCANS, SCAN_RATE, odin_confidence_threshold;
  
  // 盲区阈值及其平方，用于快速距离判定。
  double blind, blind_sqr;
  // 是否启用特征提取；消息中是否已自带偏移时间。
  bool feature_enabled, given_offset_time;
  // 调试用发布器，可发布完整/平面/角点云。
  std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> pub_full;
  std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> pub_surf;
  std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> pub_corn;

private:
  // 按不同雷达驱动格式解析原始消息。
  void avia_handler(const livox_ros_driver2::msg::CustomMsg::SharedPtr &msg);
  void oust64_handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);
  void velodyne_handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);
  void xt32_handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);
  void Pandar128_handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);
  void odin1_handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);
  void l515_handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);
  // 针对单条 scan line 做几何特征分类。
  void give_feature(PointCloudXYZI &pl, vector<orgtype> &types);
  // 发布调试点云。
  void pub_func(PointCloudXYZI &pl, const rclcpp::Time &ct);
  // 判断从当前点开始的一段邻域是否构成平面。
  int plane_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, uint &i_nex, Eigen::Vector3d &curr_direct);
  // 判断是否为小平面片。
  bool small_plane(const PointCloudXYZI &pl, vector<orgtype> &types, uint i_cur, uint &i_nex, Eigen::Vector3d &curr_direct);
  // 判断当前点是否属于跳边缘。
  bool edge_jump_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, Surround nor_dir);

  // 每组候选点的处理窗口大小。
  int group_size;
  // 若干几何阈值，控制平面/边缘判别逻辑。
  double disA, disB, inf_bound;
  double limit_maxmid, limit_midmin, limit_maxmin;
  double p2l_ratio;
  double jump_up_limit, jump_down_limit;
  double cos160;
  double edgea, edgeb;
  double smallp_intersect, smallp_ratio;
  // 临时分量缓存，避免循环内重复分配。
  double vx, vy, vz;
};
typedef std::shared_ptr<Preprocess> PreprocessPtr;

#endif // PREPROCESS_H_
