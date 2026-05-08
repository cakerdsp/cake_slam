/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef VOXEL_MAP_H_
#define VOXEL_MAP_H_

#include "cake_slam/common_lib.h"
#include <Eigen/Dense>
#include <fstream>
#include <math.h>
#include <mutex>
#include <omp.h>
#include <pcl/common/io.h>
#include <rclcpp/rclcpp.hpp>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#define VOXELMAP_HASH_P 116101
#define VOXELMAP_MAX_N 10000000000

// 体素地图相关配置。
// 这些参数控制体素层级、平面初始化、残差构建以及局部地图滑动范围。
typedef struct VoxelMapConfig
{
  double max_voxel_size_;
  int max_layer_;
  int max_iterations_;
  std::vector<int64_t> layer_init_num_;
  int max_points_num_;
  double planner_threshold_;
  double beam_err_;
  double dept_err_;
  double sigma_num_;
  bool is_pub_plane_map_;

  // config of local map sliding
  double sliding_thresh;
  bool map_sliding_en;
  int half_map_size;
} VoxelMapConfig;

// 单个点到局部平面的匹配结果。
// 该结构是状态估计中构造残差的直接输入。
typedef struct PointToPlane
{
  // 点在机体系下的位置。
  Eigen::Vector3d point_b_;
  // 点在世界系下的位置。
  Eigen::Vector3d point_w_;
  // 匹配到的局部平面法向量。
  Eigen::Vector3d normal_;
  // 匹配到的局部平面中心。
  Eigen::Vector3d center_;
  // 平面参数协方差。
  Eigen::Matrix<double, 6, 6> plane_var_;
  // 点在机体系下的协方差。
  M3D body_cov_;
  // 当前匹配发生在八叉树的哪一层。
  int layer_;
  // 平面方程常数项。
  double d_;
  // 平面最小特征值或质量指标。
  double eigen_value_;
  // 当前匹配是否有效。
  bool is_valid_;
  // 点到平面的有符号或绝对距离。
  float dis_to_plane_;
} PointToPlane;

// 一个体素内拟合出的局部平面模型。
typedef struct VoxelPlane
{
  // 平面质心。
  Eigen::Vector3d center_;
  // 主法向量。
  Eigen::Vector3d normal_;
  // 平面局部 y 方向。
  Eigen::Vector3d y_normal_;
  // 平面局部 x 方向。
  Eigen::Vector3d x_normal_;
  // 点云协方差。
  Eigen::Matrix3d covariance_;
  // 平面参数协方差。
  Eigen::Matrix<double, 6, 6> plane_var_;
  // 平面半径，用于描述覆盖范围。
  float radius_ = 0;
  // 三个特征值，用于判断是否可近似为平面。
  float min_eigen_value_ = 1;
  float mid_eigen_value_ = 1;
  float max_eigen_value_ = 1;
  // 平面方程常数项。
  float d_ = 0;
  // 当前平面由多少个点拟合得到。
  int points_size_ = 0;
  // 是否被判定为有效平面。
  bool is_plane_ = false;
  // 是否完成初始化。
  bool is_init_ = false;
  // 平面唯一标识，用于可视化。
  int id_ = 0;
  // 本次更新中是否被修改。
  bool is_update_ = false;
  VoxelPlane()
  {
    plane_var_ = Eigen::Matrix<double, 6, 6>::Zero();
    covariance_ = Eigen::Matrix3d::Zero();
    center_ = Eigen::Vector3d::Zero();
    normal_ = Eigen::Vector3d::Zero();
  }
} VoxelPlane;

class VOXEL_LOCATION
{
public:
  // 体素在三维整数网格中的索引。
  int64_t x, y, z;

  VOXEL_LOCATION(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}

  bool operator==(const VOXEL_LOCATION &other) const { return (x == other.x && y == other.y && z == other.z); }
};

// Hash value
namespace std
{
template <> struct hash<VOXEL_LOCATION>
{
  int64_t operator()(const VOXEL_LOCATION &s) const
  {
    using std::hash;
    using std::size_t;
    return ((((s.z) * VOXELMAP_HASH_P) % VOXELMAP_MAX_N + (s.y)) * VOXELMAP_HASH_P) % VOXELMAP_MAX_N + (s.x);
  }
};
} // namespace std

// 下采样点结构，主要服务于某些局部统计过程。
struct DS_POINT
{
  float xyz[3];
  float intensity;
  int count = 0;
};

// 根据距离和角度误差模型计算点在机体系下的协方差。
void calcBodyCov(Eigen::Vector3d &pb, const float range_inc, const float degree_inc, Eigen::Matrix3d &cov);

class VoxelOctoTree
{

public:
  VoxelOctoTree() = default;
  // 当前体素/子体素暂存的点。
  std::vector<pointWithVar> temp_points_;
  // 当前体素对应的平面模型。
  VoxelPlane *plane_ptr_;
  // 当前八叉树层级。
  int layer_;
  int octo_state_; // 0 表示叶子节点，1 表示仍有子节点。
  VoxelOctoTree *leaves_[8];
  double voxel_center_[3]; // 当前体素中心坐标。
  std::vector<int> layer_init_num_; // 各层初始化平面所需的最小点数。
  float quater_length_; // 当前体素边长的一半。
  float planer_threshold_; // 平面判定阈值。
  int points_size_threshold_; // 初始化平面所需点数门限。
  int update_size_threshold_; // 触发更新所需新增点数门限。
  int max_points_num_; // 当前体素允许缓存的最大点数。
  int max_layer_; // 八叉树最大层数。
  int new_points_; // 自上次更新以来新增点数。
  bool init_octo_; // 是否已完成八叉树初始化。
  bool update_enable_; // 是否允许继续更新。

  VoxelOctoTree(int max_layer, int layer, int points_size_threshold, int max_points_num, float planer_threshold)
      : max_layer_(max_layer), layer_(layer), points_size_threshold_(points_size_threshold), max_points_num_(max_points_num),
        planer_threshold_(planer_threshold)
  {
    temp_points_.clear();
    octo_state_ = 0;
    new_points_ = 0;
    update_size_threshold_ = 5;
    init_octo_ = false;
    update_enable_ = true;
    for (int i = 0; i < 8; i++)
    {
      leaves_[i] = nullptr;
    }
    plane_ptr_ = new VoxelPlane;
  }

  ~VoxelOctoTree()
  {
    for (int i = 0; i < 8; i++)
    {
      delete leaves_[i];
    }
    delete plane_ptr_;
  }
  // 用点集初始化平面模型。
  void init_plane(const std::vector<pointWithVar> &points, VoxelPlane *plane);
  // 初始化当前体素节点。
  void init_octo_tree();
  // 将当前节点继续细分为八叉树子节点。
  void cut_octo_tree();
  // 向已有节点中插入新点并按需更新平面。
  void UpdateOctoTree(const pointWithVar &pv);

  // 查找与世界点 pw 对应的最合适子节点。
  VoxelOctoTree *find_correspond(Eigen::Vector3d pw);
  // 插入点并返回最终落入的节点。
  VoxelOctoTree *Insert(const pointWithVar &pv);
};

class VoxelMapManager
{
public:
  VoxelMapManager() = default;
  // 地图配置。
  VoxelMapConfig config_setting_;
  // 当前处理到的帧编号。
  int current_frame_id_ = 0;
  // 体素地图可视化发布器。
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr voxel_map_pub_;
  // 全局体素索引到八叉树节点的哈希表。
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map_;

  // 当前帧相关点云缓存。
  PointCloudXYZI::Ptr feats_undistort_;
  PointCloudXYZI::Ptr feats_down_body_;
  PointCloudXYZI::Ptr feats_down_world_;

  // 外参：body 到 lidar。
  M3D extR_;
  V3D extT_;
  // 当前帧和平均的耗时统计。
  float build_residual_time, ekf_time;
  float ave_build_residual_time = 0.0;
  float ave_ekf_time = 0.0;
  int scan_count = 0;
  // 当前状态估计。
  StatesGroup state_;
  // 上一时刻位置，用于统计/触发滑窗。
  V3D position_last_;

  // 上一次执行地图滑动时的位置。
  V3D last_slide_position = {0,0,0};

  // 可视化用四元数缓存。
  geometry_msgs::msg::Quaternion geoQuat_;

  // 当前下采样点数和有效匹配点数。
  int feats_down_size_;
  int effct_feat_num_;
  // 点云误差传播与残差构建过程中的中间缓存。
  std::vector<M3D> cross_mat_list_;
  std::vector<M3D> body_cov_list_;
  std::vector<pointWithVar> pv_list_;
  std::vector<PointToPlane> ptpl_list_;

  VoxelMapManager(VoxelMapConfig &config_setting, std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &voxel_map)
      : config_setting_(config_setting), voxel_map_(voxel_map)
  {
    current_frame_id_ = 0;
    feats_undistort_.reset(new PointCloudXYZI());
    feats_down_body_.reset(new PointCloudXYZI());
    feats_down_world_.reset(new PointCloudXYZI());
  };

  // 进行一次基于点到面的误差状态估计。
  void StateEstimation(StatesGroup &state_propagat);
  // 将机体系点云变换到世界系。
  void TransformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud,
                      pcl::PointCloud<pcl::PointXYZI>::Ptr &trans_cloud);

  // 用当前帧点云构建初始体素地图。
  void BuildVoxelMap();
  // 为可视化生成体素颜色。
  V3F RGBFromVoxel(const V3D &input_point);

  // 将新点并入体素地图。
  void UpdateVoxelMap(const std::vector<pointWithVar> &input_points);

  // 并行构建所有点到平面的残差候选。
  void BuildResidualListOMP(std::vector<pointWithVar> &pv_list, std::vector<PointToPlane> &ptpl_list);

  // 为单个点搜索对应平面并构造残差项。
  void build_single_residual(pointWithVar &pv, const VoxelOctoTree *current_octo, const int current_layer, bool &is_sucess, double &prob,
                             PointToPlane &single_ptpl);

  // 发布体素地图可视化。
  void pubVoxelMap();

  // 执行局部地图滑窗。
  void mapSliding();
  // 清理超出局部地图边界的体素。
  void clearMemOutOfMap(const int& x_max,const int& x_min,const int& y_max,const int& y_min,const int& z_max,const int& z_min );

private:
  // 递归收集可发布的平面。
  void GetUpdatePlane(const VoxelOctoTree *current_octo, const int pub_max_voxel_layer, std::vector<VoxelPlane> &plane_list);

  // 发布单个平面 Marker。
  void pubSinglePlane(visualization_msgs::msg::MarkerArray &plane_pub, const std::string plane_ns, const VoxelPlane &single_plane, const float alpha,
                      const Eigen::Vector3d rgb);
  // 由平面局部坐标轴计算四元数。
  void CalcVectQuation(const Eigen::Vector3d &x_vec, const Eigen::Vector3d &y_vec, const Eigen::Vector3d &z_vec, geometry_msgs::msg::Quaternion &q);

  // Jet 伪彩映射。
  void mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b);
};
typedef std::shared_ptr<VoxelMapManager> VoxelMapManagerPtr;

#endif // VOXEL_MAP_H_
