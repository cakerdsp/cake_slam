#pragma once

#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "cake_slam/common_lib.h"
#include "cake_slam/config.h"
#include "cake_slam/imu_sample.h"
#include "cake_slam/lidar_preprocess.h"
#include "cake_slam/lio_core.h"
#include "cake_slam/vision/estimator/estimator.h"

namespace cake_slam {

// 主节点负责 ROS 输入、传感器同步和 LIO/VIO 顺序调度。
// 设计参考 FAST-LIVO2 的 LIVMapper，但视觉更新改为 VINS-Fusion 风格：
// 1. 图像只缓存单目原始图；
// 2. VIO 更新时把同一时间段 IMU 队列送给 Estimator 做预积分；
// 3. 特征提取留在 Estimator::inputImage 内部完成。
class SlamNode : public rclcpp::Node
{
public:
  explicit SlamNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

private:
  struct ImagePacket
  {
    double stamp = 0.0;
    cv::Mat mono;
    cv::Mat color;
  };

  // -------------------- 初始化流程 --------------------
  void initialize();
  void loadConfiguration();
  void configureModules();
  void createSubscriptions();
  void createPublishers();
  void createProcessingTimer();

  // -------------------- ROS 回调 --------------------
  void livoxPointCloudCallback(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr msg);
  void standardPointCloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
  void imuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr msg);
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);

  // -------------------- 主循环与同步 --------------------
  void processingLoop();
  bool syncPackages(FusionMeasureGroup &meas);
  bool syncLioOnly(FusionMeasureGroup &meas);
  bool syncLoOnly(FusionMeasureGroup &meas);
  bool syncLivo(FusionMeasureGroup &meas);
  bool buildLioMeasureToTime(FusionMeasureGroup &meas, double update_time);
  bool buildVioMeasure(FusionMeasureGroup &meas);

  // -------------------- 模块更新 --------------------
  void handleLIO();
  void handleVIO();
  void handleFirstFrame();
  void gravityAlignment();

  // -------------------- 缓存与时间工具 --------------------
  void clearAllBuffersLocked();
  void trimBuffersLocked();
  void resetSyncStateLocked();
  double newestLidarEndTimeLocked() const;
  double cloudEndTime(const PointCloudXYZI::Ptr &cloud, double begin_time) const;
  static double stampToSec(const builtin_interfaces::msg::Time &stamp);
  static builtin_interfaces::msg::Time secToStamp(double stamp);
  cv::Mat monoImageFromMsg(const sensor_msgs::msg::Image::ConstSharedPtr &msg) const;
  cv::Mat colorImageFromMsg(const sensor_msgs::msg::Image::ConstSharedPtr &msg) const;
  void loadExtrinsicsForMain();
  Eigen::Vector3d lidarToWorld(const Eigen::Vector3d &point_lidar) const;
  void publishOdometry(double stamp);
  void publishPath(double stamp);
  void publishMavrosPose(double stamp);
  void publishTf(double stamp);
  void publishClouds(double stamp);
  void publishColoredCloud(double stamp, const PointCloudXYZI::Ptr &cloud_lidar);
  void publishVisualSubmap(double stamp);
  void imuPropagationTimer();
  void propagateImuOnce(StatesGroup &state, double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr) const;

  Config config_;
  std::string config_file_;
  bool configured_ = false;
  bool use_lidar_ = true;
  bool use_image_ = true;
  bool use_imu_ = true;
  SLAM_MODE slam_mode_ = LIVO;

  // LiDAR 预处理和两个后端模块。
  PreprocessPtr preprocess_;
  LioCore lio_;
  std::unique_ptr<Estimator> vio_;

  // 主状态是主程序对外发布的唯一状态来源。
  StatesGroup state_;
  StatesGroup state_propagat_;
  StatesGroup latest_ekf_state_;
  StatesGroup imu_propagate_state_;
  bool ekf_finish_once_ = false;
  bool state_update_flag_ = false;
  bool first_frame_handled_ = false;
  bool gravity_align_finished_ = false;
  double first_lidar_time_ = -1.0;
  double latest_ekf_time_ = -1.0;
  double last_imu_prop_time_ = -1.0;

  Eigen::Matrix3d lidar_to_imu_R_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d lidar_to_imu_t_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d lidar_to_camera_R_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d lidar_to_camera_t_ = Eigen::Vector3d::Zero();

  // 输入缓存。IMU 使用两个队列，让 LIO 和 VIO 分别消费同一批原始测量。
  std::mutex buffer_mutex_;
  std::deque<PointCloudXYZI::Ptr> lidar_buffer_;
  std::deque<double> lidar_time_buffer_;
  std::deque<ImuSample> imu_lio_buffer_;
  std::deque<ImuSample> imu_vio_buffer_;
  std::deque<ImuSample> imu_prop_buffer_;
  std::deque<ImagePacket> image_buffer_;
  ImuSample newest_imu_;
  bool new_imu_ = false;

  // 当前同步上下文，保留点云切割缓存和最近一次更新状态。
  FusionMeasureGroup measures_;
  bool lidar_pushed_ = false;
  double last_lidar_time_ = -1.0;
  double last_imu_time_ = -1.0;
  double last_image_time_ = -1.0;
  double last_vio_update_time_ = -1.0;
  cv::Mat latest_sync_color_image_;
  double latest_sync_image_time_ = -1.0;

  nav_msgs::msg::Path path_msg_;

  rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_livox_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_image_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_mavros_pose_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_imu_prop_odom_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_registered_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_map_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_colored_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_visual_submap_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_effect_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr processing_timer_;
  rclcpp::TimerBase::SharedPtr imu_prop_timer_;
};

} // namespace cake_slam
