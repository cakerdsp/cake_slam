#pragma once

// 模块功能：系统主融合节点接口定义，负责 ROS2 话题订阅、
// 数据同步与 LIO/VIO 模块调度及结果发布。

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <image_transport/image_transport.hpp>
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
#include "cake_slam/lidar_visual_selector.h"
#include "cake_slam/lidar_visual_types.h"
#include "cake_slam/lio_core.h"
#include "cake_slam/vision/estimator/estimator.h"

namespace cake_slam {

/**
 * @brief ROS2 main fusion node for Cake-SLAM.
 *
 * SlamNode owns ROS I/O, FAST-LIVO2-style time slicing, LIO/VIO scheduling,
 * and public outputs. Sensor callbacks are deliberately light: they only push
 * raw messages into buffers. Decoding, synchronization, LIO, VIO, and coloring
 * run in sync_process_thread_.
 */
class SlamNode : public rclcpp::Node
{
public:
  explicit SlamNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
  ~SlamNode() override;

private:
  /**
   * @brief Decoded mono/color image packet consumed by the sync thread.
   */
  struct ImagePacket
  {
    double stamp = 0.0; ///< Image timestamp after configured offset [s].
    cv::Mat mono;       ///< Mono image for FeatureTracker.
    cv::Mat color;      ///< BGR image for LiDAR point coloring.
  };

  // Initialization.
  void initialize();
  void loadConfiguration();
  void configureModules();
  void createSubscriptions();
  void createPublishers();
  void startSyncProcessThread();
  void stopSyncProcessThread();

  // ROS callbacks. These callbacks must stay non-blocking and buffer only.
  void livoxPointCloudCallback(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr msg);
  void standardPointCloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
  void imuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr msg);
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);

  // Synchronization and scheduling.
  void syncProcessLoop();
  void drainRawInputBuffers();
  void enqueueProcessedCloud(double stamp, const PointCloudXYZI::Ptr &cloud);
  void enqueueImagePacket(ImagePacket &&packet);
  bool syncPackages(FusionMeasureGroup &meas);
  bool syncLioOnly(FusionMeasureGroup &meas);
  bool syncLoOnly(FusionMeasureGroup &meas);
  bool syncLivo(FusionMeasureGroup &meas);
  bool syncVioOnly(FusionMeasureGroup &meas);
  bool buildLioMeasureToTime(FusionMeasureGroup &meas, double update_time);
  bool buildVioMeasure(FusionMeasureGroup &meas);

  // Backend updates.
  void handleLIO();
  void handleVIO();
  void handleFirstFrame();
  void gravityAlignment();

  // Buffer and time helpers.
  void clearAllBuffersLocked();
  void trimBuffersLocked();
  void resetSyncStateLocked();
  double newestLidarEndTimeLocked() const;
  double cloudEndTime(const PointCloudXYZI::Ptr &cloud, double begin_time) const;
  static double stampToSec(const builtin_interfaces::msg::Time &stamp);
  static builtin_interfaces::msg::Time secToStamp(double stamp);
  cv::Mat monoImageFromMsg(const sensor_msgs::msg::Image::ConstSharedPtr &msg) const;
  cv::Mat colorImageFromMsg(const sensor_msgs::msg::Image::ConstSharedPtr &msg) const;
  cv::Mat resizeImageToConfig(const cv::Mat &image, const char *image_kind) const;
  void loadExtrinsicsForMain();
  LioPosePrior makeLioPosePrior(double stamp) const;
  void buildLidarVisualCandidates(double stamp);
  Eigen::Vector3d lidarToWorld(const Eigen::Vector3d &point_lidar) const;

  // Publishers.
  void publishRawCloud(double stamp, const PointCloudXYZI::Ptr &cloud_lidar);
  void publishOdometry(double stamp);
  void publishPath(double stamp);
  void publishMavrosPose(double stamp);
  void publishTf(double stamp);
  void publishClouds(double stamp);
  void publishColoredCloud(double stamp, const PointCloudXYZI::Ptr &cloud_lidar);
  void publishFeatureImage(double stamp);
  void publishVisualSubmap(double stamp);
  void imuPropagationTimer();
  void propagateImuOnce(StatesGroup &state, double dt, const Eigen::Vector3d &acc,
                        const Eigen::Vector3d &gyr) const;

  Config config_;
  std::string config_file_;
  bool configured_ = false;
  bool use_lidar_ = true;
  bool use_image_ = true;
  bool use_imu_ = true;
  SLAM_MODE slam_mode_ = LIVO;

  // Owned processing modules.
  PreprocessPtr preprocess_;
  LioCore lio_;
  LidarVisualSelector lidar_visual_selector_;
  std::unique_ptr<Estimator> vio_;

  // State exposed by ROS publishers. Units: position [m], velocity [m/s].
  StatesGroup state_;
  StatesGroup latest_ekf_state_;
  StatesGroup imu_propagate_state_;
  bool ekf_finish_once_ = false;
  bool state_update_flag_ = false;
  bool first_frame_handled_ = false;
  bool gravity_align_finished_ = false;
  double first_lidar_time_ = -1.0;
  double latest_ekf_time_ = -1.0;
  double last_imu_prop_time_ = -1.0;

  // Extrinsics used by the main node.
  Eigen::Matrix3d lidar_to_imu_R_ = Eigen::Matrix3d::Identity();     ///< R_I_L.
  Eigen::Vector3d lidar_to_imu_t_ = Eigen::Vector3d::Zero();         ///< t_I_L [m].
  Eigen::Matrix3d lidar_to_camera_R_ = Eigen::Matrix3d::Identity();  ///< R_C_L.
  Eigen::Vector3d lidar_to_camera_t_ = Eigen::Vector3d::Zero();      ///< t_C_L [m].

  // Raw input buffers and decoded buffers.
  std::mutex buffer_mutex_;
  std::deque<PointCloudXYZI::Ptr> lidar_buffer_;
  std::deque<double> lidar_time_buffer_;
  std::deque<livox_ros_driver2::msg::CustomMsg::ConstSharedPtr> raw_livox_buffer_;
  std::deque<sensor_msgs::msg::PointCloud2::ConstSharedPtr> raw_cloud_buffer_;
  std::deque<sensor_msgs::msg::Image::ConstSharedPtr> raw_image_buffer_;
  std::deque<ImuSample> imu_lio_buffer_;
  std::deque<ImuSample> imu_vio_buffer_;
  std::deque<ImuSample> imu_prop_buffer_;
  std::deque<ImagePacket> image_buffer_;
  ImuSample newest_imu_;
  bool new_imu_ = false;
  std::condition_variable buffer_cv_;
  std::atomic_bool sync_thread_running_{false};
  std::thread sync_process_thread_;

  // Synchronizer state and latest data bound to the current image.
  FusionMeasureGroup measures_;
  bool lidar_pushed_ = false;
  double last_lidar_time_ = -1.0;
  double last_imu_time_ = -1.0;
  double last_image_time_ = -1.0;
  double last_vio_update_time_ = -1.0;
  cv::Mat latest_sync_mono_image_;
  cv::Mat latest_sync_color_image_;
  std::vector<LidarVisualCandidate> pending_lidar_visual_candidates_;
  LioPosePrior pending_lio_pose_prior_;

  nav_msgs::msg::Path path_msg_;

  rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_livox_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_image_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_mavros_pose_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_imu_prop_odom_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_raw_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_registered_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_map_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_colored_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_visual_submap_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_effect_;
  image_transport::Publisher pub_feature_image_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr imu_prop_timer_;
};

} // namespace cake_slam
