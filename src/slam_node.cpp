// 模块功能：系统融合节点实现，
// 负责数据订阅、同步、模块调度与结果发布。

#include "cake_slam/slam.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <system_error>

#include <opencv2/imgproc.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <std_msgs/msg/header.hpp>

#include "cake_slam/utils/logging.h"
#include "cake_slam/vision/estimator/parameters.h"

using namespace std::chrono_literals;

namespace cake_slam {

SlamNode::SlamNode(const rclcpp::NodeOptions &options)
    : rclcpp::Node("cake_slam", options)
{
  initialize();
}

SlamNode::~SlamNode()
{
  stopSyncProcessThread();
}

// 初始化节点：加载配置、创建模块与启动同步线程。
void SlamNode::initialize()
{
  // 初始化顺序保持和 FAST-LIVO2 主程序接近：
  // 1. 读取统一 yaml；
  // 2. 配置 LiDAR/LIO 和 VINS 风格视觉后端；
  // 3. 创建订阅；
  // 4. 启动 sync_process 独立线程。
  loadConfiguration();
  configureModules();
  createSubscriptions();
  createPublishers();
  openTrajectoryLogs();
  configured_ = true;
  startSyncProcessThread();
}

// 读取统一配置文件，并决定当前运行模式。
void SlamNode::loadConfiguration()
{
  // 主程序只从 ROS 参数拿配置文件路径，具体 LiDAR/IMU/视觉参数统一从该 yaml 读取。
  // 这样避免 FAST-LIVO2 中视觉内参/相机模型由主程序单独加载的做法。
  config_file_ = declare_parameter<std::string>("config_file", "");
  if (config_file_.empty()) {
    RCLCPP_FATAL(get_logger(), "Parameter 'config_file' is required.");
    throw std::runtime_error("cake_slam requires a config_file parameter");
  }

  if (!LoadConfig(config_file_, config_)) {
    RCLCPP_FATAL(get_logger(), "Failed to load config file: %s", config_file_.c_str());
    throw std::runtime_error("failed to load cake_slam config");
  }

  // VINS-Fusion 视觉模块仍使用全局参数表；这里让它也从同一个 yaml 读取。
  readParameters(config_file_);

  use_lidar_ = config_.common.lidar_enable && !config_.lidar.topic.empty();
  use_image_ = config_.common.image_enable && !config_.vision.image_topic.empty();
  use_imu_ = config_.imu.enable && !config_.imu.topic.empty();

  if (!use_lidar_ && !use_image_) {
    RCLCPP_FATAL(get_logger(), "No enabled odometry input: enable LiDAR or image input in the config.");
    throw std::runtime_error("cake_slam requires LiDAR or image input");
  }
  if (!use_lidar_ && use_image_ && !use_imu_) {
    RCLCPP_FATAL(get_logger(), "Pure visual mode currently requires IMU input.");
    throw std::runtime_error("pure visual mode requires IMU input");
  }

  if (use_lidar_ && use_image_) {
    slam_mode_ = LIVO;
  } else if (use_lidar_ && use_imu_) {
    slam_mode_ = ONLY_LIO;
  } else if (use_lidar_) {
    slam_mode_ = ONLY_LO;
  } else {
    slam_mode_ = ONLY_VIO;
  }

  CAKE_INFO("Loaded config: lidar=%d image=%d imu=%d mode=%d hilti_en=%d image_interval=%d imu.acc_scale=%.6f frame.world=%s frame.body=%s frame.lidar=%s frame.camera=%s",
            use_lidar_, use_image_, use_imu_, static_cast<int>(slam_mode_),
            config_.common.hilti_en ? 1 : 0, config_.vision.image_process_interval,
            config_.imu.acc_scale,
            config_.frame.world.c_str(), config_.frame.body.c_str(),
            config_.frame.lidar.c_str(), config_.frame.camera.c_str());
}

// 配置 LIO/VIO 模块与外参设置。
void SlamNode::configureModules()
{
  loadExtrinsicsForMain();

  if (use_lidar_) {
    // LiDAR 预处理独立放在主程序中，和 FAST-LIVO2 一样先把 ROS 点云转成项目点类型。
    preprocess_.reset(new Preprocess());
    preprocess_->set(config_.lidar.feature_extract, config_.lidar.type, config_.lidar.blind,
                     config_.lidar.point_filter_num);
    preprocess_->N_SCANS = config_.lidar.scan_line;
    preprocess_->SCAN_RATE = config_.lidar.scan_rate;
    preprocess_->blind = config_.lidar.blind;
    preprocess_->blind_sqr = config_.lidar.blind * config_.lidar.blind;

    // The local voxel map is owned by SlamNode and injected into LIO, so later
    // local BA can update the same map without going through the odometry core.
    local_voxel_map_ = std::make_shared<LocalVoxelMap>();
    local_voxel_map_->Configure(config_, lidar_to_imu_R_, lidar_to_imu_t_);
    lio_.SetLocalMap(local_voxel_map_);
    lio_.Configure(config_);
    if (use_image_) {
      lidar_visual_selector_.Configure(config_);
      lidar_visual_selector_.SetExtrinsics(lidar_to_imu_R_, lidar_to_imu_t_,
                                           lidar_to_camera_R_, lidar_to_camera_t_);
    }
  }

  // 视觉后端完全走 VINS-Fusion 风格：
  // setParameter() 会读取 readParameters() 写好的全局变量，包括相机内参文件路径、IMU 噪声、外参等。
  if (use_image_) {
    vio_ = std::make_unique<Estimator>();
    vio_->setParameter();
    if (!use_lidar_) {
      CAKE_INFO("Image config: topic=%s target=%dx%d image_time_offset=%.6f td=%.6f cam0_calib=%s",
                config_.vision.image_topic.c_str(), config_.vision.image_width, config_.vision.image_height,
                config_.common.image_time_offset, config_.time_offset.td, config_.vision.cam0_calib.c_str());
    }
  }
}

// 创建传感器订阅者并绑定回调。
void SlamNode::createSubscriptions()
{
  rclcpp::SensorDataQoS qos;
  qos.keep_last(static_cast<size_t>(std::max(1, config_.common.max_buffer_size)));

  // LiDAR 支持 Livox CustomMsg 和标准 PointCloud2 两种入口。
  if (use_lidar_) {
    if (config_.lidar.type == AVIA) {
      sub_livox_ = create_subscription<livox_ros_driver2::msg::CustomMsg>(
          config_.lidar.topic, qos,
          std::bind(&SlamNode::livoxPointCloudCallback, this, std::placeholders::_1));
    } else {
      sub_cloud_ = create_subscription<sensor_msgs::msg::PointCloud2>(
          config_.lidar.topic, qos,
          std::bind(&SlamNode::standardPointCloudCallback, this, std::placeholders::_1));
    }
  }

  if (use_imu_) {
    sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(
        config_.imu.topic, qos,
        std::bind(&SlamNode::imuCallback, this, std::placeholders::_1));
  }

  // 当前主程序只接单目图像；双目入口故意不保留，避免和当前需求混杂。
  if (use_image_) {
    sub_image_ = create_subscription<sensor_msgs::msg::Image>(
        config_.vision.image_topic, qos,
        std::bind(&SlamNode::imageCallback, this, std::placeholders::_1));
  }
}

// 创建核心输出话题发布器。
void SlamNode::createPublishers()
{
  pub_cloud_raw_ = create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_raw", 100);
  pub_cloud_registered_ = create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 100);
  pub_cloud_map_ = create_publisher<sensor_msgs::msg::PointCloud2>("/Laser_map", 100);
  pub_cloud_colored_ = create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_colored", 100);
  pub_cloud_visual_submap_ = create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_visual_sub_map", 100);
  pub_vio_landmarks_ = create_publisher<sensor_msgs::msg::PointCloud2>(config_.visualization.vio_landmarks_topic, 10);
  pub_vio_window_path_ = create_publisher<nav_msgs::msg::Path>(config_.visualization.vio_window_path_topic, 10);
  pub_vio_window_poses_ = create_publisher<geometry_msgs::msg::PoseArray>(config_.visualization.vio_window_poses_topic, 10);
  pub_cloud_effect_ = create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_effected", 100);
  pub_lio_odom_ = create_publisher<nav_msgs::msg::Odometry>(config_.visualization.lio_odom_topic, 10);
  pub_lio_path_ = create_publisher<nav_msgs::msg::Path>(config_.visualization.lio_path_topic, 10);
  pub_mavros_pose_ = create_publisher<geometry_msgs::msg::PoseStamped>(config_.visualization.mavros_pose_topic, 10);
  pub_vio_odom_ = create_publisher<nav_msgs::msg::Odometry>(config_.visualization.vio_odom_topic, 10);
  pub_vio_path_ = create_publisher<nav_msgs::msg::Path>(config_.visualization.vio_path_topic, 10);
  pub_vio_pose_ = create_publisher<geometry_msgs::msg::PoseStamped>(config_.visualization.vio_pose_topic, 10);
  pub_imu_prop_odom_ = create_publisher<nav_msgs::msg::Odometry>("/cake_slam/imu_propagate", 1000);
  pub_feature_image_ = image_transport::create_publisher(this, "/cake_slam/feature_image");
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
  static_tf_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);
  publishStaticTf();
}

// Open optional TUM-format trajectory logs for later ATE/RPE evaluation.
void SlamNode::openTrajectoryLogs()
{
  if (!config_.output.record_trajectory) {
    return;
  }

  const auto resolve_output_file = [this](const std::string &file_name) {
    std::filesystem::path file_path(file_name);
    if (file_path.is_absolute()) {
      return file_path;
    }
    const std::filesystem::path output_dir = config_.output.path.empty()
        ? std::filesystem::path(".")
        : std::filesystem::path(config_.output.path);
    return output_dir / file_path;
  };

  const auto open_file = [](std::ofstream &stream, const std::filesystem::path &file_path, const char *label) {
    std::error_code ec;
    const std::filesystem::path parent = file_path.parent_path();
    if (!parent.empty()) {
      std::filesystem::create_directories(parent, ec);
      if (ec) {
        CAKE_INFO("Trajectory logging disabled for %s: cannot create directory %s (%s)",
                  label, parent.string().c_str(), ec.message().c_str());
        return;
      }
    }

    stream.open(file_path, std::ios::out);
    if (!stream.is_open()) {
      CAKE_INFO("Trajectory logging disabled for %s: cannot open %s",
                label, file_path.string().c_str());
      return;
    }
    stream << std::fixed << std::setprecision(9);
    CAKE_INFO("Trajectory logging enabled for %s: %s", label, file_path.string().c_str());
  };

  if (use_lidar_) {
    open_file(lio_trajectory_file_, resolve_output_file(config_.output.lio_trajectory_file), "LIO");
  }
  if (use_image_) {
    open_file(vio_trajectory_file_, resolve_output_file(config_.output.vio_trajectory_file), "VIO");
  }
}

// 启动数据同步与主处理线程。
void SlamNode::startSyncProcessThread()
{
  sync_thread_running_.store(true);
  sync_process_thread_ = std::thread(&SlamNode::syncProcessLoop, this);
  if (config_.common.imu_propagation_enable) {
    imu_prop_timer_ = create_wall_timer(4ms, std::bind(&SlamNode::imuPropagationTimer, this));
  }
}

// 停止同步线程并回收资源。
void SlamNode::stopSyncProcessThread()
{
  sync_thread_running_.store(false);
  buffer_cv_.notify_all();
  if (sync_process_thread_.joinable()) {
    sync_process_thread_.join();
  }
}

void SlamNode::livoxPointCloudCallback(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(buffer_mutex_);
  raw_livox_buffer_.push_back(msg);
  trimBuffersLocked();
  buffer_cv_.notify_one();
}

void SlamNode::standardPointCloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(buffer_mutex_);
  raw_cloud_buffer_.push_back(msg);
  trimBuffersLocked();
  buffer_cv_.notify_one();
}

void SlamNode::imuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr msg)
{
  if (!use_imu_) {
    return;
  }

  ImuSample sample = ImuSampleFromMsg(msg);
  sample.stamp -= config_.common.imu_time_offset;
  sample.acc *= config_.imu.acc_scale;
  CAKE_INFO_THROTTLE_MS(
      2000,
      "IMU sample: stamp=%.6f acc_norm=%.6f acc_z=%.6f acc_scale=%.6f",
      sample.stamp, sample.acc.norm(), sample.acc.z(), config_.imu.acc_scale);

  // 可选兼容某些驱动的整秒跳变修正，逻辑保持和 FAST-LIVO2 主程序一致。
  if (config_.common.ros_driver_bug_fix && last_lidar_time_ > 0.0) {
    sample.stamp += std::round(last_lidar_time_ - sample.stamp);
  }

  std::lock_guard<std::mutex> lock(buffer_mutex_);
  if (last_imu_time_ > 0.0 && sample.stamp < last_imu_time_) {
    RCLCPP_ERROR(get_logger(), "IMU timestamp moved backwards; clearing IMU buffers.");
    imu_lio_buffer_.clear();
    imu_vio_buffer_.clear();
    last_imu_time_ = -1.0;
  }

  // 同一条 IMU 按已启用的消费者分别入队；纯视觉模式不再占用 LIO 队列。
  if (use_lidar_) {
    imu_lio_buffer_.push_back(sample);
  }
  if (use_image_) {
    imu_vio_buffer_.push_back(sample);
  }
  if (ekf_finish_once_) {
    imu_prop_buffer_.push_back(sample);
    newest_imu_ = sample;
    new_imu_ = true;
  }
  last_imu_time_ = sample.stamp;
  trimBuffersLocked();
  buffer_cv_.notify_one();
}

void SlamNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  if (!use_image_) {
    return;
  }

  std::lock_guard<std::mutex> lock(buffer_mutex_);
  ++hilti_image_counter_;
  int image_process_interval = std::max(1, config_.vision.image_process_interval);
  if (image_process_interval == 1 && config_.common.hilti_en) {
    image_process_interval = 4;
  }
  if ((hilti_image_counter_ - 1) % static_cast<std::uint64_t>(image_process_interval) != 0) {
    return;
  }
  raw_image_buffer_.push_back(msg);
  trimBuffersLocked();
  buffer_cv_.notify_one();
}

// 同步线程主循环：解码、对齐并驱动 LIO/VIO 更新。
void SlamNode::syncProcessLoop()
{
  // Heavy work lives in this worker thread. ROS callbacks only enqueue raw
  // messages, while this loop decodes, synchronizes, optimizes, colors, and
  // publishes.
  while (rclcpp::ok() && sync_thread_running_.load()) {
    drainRawInputBuffers();

    bool did_work = false;
    while (syncPackages(measures_)) {
      did_work = true;
      switch (measures_.lio_vio_flg) {
        case LIO:
        case LO:
          handleLIO();
          break;
        case VIO:
          handleVIO();
          break;
        default:
          break;
      }
      drainRawInputBuffers();
    }

    if (!did_work) {
      std::unique_lock<std::mutex> lock(buffer_mutex_);
      buffer_cv_.wait_for(lock, 2ms, [this]() {
        return !sync_thread_running_.load() || !raw_livox_buffer_.empty() ||
               !raw_cloud_buffer_.empty() || !raw_image_buffer_.empty() ||
               !lidar_buffer_.empty() || !image_buffer_.empty();
      });
    }
  }
}

// 取出原始消息并完成解码/预处理。
void SlamNode::drainRawInputBuffers()
{
  // Pop raw messages under the mutex, then decode/preprocess them outside the
  // critical section so sensor callbacks remain lightweight.
  while (sync_thread_running_.load()) {
    livox_ros_driver2::msg::CustomMsg::ConstSharedPtr livox_msg;
    sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_msg;
    sensor_msgs::msg::Image::ConstSharedPtr image_msg;

    {
      std::lock_guard<std::mutex> lock(buffer_mutex_);
      if (!raw_livox_buffer_.empty()) {
        livox_msg = raw_livox_buffer_.front();
        raw_livox_buffer_.pop_front();
      } else if (!raw_cloud_buffer_.empty()) {
        cloud_msg = raw_cloud_buffer_.front();
        raw_cloud_buffer_.pop_front();
      } else if (!raw_image_buffer_.empty()) {
        image_msg = raw_image_buffer_.front();
        raw_image_buffer_.pop_front();
      } else {
        break;
      }
    }

    if (livox_msg) {
      auto mutable_msg = std::make_shared<livox_ros_driver2::msg::CustomMsg>(*livox_msg);
      PointCloudXYZI::Ptr cloud(new PointCloudXYZI());
      preprocess_->process(mutable_msg, cloud);
      enqueueProcessedCloud(stampToSec(livox_msg->header.stamp), cloud);
    } else if (cloud_msg) {
      PointCloudXYZI::Ptr cloud(new PointCloudXYZI());
      preprocess_->process(cloud_msg, cloud);
      enqueueProcessedCloud(stampToSec(cloud_msg->header.stamp), cloud);
    } else if (image_msg) {
      ImagePacket packet;
      packet.stamp = stampToSec(image_msg->header.stamp) + config_.common.image_time_offset;
      packet.mono = monoImageFromMsg(image_msg);
      packet.color = colorImageFromMsg(image_msg);
      enqueueImagePacket(std::move(packet));
    }
  }
}

void SlamNode::enqueueProcessedCloud(double stamp, const PointCloudXYZI::Ptr &cloud)
{
  if (!cloud || cloud->empty()) {
    RCLCPP_WARN(get_logger(), "Dropped empty LiDAR point cloud.");
    return;
  }

  std::lock_guard<std::mutex> lock(buffer_mutex_);
  if (last_lidar_time_ > 0.0 && stamp < last_lidar_time_) {
    RCLCPP_ERROR(get_logger(), "LiDAR timestamp moved backwards; clearing synchronized buffers.");
    clearAllBuffersLocked();
  }
  lidar_buffer_.push_back(cloud);
  lidar_time_buffer_.push_back(stamp);
  last_lidar_time_ = stamp;
  trimBuffersLocked();
  buffer_cv_.notify_one();
}

void SlamNode::enqueueImagePacket(ImagePacket &&packet)
{
  if (packet.mono.empty()) {
    RCLCPP_WARN(get_logger(), "Dropped empty image packet.");
    return;
  }

  std::lock_guard<std::mutex> lock(buffer_mutex_);
  if (last_image_time_ > 0.0 && packet.stamp < last_image_time_) {
    RCLCPP_ERROR(get_logger(), "Image timestamp moved backwards; clearing image buffer.");
    image_buffer_.clear();
  }
  image_buffer_.push_back(std::move(packet));
  last_image_time_ = image_buffer_.back().stamp;
  trimBuffersLocked();
  buffer_cv_.notify_one();
}

// 根据运行模式选择同步策略。
bool SlamNode::syncPackages(FusionMeasureGroup &meas)
{
  switch (slam_mode_) {
    case LIVO:
      return syncLivo(meas);
    case ONLY_LIO:
      return syncLioOnly(meas);
    case ONLY_LO:
      return syncLoOnly(meas);
    case ONLY_VIO:
      return syncVioOnly(meas);
    default:
      return false;
  }
}

bool SlamNode::syncLioOnly(FusionMeasureGroup &meas)
{
  std::lock_guard<std::mutex> lock(buffer_mutex_);
  if (lidar_buffer_.empty() || (use_imu_ && imu_lio_buffer_.empty())) {
    return false;
  }

  if (meas.last_lio_update_time < 0.0) {
    meas.last_lio_update_time = lidar_time_buffer_.front();
  }
  if (!lidar_pushed_) {
    meas.lidar = lidar_buffer_.front();
    if (!meas.lidar || meas.lidar->points.size() <= 1) {
      return false;
    }
    meas.lidar_frame_beg_time = lidar_time_buffer_.front();
    meas.lidar_frame_end_time = cloudEndTime(meas.lidar, meas.lidar_frame_beg_time);
    meas.pcl_proc_cur = meas.lidar;
    lidar_pushed_ = true;
  }

  if (use_imu_ && last_imu_time_ < meas.lidar_frame_end_time) {
    return false;
  }

  FusionMeasure m;
  m.lio_time = meas.lidar_frame_end_time;
  while (!imu_lio_buffer_.empty() && imu_lio_buffer_.front().stamp <= m.lio_time) {
    m.imu.push_back(imu_lio_buffer_.front());
    imu_lio_buffer_.pop_front();
  }

  lidar_buffer_.pop_front();
  lidar_time_buffer_.pop_front();
  lidar_pushed_ = false;
  meas.measures.clear();
  meas.measures.push_back(m);
  meas.lio_vio_flg = LIO;
  return true;
}

bool SlamNode::syncLoOnly(FusionMeasureGroup &meas)
{
  std::lock_guard<std::mutex> lock(buffer_mutex_);
  if (lidar_buffer_.empty()) {
    return false;
  }

  FusionMeasure m;
  meas.lidar = lidar_buffer_.front();
  meas.lidar_frame_beg_time = lidar_time_buffer_.front();
  meas.lidar_frame_end_time = cloudEndTime(meas.lidar, meas.lidar_frame_beg_time);
  meas.pcl_proc_cur = meas.lidar;
  m.lio_time = meas.lidar_frame_end_time;

  lidar_buffer_.pop_front();
  lidar_time_buffer_.pop_front();
  meas.measures.clear();
  meas.measures.push_back(m);
  meas.lio_vio_flg = LO;
  return true;
}

bool SlamNode::syncLivo(FusionMeasureGroup &meas)
{
  std::lock_guard<std::mutex> lock(buffer_mutex_);
  if (lidar_buffer_.empty() || image_buffer_.empty() ||
      (use_imu_ && (imu_lio_buffer_.empty() || imu_vio_buffer_.empty()))) {
    return false;
  }

  // FAST-LIVO2 的 LIVO 调度方式：
  // WAIT/VIO 之后先把 LiDAR 切到下一帧图像时间，做 LIO；
  // LIO 之后再用同一张图像做一次 VIO。
  // WAIT/VIO -> run LIO up to the next image time; LIO -> run VIO on that same
  // image with the freshly computed LIO state/covariance prior.
  switch (meas.lio_vio_flg) {
    case WAIT:
    case VIO:
      return buildLioMeasureToTime(meas, image_buffer_.front().stamp);
    case LIO:
      return buildVioMeasure(meas);
    default:
      return false;
  }
}

bool SlamNode::syncVioOnly(FusionMeasureGroup &meas)
{
  std::lock_guard<std::mutex> lock(buffer_mutex_);
  if (image_buffer_.empty() || (use_imu_ && imu_vio_buffer_.empty())) {
    return false;
  }

  const double image_time = image_buffer_.front().stamp;
  const double imu_cut_time = image_time + config_.time_offset.td;
  if (use_imu_ && imu_cut_time > last_imu_time_) {
    return false;
  }
  if (last_vio_update_time_ > 0.0 && image_time <= last_vio_update_time_) {
    image_buffer_.pop_front();
    return false;
  }

  FusionMeasure m;
  m.vio_time = image_time;
  m.lio_time = -1.0;
  m.img = image_buffer_.front().mono;

  while (!imu_vio_buffer_.empty() && imu_vio_buffer_.front().stamp < imu_cut_time) {
    m.imu.push_back(imu_vio_buffer_.front());
    imu_vio_buffer_.pop_front();
  }
  if (use_imu_ && !imu_vio_buffer_.empty()) {
    m.imu.push_back(imu_vio_buffer_.front());
    imu_vio_buffer_.pop_front();
  }

  CAKE_INFO_THROTTLE_MS(
      2000,
      "VIO-only sync image: image=%.6f imu_cut=%.6f imu_count=%zu vio_imu_queue=%zu image_queue=%zu",
      image_time, imu_cut_time, m.imu.size(), imu_vio_buffer_.size(), image_buffer_.size());
  image_buffer_.pop_front();

  meas.measures.clear();
  meas.measures.push_back(m);
  meas.lio_vio_flg = VIO;
  return true;
}

// 将 LiDAR 点云切到指定时刻，构建一包 LIO 测量。
bool SlamNode::buildLioMeasureToTime(FusionMeasureGroup &meas, double update_time)
{
  // Slice LiDAR data at the selected image timestamp. Points after update_time
  // stay in pcl_proc_next and become the prefix of the next LIO packet.
  if (meas.last_lio_update_time < 0.0) {
    meas.last_lio_update_time = lidar_time_buffer_.front();
  }

  // 丢掉早于当前 LIO 时间的旧图像，避免同步器被过期图像卡住。
  while (!image_buffer_.empty() && image_buffer_.front().stamp <= meas.last_lio_update_time + 1e-5) {
    image_buffer_.pop_front();
  }
  if (image_buffer_.empty()) {
    return false;
  }
  update_time = image_buffer_.front().stamp;
  latest_sync_mono_image_ = image_buffer_.front().mono.clone();
  latest_sync_color_image_ = image_buffer_.front().color.clone();

  if (update_time > newestLidarEndTimeLocked()) {
    return false;
  }
  if (use_imu_ && update_time > last_imu_time_) {
    return false;
  }

  FusionMeasure m;
  m.lio_time = update_time;
  while (!imu_lio_buffer_.empty() && imu_lio_buffer_.front().stamp <= m.lio_time) {
    if (imu_lio_buffer_.front().stamp > meas.last_lio_update_time) {
      m.imu.push_back(imu_lio_buffer_.front());
    }
    imu_lio_buffer_.pop_front();
  }

  // 点云切割沿用 FAST-LIVO2 的 pcl_proc_cur/pcl_proc_next 思路：
  // 当前图像时间之前的点进入本次 LIO，之后的点暂存到下一次切包。
  *(meas.pcl_proc_cur) = *(meas.pcl_proc_next);
  PointCloudXYZI().swap(*meas.pcl_proc_next);
  const int reserve_size = static_cast<int>(meas.pcl_proc_cur->size()) +
                           24000 * static_cast<int>(lidar_buffer_.size());
  meas.pcl_proc_cur->reserve(reserve_size);
  meas.pcl_proc_next->reserve(reserve_size);

  while (!lidar_buffer_.empty()) {
    if (lidar_time_buffer_.front() > update_time) {
      break;
    }

    const PointCloudXYZI::Ptr cloud = lidar_buffer_.front();
    const double frame_begin = lidar_time_buffer_.front();
    const float split_offset_ms = static_cast<float>((update_time - frame_begin) * 1000.0);

    for (const auto &src_pt : cloud->points) {
      PointType pt = src_pt;
      if (src_pt.curvature < split_offset_ms) {
        pt.curvature += static_cast<float>((frame_begin - meas.last_lio_update_time) * 1000.0);
        meas.pcl_proc_cur->points.push_back(pt);
      } else {
        pt.curvature += static_cast<float>((frame_begin - update_time) * 1000.0);
        meas.pcl_proc_next->points.push_back(pt);
      }
    }

    lidar_buffer_.pop_front();
    lidar_time_buffer_.pop_front();
  }

  CAKE_INFO_THROTTLE_MS(
      2000,
      "LIVO sync LIO cut: last_lio=%.6f image=%.6f dt=%.6f imu_last=%.6f cur_pts=%zu next_pts=%zu lidar_queue=%zu image_queue=%zu",
      meas.last_lio_update_time, update_time, update_time - meas.last_lio_update_time, last_imu_time_,
      meas.pcl_proc_cur->size(), meas.pcl_proc_next->size(), lidar_buffer_.size(), image_buffer_.size());

  meas.measures.clear();
  meas.measures.push_back(m);
  meas.lio_vio_flg = LIO;
  return true;
}

// 构建单次 VIO 测量包并同步 IMU。
bool SlamNode::buildVioMeasure(FusionMeasureGroup &meas)
{
  // Consume the image that defined the preceding LIO cut time. VIO has its own
  // IMU queue, so LIO and VIO can integrate the same physical measurements.
  if (image_buffer_.empty()) {
    return false;
  }

  const double image_time = image_buffer_.front().stamp;
  const double imu_cut_time = image_time + config_.time_offset.td;
  if (use_imu_ && imu_cut_time > last_imu_time_) {
    return false;
  }
  if (last_vio_update_time_ > 0.0 && image_time <= last_vio_update_time_) {
    image_buffer_.pop_front();
    return false;
  }

  FusionMeasure m;
  m.vio_time = image_time;
  m.lio_time = meas.last_lio_update_time;
  m.img = image_buffer_.front().mono;

  // VIO 使用独立 IMU 队列，保留 VINS-Fusion 两帧图像间预积分所需的原始测量。
  while (!imu_vio_buffer_.empty() && imu_vio_buffer_.front().stamp < imu_cut_time) {
    m.imu.push_back(imu_vio_buffer_.front());
    imu_vio_buffer_.pop_front();
  }
  if (use_imu_ && !imu_vio_buffer_.empty()) {
    m.imu.push_back(imu_vio_buffer_.front());
    imu_vio_buffer_.pop_front();
  }
  CAKE_INFO_THROTTLE_MS(
      2000,
      "LIVO sync VIO image: image=%.6f imu_cut=%.6f lio_prior=%.6f imu_count=%zu vio_imu_queue=%zu image_queue=%zu",
      image_time, imu_cut_time, meas.last_lio_update_time, m.imu.size(), imu_vio_buffer_.size(), image_buffer_.size());
  image_buffer_.pop_front();

  meas.measures.clear();
  meas.measures.push_back(m);
  meas.lio_vio_flg = VIO;
  lidar_pushed_ = false;
  return true;
}

// 执行一次 LIO 更新并输出里程计/点云。
void SlamNode::handleLIO()
{
  // Sequential update stage 1: run LIO, publish state/cloud outputs, and prepare
  // direct priors for the next VIO image update.
  if (measures_.measures.empty()) {
    return;
  }

  const auto t_lio_start = std::chrono::steady_clock::now();
  const double update_time = measures_.measures.back().lio_time;
  handleFirstFrame();
  const StatesGroup lio_input_state = state_;
  lio_.SetState(state_);

  // LioCore 内部会复制测量包，因此主程序同步上下文中的 last_lio_update_time
  // 需要在这里显式推进，后续 LIVO 点云切割才有正确起点。
  const auto t_lio_process_start = std::chrono::steady_clock::now();
  lio_.ProcessMeasurement(measures_);
  measures_.last_lio_update_time = update_time;
  state_ = lio_.GetState();
  const bool lio_imu_initialized = !use_imu_ || lio_.IsImuInitialized();
  if (lio_imu_initialized) {
    gravityAlignment();
  }
  latest_ekf_state_ = state_;
  latest_ekf_time_ = update_time;
  const auto t_lio_process_end = std::chrono::steady_clock::now();

  if (lio_imu_initialized) {
    if (!lio_full_state_prior_ready_) {
      lio_full_state_history_.clear();
      lio_full_state_prior_ready_ = true;
      std::printf("LIO PRIOR GATE reset history after IMU initialization and gravity alignment: stamp=%.6f\n",
                  update_time);
    }
    pending_lio_pose_prior_ = makeLioPosePrior(update_time);
    recordLioFullStatePrior(update_time);
    buildLidarVisualCandidates(update_time);
  } else {
    lio_full_state_prior_ready_ = false;
    lio_full_state_history_.clear();
    pending_lio_pose_prior_ = LioPosePrior();
    pending_lidar_depth_frame_ = LidarDepthFrame();
    pending_lidar_visual_candidates_.clear();
    static double last_lio_prior_skip_log_time = -1.0;
    if (last_lio_prior_skip_log_time < 0.0 ||
        update_time - last_lio_prior_skip_log_time >= 2.0) {
      std::printf("LIO PRIOR GATE skip priors before IMU initialization: stamp=%.6f\n",
                  update_time);
      last_lio_prior_skip_log_time = update_time;
    }
  }
  static bool has_last_lio_output_state = false;
  static double last_lio_output_z = 0.0;
  static double last_lio_output_time = -1.0;
  const double lio_output_z = state_.pos_end.z();
  const double lio_dz_from_input = lio_output_z - lio_input_state.pos_end.z();
  const double lio_dz_from_prev =
      has_last_lio_output_state ? lio_output_z - last_lio_output_z
                                : std::numeric_limits<double>::quiet_NaN();
  const double lio_dt_from_prev =
      has_last_lio_output_state ? update_time - last_lio_output_time
                                : std::numeric_limits<double>::quiet_NaN();
  std::printf("LIO STATE DEBUG stamp=%.6f input_z=% .6f output_z=% .6f dz_input=% .6f dz_prev=% .6f dt_prev=%.6f vel_z=% .6f gravity_z=% .6f imu_init=%d visual_candidates=%zu lio_history=%zu\n",
              update_time,
              lio_input_state.pos_end.z(),
              lio_output_z,
              lio_dz_from_input,
              lio_dz_from_prev,
              lio_dt_from_prev,
              state_.vel_end.z(),
              state_.gravity.z(),
              lio_imu_initialized ? 1 : 0,
              pending_lidar_visual_candidates_.size(),
              lio_full_state_history_.size());
  has_last_lio_output_state = true;
  last_lio_output_z = lio_output_z;
  last_lio_output_time = update_time;
  const auto t_lio_prior_end = std::chrono::steady_clock::now();

  state_update_flag_ = true;
  ekf_finish_once_ = true;

  publishClouds(update_time);
  publishLioOdometry(update_time);
  publishLioPath(update_time);
  publishLioTf(update_time);
  publishLioMavrosPose(update_time);

  const auto t_lio_end = std::chrono::steady_clock::now();
  static int lio_time_count = 0;
  static double lio_average_total = 0.0;
  const double process_sec = std::chrono::duration<double>(t_lio_process_end - t_lio_process_start).count();
  const double prior_sec = std::chrono::duration<double>(t_lio_prior_end - t_lio_process_end).count();
  const double publish_sec = std::chrono::duration<double>(t_lio_end - t_lio_prior_end).count();
  const double total_sec = std::chrono::duration<double>(t_lio_end - t_lio_start).count();
  lio_time_count++;
  lio_average_total += (total_sec - lio_average_total) / static_cast<double>(lio_time_count);
  PrintTimeTable("LIO Mapping Time", BOLDCYAN,
                 {{"IESKF+VoxelMap", process_sec},
                  {"BuildVisualPrior", prior_sec},
                  {"PublishOutputs", publish_sec}},
                 total_sec, lio_average_total);
  CAKE_INFO("LIO summary: stamp=%.6f imu=%zu cur_pts=%zu next_pts=%zu lidar_prior_candidates=%zu mode=%d",
            update_time, measures_.measures.back().imu.size(),
            measures_.pcl_proc_cur ? measures_.pcl_proc_cur->size() : 0,
            measures_.pcl_proc_next ? measures_.pcl_proc_next->size() : 0,
            pending_lidar_visual_candidates_.size(),
            static_cast<int>(slam_mode_));
}

// Run one VIO update. In LIVO mode this is feed-forward: LIO exports priors to
// VIO, and VIO publishes its optimized state without writing back into LioCore.
void SlamNode::handleVIO()
{
  // Sequential update stage 2: feed IMU samples, LiDAR visual candidates, and
  // the LIO pose prior directly into Estimator::inputImage().
  if (!vio_ || measures_.measures.empty()) {
    return;
  }

  const auto t_vio_start = std::chrono::steady_clock::now();
  const FusionMeasure &measure = measures_.measures.back();
  const size_t lidar_candidate_count = pending_lidar_visual_candidates_.size();
  const bool has_lio_prior = use_lidar_ && pending_lio_pose_prior_.valid;
  const LioFullStatePrior lio_full_prior = use_lidar_ ? FindClosestLioState(measure.vio_time) : LioFullStatePrior();
  const bool has_lio_full_prior = lio_full_prior.valid;
  const double lio_full_prior_dt = has_lio_full_prior ? std::abs(lio_full_prior.timestamp - measure.vio_time) : -1.0;
  const bool vio_feed_forward = use_lidar_;

  // VINS-Fusion 风格入口：先送 IMU，再送单目图像。
  // inputImage 内部会调用 featureTracker.trackImage，外部不负责提特征。
  for (const auto &imu : measure.imu) {
    vio_->inputImuSample(imu);
  }
  const auto t_vio_imu_end = std::chrono::steady_clock::now();
  auto t_vio_image_end = t_vio_imu_end;
  auto t_vio_publish_end = t_vio_imu_end;
  if (!measure.img.empty()) {
    if (last_vio_update_time_ < 0.0) {
      vio_->initFirstPose(state_.pos_end, state_.rot_end);
    }
    vio_->inputImage(measure.vio_time, measure.img,
                     pending_lidar_visual_candidates_, pending_lidar_depth_frame_,
                     pending_lio_pose_prior_, lio_full_prior);
    publishFeatureImage(measure.vio_time);
    t_vio_image_end = std::chrono::steady_clock::now();
    pending_lidar_visual_candidates_.clear();
    pending_lidar_depth_frame_ = LidarDepthFrame();
    pending_lio_pose_prior_ = LioPosePrior();
    last_vio_update_time_ = measure.vio_time;
    if (vio_->solver_flag == Estimator::NON_LINEAR) {
      StatesGroup vio_state = vio_->getLatestState();
      const int opt_features = vio_->getLastOptimizationFeatureCount();
      const int opt_lidar_features = vio_->getLastOptimizationLidarFeatureCount();
      const int opt_visual_residuals = vio_->getLastOptimizationVisualResidualCount();
      const StatesGroup lio_state_before_vio = state_;
      if (use_lidar_) {
        const Eigen::Matrix3d dR = lio_state_before_vio.rot_end.transpose() * vio_state.rot_end;
        const double cos_angle = std::clamp((dR.trace() - 1.0) * 0.5, -1.0, 1.0);
        constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;
        const double rot_delta_deg = std::acos(cos_angle) * kRadToDeg;
        const double pos_delta = (vio_state.pos_end - lio_state_before_vio.pos_end).norm();
        const double z_delta = vio_state.pos_end.z() - lio_state_before_vio.pos_end.z();
        std::printf("VIO FEEDFORWARD DEBUG stamp=%.6f publish_vio=1 write_lio=0 pos_delta=%.6f z_delta=% .6f rot_delta_deg=%.6f tracked=%d depth_prior=%d opt_features=%d opt_depth=%d visual_residuals=%d lio_z=% .6f vio_z=% .6f lio_full_prior=%d lio_full_dt=%.6f lidar_candidates=%zu\n",
                    measure.vio_time,
                    pos_delta,
                    z_delta,
                    rot_delta_deg,
                    vio_->getLastTrackedFeatureCount(),
                    vio_->getLastDepthFeatureCount(),
                    opt_features,
                    opt_lidar_features,
                    opt_visual_residuals,
                    lio_state_before_vio.pos_end.z(),
                    vio_state.pos_end.z(),
                    has_lio_full_prior ? 1 : 0,
                    lio_full_prior_dt,
                    lidar_candidate_count);
        CAKE_INFO_THROTTLE_MS(
            2000,
            "VIO feed-forward: stamp=%.6f write_lio=0 pos_delta=%.4f rot_delta_deg=%.3f tracked=%d depth_prior=%d visual_residuals=%d "
            "lio_pos=(%.3f %.3f %.3f) vio_pos=(%.3f %.3f %.3f)",
            measure.vio_time, pos_delta, rot_delta_deg, vio_->getLastTrackedFeatureCount(),
            vio_->getLastDepthFeatureCount(), opt_visual_residuals, lio_state_before_vio.pos_end.x(),
            lio_state_before_vio.pos_end.y(), lio_state_before_vio.pos_end.z(),
            vio_state.pos_end.x(), vio_state.pos_end.y(), vio_state.pos_end.z());
      } else {
        CAKE_INFO_THROTTLE_MS(
            2000,
            "VIO-only state: stamp=%.6f tracked=%d depth_prior=%d visual_residuals=%d pos=(%.3f %.3f %.3f)",
            measure.vio_time, vio_->getLastTrackedFeatureCount(), vio_->getLastDepthFeatureCount(),
            opt_visual_residuals, vio_state.pos_end.x(), vio_state.pos_end.y(), vio_state.pos_end.z());
      }

      if (!use_lidar_) {
        state_ = vio_state;
        latest_ekf_state_ = vio_state;
        latest_ekf_time_ = measure.vio_time;
        state_update_flag_ = true;
        ekf_finish_once_ = true;
        publishLioOdometry(measure.vio_time);
        publishLioPath(measure.vio_time);
        publishLioTf(measure.vio_time);
        publishLioMavrosPose(measure.vio_time);
      }
      publishVioOdometry(measure.vio_time, vio_state);
      publishVioPath(measure.vio_time, vio_state);
      publishVioPose(measure.vio_time, vio_state);
      publishVisualSubmap(measure.vio_time);
      publishVioWindowVisualization(measure.vio_time);
      t_vio_publish_end = std::chrono::steady_clock::now();
    }
  }

  const auto t_vio_end = std::chrono::steady_clock::now();
  if (t_vio_publish_end < t_vio_image_end) {
    t_vio_publish_end = t_vio_image_end;
  }
  static int vio_time_count = 0;
  static double vio_average_total = 0.0;
  const double imu_sec = std::chrono::duration<double>(t_vio_imu_end - t_vio_start).count();
  const double image_sec = std::chrono::duration<double>(t_vio_image_end - t_vio_imu_end).count();
  const double publish_sec = std::chrono::duration<double>(t_vio_publish_end - t_vio_image_end).count();
  const double total_sec = std::chrono::duration<double>(t_vio_end - t_vio_start).count();
  vio_time_count++;
  vio_average_total += (total_sec - vio_average_total) / static_cast<double>(vio_time_count);
  PrintTimeTable("VIO Time", BOLDGREEN,
                 {{"InputIMU", imu_sec},
                  {"TrackAndOptimize", image_sec},
                  {"PublishOutputs", publish_sec}},
                 total_sec, vio_average_total);
  CAKE_INFO("VIO summary: stamp=%.6f imu=%zu tracked=%d depth_prior=%d opt_features=%d opt_depth=%d visual_residuals=%d lidar_candidates=%zu lio_prior=%d lio_full_prior=%d lio_full_dt=%.6f feed_forward=%d write_lio=0 mode=%d",
            measure.vio_time, measure.imu.size(), vio_->getLastTrackedFeatureCount(),
            vio_->getLastDepthFeatureCount(), vio_->getLastOptimizationFeatureCount(),
            vio_->getLastOptimizationLidarFeatureCount(), vio_->getLastOptimizationVisualResidualCount(),
            lidar_candidate_count, static_cast<int>(has_lio_prior),
            static_cast<int>(has_lio_full_prior), lio_full_prior_dt,
            vio_feed_forward ? 1 : 0,
            static_cast<int>(slam_mode_));
  CAKE_INFO_THROTTLE_MS(
      2000,
      "VIO feature tracking: prev=%d tracked_after_flow=%d prev_depth=%d tracked_depth=%d lio_prior_reject=%d added_lidar=%d added_visual=%d pending_lidar_candidates=%d",
      vio_->getLastPrevTrackCount(), vio_->getLastTrackedAfterFlowCount(),
      vio_->getLastPrevLidarTrackCount(), vio_->getLastTrackedLidarCount(),
      vio_->getLastRejectedByLioPriorCount(), vio_->getLastAddedLidarCount(),
      vio_->getLastAddedVisualCount(), vio_->getLastPendingLidarCandidateCount());
}

void SlamNode::clearAllBuffersLocked()
{
  lidar_buffer_.clear();
  lidar_time_buffer_.clear();
  raw_livox_buffer_.clear();
  raw_cloud_buffer_.clear();
  raw_image_buffer_.clear();
  imu_lio_buffer_.clear();
  imu_vio_buffer_.clear();
  imu_prop_buffer_.clear();
  image_buffer_.clear();
  resetSyncStateLocked();
}

void SlamNode::trimBuffersLocked()
{
  const size_t max_size = static_cast<size_t>(std::max(1, config_.common.max_buffer_size));
  while (lidar_buffer_.size() > max_size) {
    lidar_buffer_.pop_front();
    lidar_time_buffer_.pop_front();
  }
  while (raw_livox_buffer_.size() > max_size) {
    raw_livox_buffer_.pop_front();
  }
  while (raw_cloud_buffer_.size() > max_size) {
    raw_cloud_buffer_.pop_front();
  }
  while (raw_image_buffer_.size() > max_size) {
    raw_image_buffer_.pop_front();
  }
  while (imu_lio_buffer_.size() > max_size) {
    imu_lio_buffer_.pop_front();
  }
  while (imu_vio_buffer_.size() > max_size) {
    imu_vio_buffer_.pop_front();
  }
  while (imu_prop_buffer_.size() > max_size) {
    imu_prop_buffer_.pop_front();
  }
  while (image_buffer_.size() > max_size) {
    image_buffer_.pop_front();
  }
}

void SlamNode::resetSyncStateLocked()
{
  measures_ = FusionMeasureGroup();
  lidar_pushed_ = false;
  last_lidar_time_ = -1.0;
  last_imu_time_ = -1.0;
  last_image_time_ = -1.0;
  last_vio_update_time_ = -1.0;
  latest_ekf_time_ = -1.0;
  last_imu_prop_time_ = -1.0;
  ekf_finish_once_ = false;
  state_update_flag_ = false;
  new_imu_ = false;
  hilti_image_counter_ = 0;
  latest_sync_mono_image_.release();
  latest_sync_color_image_.release();
  pending_lidar_visual_candidates_.clear();
  pending_lio_pose_prior_ = LioPosePrior();
  lio_full_state_history_.clear();
  lio_full_state_prior_ready_ = false;
  lio_path_msg_.poses.clear();
  vio_path_msg_.poses.clear();
}

double SlamNode::newestLidarEndTimeLocked() const
{
  if (lidar_buffer_.empty()) {
    return -1.0;
  }
  return cloudEndTime(lidar_buffer_.back(), lidar_time_buffer_.back());
}

double SlamNode::cloudEndTime(const PointCloudXYZI::Ptr &cloud, double begin_time) const
{
  if (!cloud || cloud->empty()) {
    return begin_time;
  }
  return begin_time + cloud->points.back().curvature / 1000.0;
}

double SlamNode::stampToSec(const builtin_interfaces::msg::Time &stamp)
{
  return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
}

builtin_interfaces::msg::Time SlamNode::secToStamp(double stamp)
{
  builtin_interfaces::msg::Time out;
  out.sec = static_cast<int32_t>(std::floor(stamp));
  out.nanosec = static_cast<uint32_t>((stamp - static_cast<double>(out.sec)) * 1e9);
  if (out.nanosec >= 1000000000U) {
    out.sec += 1;
    out.nanosec -= 1000000000U;
  }
  return out;
}

cv::Mat SlamNode::monoImageFromMsg(const sensor_msgs::msg::Image::ConstSharedPtr &msg) const
{
  // 和 VINS-Fusion-ROS2 主程序保持一致，统一转换成 MONO8 后交给前端。
  cv_bridge::CvImageConstPtr ptr;
  if (msg->encoding == "8UC1") {
    sensor_msgs::msg::Image mono_msg;
    mono_msg.header = msg->header;
    mono_msg.height = msg->height;
    mono_msg.width = msg->width;
    mono_msg.is_bigendian = msg->is_bigendian;
    mono_msg.step = msg->step;
    mono_msg.data = msg->data;
    mono_msg.encoding = sensor_msgs::image_encodings::MONO8;
    ptr = cv_bridge::toCvCopy(mono_msg, sensor_msgs::image_encodings::MONO8);
  } else {
    ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
  }
  return resizeImageToConfig(ptr->image, "mono");
}

cv::Mat SlamNode::colorImageFromMsg(const sensor_msgs::msg::Image::ConstSharedPtr &msg) const
{
  cv_bridge::CvImageConstPtr ptr;
  if (msg->encoding == sensor_msgs::image_encodings::BGR8) {
    ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
  } else if (msg->encoding == sensor_msgs::image_encodings::RGB8) {
    cv_bridge::CvImageConstPtr rgb = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::RGB8);
    cv::Mat bgr;
    cv::cvtColor(rgb->image, bgr, cv::COLOR_RGB2BGR);
    return resizeImageToConfig(bgr, "color");
  } else {
    ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  return resizeImageToConfig(ptr->image, "color");
}

cv::Mat SlamNode::resizeImageToConfig(const cv::Mat &image, const char *image_kind) const
{
  if (image.empty()) {
    return image.clone();
  }

  const int target_width = config_.vision.image_width;
  const int target_height = config_.vision.image_height;
  if (target_width <= 0 || target_height <= 0 ||
      (image.cols == target_width && image.rows == target_height)) {
    return image.clone();
  }

  auto clock_ptr = std::const_pointer_cast<rclcpp::Clock>(this->get_clock());
  RCLCPP_WARN_THROTTLE(
      get_logger(), *clock_ptr, 5000,
      "Resizing %s image from %dx%d to configured %dx%d. Make sure cam0_calib uses the same target size.",
      image_kind, image.cols, image.rows, target_width, target_height);

  cv::Mat resized;
  cv::resize(image, resized, cv::Size(target_width, target_height), 0.0, 0.0, cv::INTER_LINEAR);
  return resized;
}

void SlamNode::loadExtrinsicsForMain()
{
  if (config_.extrinsic.lidar_R.size() >= 9) {
    lidar_to_imu_R_ << config_.extrinsic.lidar_R[0], config_.extrinsic.lidar_R[1], config_.extrinsic.lidar_R[2],
                       config_.extrinsic.lidar_R[3], config_.extrinsic.lidar_R[4], config_.extrinsic.lidar_R[5],
                       config_.extrinsic.lidar_R[6], config_.extrinsic.lidar_R[7], config_.extrinsic.lidar_R[8];
  }
  if (config_.extrinsic.lidar_T.size() >= 3) {
    lidar_to_imu_t_ << config_.extrinsic.lidar_T[0], config_.extrinsic.lidar_T[1], config_.extrinsic.lidar_T[2];
  }
  if (config_.extrinsic.camera_R.size() >= 9) {
    lidar_to_camera_R_ << config_.extrinsic.camera_R[0], config_.extrinsic.camera_R[1], config_.extrinsic.camera_R[2],
                          config_.extrinsic.camera_R[3], config_.extrinsic.camera_R[4], config_.extrinsic.camera_R[5],
                          config_.extrinsic.camera_R[6], config_.extrinsic.camera_R[7], config_.extrinsic.camera_R[8];
  }
  if (config_.extrinsic.camera_T.size() >= 3) {
    lidar_to_camera_t_ << config_.extrinsic.camera_T[0], config_.extrinsic.camera_T[1], config_.extrinsic.camera_T[2];
  }

  if (config_.extrinsic.body_T_cam0.size() >= 16) {
    const auto &T = config_.extrinsic.body_T_cam0;
    camera_to_imu_R_ << T[0], T[1], T[2],
                        T[4], T[5], T[6],
                        T[8], T[9], T[10];
    camera_to_imu_t_ << T[3], T[7], T[11];
  } else {
    camera_to_imu_R_ = lidar_to_imu_R_ * lidar_to_camera_R_.transpose();
    camera_to_imu_t_ = lidar_to_imu_t_ - camera_to_imu_R_ * lidar_to_camera_t_;
  }

  CAKE_INFO("Main extrinsics: R_I_L=[%.6f %.6f %.6f; %.6f %.6f %.6f; %.6f %.6f %.6f], t_I_L=[%.6f %.6f %.6f]",
            lidar_to_imu_R_(0, 0), lidar_to_imu_R_(0, 1), lidar_to_imu_R_(0, 2),
            lidar_to_imu_R_(1, 0), lidar_to_imu_R_(1, 1), lidar_to_imu_R_(1, 2),
            lidar_to_imu_R_(2, 0), lidar_to_imu_R_(2, 1), lidar_to_imu_R_(2, 2),
            lidar_to_imu_t_.x(), lidar_to_imu_t_.y(), lidar_to_imu_t_.z());
  CAKE_INFO("Main extrinsics: R_C_L=[%.6f %.6f %.6f; %.6f %.6f %.6f; %.6f %.6f %.6f], t_C_L=[%.6f %.6f %.6f]",
            lidar_to_camera_R_(0, 0), lidar_to_camera_R_(0, 1), lidar_to_camera_R_(0, 2),
            lidar_to_camera_R_(1, 0), lidar_to_camera_R_(1, 1), lidar_to_camera_R_(1, 2),
            lidar_to_camera_R_(2, 0), lidar_to_camera_R_(2, 1), lidar_to_camera_R_(2, 2),
            lidar_to_camera_t_.x(), lidar_to_camera_t_.y(), lidar_to_camera_t_.z());
  CAKE_INFO("Main extrinsics: R_I_C=[%.6f %.6f %.6f; %.6f %.6f %.6f; %.6f %.6f %.6f], t_I_C=[%.6f %.6f %.6f]",
            camera_to_imu_R_(0, 0), camera_to_imu_R_(0, 1), camera_to_imu_R_(0, 2),
            camera_to_imu_R_(1, 0), camera_to_imu_R_(1, 1), camera_to_imu_R_(1, 2),
            camera_to_imu_R_(2, 0), camera_to_imu_R_(2, 1), camera_to_imu_R_(2, 2),
            camera_to_imu_t_.x(), camera_to_imu_t_.y(), camera_to_imu_t_.z());
  CAKE_INFO("Image config: topic=%s target=%dx%d image_time_offset=%.6f td=%.6f cam0_calib=%s lidar_depth_source=%d lidar_prior_feature_enable=%d visual_depth_prior=%d",
            config_.vision.image_topic.c_str(), config_.vision.image_width, config_.vision.image_height,
            config_.common.image_time_offset, config_.time_offset.td, config_.vision.cam0_calib.c_str(),
            config_.vision.lidar_depth_source,
            config_.vision.lidar_prior_feature_enable ? 1 : 0,
            config_.vision.visual_feature_depth_prior_enable ? 1 : 0);
}

LioPosePrior SlamNode::makeLioPosePrior(double stamp) const
{
  // Crop the StatesGroup covariance as [delta_theta(rad), delta_p(m)], inflate
  // rotation/translation blocks, invert, then store a square-root information
  // matrix for the Ceres prior residual.
  LioPosePrior prior;
  prior.valid = true;
  prior.timestamp = stamp;
  prior.R_WB = state_.rot_end;
  prior.p_WB = state_.pos_end;

  Eigen::Matrix<double, 6, 6> cov_pose =
      state_.cov.block<6, 6>(0, 0).cast<double>();
  cov_pose = 0.5 * (cov_pose + cov_pose.transpose());
  cov_pose.topLeftCorner<3, 3>() *= 5.0;
  cov_pose.bottomRightCorner<3, 3>() *= 2.0;

  const double min_var = std::max(1e-12, config_.vision.min_lio_pose_prior_var);
  for (int i = 0; i < 3; ++i) {
    cov_pose(i, i) = std::max(min_var, cov_pose(i, i));
    cov_pose(i + 3, i + 3) = std::max(min_var, cov_pose(i + 3, i + 3));
  }

  const Eigen::Matrix<double, 6, 6> information = cov_pose.inverse();
  Eigen::LLT<Eigen::Matrix<double, 6, 6>> llt(information);
  if (llt.info() == Eigen::Success) {
    prior.sqrt_information = llt.matrixL().transpose();
  } else {
    prior.sqrt_information.setZero();
    for (int i = 0; i < 6; ++i) {
      prior.sqrt_information(i, i) = 1.0 / std::sqrt(std::max(min_var, cov_pose(i, i)));
    }
  }
  return prior;
}

LioFullStatePrior SlamNode::makeLioFullStatePrior(double stamp) const
{
  LioFullStatePrior prior;
  prior.valid = true;
  prior.timestamp = stamp;

  const StatesGroup &lio_state = lio_.GetState();
  prior.R_WB = lio_state.rot_end;
  prior.p_WB = lio_state.pos_end;
  prior.v_WB = lio_state.vel_end;
  prior.ba = lio_state.bias_a;
  prior.bg = lio_state.bias_g;

  // StatesGroup covariance order is [rot, pos, inv_expo, vel, bg, ba, gravity].
  // LioFullStatePriorResidual order is [pos, rot, vel, ba, bg].
  constexpr int kFullPriorStateIndex[15] = {
      3, 4, 5,    // pos
      0, 1, 2,    // rot
      7, 8, 9,    // vel
      13, 14, 15, // ba
      10, 11, 12  // bg
  };

  Eigen::Matrix<double, 15, 15> cov_full;
  cov_full.setZero();
  for (int row = 0; row < 15; ++row) {
    for (int col = 0; col < 15; ++col) {
      const double value = lio_state.cov(kFullPriorStateIndex[row], kFullPriorStateIndex[col]);
      cov_full(row, col) = std::isfinite(value) ? value : 0.0;
    }
  }
  cov_full = (0.5 * (cov_full + cov_full.transpose())).eval();

  Eigen::Matrix<double, 15, 1> cov_inflation;
  cov_inflation.segment<3>(0).setConstant(std::sqrt(2.0));  // pos
  cov_inflation.segment<3>(3).setConstant(std::sqrt(5.0));  // rot
  cov_inflation.segment<3>(6).setConstant(std::sqrt(5.0));  // vel
  cov_inflation.segment<3>(9).setConstant(std::sqrt(10.0)); // ba
  cov_inflation.segment<3>(12).setConstant(std::sqrt(10.0)); // bg
  cov_full = cov_inflation.asDiagonal() * cov_full * cov_inflation.asDiagonal();
  cov_full = (0.5 * (cov_full + cov_full.transpose())).eval();

  const double config_min_var = std::max(1e-12, config_.vision.min_lio_pose_prior_var);
  constexpr double kMinVariance[15] = {
      1e-4, 1e-4, 1e-4,       // pos: 0.01 m
      4e-6, 4e-6, 4e-6,       // rot: 0.002 rad
      2.5e-3, 2.5e-3, 2.5e-3, // vel: 0.05 m/s
      4e-6, 4e-6, 4e-6,       // ba
      1e-6, 1e-6, 1e-6        // bg
  };
  for (int i = 0; i < 15; ++i) {
    const double min_var = std::max(config_min_var, kMinVariance[i]);
    if (!std::isfinite(cov_full(i, i)) || cov_full(i, i) < min_var) {
      cov_full(i, i) = min_var;
    }
  }

  Eigen::LLT<Eigen::Matrix<double, 15, 15>> cov_llt(cov_full);
  if (cov_llt.info() == Eigen::Success) {
    const Eigen::Matrix<double, 15, 15> information =
        cov_llt.solve(Eigen::Matrix<double, 15, 15>::Identity());
    const Eigen::Matrix<double, 15, 15> information_sym =
        (0.5 * (information + information.transpose())).eval();
    Eigen::LLT<Eigen::Matrix<double, 15, 15>> info_llt(information_sym);
    if (info_llt.info() == Eigen::Success) {
      prior.sqrt_information = info_llt.matrixL().transpose();
    } else {
      prior.sqrt_information.setZero();
      for (int i = 0; i < 15; ++i) {
        prior.sqrt_information(i, i) = 1.0 / std::sqrt(cov_full(i, i));
      }
    }
  } else {
    prior.sqrt_information.setZero();
    for (int i = 0; i < 15; ++i) {
      prior.sqrt_information(i, i) = 1.0 / std::sqrt(cov_full(i, i));
    }
  }
  return prior;
}

void SlamNode::recordLioFullStatePrior(double stamp)
{
  if (use_imu_ && !lio_.IsImuInitialized()) {
    lio_full_state_history_.clear();
    return;
  }

  LioFullStatePrior prior = makeLioFullStatePrior(stamp);
  if (!lio_full_state_history_.empty() && stamp < lio_full_state_history_.back().timestamp) {
    lio_full_state_history_.clear();
  }
  lio_full_state_history_.push_back(prior);

  constexpr size_t kMaxLioFullStateHistory = 2000;
  while (lio_full_state_history_.size() > kMaxLioFullStateHistory) {
    lio_full_state_history_.pop_front();
  }
}

LioFullStatePrior SlamNode::FindClosestLioState(double stamp) const
{
  LioFullStatePrior closest;
  if (use_imu_ && !lio_.IsImuInitialized()) {
    closest.timestamp = stamp;
    return closest;
  }

  double best_dt = std::numeric_limits<double>::max();
  for (const auto &prior : lio_full_state_history_) {
    const double dt = std::abs(prior.timestamp - stamp);
    if (dt < best_dt) {
      best_dt = dt;
      closest = prior;
    }
  }
  if (best_dt > 0.01) {
    closest.valid = false;
    closest.timestamp = stamp;
  }
  return closest;
}

void SlamNode::buildLidarVisualCandidates(double stamp)
{
  // Build sparse depth assistance for the synchronized image. In scan mode the
  // current LIO cloud is projected directly. In LocalVoxelMap mode the same
  // cloud is used only for occlusion while depth candidates come from map voxels.
  pending_lidar_visual_candidates_.clear();
  pending_lidar_depth_frame_ = LidarDepthFrame();
  const bool depth_assist_requested =
      config_.vision.lidar_prior_feature_enable ||
      config_.vision.visual_feature_depth_prior_enable;
  if (!use_image_ || !vio_ || !config_.vision.lidar_depth_enable ||
      config_.vision.lidar_depth_source == 2 ||
      !depth_assist_requested || latest_sync_mono_image_.empty()) {
    const char *reason = "disabled";
    if (!use_image_)
      reason = "image_disabled";
    else if (!vio_)
      reason = "vio_missing";
    else if (!config_.vision.lidar_depth_enable)
      reason = "lidar_depth_disabled";
    else if (config_.vision.lidar_depth_source == 2)
      reason = "lidar_depth_source_off";
    else if (!depth_assist_requested)
      reason = "depth_assist_disabled";
    else if (latest_sync_mono_image_.empty())
      reason = "image_empty";
    std::printf("LIDAR CANDIDATE DEBUG stamp=%.6f skip=1 reason=%s use_image=%d vio=%d lidar_depth_enable=%d lidar_depth_source=%d lidar_prior_feature_enable=%d visual_depth_prior=%d image_empty=%d state_z=% .6f\n",
                stamp,
                reason,
                use_image_ ? 1 : 0,
                vio_ ? 1 : 0,
                config_.vision.lidar_depth_enable ? 1 : 0,
                config_.vision.lidar_depth_source,
                config_.vision.lidar_prior_feature_enable ? 1 : 0,
                config_.vision.visual_feature_depth_prior_enable ? 1 : 0,
                latest_sync_mono_image_.empty() ? 1 : 0,
                state_.pos_end.z());
    return;
  }

  auto camera = vio_->getCameraModel();
  if (!camera) {
    std::printf("LIDAR CANDIDATE DEBUG stamp=%.6f skip=1 reason=no_camera image_size=%dx%d state_z=% .6f\n",
                stamp,
                latest_sync_mono_image_.cols,
                latest_sync_mono_image_.rows,
                state_.pos_end.z());
    return;
  }

  PointCloudXYZI::Ptr source_cloud = lio_.GetDownsampledCloud();
  if (!source_cloud || source_cloud->empty()) {
    source_cloud = lio_.GetUndistortedCloud();
  }
  const size_t source_points = source_cloud ? source_cloud->size() : 0;
  if (!source_cloud || source_cloud->empty()) {
    std::printf("LIDAR CANDIDATE DEBUG stamp=%.6f skip=1 reason=empty_cloud image_size=%dx%d state_z=% .6f\n",
                stamp,
                latest_sync_mono_image_.cols,
                latest_sync_mono_image_.rows,
                state_.pos_end.z());
    return;
  }
  if (config_.vision.lidar_depth_source == 0) {
    pending_lidar_visual_candidates_ = lidar_visual_selector_.SelectFromLocalMap(
        local_voxel_map_, source_cloud, state_, latest_sync_mono_image_,
        vio_->getImageValidMask(), camera, &pending_lidar_depth_frame_);
  } else {
    pending_lidar_visual_candidates_ = lidar_visual_selector_.Select(
        source_cloud, state_, latest_sync_mono_image_, vio_->getImageValidMask(),
        camera, &pending_lidar_depth_frame_);
  }

  const auto &stats = lidar_visual_selector_.lastStats();
  const double depth_ratio = stats.input_points > 0
                                 ? static_cast<double>(stats.positive_depth) /
                                       static_cast<double>(stats.input_points)
                                 : 0.0;
  const double image_ratio = stats.positive_depth > 0
                                 ? static_cast<double>(stats.in_image) /
                                       static_cast<double>(stats.positive_depth)
                                 : 0.0;
  const double texture_ratio = stats.zbuffer_kept > 0
                                   ? static_cast<double>(stats.texture_kept) /
                                         static_cast<double>(stats.zbuffer_kept)
                                   : 0.0;
  const double mask_ratio = stats.texture_kept > 0
                                ? static_cast<double>(stats.mask_kept) /
                                      static_cast<double>(stats.texture_kept)
                                : 0.0;
  std::printf("LIDAR CANDIDATE DEBUG stamp=%.6f skip=0 source_mode=%d occlusion_scan_points=%zu candidate_input=%d positive_depth=%d in_image=%d zbuffer=%d texture=%d mask_selected=%d selected=%zu depth_ratio=%.3f image_ratio=%.3f texture_ratio=%.3f mask_ratio=%.3f occlusion_reject=%d local_map_samples=%d depth_cells=%d image_size=%dx%d state_z=% .6f min_depth=%.3f max_depth=%.3f mask_radius=%.3f score_min=%.6g timing_ms={total=%.3f map_collect=%.3f project=%.3f occlusion=%.3f texture=%.3f select=%.3f}\n",
              stamp,
              stats.source_mode,
              source_points,
              stats.input_points,
              stats.positive_depth,
              stats.in_image,
              stats.zbuffer_kept,
              stats.texture_kept,
              stats.mask_kept,
              pending_lidar_visual_candidates_.size(),
              depth_ratio,
              image_ratio,
              texture_ratio,
              mask_ratio,
              stats.occlusion_reject,
              stats.source_mode == 0 ? stats.input_points : 0,
              stats.visual_depth_cells,
              latest_sync_mono_image_.cols,
              latest_sync_mono_image_.rows,
              state_.pos_end.z(),
              config_.vision.min_lidar_depth,
              config_.vision.max_lidar_depth,
              config_.vision.lidar_mask_radius,
              config_.vision.shi_tomasi_min_score,
              stats.total_ms,
              stats.map_collect_ms,
              stats.project_ms,
              stats.occlusion_ms,
              stats.texture_ms,
              stats.select_ms);
  CAKE_INFO_THROTTLE_MS(
      2000,
      "LiDAR visual candidates: candidate_input=%d positive_depth=%d in_image=%d zbuffer=%d texture=%d mask_selected=%d image_size=%dx%d",
      stats.input_points, stats.positive_depth, stats.in_image, stats.zbuffer_kept,
      stats.texture_kept, stats.mask_kept, latest_sync_mono_image_.cols, latest_sync_mono_image_.rows);
}

Eigen::Vector3d SlamNode::lidarToWorld(const Eigen::Vector3d &point_lidar) const
{
  const Eigen::Vector3d point_imu = lidar_to_imu_R_ * point_lidar + lidar_to_imu_t_;
  return state_.rot_end * point_imu + state_.pos_end;
}

void SlamNode::handleFirstFrame()
{
  if (first_frame_handled_) {
    return;
  }
  first_lidar_time_ = measures_.last_lio_update_time;
  first_frame_handled_ = true;
  CAKE_INFO("First LiDAR frame time: %.6f", first_lidar_time_);
}

void SlamNode::gravityAlignment()
{
  if (!config_.common.gravity_align_enable || gravity_align_finished_ || state_.gravity.norm() < 1.0) {
    return;
  }

  const Eigen::Vector3d target_g(0.0, 0.0, -state_.gravity.norm());
  const Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(state_.gravity, target_g);
  const Eigen::Matrix3d R = q.toRotationMatrix();
  state_.pos_end = R * state_.pos_end;
  state_.rot_end = R * state_.rot_end;
  state_.vel_end = R * state_.vel_end;
  state_.gravity = R * state_.gravity;
  lio_.SetState(state_);
  gravity_align_finished_ = true;
  CAKE_INFO("Gravity alignment finished.");
}

void SlamNode::writeTumTrajectory(std::ofstream &stream, double stamp, const StatesGroup &state)
{
  if (!stream.is_open()) {
    return;
  }

  Eigen::Quaterniond q(state.rot_end);
  q.normalize();
  stream << std::fixed << std::setprecision(9)
         << stamp << " "
         << state.pos_end.x() << " " << state.pos_end.y() << " " << state.pos_end.z() << " "
         << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << '\n';
}

void SlamNode::publishLioOdometry(double stamp)
{
  if (!pub_lio_odom_) {
    return;
  }
  nav_msgs::msg::Odometry odom;
  odom.header.stamp = secToStamp(stamp);
  odom.header.frame_id = config_.frame.world;
  odom.child_frame_id = config_.frame.body;
  odom.pose.pose.position.x = state_.pos_end.x();
  odom.pose.pose.position.y = state_.pos_end.y();
  odom.pose.pose.position.z = state_.pos_end.z();
  const Eigen::Quaterniond q(state_.rot_end);
  odom.pose.pose.orientation.x = q.x();
  odom.pose.pose.orientation.y = q.y();
  odom.pose.pose.orientation.z = q.z();
  odom.pose.pose.orientation.w = q.w();
  odom.twist.twist.linear.x = state_.vel_end.x();
  odom.twist.twist.linear.y = state_.vel_end.y();
  odom.twist.twist.linear.z = state_.vel_end.z();
  pub_lio_odom_->publish(odom);
  writeTumTrajectory(lio_trajectory_file_, stamp, state_);
}

void SlamNode::publishLioPath(double stamp)
{
  if (!pub_lio_path_) {
    return;
  }
  geometry_msgs::msg::PoseStamped pose;
  pose.header.stamp = secToStamp(stamp);
  pose.header.frame_id = config_.frame.world;
  pose.pose.position.x = state_.pos_end.x();
  pose.pose.position.y = state_.pos_end.y();
  pose.pose.position.z = state_.pos_end.z();
  const Eigen::Quaterniond q(state_.rot_end);
  pose.pose.orientation.x = q.x();
  pose.pose.orientation.y = q.y();
  pose.pose.orientation.z = q.z();
  pose.pose.orientation.w = q.w();
  lio_path_msg_.header = pose.header;
  lio_path_msg_.poses.push_back(pose);
  pub_lio_path_->publish(lio_path_msg_);
}

void SlamNode::publishLioMavrosPose(double stamp)
{
  if (!pub_mavros_pose_) {
    return;
  }
  geometry_msgs::msg::PoseStamped pose;
  pose.header.stamp = secToStamp(stamp);
  pose.header.frame_id = config_.frame.world;
  pose.pose.position.x = state_.pos_end.x();
  pose.pose.position.y = state_.pos_end.y();
  pose.pose.position.z = state_.pos_end.z();
  const Eigen::Quaterniond q(state_.rot_end);
  pose.pose.orientation.x = q.x();
  pose.pose.orientation.y = q.y();
  pose.pose.orientation.z = q.z();
  pose.pose.orientation.w = q.w();
  pub_mavros_pose_->publish(pose);
}

void SlamNode::publishLioTf(double stamp)
{
  if (!tf_broadcaster_) {
    return;
  }
  geometry_msgs::msg::TransformStamped tf;
  tf.header.stamp = secToStamp(stamp);
  tf.header.frame_id = config_.frame.world;
  tf.child_frame_id = config_.frame.body;
  tf.transform.translation.x = state_.pos_end.x();
  tf.transform.translation.y = state_.pos_end.y();
  tf.transform.translation.z = state_.pos_end.z();
  const Eigen::Quaterniond q(state_.rot_end);
  tf.transform.rotation.x = q.x();
  tf.transform.rotation.y = q.y();
  tf.transform.rotation.z = q.z();
  tf.transform.rotation.w = q.w();
  tf_broadcaster_->sendTransform(tf);
}

void SlamNode::publishVioOdometry(double stamp, const StatesGroup &state)
{
  if (!pub_vio_odom_) {
    return;
  }
  nav_msgs::msg::Odometry odom;
  odom.header.stamp = secToStamp(stamp);
  odom.header.frame_id = config_.frame.world;
  odom.child_frame_id = config_.frame.body;
  odom.pose.pose.position.x = state.pos_end.x();
  odom.pose.pose.position.y = state.pos_end.y();
  odom.pose.pose.position.z = state.pos_end.z();
  const Eigen::Quaterniond q(state.rot_end);
  odom.pose.pose.orientation.x = q.x();
  odom.pose.pose.orientation.y = q.y();
  odom.pose.pose.orientation.z = q.z();
  odom.pose.pose.orientation.w = q.w();
  odom.twist.twist.linear.x = state.vel_end.x();
  odom.twist.twist.linear.y = state.vel_end.y();
  odom.twist.twist.linear.z = state.vel_end.z();
  pub_vio_odom_->publish(odom);
  writeTumTrajectory(vio_trajectory_file_, stamp, state);
}

void SlamNode::publishVioPath(double stamp, const StatesGroup &state)
{
  if (!pub_vio_path_) {
    return;
  }
  geometry_msgs::msg::PoseStamped pose;
  pose.header.stamp = secToStamp(stamp);
  pose.header.frame_id = config_.frame.world;
  pose.pose.position.x = state.pos_end.x();
  pose.pose.position.y = state.pos_end.y();
  pose.pose.position.z = state.pos_end.z();
  const Eigen::Quaterniond q(state.rot_end);
  pose.pose.orientation.x = q.x();
  pose.pose.orientation.y = q.y();
  pose.pose.orientation.z = q.z();
  pose.pose.orientation.w = q.w();
  vio_path_msg_.header = pose.header;
  vio_path_msg_.poses.push_back(pose);
  pub_vio_path_->publish(vio_path_msg_);
}

void SlamNode::publishVioPose(double stamp, const StatesGroup &state)
{
  if (!pub_vio_pose_) {
    return;
  }
  geometry_msgs::msg::PoseStamped pose;
  pose.header.stamp = secToStamp(stamp);
  pose.header.frame_id = config_.frame.world;
  pose.pose.position.x = state.pos_end.x();
  pose.pose.position.y = state.pos_end.y();
  pose.pose.position.z = state.pos_end.z();
  const Eigen::Quaterniond q(state.rot_end);
  pose.pose.orientation.x = q.x();
  pose.pose.orientation.y = q.y();
  pose.pose.orientation.z = q.z();
  pose.pose.orientation.w = q.w();
  pub_vio_pose_->publish(pose);
}

void SlamNode::publishStaticTf()
{
  if (!static_tf_broadcaster_) {
    return;
  }

  std::vector<geometry_msgs::msg::TransformStamped> transforms;
  const auto stamp = now();

  const auto make_transform =
      [stamp](const std::string &parent, const std::string &child,
              const Eigen::Matrix3d &R_parent_child,
              const Eigen::Vector3d &t_parent_child) {
        geometry_msgs::msg::TransformStamped tf;
        tf.header.stamp = stamp;
        tf.header.frame_id = parent;
        tf.child_frame_id = child;
        tf.transform.translation.x = t_parent_child.x();
        tf.transform.translation.y = t_parent_child.y();
        tf.transform.translation.z = t_parent_child.z();
        const Eigen::Quaterniond q(R_parent_child);
        tf.transform.rotation.x = q.x();
        tf.transform.rotation.y = q.y();
        tf.transform.rotation.z = q.z();
        tf.transform.rotation.w = q.w();
        return tf;
      };

  if (!config_.frame.lidar.empty() && config_.frame.lidar != config_.frame.body) {
    transforms.push_back(make_transform(config_.frame.body, config_.frame.lidar,
                                        lidar_to_imu_R_, lidar_to_imu_t_));
  }
  if (!config_.frame.camera.empty() && config_.frame.camera != config_.frame.body) {
    transforms.push_back(make_transform(config_.frame.body, config_.frame.camera,
                                        camera_to_imu_R_, camera_to_imu_t_));
  }

  if (!transforms.empty()) {
    static_tf_broadcaster_->sendTransform(transforms);
    CAKE_INFO("Static TF published: world=%s body=%s lidar=%s camera=%s",
              config_.frame.world.c_str(), config_.frame.body.c_str(),
              config_.frame.lidar.c_str(), config_.frame.camera.c_str());
  }
}

void SlamNode::publishRawCloud(double stamp, const PointCloudXYZI::Ptr &cloud_lidar)
{
  if (!pub_cloud_raw_ || !cloud_lidar || cloud_lidar->empty()) {
    return;
  }
  sensor_msgs::msg::PointCloud2 msg;
  pcl::toROSMsg(*cloud_lidar, msg);
  msg.header.stamp = secToStamp(stamp);
  msg.header.frame_id = config_.frame.lidar;
  pub_cloud_raw_->publish(msg);
}

void SlamNode::publishClouds(double stamp)
{
  publishRawCloud(stamp, measures_.pcl_proc_cur);

  const PointCloudXYZI::Ptr cloud_world = lio_.GetDownsampledWorldCloud();
  if (pub_cloud_registered_ && cloud_world && !cloud_world->empty()) {
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*cloud_world, msg);
    msg.header.stamp = secToStamp(stamp);
    msg.header.frame_id = config_.frame.world;
    pub_cloud_registered_->publish(msg);
    if (pub_cloud_map_) {
      pub_cloud_map_->publish(msg);
    }
  }
  if (pub_cloud_effect_) {
    const auto &effect_points = lio_.GetEffectPoints();
    PointCloudXYZI effect_cloud;
    effect_cloud.reserve(effect_points.size());
    for (const auto &ptpl : effect_points) {
      PointType pt;
      pt.x = static_cast<float>(ptpl.point_w_.x());
      pt.y = static_cast<float>(ptpl.point_w_.y());
      pt.z = static_cast<float>(ptpl.point_w_.z());
      pt.intensity = static_cast<float>(std::abs(ptpl.dis_to_plane_));
      effect_cloud.push_back(pt);
    }
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(effect_cloud, msg);
    msg.header.stamp = secToStamp(stamp);
    msg.header.frame_id = config_.frame.world;
    pub_cloud_effect_->publish(msg);
  }
  publishColoredCloud(stamp, lio_.GetDownsampledCloud());
}

void SlamNode::publishColoredCloud(double stamp, const PointCloudXYZI::Ptr &cloud_lidar)
{
  // Project each LiDAR point with T_C_L into the synchronized color image.
  // Only valid colored projections are published, matching FAST-LIVO2's RGB cloud
  // behavior and avoiding gray filler points that hide calibration errors.
  if (!pub_cloud_colored_ || !cloud_lidar || cloud_lidar->empty()) {
    return;
  }

  const auto camera = vio_ ? vio_->getCameraModel() : camodocal::CameraConstPtr();
  const bool can_color = camera && !latest_sync_color_image_.empty();
  if (!can_color) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "Skip colored cloud: camera=%d color_image_empty=%d",
                         camera ? 1 : 0, latest_sync_color_image_.empty() ? 1 : 0);
    return;
  }

  PointCloudXYZRGB colored;
  colored.reserve(cloud_lidar->size());
  size_t positive_depth = 0;
  size_t in_image = 0;
  for (const auto &pt : cloud_lidar->points) {
    const Eigen::Vector3d p_lidar(pt.x, pt.y, pt.z);
    const Eigen::Vector3d p_cam = lidar_to_camera_R_ * p_lidar + lidar_to_camera_t_;
    if (p_cam.z() <= config_.vision.min_lidar_depth) {
      continue;
    }
    positive_depth++;

    Eigen::Vector2d uv;
    camera->spaceToPlane(p_cam, uv);
    const int u = static_cast<int>(std::round(uv.x()));
    const int v = static_cast<int>(std::round(uv.y()));
    if (u < 0 || v < 0 || u >= latest_sync_color_image_.cols || v >= latest_sync_color_image_.rows) {
      continue;
    }
    in_image++;

    const cv::Vec3b bgr = latest_sync_color_image_.at<cv::Vec3b>(v, u);
    PointTypeRGB out;
    const Eigen::Vector3d p_world = lidarToWorld(p_lidar);
    out.x = static_cast<float>(p_world.x());
    out.y = static_cast<float>(p_world.y());
    out.z = static_cast<float>(p_world.z());
    out.r = bgr[2];
    out.g = bgr[1];
    out.b = bgr[0];
    colored.push_back(out);
  }
  if (colored.empty()) {
    RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Colored cloud empty: input=%zu positive_depth=%zu in_image=%zu image_size=%dx%d. Check image scale/intrinsics/extrinsics/time offset.",
        cloud_lidar->size(), positive_depth, in_image, latest_sync_color_image_.cols, latest_sync_color_image_.rows);
    return;
  }

  CAKE_INFO_THROTTLE_MS(
      2000,
      "Colored cloud: input=%zu positive_depth=%zu in_image=%zu published=%zu image_size=%dx%d",
      cloud_lidar->size(), positive_depth, in_image, colored.size(),
      latest_sync_color_image_.cols, latest_sync_color_image_.rows);

  sensor_msgs::msg::PointCloud2 msg;
  pcl::toROSMsg(colored, msg);
  msg.header.stamp = secToStamp(stamp);
  msg.header.frame_id = config_.frame.world;
  pub_cloud_colored_->publish(msg);
}

void SlamNode::publishFeatureImage(double stamp)
{
  if (!vio_) {
    return;
  }

  cv::Mat debug_image = vio_->getFeatureDebugImage();
  if (debug_image.empty()) {
    return;
  }
  if (debug_image.channels() == 1) {
    cv::cvtColor(debug_image, debug_image, cv::COLOR_GRAY2BGR);
  }

  std_msgs::msg::Header header;
  header.stamp = secToStamp(stamp);
  header.frame_id = config_.frame.camera;
  auto msg = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, debug_image).toImageMsg();
  pub_feature_image_.publish(*msg);

  const int total = vio_->getLastTrackedFeatureCount();
  const int with_depth = vio_->getLastDepthFeatureCount();
  const double ratio = total > 0 ? static_cast<double>(with_depth) / total : 0.0;
  CAKE_INFO_THROTTLE_MS(
      2000,
      "Feature image: stamp=%.6f tracked=%d depth_prior=%d depth_ratio=%.3f size=%dx%d",
      stamp, total, with_depth, ratio, debug_image.cols, debug_image.rows);
}

void SlamNode::publishVisualSubmap(double stamp)
{
  if (!pub_cloud_visual_submap_ || !vio_ || vio_->solver_flag != Estimator::NON_LINEAR) {
    return;
  }

  PointCloudXYZI cloud;
  for (const auto &feature : vio_->f_manager.feature) {
    const int used_num = static_cast<int>(feature.feature_per_frame.size());
    if (used_num < 2 || feature.solve_flag != 1 || feature.estimated_depth <= 0.0) {
      continue;
    }
    const int imu_i = feature.start_frame;
    if (imu_i < 0 || imu_i > WINDOW_SIZE) {
      continue;
    }
    const Eigen::Vector3d point_cam = feature.feature_per_frame[0].point * feature.estimated_depth;
    const Eigen::Vector3d point_body = vio_->getCameraToImuRotation() * point_cam + vio_->getCameraToImuTranslation();
    const Eigen::Vector3d point_world = vio_->states_[imu_i].rot_end * point_body + vio_->states_[imu_i].pos_end;
    PointType pt;
    pt.x = static_cast<float>(point_world.x());
    pt.y = static_cast<float>(point_world.y());
    pt.z = static_cast<float>(point_world.z());
    pt.intensity = 1.0f;
    cloud.push_back(pt);
  }

  sensor_msgs::msg::PointCloud2 msg;
  pcl::toROSMsg(cloud, msg);
  msg.header.stamp = secToStamp(stamp);
  msg.header.frame_id = config_.frame.world;
  pub_cloud_visual_submap_->publish(msg);
}

void SlamNode::publishVioWindowVisualization(double stamp)
{
  if (!vio_ || vio_->solver_flag != Estimator::NON_LINEAR) {
    return;
  }

  std_msgs::msg::Header header;
  header.stamp = secToStamp(stamp);
  header.frame_id = config_.frame.world;

  if (pub_vio_landmarks_) {
    PointCloudXYZI cloud;
    for (const auto &feature : vio_->f_manager.feature) {
      const int used_num = static_cast<int>(feature.feature_per_frame.size());
      if (used_num < 2 || feature.solve_flag != 1 || feature.estimated_depth <= 0.0) {
        continue;
      }
      const int imu_i = feature.start_frame;
      if (imu_i < 0 || imu_i > WINDOW_SIZE) {
        continue;
      }
      const Eigen::Vector3d point_cam = feature.feature_per_frame[0].point * feature.estimated_depth;
      const Eigen::Vector3d point_body = vio_->getCameraToImuRotation() * point_cam + vio_->getCameraToImuTranslation();
      const Eigen::Vector3d point_world = vio_->states_[imu_i].rot_end * point_body + vio_->states_[imu_i].pos_end;

      PointType pt;
      pt.x = static_cast<float>(point_world.x());
      pt.y = static_cast<float>(point_world.y());
      pt.z = static_cast<float>(point_world.z());
      pt.intensity = feature.has_lidar_depth_prior ? 2.0f : 1.0f;
      cloud.push_back(pt);
    }

    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(cloud, msg);
    msg.header = header;
    pub_vio_landmarks_->publish(msg);
  }

  geometry_msgs::msg::PoseArray pose_array;
  nav_msgs::msg::Path window_path;
  pose_array.header = header;
  window_path.header = header;

  for (int i = 0; i <= WINDOW_SIZE; ++i) {
    geometry_msgs::msg::Pose pose;
    pose.position.x = vio_->states_[i].pos_end.x();
    pose.position.y = vio_->states_[i].pos_end.y();
    pose.position.z = vio_->states_[i].pos_end.z();
    const Eigen::Quaterniond q(vio_->states_[i].rot_end);
    pose.orientation.x = q.x();
    pose.orientation.y = q.y();
    pose.orientation.z = q.z();
    pose.orientation.w = q.w();
    pose_array.poses.push_back(pose);

    geometry_msgs::msg::PoseStamped pose_stamped;
    pose_stamped.header = header;
    pose_stamped.header.stamp = secToStamp(vio_->Headers[i]);
    pose_stamped.pose = pose;
    window_path.poses.push_back(pose_stamped);
  }

  if (pub_vio_window_poses_) {
    pub_vio_window_poses_->publish(pose_array);
  }
  if (pub_vio_window_path_) {
    pub_vio_window_path_->publish(window_path);
  }
}

void SlamNode::propagateImuOnce(StatesGroup &state, double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr) const
{
  if (dt <= 0.0) {
    return;
  }
  const Eigen::Vector3d gyr_unbiased = gyr - state.bias_g;
  const Eigen::Vector3d acc_unbiased = acc - state.bias_a;
  state.rot_end = state.rot_end * Exp(gyr_unbiased, dt);
  const Eigen::Vector3d acc_world = state.rot_end * acc_unbiased + state.gravity;
  state.pos_end = state.pos_end + state.vel_end * dt + 0.5 * acc_world * dt * dt;
  state.vel_end = state.vel_end + acc_world * dt;
}

void SlamNode::imuPropagationTimer()
{
  if (!config_.common.imu_propagation_enable || !new_imu_ || !ekf_finish_once_ || latest_ekf_time_ < 0.0) {
    return;
  }

  std::lock_guard<std::mutex> lock(buffer_mutex_);
  new_imu_ = false;
  if (state_update_flag_) {
    imu_propagate_state_ = latest_ekf_state_;
    while (!imu_prop_buffer_.empty() && imu_prop_buffer_.front().stamp < latest_ekf_time_) {
      imu_prop_buffer_.pop_front();
    }
    double last_time = latest_ekf_time_;
    for (const auto &imu : imu_prop_buffer_) {
      propagateImuOnce(imu_propagate_state_, imu.stamp - last_time, imu.acc, imu.gyr);
      last_time = imu.stamp;
    }
    last_imu_prop_time_ = last_time;
    state_update_flag_ = false;
  } else {
    if (last_imu_prop_time_ < latest_ekf_time_) {
      last_imu_prop_time_ = latest_ekf_time_;
    }
    propagateImuOnce(imu_propagate_state_, newest_imu_.stamp - last_imu_prop_time_, newest_imu_.acc, newest_imu_.gyr);
    last_imu_prop_time_ = newest_imu_.stamp;
  }

  nav_msgs::msg::Odometry odom;
  odom.header.stamp = secToStamp(newest_imu_.stamp);
  odom.header.frame_id = config_.frame.world;
  odom.child_frame_id = config_.frame.body;
  odom.pose.pose.position.x = imu_propagate_state_.pos_end.x();
  odom.pose.pose.position.y = imu_propagate_state_.pos_end.y();
  odom.pose.pose.position.z = imu_propagate_state_.pos_end.z();
  const Eigen::Quaterniond q(imu_propagate_state_.rot_end);
  odom.pose.pose.orientation.x = q.x();
  odom.pose.pose.orientation.y = q.y();
  odom.pose.pose.orientation.z = q.z();
  odom.pose.pose.orientation.w = q.w();
  odom.twist.twist.linear.x = imu_propagate_state_.vel_end.x();
  odom.twist.twist.linear.y = imu_propagate_state_.vel_end.y();
  odom.twist.twist.linear.z = imu_propagate_state_.vel_end.z();
  pub_imu_prop_odom_->publish(odom);
}

} // namespace cake_slam

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<cake_slam::SlamNode>());
  rclcpp::shutdown();
  return 0;
}
