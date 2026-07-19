/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef LIV_MAPPER_MULTI_CAM_H
#define LIV_MAPPER_MULTI_CAM_H

#include "IMU_Processing_multi_cam.h"
#include "vio_multi_cam.h"
#include "preprocess_multi_cam.h"
#ifdef PRE_ROS_IRON
#include <cv_bridge/cv_bridge.h>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <image_transport/image_transport.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/CompressedImage.h>
#include <vikit/camera_loader.h>
#include <cstdint>
#include <map>

struct CameraInputConfig
{
  std::string img_topic;
  std::string camera_namespace;
  bool image_undistort_en = false;
  std::string raw_camera_namespace;
  cv::Mat undistort_map_x;
  cv::Mat undistort_map_y;
  std::vector<double> Rcl;
  std::vector<double> Pcl;
  bool online_extrinsic_en = true;
  int time_offset_group = 0;
  bool online_time_offset_en = true;
  std::string camera_model_type = "Pinhole";
  double k1 = 0.0;
  double k2 = 0.0;
  double k3 = 0.0;
  double k4 = 0.0;
  double xi = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
};

struct PendingImageGroup
{
  uint64_t stamp_ns = 0;
  double timestamp = 0.0;
  std::vector<cv::Mat> images;
  std::vector<uint8_t> arrived;
  std::vector<uint64_t> image_stamp_ns;
  std::vector<double> raw_timestamps;
  std::vector<double> corrected_timestamps;
  std::vector<double> capture_timestamps;
  std::vector<double> td_used;
  std::vector<int> time_offset_group;

  bool isComplete() const
  {
    if (arrived.empty()) return false;
    for (uint8_t value : arrived)
      if (!value) return false;
    return true;
  }
};

class LIVMapper
{
public:
  LIVMapper(ros::NodeHandle &nh, std::string node_name);
  ~LIVMapper();
  void initializeSubscribersAndPublishers(ros::NodeHandle &nh, image_transport::ImageTransport &it_);
  void initializeComponents(ros::NodeHandle &nh);
  void initializeFiles();
  void run(ros::NodeHandle &nh);
  void gravityAlignment();
  void handleFirstFrame();
  void stateEstimationAndMapping();
  void handleVIO();
  void handleLIO();
  void savePCD();
  void processImu();
  
  bool sync_packages(LidarMeasureGroup &meas);
  void prop_imu_once(StatesGroup &imu_prop_state, const double dt, V3D acc_avr, V3D angvel_avr);
  void imu_prop_callback();
  void transformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud);
  void pointBodyToWorld(const PointType &pi, PointType &po);
 
  void RGBpointBodyToWorld(PointType const *const pi, PointType *const po);
  void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg_in);
  void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in);
  void img_cbk(int camera_id, const sensor_msgs::Image::ConstPtr &msg_in);
  void compressed_img_cbk(int camera_id, const sensor_msgs::CompressedImage::ConstPtr &msg_in);
  void publish_img_rgb(VIOManagerPtr vio_manager);
  void publish_optical_flow_image(const image_transport::Publisher &pubImage, VIOManagerPtr vio_manager);
  void publish_triangulated_points(const ros::Publisher &pubCloud,
                                   const PointCloudXYZI::Ptr &cloud);
  void publish_frame_world(const ros::Publisher &pubLaserCloudFullRes, VIOManagerPtr vio_manager);
  void publish_visual_sub_map(const ros::Publisher &pubSubVisualMap);
  void publish_effect_world(const ros::Publisher &pubLaserCloudEffect, const std::vector<PointToPlane> &ptpl_list);
  void publish_odometry(const ros::Publisher &pmavros_pose_publisherubOdomAftMapped);
  void publish_mavros(const ros::Publisher &mavros_pose_publisher);
  void publish_path(const ros::Publisher &pubPath);
  void readParameters(ros::NodeHandle &nh);
  template <typename T> void set_posestamp(T &out);
  template <typename T> void pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi, Eigen::Matrix<T, 3, 1> &po);
  template <typename T> Eigen::Matrix<T, 3, 1> pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi);
  cv::Mat getImageFromMsg(const sensor_msgs::Image::ConstPtr &img_msg);
  void handleImageFrame(int camera_id, const ros::Time &stamp, const cv::Mat &img_cur);
  void flushCompletedImageGroupsLocked();

  std::mutex mtx_buffer, mtx_buffer_imu_prop;
  std::condition_variable sig_buffer;

  SLAM_MODE slam_mode_;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map;
  
  string root_dir;
  string lid_topic, imu_topic, seq_name;
  int num_cameras = 1;
  bool require_all_cameras = true;
  int multi_cam_sync_queue_size = 5;
  double multi_cam_sync_tolerance_ms = 0.0;
  std::vector<CameraInputConfig> camera_configs;
  V3D extT;
  M3D extR;

  int feats_down_size = 0, max_iterations = 0;

  double res_mean_last = 0.05;
  double gyr_cov = 0, acc_cov = 0, inv_expo_cov = 0;
  double blind_rgb_points = 0.0;
  double last_timestamp_lidar = -1.0, last_timestamp_imu = -1.0;
  std::vector<double> last_timestamp_img_by_camera;
  uint64_t multi_cam_downclock_counter = 0;
  double filter_size_surf_min = 0;
  double filter_size_pcd = 0;
  double _first_lidar_time = 0.0;
  double match_time = 0, solve_time = 0, solve_const_H_time = 0;

  bool lidar_map_inited = false, pcd_save_en = false, pub_effect_point_en = false, pose_output_en = false, ros_driver_fix_en = false;
  int pcd_save_interval = -1, pcd_index = 0;
  int pub_scan_num = 1;

  StatesGroup imu_propagate, latest_ekf_state;

  bool new_imu = false, state_update_flg = false, imu_prop_enable = true, ekf_finish_once = false;
  deque<sensor_msgs::Imu> prop_imu_buffer;
  sensor_msgs::Imu newest_imu;
  double latest_ekf_time;
  nav_msgs::Odometry imu_prop_odom;
  ros::Publisher pubImuPropOdom;
  double imu_time_offset = 0.0;

  bool gravity_align_en = false, gravity_align_finished = false;

  bool sync_jump_flag = false;

  bool lidar_pushed = false, imu_en, gravity_est_en, flg_reset = false, ba_bg_est_en = true;
  bool dense_map_en = false;
  int img_en = 1, imu_int_frame = 3;
  bool normal_en = true;
  bool exposure_estimate_en = false;
  double inv_expo_time_init = 1.0;
  double exposure_time_init = 0.0;
  bool inverse_composition_en = false;
  bool raycast_en = false;
  bool ncc_en = false;
  double ncc_thre = 0.8;
  std::vector<double> ncc_thre_by_level;
  bool usage_stats_en = false;
  int usage_stats_window = 100;
  bool directional_update_en = false;
  double directional_drop_variance_reduction = 0.05;
  double directional_full_variance_reduction = 0.50;
  int lidar_en = 1;
  bool is_first_frame = false;
  int grid_size, patch_size, grid_n_width, grid_n_height, patch_pyrimid_level;
  bool virtual_fisheye_patch_en = false;
  bool virtual_sparse_patch_en = false;
  bool virtual_s2_optimize_en = false;
  bool visual_ref_post_ekf_build_en = false;
  bool visual_map_manage_en = false;
  bool visual_map_manage_log_en = true;
  bool visual_map_manage_shadow_en = false;
  bool visual_ref_current_select_en = true;
  bool visual_ref_fallback_en = true;
  bool visual_ref_lifecycle_en = true;
  bool visual_ref_view_coverage_en = true;
  bool visual_ref_nis_en = true;
  bool visual_point_seed_validation_en = true;
  bool visual_point_footprint_redundancy_en = true;
  bool visual_point_information_prune_en = true;
  bool visual_point_replacement_en = true;
  bool visual_map_retirement_apply_en = true;
  bool raw_camera_model_jacobian_en = false;
  bool cross_camera_reference_en = false;
  bool cross_camera_current_residual_en = false;
  int visual_ref_max_candidates = 2;
  int visual_ref_validate_min_tests = 3;
  double visual_ref_validate_min_ratio = 0.67;
  int visual_ref_retire_reject_count = 5;
  int visual_ref_max_count = 8;
  double visual_ref_coverage_angle_deg = 12.0;
  double visual_ref_max_anisotropy = 4.0;
  double visual_ref_nis_max_per_dof = 3.0;
  int visual_point_seed_min_tests = 3;
  double visual_point_seed_min_ratio = 0.67;
  double visual_point_footprint_iou = 0.60;
  double visual_point_information_retain = 0.90;
  int visual_point_suspect_reject_count = 8;
  int visual_point_stale_frames = 200;
  bool online_extrinsic_en = false;
  bool online_extrinsic_rot_en = true;
  bool online_extrinsic_trans_en = true;
  bool online_extrinsic_prior_factor_en = false;
  std::vector<int64_t> online_extrinsic_camera_mask;
  int online_extrinsic_start_frame = 100;
  int online_extrinsic_min_tracks = 20;
  double online_extrinsic_prior_rot_std_deg = 0.5;
  double online_extrinsic_prior_trans_std_m = 0.02;
  double online_extrinsic_max_rot_update_deg = 0.02;
  double online_extrinsic_max_trans_update_m = 0.0001;
  bool online_time_offset_en = false;
  int num_time_offset_groups = 1;
  std::vector<int64_t> online_time_offset_group_mask;
  int online_time_offset_start_frame = 100;
  int online_time_offset_min_tracks = 30;
  double online_time_offset_min_pixel_velocity = 15.0;
  double online_time_offset_prior_std_ms = 5.0;
  double online_time_offset_process_noise_ms_sqrt_s = 0.0;
  double online_time_offset_max_update_ms = 0.2;
  double online_time_offset_max_abs_ms = 50.0;
  int online_time_offset_min_update_interval = 5;
  double virtual_focal_length = 300.0;
  int virtual_patch_margin = 4;
  int virtual_max_search_level = 1;
  std::string virtual_patch_resampling_mode = "forward_splat";
  std::string virtual_interp_mode = "bilinear";
  int virtual_raw_window_half_size = 48;
  double virtual_splat_min_weight = 1.0e-6;
  bool virtual_splat_require_full_core_coverage = true;
  bool virtual_splat_debug_compare_pull_exact = false;
  bool visual_geom_filter_en = false;
  bool visual_geom_filter_log_en = false;
  bool visual_geom_filter_require_point_cov = true;
  double visual_geom_filter_voxel_size = 0.5;
  int visual_geom_filter_min_plane_points = 5;
  double visual_geom_filter_radius_multiplier = 3.0;
  double visual_geom_filter_min_normal_cos = 0.8660254037844386;
  double visual_geom_filter_max_chi2 = 9.0;
  double visual_geom_filter_min_sigma = 1.0e-12;
  double visual_geom_filter_max_point_cov_trace = -1.0;
  double visual_geom_filter_max_normal_cov = -1.0;
  bool draw_rejected_points_en = false;
  bool ref_patch_dump_en = false;
  int ref_patch_dump_random_seed = -1;
  int ref_patch_dump_max_candidate_skip = 50;
  double ref_patch_dump_ncc_threshold = 0.6;
  bool runtime_support_dump_en = false;
  std::string runtime_support_dump_folder = "runtime_support";
  int frontend_mode = 0;
  int optical_flow_max_cnt = 250;
  int optical_flow_min_dist = 20;
  int optical_flow_min_track_len_for_triangulation = 3;
  int optical_flow_track_history_size = 20;
  double optical_flow_quality_level = 0.01;
  double optical_flow_f_threshold = 0.5;
  bool optical_flow_flow_back = true;
  std::string optical_flow_feature_image_topic = "/fast_livo/feature_image";
  std::string optical_flow_triangulated_points_topic = "/fast_livo/triangulated_points";
  int outlier_threshold;
  double plot_time;
  int frame_cnt;
  double img_time_offset = 0.0;
  std::vector<double> img_time_offset_groups;
  deque<PointCloudXYZI::Ptr> lid_raw_data_buffer;
  deque<double> lid_header_time_buffer;
  deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
  std::map<uint64_t, PendingImageGroup> pending_images;
  deque<MultiCameraFrame> multi_cam_frame_buffer;
  int next_vio_frame_id = 0;
  vector<pointWithVar> _pv_list;
  vector<double> extrinT;
  vector<double> extrinR;
  int IMG_POINT_COV;

  PointCloudXYZI::Ptr visual_sub_map;
  PointCloudXYZI::Ptr feats_undistort;
  PointCloudXYZI::Ptr feats_down_body;
  PointCloudXYZI::Ptr feats_down_world;
  PointCloudXYZI::Ptr pcl_w_wait_pub;
  PointCloudXYZI::Ptr pcl_wait_pub;
  PointCloudXYZRGB::Ptr pcl_wait_save;
  PointCloudXYZI::Ptr pcl_wait_save_intensity;

  ofstream fout_pre, fout_out, fout_pcd_pos, fout_points;

  pcl::VoxelGrid<PointType> downSizeFilterSurf;

  V3D euler_cur;

  LidarMeasureGroup LidarMeasures;
  StatesGroup _state;
  StatesGroup  state_propagat;

  nav_msgs::Path path;
  nav_msgs::Odometry odomAftMapped;
  geometry_msgs::Quaternion geoQuat;
  geometry_msgs::PoseStamped msg_body_pose;

  PreprocessPtr p_pre;
  ImuProcessPtr p_imu;
  VoxelMapManagerPtr voxelmap_manager;
  VIOManagerPtr vio_manager;

  ros::Publisher plane_pub;
  ros::Publisher voxel_pub;
  ros::Subscriber sub_pcl;
  ros::Subscriber sub_imu;
  std::vector<ros::Subscriber> sub_imgs;
  std::vector<ros::Subscriber> sub_imgs_compressed;
  ros::Publisher pubLaserCloudFullRes;
  ros::Publisher pubNormal;
  ros::Publisher pubSubVisualMap;
  ros::Publisher pubLaserCloudEffect;
  ros::Publisher pubLaserCloudMap;
  ros::Publisher pubOdomAftMapped;
  ros::Publisher pubPath;
  ros::Publisher pubLaserCloudDyn;
  ros::Publisher pubLaserCloudDynRmed;
  ros::Publisher pubLaserCloudDynDbg;
  std::vector<image_transport::Publisher> pubImages;
  image_transport::Publisher pubOpticalFlowImage;
  ros::Publisher pubTriangulatedPoints;
  ros::Publisher mavros_pose_publisher;
  ros::Timer imu_prop_timer;
  ros::NodeHandle node;

  int frame_num = 0;
  double aver_time_consu = 0;
  double aver_time_icp = 0;
  double aver_time_map_inre = 0;
  bool colmap_output_en = false;
  bool image_downclock_en = false;
  int image_downclock_factor = 1;
};
#endif
