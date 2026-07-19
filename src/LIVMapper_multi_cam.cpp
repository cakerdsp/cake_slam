/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "LIVMapper_multi_cam.h"
#include "utils/ros1_param.h"
#include <cmath>
#include <filesystem>
#include <limits>
#include <vikit/abstract_camera.h>
#include <vikit/camera_loader.h>

using namespace Sophus;

namespace
{
constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
void suppressRosInfoLogs(const ros::NodeHandle &nh)
{
  (void)nh;
}


bool hasLaterCompleteImageGroupLocked(const std::map<uint64_t, PendingImageGroup> &pending_images,
                                      uint64_t stamp_ns)
{
  for (auto it = pending_images.upper_bound(stamp_ns); it != pending_images.end(); ++it)
  {
    if (it->second.isComplete()) return true;
  }
  return false;
}

std::string formatMissingCameraIdsLocked(const PendingImageGroup &group)
{
  std::string missing_camera_ids;
  for (size_t camera_id = 0; camera_id < group.arrived.size(); ++camera_id)
  {
    if (group.arrived[camera_id]) continue;
    if (!missing_camera_ids.empty()) missing_camera_ids += ",";
    missing_camera_ids += std::to_string(camera_id);
  }
  return missing_camera_ids.empty() ? "unknown" : missing_camera_ids;
}

bool buildImageUndistortMaps(vk::AbstractCamera &raw_camera,
                             vk::AbstractCamera &undistorted_camera,
                             cv::Mat &map_x, cv::Mat &map_y,
                             std::string &error_message)
{
  const int width = undistorted_camera.width();
  const int height = undistorted_camera.height();
  if (width <= 0 || height <= 0)
  {
    error_message = "undistorted camera width/height must be positive";
    return false;
  }

  map_x.create(height, width, CV_32FC1);
  map_y.create(height, width, CV_32FC1);
  int valid_samples = 0;
  constexpr double kMinRayNorm = 1.0e-12;
  for (int y = 0; y < height; ++y)
  {
    float *map_x_row = map_x.ptr<float>(y);
    float *map_y_row = map_y.ptr<float>(y);
    for (int x = 0; x < width; ++x)
    {
      const V3D ray = undistorted_camera.cam2world(static_cast<double>(x), static_cast<double>(y));
      const double ray_norm = ray.norm();
      if (!ray.array().isFinite().all() || !std::isfinite(ray_norm) || ray_norm <= kMinRayNorm)
      {
        map_x_row[x] = -1.0f;
        map_y_row[x] = -1.0f;
        continue;
      }

      const V2D raw_px = raw_camera.world2cam(ray);
      if (!raw_px.array().isFinite().all())
      {
        map_x_row[x] = -1.0f;
        map_y_row[x] = -1.0f;
        continue;
      }

      map_x_row[x] = static_cast<float>(raw_px[0]);
      map_y_row[x] = static_cast<float>(raw_px[1]);
      ++valid_samples;
    }
  }

  if (valid_samples == 0)
  {
    error_message = "undistort remap has no valid samples";
    return false;
  }
  return true;
}
} // namespace

LIVMapper::LIVMapper(ros::NodeHandle &nh, std::string node_name)
    : node(nh),
      extT(0, 0, 0),
      extR(M3D::Identity())
{
  (void)node_name;
  suppressRosInfoLogs(this->node);
  extrinT.assign(3, 0.0);
  extrinR.assign(9, 0.0);

  p_pre.reset(new Preprocess());
  p_imu.reset(new ImuProcess());

  readParameters(this->node);
  _state.configureCameras(num_cameras, inv_expo_time_init, num_time_offset_groups, img_time_offset);
  state_propagat.configureCameras(num_cameras, inv_expo_time_init, num_time_offset_groups, img_time_offset);
  imu_propagate.configureCameras(num_cameras, inv_expo_time_init, num_time_offset_groups, img_time_offset);
  latest_ekf_state.configureCameras(num_cameras, inv_expo_time_init, num_time_offset_groups, img_time_offset);
  auto apply_time_offset_init = [this](StatesGroup &state_value) {
    for (int group_id = 0; group_id < state_value.num_time_offset_groups; ++group_id)
    {
      const double td_init = group_id < static_cast<int>(img_time_offset_groups.size())
                                 ? img_time_offset_groups[group_id]
                                 : img_time_offset;
      state_value.setTimeOffset(group_id, td_init);
    }
  };
  apply_time_offset_init(_state);
  apply_time_offset_init(state_propagat);
  apply_time_offset_init(imu_propagate);
  apply_time_offset_init(latest_ekf_state);
  VoxelMapConfig voxel_config;
  loadVoxelConfig(this->node, voxel_config);

  visual_sub_map.reset(new PointCloudXYZI());
  feats_undistort.reset(new PointCloudXYZI());
  feats_down_body.reset(new PointCloudXYZI());
  feats_down_world.reset(new PointCloudXYZI());
  pcl_w_wait_pub.reset(new PointCloudXYZI());
  pcl_wait_pub.reset(new PointCloudXYZI());
  pcl_wait_save.reset(new PointCloudXYZRGB());
  pcl_wait_save_intensity.reset(new PointCloudXYZI());
  voxelmap_manager.reset(new VoxelMapManager(voxel_config, voxel_map));
  vio_manager.reset(new VIOManager());
  root_dir = ROOT_DIR;
  initializeFiles();
  initializeComponents(this->node);          // initialize components errors
  path.header.stamp = ros::Time::now();
  path.header.frame_id = "camera_init";
}

LIVMapper::~LIVMapper() {}

void LIVMapper::readParameters(ros::NodeHandle &nh)
{
  // declare parameters
  auto try_declare = [&nh]<typename ParameterT>(const std::string & name,
    const ParameterT & default_value)
  {
    return declareOrGetRosParam(nh, name, default_value);
  };

  // declare parameter
  try_declare.template operator()<std::string>("common.lid_topic", "/livox/lidar");
  try_declare.template operator()<std::string>("common.imu_topic", "/livox/imu");
  try_declare.template operator()<bool>("common.ros_driver_bug_fix", false);
  try_declare.template operator()<int>("common.img_en", 1);
  try_declare.template operator()<int>("common.lidar_en", 1);
  try_declare.template operator()<int>("common.num_cameras", 1);
  try_declare.template operator()<bool>("common.require_all_cameras", true);
  try_declare.template operator()<int>("common.multi_cam_sync_queue_size", 5);
  try_declare.template operator()<double>("common.multi_cam_sync_tolerance_ms", 0.0);
  try_declare.template operator()<bool>("common.directional_update_en", false);
  try_declare.template operator()<double>("common.directional_drop_variance_reduction", 0.05);
  try_declare.template operator()<double>("common.directional_full_variance_reduction", 0.50);

  try_declare.template operator()<bool>("vio.normal_en", true);
  try_declare.template operator()<bool>("vio.inverse_composition_en", false);
  try_declare.template operator()<int>("vio.max_iterations", 5);
  try_declare.template operator()<int>("vio.img_point_cov", 100);
  try_declare.template operator()<bool>("vio.raycast_en", false);
  try_declare.template operator()<bool>("vio.exposure_estimate_en", true);
  try_declare.template operator()<double>("vio.inv_expo_time_init", 1.0);
  try_declare.template operator()<double>("vio.inv_expo_cov", 0.1);
  try_declare.template operator()<int>("vio.grid_size", 5);
  try_declare.template operator()<int>("vio.grid_n_height", 17);
  try_declare.template operator()<int>("vio.patch_pyrimid_level", 3);
  try_declare.template operator()<int>("vio.patch_size", 8);
  try_declare.template operator()<bool>("vio.virtual_fisheye_patch_en", false);
  try_declare.template operator()<bool>("vio.virtual_sparse_patch_en", false);
  try_declare.template operator()<bool>("vio.virtual_s2_optimize_en", false);
  try_declare.template operator()<bool>("vio.visual_ref_post_ekf_build_en", false);
  try_declare.template operator()<bool>("vio.visual_map_manage_en", false);
  try_declare.template operator()<bool>("vio.visual_map_manage_log_en", true);
  try_declare.template operator()<bool>("vio.visual_map_manage_shadow_en", false);
  try_declare.template operator()<bool>("vio.visual_ref_current_select_en", true);
  try_declare.template operator()<bool>("vio.visual_ref_fallback_en", true);
  try_declare.template operator()<bool>("vio.visual_ref_lifecycle_en", true);
  try_declare.template operator()<bool>("vio.visual_ref_view_coverage_en", true);
  try_declare.template operator()<bool>("vio.visual_ref_nis_en", true);
  try_declare.template operator()<bool>("vio.visual_point_seed_validation_en", true);
  try_declare.template operator()<bool>("vio.visual_point_footprint_redundancy_en", true);
  try_declare.template operator()<bool>("vio.visual_point_information_prune_en", true);
  try_declare.template operator()<bool>("vio.visual_point_replacement_en", true);
  try_declare.template operator()<bool>("vio.visual_map_retirement_apply_en", true);
  try_declare.template operator()<int>("vio.visual_ref_max_candidates", 2);
  try_declare.template operator()<int>("vio.visual_ref_validate_min_tests", 3);
  try_declare.template operator()<double>("vio.visual_ref_validate_min_ratio", 0.67);
  try_declare.template operator()<int>("vio.visual_ref_retire_reject_count", 5);
  try_declare.template operator()<int>("vio.visual_ref_max_count", 8);
  try_declare.template operator()<double>("vio.visual_ref_coverage_angle_deg", 12.0);
  try_declare.template operator()<double>("vio.visual_ref_max_anisotropy", 4.0);
  try_declare.template operator()<double>("vio.visual_ref_nis_max_per_dof", 3.0);
  try_declare.template operator()<int>("vio.visual_point_seed_min_tests", 3);
  try_declare.template operator()<double>("vio.visual_point_seed_min_ratio", 0.67);
  try_declare.template operator()<double>("vio.visual_point_footprint_iou", 0.60);
  try_declare.template operator()<double>("vio.visual_point_information_retain", 0.90);
  try_declare.template operator()<int>("vio.visual_point_suspect_reject_count", 8);
  try_declare.template operator()<int>("vio.visual_point_stale_frames", 200);
  try_declare.template operator()<bool>("vio.raw_camera_model_jacobian_en", false);
  try_declare.template operator()<double>("vio.virtual_focal_length", 300.0);
  try_declare.template operator()<int>("vio.virtual_patch_margin", 4);
  try_declare.template operator()<int>("vio.virtual_max_search_level", 1);
  try_declare.template operator()<std::string>("vio.virtual_patch_resampling_mode", "forward_splat");
  try_declare.template operator()<std::string>("vio.virtual_interp_mode", "bilinear");
  try_declare.template operator()<int>("vio.virtual_raw_window_half_size", 48);
  try_declare.template operator()<double>("vio.virtual_splat_min_weight", 1.0e-6);
  try_declare.template operator()<bool>("vio.virtual_splat_require_full_core_coverage", true);
  try_declare.template operator()<bool>("vio.virtual_splat_debug_compare_pull_exact", false);
  try_declare.template operator()<bool>("vio.visual_geom_filter_en", false);
  try_declare.template operator()<bool>("vio.visual_geom_filter_log_en", false);
  try_declare.template operator()<bool>("vio.visual_geom_filter_require_point_cov", true);
  try_declare.template operator()<double>("vio.visual_geom_filter_voxel_size", 0.5);
  try_declare.template operator()<int>("vio.visual_geom_filter_min_plane_points", 5);
  try_declare.template operator()<double>("vio.visual_geom_filter_radius_multiplier", 3.0);
  try_declare.template operator()<double>("vio.visual_geom_filter_min_normal_cos", 0.8660254037844386);
  try_declare.template operator()<double>("vio.visual_geom_filter_max_chi2", 9.0);
  try_declare.template operator()<double>("vio.visual_geom_filter_min_sigma", 1.0e-12);
  try_declare.template operator()<double>("vio.visual_geom_filter_max_point_cov_trace", -1.0);
  try_declare.template operator()<double>("vio.visual_geom_filter_max_normal_cov", -1.0);
  try_declare.template operator()<bool>("vio.draw_rejected_points_en", false);
  try_declare.template operator()<bool>("vio.ncc_en", false);
  try_declare.template operator()<double>("vio.ncc_thre", 0.8);
  try_declare.template operator()<std::vector<double>>("vio.ncc_thre_by_level", std::vector<double>{});
  try_declare.template operator()<bool>("vio.usage_stats_en", false);
  try_declare.template operator()<int>("vio.usage_stats_window", 100);
  try_declare.template operator()<bool>("vio.cross_camera_reference_en", false);
  try_declare.template operator()<bool>("vio.cross_camera_current_residual_en", false);
  try_declare.template operator()<bool>("vio.online_extrinsic_en", false);
  try_declare.template operator()<bool>("vio.online_extrinsic_rot_en", true);
  try_declare.template operator()<bool>("vio.online_extrinsic_trans_en", true);
  try_declare.template operator()<bool>("vio.online_extrinsic_prior_factor_en", false);
  try_declare.template operator()<std::vector<int>>("vio.online_extrinsic_camera_mask", std::vector<int>{});
  try_declare.template operator()<int>("vio.online_extrinsic_start_frame", 100);
  try_declare.template operator()<int>("vio.online_extrinsic_min_tracks", 20);
  try_declare.template operator()<double>("vio.online_extrinsic_prior_rot_std_deg", 0.5);
  try_declare.template operator()<double>("vio.online_extrinsic_prior_trans_std_m", 0.02);
  try_declare.template operator()<double>("vio.online_extrinsic_max_rot_update_deg", 0.02);
  try_declare.template operator()<double>("vio.online_extrinsic_max_trans_update_m", 0.0001);
  try_declare.template operator()<bool>("vio.online_time_offset_en", false);
  try_declare.template operator()<std::vector<int>>("vio.online_time_offset_group_mask", std::vector<int>{});
  try_declare.template operator()<int>("vio.online_time_offset_start_frame", 100);
  try_declare.template operator()<int>("vio.online_time_offset_min_tracks", 30);
  try_declare.template operator()<double>("vio.online_time_offset_min_pixel_velocity", 15.0);
  try_declare.template operator()<double>("vio.online_time_offset_prior_std_ms", 5.0);
  try_declare.template operator()<double>("vio.online_time_offset_process_noise_ms_sqrt_s", 0.0);
  try_declare.template operator()<double>("vio.online_time_offset_max_update_ms", 0.2);
  try_declare.template operator()<double>("vio.online_time_offset_max_abs_ms", 50.0);
  try_declare.template operator()<int>("vio.online_time_offset_min_update_interval", 5);
  try_declare.template operator()<int>("vio.outlier_threshold", 100);
  try_declare.template operator()<int>("vio.frontend_mode", 0);
  try_declare.template operator()<int>("vio.opticalflow.max_cnt", 250);
  try_declare.template operator()<int>("vio.opticalflow.min_dist", 20);
  try_declare.template operator()<int>("vio.opticalflow.min_track_len_for_triangulation", 3);
  try_declare.template operator()<int>("vio.opticalflow.track_history_size", 20);
  try_declare.template operator()<double>("vio.opticalflow.quality_level", 0.01);
  try_declare.template operator()<double>("vio.opticalflow.f_threshold", 0.5);
  try_declare.template operator()<bool>("vio.opticalflow.flow_back", true);
  try_declare.template operator()<std::string>("vio.opticalflow.feature_image_topic", "/fast_livo/feature_image");
  try_declare.template operator()<std::string>("vio.opticalflow.triangulated_points_topic", "/fast_livo/triangulated_points");
  try_declare.template operator()<double>("time_offset.exposure_time_init", 0.0);
  try_declare.template operator()<double>("time_offset.img_time_offset", 0.0);
  try_declare.template operator()<std::vector<double>>("time_offset.img_time_offset_groups", std::vector<double>{});
  try_declare.template operator()<bool>("uav.imu_rate_odom", false);
  try_declare.template operator()<bool>("uav.gravity_align_en", false);

  try_declare.template operator()<std::string>("evo.seq_name", "01");
  try_declare.template operator()<bool>("evo.pose_output_en", false);
  try_declare.template operator()<double>("imu.gyr_cov", 1.0);
  try_declare.template operator()<double>("imu.acc_cov", 1.0);
  try_declare.template operator()<int>("imu.imu_int_frame", 30);
  try_declare.template operator()<bool>("imu.imu_en", true);
  try_declare.template operator()<bool>("imu.gravity_est_en", true);
  try_declare.template operator()<bool>("imu.ba_bg_est_en", true);

  try_declare.template operator()<double>("preprocess.blind", 0.01);
  try_declare.template operator()<double>("preprocess.filter_size_surf", 0.5);
  try_declare.template operator()<int>("preprocess.lidar_type", AVIA);
  try_declare.template operator()<int>("preprocess.scan_line",6);
  try_declare.template operator()<int>("preprocess.point_filter_num", 3);
  try_declare.template operator()<int>("preprocess.scan_rate", 10);
  try_declare.template operator()<bool>("preprocess.feature_extract_enabled", false);
  try_declare.template operator()<bool>("preprocess.image_downclock_en", false);
  try_declare.template operator()<int>("preprocess.image_downclock_factor", 1);

  try_declare.template operator()<int>("pcd_save.interval", -1);
  try_declare.template operator()<bool>("pcd_save.pcd_save_en", false);
  try_declare.template operator()<bool>("pcd_save.colmap_output_en", false);
  try_declare.template operator()<double>("pcd_save.filter_size_pcd", 0.5);
  try_declare.template operator()<vector<double>>("extrin_calib.extrinsic_T", vector<double>{});
  try_declare.template operator()<vector<double>>("extrin_calib.extrinsic_R", vector<double>{});
  try_declare.template operator()<double>("debug.plot_time", -10);
  try_declare.template operator()<int>("debug.frame_cnt", 6);
  try_declare.template operator()<bool>("debug.ref_patch_dump_en", false);
  try_declare.template operator()<int>("debug.ref_patch_dump_random_seed", -1);
  try_declare.template operator()<int>("debug.ref_patch_dump_max_candidate_skip", 50);
  try_declare.template operator()<double>("debug.ref_patch_dump_ncc_threshold", 0.6);
  try_declare.template operator()<bool>("debug.runtime_support_dump_en", false);
  try_declare.template operator()<std::string>("debug.runtime_support_dump_folder", "runtime_support");

  try_declare.template operator()<double>("publish.blind_rgb_points", 0.01);
  try_declare.template operator()<int>("publish.pub_scan_num", 1);
  try_declare.template operator()<bool>("publish.pub_effect_point_en", false);
  try_declare.template operator()<bool>("publish.dense_map_en", false);

  // get parameter
  getRosParam(this->node, "common.lid_topic", lid_topic);
  getRosParam(this->node, "common.imu_topic", imu_topic);
  getRosParam(this->node, "common.ros_driver_bug_fix", ros_driver_fix_en);
  getRosParam(this->node, "common.img_en", img_en);
  getRosParam(this->node, "common.lidar_en", lidar_en);
  getRosParam(this->node, "common.num_cameras", num_cameras);
  getRosParam(this->node, "common.require_all_cameras", require_all_cameras);
  getRosParam(this->node, "common.multi_cam_sync_queue_size", multi_cam_sync_queue_size);
  getRosParam(this->node, "common.multi_cam_sync_tolerance_ms", multi_cam_sync_tolerance_ms);
  getRosParam(this->node, "common.directional_update_en", directional_update_en);
  getRosParam(this->node, "common.directional_drop_variance_reduction", directional_drop_variance_reduction);
  getRosParam(this->node, "common.directional_full_variance_reduction", directional_full_variance_reduction);
  if (num_cameras < 1) throw std::runtime_error("common.num_cameras must be at least 1");
  if (!require_all_cameras) throw std::runtime_error("partial camera frames are not supported; common.require_all_cameras must be true");
  if (multi_cam_sync_queue_size < 1) throw std::runtime_error("common.multi_cam_sync_queue_size must be at least 1");
  if (!std::isfinite(multi_cam_sync_tolerance_ms) || multi_cam_sync_tolerance_ms < 0.0)
    throw std::runtime_error("common.multi_cam_sync_tolerance_ms must be finite and non-negative");
  if (directional_update_en &&
      (!std::isfinite(directional_drop_variance_reduction) ||
       !std::isfinite(directional_full_variance_reduction) ||
       directional_drop_variance_reduction < 0.0 ||
       directional_full_variance_reduction > 1.0 ||
       directional_full_variance_reduction <= directional_drop_variance_reduction))
    throw std::runtime_error(
        "directional thresholds must satisfy 0 <= common.directional_drop_variance_reduction "
        "< common.directional_full_variance_reduction <= 1");

  camera_configs.resize(num_cameras);
  last_timestamp_img_by_camera.assign(num_cameras, -1.0);
  for (int camera_id = 0; camera_id < num_cameras; ++camera_id)
  {
    const std::string camera_cfg_ns = "cameras.camera" + std::to_string(camera_id);
    const std::string topic_key = camera_cfg_ns + ".img_topic";
    const std::string namespace_key = camera_cfg_ns + ".camera_ns";
    const std::string image_undistort_key = camera_cfg_ns + ".image_undistort_en";
    const std::string raw_namespace_key = camera_cfg_ns + ".raw_camera_ns";
    const std::string rotation_key = camera_cfg_ns + ".Rcl";
    const std::string translation_key = camera_cfg_ns + ".Pcl";
    const std::string online_extrinsic_key = camera_cfg_ns + ".online_extrinsic_en";
    const std::string time_offset_group_key = camera_cfg_ns + ".time_offset_group";
    const std::string online_time_offset_key = camera_cfg_ns + ".online_time_offset_en";
    try_declare.template operator()<std::string>(topic_key, "");
    try_declare.template operator()<std::string>(namespace_key, "");
    try_declare.template operator()<bool>(image_undistort_key, false);
    try_declare.template operator()<std::string>(raw_namespace_key, "");
    try_declare.template operator()<vector<double>>(rotation_key, vector<double>{});
    try_declare.template operator()<vector<double>>(translation_key, vector<double>{});
    try_declare.template operator()<bool>(online_extrinsic_key, true);
    try_declare.template operator()<int>(time_offset_group_key, camera_id);
    try_declare.template operator()<bool>(online_time_offset_key, true);
    getRosParam(nh, topic_key, camera_configs[camera_id].img_topic);
    getRosParam(nh, namespace_key, camera_configs[camera_id].camera_namespace);
    getRosParam(nh, image_undistort_key, camera_configs[camera_id].image_undistort_en);
    getRosParam(nh, raw_namespace_key, camera_configs[camera_id].raw_camera_namespace);
    getRosParam(nh, rotation_key, camera_configs[camera_id].Rcl);
    getRosParam(nh, translation_key, camera_configs[camera_id].Pcl);
    getRosParam(nh, online_extrinsic_key, camera_configs[camera_id].online_extrinsic_en);
    getRosParam(nh, time_offset_group_key, camera_configs[camera_id].time_offset_group);
    getRosParam(nh, online_time_offset_key, camera_configs[camera_id].online_time_offset_en);
    if (camera_configs[camera_id].img_topic.empty())
      throw std::runtime_error("missing required parameter " + topic_key);
    if (camera_configs[camera_id].camera_namespace.empty())
      throw std::runtime_error("missing required parameter " + namespace_key);
    if (camera_configs[camera_id].image_undistort_en && camera_configs[camera_id].raw_camera_namespace.empty())
      throw std::runtime_error("missing required parameter " + raw_namespace_key + " when " + image_undistort_key + " is true");
    if (camera_configs[camera_id].Rcl.size() != 9)
      throw std::runtime_error(rotation_key + " must contain exactly 9 values");
    if (camera_configs[camera_id].Pcl.size() != 3)
      throw std::runtime_error(translation_key + " must contain exactly 3 values");
    if (camera_configs[camera_id].time_offset_group < 0)
      throw std::runtime_error(time_offset_group_key + " must be non-negative");
    num_time_offset_groups = std::max(num_time_offset_groups, camera_configs[camera_id].time_offset_group + 1);

    const std::string camera_param_ns = camera_configs[camera_id].camera_namespace;
    try_declare.template operator()<std::string>(camera_param_ns + ".model", "Pinhole");
    try_declare.template operator()<double>(camera_param_ns + ".k1", 0.0);
    try_declare.template operator()<double>(camera_param_ns + ".k2", 0.0);
    try_declare.template operator()<double>(camera_param_ns + ".k3", 0.0);
    try_declare.template operator()<double>(camera_param_ns + ".k4", 0.0);
    try_declare.template operator()<double>(camera_param_ns + ".xi", 0.0);
    try_declare.template operator()<double>(camera_param_ns + ".p1", 0.0);
    try_declare.template operator()<double>(camera_param_ns + ".p2", 0.0);
    getRosParam(nh, camera_param_ns + ".model", camera_configs[camera_id].camera_model_type);
    getRosParam(nh, camera_param_ns + ".k1", camera_configs[camera_id].k1);
    getRosParam(nh, camera_param_ns + ".k2", camera_configs[camera_id].k2);
    getRosParam(nh, camera_param_ns + ".k3", camera_configs[camera_id].k3);
    getRosParam(nh, camera_param_ns + ".k4", camera_configs[camera_id].k4);
    getRosParam(nh, camera_param_ns + ".xi", camera_configs[camera_id].xi);
    getRosParam(nh, camera_param_ns + ".p1", camera_configs[camera_id].p1);
    getRosParam(nh, camera_param_ns + ".p2", camera_configs[camera_id].p2);
  }

  getRosParam(this->node, "vio.normal_en", normal_en);
  getRosParam(this->node, "vio.inverse_composition_en", inverse_composition_en);
  getRosParam(this->node, "vio.max_iterations", max_iterations);
  getRosParam(this->node, "vio.img_point_cov", IMG_POINT_COV);
  getRosParam(this->node, "vio.raycast_en", raycast_en);
  getRosParam(this->node, "vio.exposure_estimate_en", exposure_estimate_en);
  getRosParam(this->node, "vio.inv_expo_time_init", inv_expo_time_init);
  getRosParam(this->node, "vio.inv_expo_cov", inv_expo_cov);
  getRosParam(this->node, "vio.grid_size", grid_size);
  getRosParam(this->node, "vio.grid_n_height", grid_n_height);
  getRosParam(this->node, "vio.patch_pyrimid_level", patch_pyrimid_level);
  getRosParam(this->node, "vio.patch_size", patch_size);
  getRosParam(this->node, "vio.virtual_fisheye_patch_en", virtual_fisheye_patch_en);
  getRosParam(this->node, "vio.virtual_sparse_patch_en", virtual_sparse_patch_en);
  getRosParam(this->node, "vio.virtual_s2_optimize_en", virtual_s2_optimize_en);
  getRosParam(this->node, "vio.visual_ref_post_ekf_build_en", visual_ref_post_ekf_build_en);
  getRosParam(this->node, "vio.visual_map_manage_en", visual_map_manage_en);
  getRosParam(this->node, "vio.visual_map_manage_log_en", visual_map_manage_log_en);
  getRosParam(this->node, "vio.visual_map_manage_shadow_en", visual_map_manage_shadow_en);
  getRosParam(this->node, "vio.visual_ref_current_select_en", visual_ref_current_select_en);
  getRosParam(this->node, "vio.visual_ref_fallback_en", visual_ref_fallback_en);
  getRosParam(this->node, "vio.visual_ref_lifecycle_en", visual_ref_lifecycle_en);
  getRosParam(this->node, "vio.visual_ref_view_coverage_en", visual_ref_view_coverage_en);
  getRosParam(this->node, "vio.visual_ref_nis_en", visual_ref_nis_en);
  getRosParam(this->node, "vio.visual_point_seed_validation_en", visual_point_seed_validation_en);
  getRosParam(this->node, "vio.visual_point_footprint_redundancy_en", visual_point_footprint_redundancy_en);
  getRosParam(this->node, "vio.visual_point_information_prune_en", visual_point_information_prune_en);
  getRosParam(this->node, "vio.visual_point_replacement_en", visual_point_replacement_en);
  getRosParam(this->node, "vio.visual_map_retirement_apply_en", visual_map_retirement_apply_en);
  getRosParam(this->node, "vio.visual_ref_max_candidates", visual_ref_max_candidates);
  getRosParam(this->node, "vio.visual_ref_validate_min_tests", visual_ref_validate_min_tests);
  getRosParam(this->node, "vio.visual_ref_validate_min_ratio", visual_ref_validate_min_ratio);
  getRosParam(this->node, "vio.visual_ref_retire_reject_count", visual_ref_retire_reject_count);
  getRosParam(this->node, "vio.visual_ref_max_count", visual_ref_max_count);
  getRosParam(this->node, "vio.visual_ref_coverage_angle_deg", visual_ref_coverage_angle_deg);
  getRosParam(this->node, "vio.visual_ref_max_anisotropy", visual_ref_max_anisotropy);
  getRosParam(this->node, "vio.visual_ref_nis_max_per_dof", visual_ref_nis_max_per_dof);
  getRosParam(this->node, "vio.visual_point_seed_min_tests", visual_point_seed_min_tests);
  getRosParam(this->node, "vio.visual_point_seed_min_ratio", visual_point_seed_min_ratio);
  getRosParam(this->node, "vio.visual_point_footprint_iou", visual_point_footprint_iou);
  getRosParam(this->node, "vio.visual_point_information_retain", visual_point_information_retain);
  getRosParam(this->node, "vio.visual_point_suspect_reject_count", visual_point_suspect_reject_count);
  getRosParam(this->node, "vio.visual_point_stale_frames", visual_point_stale_frames);
  getRosParam(this->node, "vio.raw_camera_model_jacobian_en", raw_camera_model_jacobian_en);
  getRosParam(this->node, "vio.virtual_focal_length", virtual_focal_length);
  getRosParam(this->node, "vio.virtual_patch_margin", virtual_patch_margin);
  getRosParam(this->node, "vio.virtual_max_search_level", virtual_max_search_level);
  getRosParam(this->node, "vio.virtual_patch_resampling_mode", virtual_patch_resampling_mode);
  getRosParam(this->node, "vio.virtual_interp_mode", virtual_interp_mode);
  getRosParam(this->node, "vio.virtual_raw_window_half_size", virtual_raw_window_half_size);
  getRosParam(this->node, "vio.virtual_splat_min_weight", virtual_splat_min_weight);
  getRosParam(this->node, "vio.virtual_splat_require_full_core_coverage", virtual_splat_require_full_core_coverage);
  getRosParam(this->node, "vio.virtual_splat_debug_compare_pull_exact", virtual_splat_debug_compare_pull_exact);
  getRosParam(this->node, "vio.visual_geom_filter_en", visual_geom_filter_en);
  getRosParam(this->node, "vio.visual_geom_filter_log_en", visual_geom_filter_log_en);
  getRosParam(this->node, "vio.visual_geom_filter_require_point_cov", visual_geom_filter_require_point_cov);
  getRosParam(this->node, "vio.visual_geom_filter_voxel_size", visual_geom_filter_voxel_size);
  getRosParam(this->node, "vio.visual_geom_filter_min_plane_points", visual_geom_filter_min_plane_points);
  getRosParam(this->node, "vio.visual_geom_filter_radius_multiplier", visual_geom_filter_radius_multiplier);
  getRosParam(this->node, "vio.visual_geom_filter_min_normal_cos", visual_geom_filter_min_normal_cos);
  getRosParam(this->node, "vio.visual_geom_filter_max_chi2", visual_geom_filter_max_chi2);
  getRosParam(this->node, "vio.visual_geom_filter_min_sigma", visual_geom_filter_min_sigma);
  getRosParam(this->node, "vio.visual_geom_filter_max_point_cov_trace", visual_geom_filter_max_point_cov_trace);
  getRosParam(this->node, "vio.visual_geom_filter_max_normal_cov", visual_geom_filter_max_normal_cov);
  getRosParam(this->node, "vio.draw_rejected_points_en", draw_rejected_points_en);
  getRosParam(this->node, "vio.ncc_en", ncc_en);
  getRosParam(this->node, "vio.ncc_thre", ncc_thre);
  getRosParam(this->node, "vio.ncc_thre_by_level", ncc_thre_by_level);
  getRosParam(this->node, "vio.usage_stats_en", usage_stats_en);
  getRosParam(this->node, "vio.usage_stats_window", usage_stats_window);
  getRosParam(this->node, "vio.cross_camera_reference_en", cross_camera_reference_en);
  getRosParam(this->node, "vio.cross_camera_current_residual_en", cross_camera_current_residual_en);
  getRosParam(this->node, "vio.online_extrinsic_en", online_extrinsic_en);
  getRosParam(this->node, "vio.online_extrinsic_rot_en", online_extrinsic_rot_en);
  getRosParam(this->node, "vio.online_extrinsic_trans_en", online_extrinsic_trans_en);
  getRosParam(this->node, "vio.online_extrinsic_prior_factor_en", online_extrinsic_prior_factor_en);
  getRosParam(this->node, "vio.online_extrinsic_camera_mask", online_extrinsic_camera_mask);
  getRosParam(this->node, "vio.online_extrinsic_start_frame", online_extrinsic_start_frame);
  getRosParam(this->node, "vio.online_extrinsic_min_tracks", online_extrinsic_min_tracks);
  getRosParam(this->node, "vio.online_extrinsic_prior_rot_std_deg", online_extrinsic_prior_rot_std_deg);
  getRosParam(this->node, "vio.online_extrinsic_prior_trans_std_m", online_extrinsic_prior_trans_std_m);
  getRosParam(this->node, "vio.online_extrinsic_max_rot_update_deg", online_extrinsic_max_rot_update_deg);
  getRosParam(this->node, "vio.online_extrinsic_max_trans_update_m", online_extrinsic_max_trans_update_m);
  getRosParam(this->node, "vio.online_time_offset_en", online_time_offset_en);
  getRosParam(this->node, "vio.online_time_offset_group_mask", online_time_offset_group_mask);
  getRosParam(this->node, "vio.online_time_offset_start_frame", online_time_offset_start_frame);
  getRosParam(this->node, "vio.online_time_offset_min_tracks", online_time_offset_min_tracks);
  getRosParam(this->node, "vio.online_time_offset_min_pixel_velocity", online_time_offset_min_pixel_velocity);
  getRosParam(this->node, "vio.online_time_offset_prior_std_ms", online_time_offset_prior_std_ms);
  getRosParam(this->node, "vio.online_time_offset_process_noise_ms_sqrt_s", online_time_offset_process_noise_ms_sqrt_s);
  getRosParam(this->node, "vio.online_time_offset_max_update_ms", online_time_offset_max_update_ms);
  getRosParam(this->node, "vio.online_time_offset_max_abs_ms", online_time_offset_max_abs_ms);
  getRosParam(this->node, "vio.online_time_offset_min_update_interval", online_time_offset_min_update_interval);
  getRosParam(this->node, "vio.outlier_threshold", outlier_threshold);
  getRosParam(this->node, "vio.frontend_mode", frontend_mode);
  getRosParam(this->node, "vio.opticalflow.max_cnt", optical_flow_max_cnt);
  getRosParam(this->node, "vio.opticalflow.min_dist", optical_flow_min_dist);
  getRosParam(this->node, "vio.opticalflow.min_track_len_for_triangulation", optical_flow_min_track_len_for_triangulation);
  getRosParam(this->node, "vio.opticalflow.track_history_size", optical_flow_track_history_size);
  getRosParam(this->node, "vio.opticalflow.quality_level", optical_flow_quality_level);
  getRosParam(this->node, "vio.opticalflow.f_threshold", optical_flow_f_threshold);
  getRosParam(this->node, "vio.opticalflow.flow_back", optical_flow_flow_back);
  getRosParam(this->node, "vio.opticalflow.feature_image_topic", optical_flow_feature_image_topic);
  getRosParam(this->node, "vio.opticalflow.triangulated_points_topic", optical_flow_triangulated_points_topic);
  getRosParam(this->node, "time_offset.exposure_time_init", exposure_time_init);
  getRosParam(this->node, "time_offset.img_time_offset", img_time_offset);
  getRosParam(this->node, "time_offset.img_time_offset_groups", img_time_offset_groups);
  getRosParam(this->node, "uav.imu_rate_odom", imu_prop_enable);
  getRosParam(this->node, "uav.gravity_align_en", gravity_align_en);

  getRosParam(this->node, "evo.seq_name", seq_name);
  getRosParam(this->node, "evo.pose_output_en", pose_output_en);
  getRosParam(this->node, "imu.gyr_cov", gyr_cov);
  getRosParam(this->node, "imu.acc_cov", acc_cov);
  getRosParam(this->node, "imu.imu_int_frame", imu_int_frame);
  getRosParam(this->node, "imu.imu_en", imu_en);
  getRosParam(this->node, "imu.gravity_est_en", gravity_est_en);
  getRosParam(this->node, "imu.ba_bg_est_en", ba_bg_est_en);

  getRosParam(this->node, "preprocess.blind", p_pre->blind);
  getRosParam(this->node, "preprocess.filter_size_surf", filter_size_surf_min);
  getRosParam(this->node, "preprocess.lidar_type", p_pre->lidar_type);
  getRosParam(this->node, "preprocess.scan_line", p_pre->N_SCANS);
  getRosParam(this->node, "preprocess.scan_rate", p_pre->SCAN_RATE);
  getRosParam(this->node, "preprocess.point_filter_num", p_pre->point_filter_num);
  getRosParam(this->node, "preprocess.feature_extract_enabled", p_pre->feature_enabled);
  getRosParam(this->node, "preprocess.image_downclock_en", image_downclock_en);
  getRosParam(this->node, "preprocess.image_downclock_factor", image_downclock_factor);
  if (image_downclock_factor < 1)
    throw std::runtime_error("preprocess.image_downclock_factor must be at least 1");

  getRosParam(this->node, "pcd_save.interval", pcd_save_interval);
  getRosParam(this->node, "pcd_save.pcd_save_en", pcd_save_en);
  getRosParam(this->node, "pcd_save.colmap_output_en", colmap_output_en);
  getRosParam(this->node, "pcd_save.filter_size_pcd", filter_size_pcd);
  getRosParam(this->node, "extrin_calib.extrinsic_T", extrinT);
  getRosParam(this->node, "extrin_calib.extrinsic_R", extrinR);
  getRosParam(this->node, "debug.plot_time", plot_time);
  getRosParam(this->node, "debug.frame_cnt", frame_cnt);
  getRosParam(this->node, "debug.ref_patch_dump_en", ref_patch_dump_en);
  getRosParam(this->node, "debug.ref_patch_dump_random_seed", ref_patch_dump_random_seed);
  getRosParam(this->node, "debug.ref_patch_dump_max_candidate_skip", ref_patch_dump_max_candidate_skip);
  getRosParam(this->node, "debug.ref_patch_dump_ncc_threshold", ref_patch_dump_ncc_threshold);
  getRosParam(this->node, "debug.runtime_support_dump_en", runtime_support_dump_en);
  getRosParam(this->node, "debug.runtime_support_dump_folder", runtime_support_dump_folder);

  getRosParam(this->node, "publish.blind_rgb_points", blind_rgb_points);
  getRosParam(this->node, "publish.pub_scan_num", pub_scan_num);
  getRosParam(this->node, "publish.pub_effect_point_en", pub_effect_point_en);
  getRosParam(this->node, "publish.dense_map_en", dense_map_en);

  if (!std::isfinite(ncc_thre) || ncc_thre < -1.0 || ncc_thre > 1.0)
    throw std::runtime_error("vio.ncc_thre must be finite and in [-1, 1]");
  if (ncc_thre_by_level.empty())
    ncc_thre_by_level.assign(std::max(1, patch_pyrimid_level), ncc_thre);
  else if (ncc_thre_by_level.size() == 1)
    ncc_thre_by_level.assign(std::max(1, patch_pyrimid_level), ncc_thre_by_level.front());
  else if (static_cast<int>(ncc_thre_by_level.size()) != std::max(1, patch_pyrimid_level))
    throw std::runtime_error("vio.ncc_thre_by_level must contain one value or exactly vio.patch_pyrimid_level values, ordered L0 first");
  for (double threshold : ncc_thre_by_level)
    if (!std::isfinite(threshold) || threshold < -1.0 || threshold > 1.0)
      throw std::runtime_error("every vio.ncc_thre_by_level value must be finite and in [-1, 1]");
  if (usage_stats_window <= 0) throw std::runtime_error("vio.usage_stats_window must be positive");

  if (num_cameras > 1 && colmap_output_en)
  {
    ROS_WARN("Multi-camera COLMAP output is not implemented; disabling pcd_save.colmap_output_en");
    colmap_output_en = false;
  }

  if (extrinT.size() != 3) throw std::runtime_error("extrin_calib.extrinsic_T must contain exactly 3 values");
  if (extrinR.size() != 9) throw std::runtime_error("extrin_calib.extrinsic_R must contain exactly 9 values");
  if (!online_extrinsic_camera_mask.empty() && static_cast<int>(online_extrinsic_camera_mask.size()) != num_cameras)
    throw std::runtime_error("vio.online_extrinsic_camera_mask must be empty or have common.num_cameras entries");
  if (raw_camera_model_jacobian_en && virtual_fisheye_patch_en)
    ROS_WARN("vio.raw_camera_model_jacobian_en is ignored because vio.virtual_fisheye_patch_en is true");

  if (online_extrinsic_camera_mask.empty())
  {
    online_extrinsic_camera_mask.resize(num_cameras, 1);
    for (int camera_id = 0; camera_id < num_cameras; ++camera_id)
      online_extrinsic_camera_mask[camera_id] = camera_configs[camera_id].online_extrinsic_en ? 1 : 0;
  }
  if (online_extrinsic_start_frame < 0) throw std::runtime_error("vio.online_extrinsic_start_frame must be non-negative");
  if (online_extrinsic_min_tracks < 0) throw std::runtime_error("vio.online_extrinsic_min_tracks must be non-negative");
  if (online_extrinsic_prior_rot_std_deg <= 0.0 || online_extrinsic_prior_trans_std_m <= 0.0)
    throw std::runtime_error("vio.online_extrinsic prior std values must be positive");
  if (online_extrinsic_max_rot_update_deg < 0.0 || online_extrinsic_max_trans_update_m < 0.0)
    throw std::runtime_error("vio.online_extrinsic max update limits must be non-negative");

  if (!online_time_offset_group_mask.empty() && static_cast<int>(online_time_offset_group_mask.size()) != num_time_offset_groups)
    throw std::runtime_error("vio.online_time_offset_group_mask must be empty or have one entry per time-offset group");
  if (!img_time_offset_groups.empty() && static_cast<int>(img_time_offset_groups.size()) != num_time_offset_groups)
    throw std::runtime_error("time_offset.img_time_offset_groups must be empty or have one entry per time-offset group");
  if (img_time_offset_groups.empty())
    img_time_offset_groups.assign(num_time_offset_groups, img_time_offset);
  if (online_time_offset_group_mask.empty())
  {
    online_time_offset_group_mask.assign(num_time_offset_groups, 0);
    for (int camera_id = 0; camera_id < num_cameras; ++camera_id)
    {
      const int group_id = camera_configs[camera_id].time_offset_group;
      if (group_id >= 0 && group_id < num_time_offset_groups && camera_configs[camera_id].online_time_offset_en)
        online_time_offset_group_mask[group_id] = 1;
    }
  }
  if (online_time_offset_start_frame < 0) throw std::runtime_error("vio.online_time_offset_start_frame must be non-negative");
  if (online_time_offset_min_tracks < 0) throw std::runtime_error("vio.online_time_offset_min_tracks must be non-negative");
  if (!std::isfinite(online_time_offset_min_pixel_velocity) || online_time_offset_min_pixel_velocity < 0.0)
    throw std::runtime_error("vio.online_time_offset_min_pixel_velocity must be finite and non-negative");
  if (online_time_offset_prior_std_ms <= 0.0 || online_time_offset_max_update_ms < 0.0 || online_time_offset_max_abs_ms < 0.0)
    throw std::runtime_error("vio.online_time_offset std/max parameters are invalid");
  if (online_time_offset_min_update_interval < 0)
    throw std::runtime_error("vio.online_time_offset_min_update_interval must be non-negative");

  p_pre->blind_sqr = p_pre->blind * p_pre->blind;
}

void LIVMapper::initializeComponents(ros::NodeHandle &nh)
{
  downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);

  // extrinT.assign({0.04165, 0.02326, -0.0284});
  // extrinR.assign({1, 0, 0, 0, 1, 0, 0, 0, 1});
  // cameraextrinT.assign({0.0194384, 0.104689,-0.0251952});
  // cameraextrinR.assign({0.00610193,-0.999863,-0.0154172,-0.00615449,0.0153796,-0.999863,0.999962,0.00619598,-0.0060598});

  extT << VEC_FROM_ARRAY(extrinT);
  extR << MAT_FROM_ARRAY(extrinR);

  voxelmap_manager->extT_ << VEC_FROM_ARRAY(extrinT);
  voxelmap_manager->extR_ << MAT_FROM_ARRAY(extrinR);

  vio_manager->configureCameras(num_cameras);
  std::vector<int> camera_time_offset_groups(num_cameras, 0);
  const double time_offset_prior_std_s = online_time_offset_prior_std_ms * 1.0e-3;
  const double time_offset_prior_var = time_offset_prior_std_s * time_offset_prior_std_s;
  for (int group_id = 0; group_id < num_time_offset_groups; ++group_id)
  {
    _state.setTimeOffsetCovariance(group_id, time_offset_prior_var);
    state_propagat.setTimeOffsetCovariance(group_id, time_offset_prior_var);
    imu_propagate.setTimeOffsetCovariance(group_id, time_offset_prior_var);
    latest_ekf_state.setTimeOffsetCovariance(group_id, time_offset_prior_var);
  }
  for (int camera_id = 0; camera_id < num_cameras; ++camera_id)
  {
    CameraInputConfig &config = camera_configs[camera_id];
    camera_time_offset_groups[camera_id] = config.time_offset_group;
    PerCameraData &ctx = vio_manager->cameras_[camera_id];
    if (!vk::camera_loader::loadFromRosNs(this->node, config.camera_namespace, ctx.cam))
      throw std::runtime_error("failed to load camera model for camera_id=" + std::to_string(camera_id) +
                               " camera_ns=" + config.camera_namespace);
    if (config.image_undistort_en)
    {
      vk::AbstractCamera *raw_camera = nullptr;
      if (!vk::camera_loader::loadFromRosNs(this->node, config.raw_camera_namespace, raw_camera))
        throw std::runtime_error("failed to load raw camera model for input undistort, camera_id=" +
                                 std::to_string(camera_id) + " raw_camera_ns=" + config.raw_camera_namespace);
      std::string remap_error;
      if (!buildImageUndistortMaps(*raw_camera, *ctx.cam, config.undistort_map_x,
                                   config.undistort_map_y, remap_error))
        throw std::runtime_error("failed to build input undistort remap for camera_id=" +
                                 std::to_string(camera_id) + ": " + remap_error);
      printf("[ Input Undistort ] camera_id=%d raw_ns=%s output_ns=%s output_size=%dx%d\n",
             camera_id, config.raw_camera_namespace.c_str(), config.camera_namespace.c_str(),
             config.undistort_map_x.cols, config.undistort_map_x.rows);
    }
    vio_manager->setCameraCalibration(camera_id, config.img_topic, config.camera_namespace, config.Rcl, config.Pcl);
    vio_manager->setCameraModelJacobianParameters(camera_id, config.camera_model_type,
                                                  config.k1, config.k2, config.k3, config.k4,
                                                  config.xi, config.p1, config.p2);
    const M3D Rcl = vio_manager->cameras_[camera_id].Rcl;
    const V3D Pcl = vio_manager->cameras_[camera_id].Pcl;
    const double rot_std_rad = online_extrinsic_prior_rot_std_deg * kDegToRad;
    const double rot_var = rot_std_rad * rot_std_rad;
    const double trans_var = online_extrinsic_prior_trans_std_m * online_extrinsic_prior_trans_std_m;
    _state.setCameraExtrinsic(camera_id, Rcl, Pcl);
    state_propagat.setCameraExtrinsic(camera_id, Rcl, Pcl);
    imu_propagate.setCameraExtrinsic(camera_id, Rcl, Pcl);
    latest_ekf_state.setCameraExtrinsic(camera_id, Rcl, Pcl);
    _state.setCameraExtrinsicCovariance(camera_id, rot_var, trans_var);
    state_propagat.setCameraExtrinsicCovariance(camera_id, rot_var, trans_var);
    imu_propagate.setCameraExtrinsicCovariance(camera_id, rot_var, trans_var);
    latest_ekf_state.setCameraExtrinsicCovariance(camera_id, rot_var, trans_var);
  }
  vio_manager->setCameraTimeOffsetGroups(camera_time_offset_groups);
  vio_manager->grid_size = grid_size;
  vio_manager->patch_size = patch_size;
  vio_manager->outlier_threshold = outlier_threshold;
  vio_manager->setImuToLidarExtrinsic(extT, extR);
  vio_manager->state = &_state;
  vio_manager->state_propagat = &state_propagat;
  vio_manager->max_iterations = max_iterations;
  vio_manager->img_point_cov = IMG_POINT_COV;
  vio_manager->normal_en = normal_en;
  vio_manager->inverse_composition_en = inverse_composition_en;
  vio_manager->raycast_en = raycast_en;
  vio_manager->grid_n_width = grid_n_width;
  vio_manager->grid_n_height = grid_n_height;
  vio_manager->patch_pyrimid_level = patch_pyrimid_level;
  vio_manager->virtual_fisheye_patch_en = virtual_fisheye_patch_en;
  vio_manager->virtual_sparse_patch_en = virtual_sparse_patch_en;
  vio_manager->virtual_s2_optimize_en = virtual_s2_optimize_en;
  vio_manager->visual_ref_post_ekf_build_en = visual_ref_post_ekf_build_en;
  vio_manager->visual_map_manage_en = visual_map_manage_en;
  vio_manager->visual_map_manage_log_en = visual_map_manage_log_en;
  vio_manager->visual_map_manage_shadow_en = visual_map_manage_shadow_en;
  vio_manager->visual_ref_current_select_en = visual_ref_current_select_en;
  vio_manager->visual_ref_fallback_en = visual_ref_fallback_en;
  vio_manager->visual_ref_lifecycle_en = visual_ref_lifecycle_en;
  vio_manager->visual_ref_view_coverage_en = visual_ref_view_coverage_en;
  vio_manager->visual_ref_nis_en = visual_ref_nis_en;
  vio_manager->visual_point_seed_validation_en = visual_point_seed_validation_en;
  vio_manager->visual_point_footprint_redundancy_en = visual_point_footprint_redundancy_en;
  vio_manager->visual_point_information_prune_en = visual_point_information_prune_en;
  vio_manager->visual_point_replacement_en = visual_point_replacement_en;
  vio_manager->visual_map_retirement_apply_en = visual_map_retirement_apply_en;
  vio_manager->visual_ref_max_candidates = visual_ref_max_candidates;
  vio_manager->visual_ref_validate_min_tests = visual_ref_validate_min_tests;
  vio_manager->visual_ref_validate_min_ratio = visual_ref_validate_min_ratio;
  vio_manager->visual_ref_retire_reject_count = visual_ref_retire_reject_count;
  vio_manager->visual_ref_max_count = visual_ref_max_count;
  vio_manager->visual_ref_coverage_angle_deg = visual_ref_coverage_angle_deg;
  vio_manager->visual_ref_max_anisotropy = visual_ref_max_anisotropy;
  vio_manager->visual_ref_nis_max_per_dof = visual_ref_nis_max_per_dof;
  vio_manager->visual_point_seed_min_tests = visual_point_seed_min_tests;
  vio_manager->visual_point_seed_min_ratio = visual_point_seed_min_ratio;
  vio_manager->visual_point_footprint_iou = visual_point_footprint_iou;
  vio_manager->visual_point_information_retain = visual_point_information_retain;
  vio_manager->visual_point_suspect_reject_count = visual_point_suspect_reject_count;
  vio_manager->visual_point_stale_frames = visual_point_stale_frames;
  vio_manager->raw_camera_model_jacobian_en = raw_camera_model_jacobian_en && !virtual_fisheye_patch_en;
  vio_manager->cross_camera_reference_en = cross_camera_reference_en;
  vio_manager->cross_camera_current_residual_en = cross_camera_current_residual_en;
  vio_manager->directional_update_en = directional_update_en;
  vio_manager->directional_drop_variance_reduction = directional_drop_variance_reduction;
  vio_manager->directional_full_variance_reduction = directional_full_variance_reduction;
  vio_manager->online_extrinsic_en = online_extrinsic_en;
  vio_manager->online_extrinsic_rot_en = online_extrinsic_rot_en;
  vio_manager->online_extrinsic_trans_en = online_extrinsic_trans_en;
  vio_manager->online_extrinsic_prior_factor_en = online_extrinsic_prior_factor_en;
  vio_manager->online_extrinsic_camera_mask = online_extrinsic_camera_mask;
  vio_manager->online_extrinsic_start_frame = online_extrinsic_start_frame;
  vio_manager->online_extrinsic_min_tracks = online_extrinsic_min_tracks;
  vio_manager->online_extrinsic_prior_rot_std_deg = online_extrinsic_prior_rot_std_deg;
  vio_manager->online_extrinsic_prior_trans_std_m = online_extrinsic_prior_trans_std_m;
  vio_manager->online_extrinsic_max_rot_update_deg = online_extrinsic_max_rot_update_deg;
  vio_manager->online_extrinsic_max_trans_update_m = online_extrinsic_max_trans_update_m;
  vio_manager->online_time_offset_en = online_time_offset_en;
  vio_manager->online_time_offset_group_mask = online_time_offset_group_mask;
  vio_manager->online_time_offset_start_frame = online_time_offset_start_frame;
  vio_manager->online_time_offset_min_tracks = online_time_offset_min_tracks;
  vio_manager->online_time_offset_min_pixel_velocity = online_time_offset_min_pixel_velocity;
  vio_manager->online_time_offset_prior_std_ms = online_time_offset_prior_std_ms;
  vio_manager->online_time_offset_max_update_ms = online_time_offset_max_update_ms;
  vio_manager->online_time_offset_max_abs_ms = online_time_offset_max_abs_ms;
  vio_manager->online_time_offset_min_update_interval = online_time_offset_min_update_interval;
  vio_manager->virtual_focal_length = virtual_focal_length;
  vio_manager->virtual_patch_margin = virtual_patch_margin;
  vio_manager->virtual_max_search_level = virtual_max_search_level;
  vio_manager->virtual_patch_resampling_mode = virtual_patch_resampling_mode;
  vio_manager->virtual_interp_mode = virtual_interp_mode;
  vio_manager->virtual_raw_window_half_size = virtual_raw_window_half_size;
  vio_manager->virtual_splat_min_weight = virtual_splat_min_weight;
  vio_manager->virtual_splat_require_full_core_coverage = virtual_splat_require_full_core_coverage;
  vio_manager->virtual_splat_debug_compare_pull_exact = virtual_splat_debug_compare_pull_exact;
  vio_manager->visual_geom_filter_en = visual_geom_filter_en;
  vio_manager->visual_geom_filter_log_en = visual_geom_filter_log_en;
  vio_manager->visual_geom_filter_require_point_cov = visual_geom_filter_require_point_cov;
  vio_manager->visual_geom_filter_voxel_size = visual_geom_filter_voxel_size;
  vio_manager->visual_geom_filter_min_plane_points = visual_geom_filter_min_plane_points;
  vio_manager->visual_geom_filter_radius_multiplier = visual_geom_filter_radius_multiplier;
  vio_manager->visual_geom_filter_min_normal_cos = visual_geom_filter_min_normal_cos;
  vio_manager->visual_geom_filter_max_chi2 = visual_geom_filter_max_chi2;
  vio_manager->visual_geom_filter_min_sigma = visual_geom_filter_min_sigma;
  vio_manager->visual_geom_filter_max_point_cov_trace = visual_geom_filter_max_point_cov_trace;
  vio_manager->visual_geom_filter_max_normal_cov = visual_geom_filter_max_normal_cov;
  vio_manager->draw_rejected_points_en = draw_rejected_points_en;
  vio_manager->ncc_en = ncc_en;
  vio_manager->ncc_thre = ncc_thre;
  vio_manager->ncc_thre_by_level = ncc_thre_by_level;
  vio_manager->usage_stats_en = usage_stats_en;
  vio_manager->usage_stats_window = usage_stats_window;
  vio_manager->ref_patch_dump_en = ref_patch_dump_en;
  vio_manager->ref_patch_dump_random_seed = ref_patch_dump_random_seed;
  vio_manager->ref_patch_dump_max_candidate_skip = ref_patch_dump_max_candidate_skip;
  vio_manager->ref_patch_dump_ncc_threshold = ref_patch_dump_ncc_threshold;
  vio_manager->runtime_support_dump_en = runtime_support_dump_en;
  vio_manager->runtime_support_dump_folder = runtime_support_dump_folder;
  vio_manager->exposure_estimate_en = exposure_estimate_en;
  vio_manager->colmap_output_en = colmap_output_en;
  vio_manager->frontend_mode = frontend_mode;
  vio_manager->optical_flow_max_cnt = optical_flow_max_cnt;
  vio_manager->optical_flow_min_dist = optical_flow_min_dist;
  vio_manager->optical_flow_min_track_len_for_triangulation = optical_flow_min_track_len_for_triangulation;
  vio_manager->optical_flow_track_history_size = optical_flow_track_history_size;
  vio_manager->optical_flow_quality_level = optical_flow_quality_level;
  vio_manager->optical_flow_f_threshold = optical_flow_f_threshold;
  vio_manager->optical_flow_flow_back = optical_flow_flow_back;
  vio_manager->initializeVIO();

  p_imu->set_extrinsic(extT, extR);
  p_imu->set_gyr_cov_scale(V3D(gyr_cov, gyr_cov, gyr_cov));
  p_imu->set_acc_cov_scale(V3D(acc_cov, acc_cov, acc_cov));
  p_imu->set_inv_expo_cov(inv_expo_cov);
  p_imu->set_time_offset_cov(std::pow(online_time_offset_process_noise_ms_sqrt_s * 1.0e-3, 2));
  p_imu->set_gyr_bias_cov(V3D(0.0001, 0.0001, 0.0001));
  p_imu->set_acc_bias_cov(V3D(0.0001, 0.0001, 0.0001));
  p_imu->set_imu_init_frame_num(imu_int_frame);

  if (!imu_en) p_imu->disable_imu();
  if (!gravity_est_en) p_imu->disable_gravity_est();
  if (!ba_bg_est_en) p_imu->disable_bias_est();
  if (!exposure_estimate_en) p_imu->disable_exposure_est();

  slam_mode_ = (img_en && lidar_en) ? LIVO : imu_en ? ONLY_LIO : ONLY_LO;
}

void LIVMapper::initializeFiles()
{
  std::filesystem::create_directories(std::string(ROOT_DIR) + "Log");
  if (pcd_save_en) std::filesystem::create_directories(std::string(ROOT_DIR) + "Log/PCD");
  if (pcd_save_en && colmap_output_en)
  {
      const std::string folderPath = std::string(ROOT_DIR) + "/scripts/colmap_output.sh";

      std::string chmodCommand = "chmod +x " + folderPath;

      int chmodRet = system(chmodCommand.c_str());
      if (chmodRet != 0) {
          std::cerr << "Failed to set execute permissions for the script." << std::endl;
          return;
      }

      int executionRet = system(folderPath.c_str());
      if (executionRet != 0) {
          std::cerr << "Failed to execute the script." << std::endl;
          return;
      }
  }
  if(colmap_output_en) fout_points.open(std::string(ROOT_DIR) + "Log/Colmap/sparse/0/points3D.txt", std::ios::out);
  if(pcd_save_interval > 0) fout_pcd_pos.open(std::string(ROOT_DIR) + "Log/PCD/scans_pos.json", std::ios::out);
  fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), std::ios::out);
  fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), std::ios::out);
}

void LIVMapper::initializeSubscribersAndPublishers(ros::NodeHandle &nh, image_transport::ImageTransport &it_)
{
  image_transport::ImageTransport it(this->node);
  if (p_pre->lidar_type == AVIA) {
    sub_pcl = this->node.subscribe(lid_topic, 200000, &LIVMapper::livox_pcl_cbk, this);
  } else {
    sub_pcl = this->node.subscribe(lid_topic, 200000, &LIVMapper::standard_pcl_cbk, this);
  }
  sub_imu = this->node.subscribe(imu_topic, 200000, &LIVMapper::imu_cbk, this);
  const std::string compressed_suffix = "/compressed";
  sub_imgs.resize(num_cameras);
  sub_imgs_compressed.resize(num_cameras);
  for (int camera_id = 0; camera_id < num_cameras; ++camera_id)
  {
    const std::string &topic = camera_configs[camera_id].img_topic;
    const bool compressed = topic.size() >= compressed_suffix.size() &&
                            topic.compare(topic.size() - compressed_suffix.size(), compressed_suffix.size(), compressed_suffix) == 0;
    if (compressed)
    {
      sub_imgs_compressed[camera_id] = this->node.subscribe<sensor_msgs::CompressedImage>(
          topic, 200000, [this, camera_id](const sensor_msgs::CompressedImage::ConstPtr &msg) {
            compressed_img_cbk(camera_id, msg);
          });
    }
    else
    {
      sub_imgs[camera_id] = this->node.subscribe<sensor_msgs::Image>(
          topic, 200000, [this, camera_id](const sensor_msgs::Image::ConstPtr &msg) { img_cbk(camera_id, msg); });
    }
    ROS_INFO("Subscribed camera_id=%d image=%s (%s)", camera_id, topic.c_str(), compressed ? "compressed" : "raw");
  }

  pubLaserCloudFullRes = this->node.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);
  pubNormal = this->node.advertise<visualization_msgs::MarkerArray>("/visualization_marker", 100);
  pubSubVisualMap = this->node.advertise<sensor_msgs::PointCloud2>("/cloud_visual_sub_map_before", 100);
  pubLaserCloudEffect = this->node.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100);
  pubLaserCloudMap = this->node.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100);
  pubOdomAftMapped = this->node.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
  pubPath = this->node.advertise<nav_msgs::Path>("/path", 10);
  plane_pub = this->node.advertise<visualization_msgs::Marker>("/planner_normal", 1);
  voxel_pub = this->node.advertise<visualization_msgs::MarkerArray>("/voxels", 1);
  pubLaserCloudDyn = this->node.advertise<sensor_msgs::PointCloud2>("/dyn_obj", 100);
  pubLaserCloudDynRmed = this->node.advertise<sensor_msgs::PointCloud2>("/dyn_obj_removed", 100);
  pubLaserCloudDynDbg = this->node.advertise<sensor_msgs::PointCloud2>("/dyn_obj_dbg_hist", 100);
  mavros_pose_publisher = this->node.advertise<geometry_msgs::PoseStamped>("/mavros/vision_pose/pose", 10);
  pubImages.reserve(num_cameras);
  for (int camera_id = 0; camera_id < num_cameras; ++camera_id)
    pubImages.push_back(it.advertise("/rgb_img/camera_" + std::to_string(camera_id), 1));
  pubOpticalFlowImage = it.advertise(optical_flow_feature_image_topic, 1);
  pubTriangulatedPoints = this->node.advertise<sensor_msgs::PointCloud2>(optical_flow_triangulated_points_topic, 10);
  pubImuPropOdom = this->node.advertise<nav_msgs::Odometry>("/LIVO2/imu_propagate", 10000);
  imu_prop_timer = this->node.createTimer(ros::Duration(0.004), [this](const ros::TimerEvent &) { imu_prop_callback(); });
  voxelmap_manager->voxel_map_pub_= this->node.advertise<visualization_msgs::MarkerArray>("/planes", 10000);
}

void LIVMapper::handleFirstFrame()
{
  if (!is_first_frame)
  {
    _first_lidar_time = LidarMeasures.last_lio_update_time;
    p_imu->first_lidar_time = _first_lidar_time; // Only for IMU data log
    is_first_frame = true;
    cout << "FIRST LIDAR FRAME!" << endl;
  }
}

void LIVMapper::gravityAlignment()
{
  if (!p_imu->imu_need_init && !gravity_align_finished)
  {
    std::cout << "Gravity Alignment Starts" << std::endl;
    V3D ez(0, 0, -1), gz(_state.gravity);
    Eigen::Quaterniond G_q_I0 = Eigen::Quaterniond::FromTwoVectors(gz, ez);
    M3D G_R_I0 = G_q_I0.toRotationMatrix();

    _state.pos_end = G_R_I0 * _state.pos_end;
    _state.rot_end = G_R_I0 * _state.rot_end;
    _state.vel_end = G_R_I0 * _state.vel_end;
    _state.gravity = G_R_I0 * _state.gravity;
    gravity_align_finished = true;
    std::cout << "Gravity Alignment Finished" << std::endl;
  }
}

void LIVMapper::processImu()
{
  // double t0 = omp_get_wtime();

  p_imu->Process2(LidarMeasures, _state, feats_undistort);
  if (vio_manager) vio_manager->setCurrentUnbiasedGyr(p_imu->unbiased_gyr);

  if (gravity_align_en) gravityAlignment();

  state_propagat = _state;
  voxelmap_manager->state_ = _state;
  voxelmap_manager->feats_undistort_ = feats_undistort;

  // double t_prop = omp_get_wtime();

  // std::cout << "[ Mapping ] feats_undistort: " << feats_undistort->size() << std::endl;
  // std::cout << "[ Mapping ] predict cov: " << _state.cov.diagonal().transpose() << std::endl;
  // std::cout << "[ Mapping ] predict sta: " << state_propagat.pos_end.transpose() << state_propagat.vel_end.transpose() << std::endl;
}

void LIVMapper::stateEstimationAndMapping()
{
  switch (LidarMeasures.lio_vio_flg)
  {
    case VIO:
      handleVIO();
      break;
    case LIO:
    case LO:
      handleLIO();
      break;
  }
}

void LIVMapper::handleVIO()
{
  int current_frontend_mode = frontend_mode;
  if (current_frontend_mode < 0 || current_frontend_mode > 2)
  {
    printf(BOLDYELLOW "[ VIO ] Unknown frontend_mode=%d, fallback to direct frontend.\n" RESET, current_frontend_mode);
    current_frontend_mode = 0;
  }

  euler_cur = RotMtoEuler(_state.rot_end);
  fout_pre << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
            << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
            << _state.bias_a.transpose() << " " << _state.inv_expo_time.transpose() << std::endl;

  if ((pcl_w_wait_pub == nullptr || pcl_w_wait_pub->empty()) && current_frontend_mode == 0)
  {
    std::cout << "[ VIO ] No point!!!" << std::endl;
    return;
  }
  if (pcl_w_wait_pub != nullptr && !pcl_w_wait_pub->empty())
  {
    std::cout << "[ VIO ] Raw feature num: " << pcl_w_wait_pub->points.size() << std::endl;
  }
  else
  {
    printf("[ VIO ] No lidar points, continue frontend_mode=%d image frontend.\n", current_frontend_mode);
  }

  const size_t vio_raw_points = pcl_w_wait_pub != nullptr ? pcl_w_wait_pub->points.size() : 0;
  // printf("[ VIO Debug ] dispatch frontend_mode=%d raw_points=%zu cameras=%d virtual=%d cross_ref=%d normal=%d inverse=%d raycast=%d\n",
  //        current_frontend_mode, vio_raw_points, vio_manager->numCameras(), vio_manager->virtual_fisheye_patch_en ? 1 : 0,
  //        vio_manager->cross_camera_reference_en ? 1 : 0, vio_manager->normal_en ? 1 : 0,
  //        vio_manager->inverse_composition_en ? 1 : 0, vio_manager->raycast_en ? 1 : 0);
  // fflush(stdout);

  if (fabs((LidarMeasures.last_lio_update_time - _first_lidar_time) - plot_time) < (frame_cnt / 2 * 0.1))
  {
    vio_manager->plot_flag = true;
  }
  else
  {
    vio_manager->plot_flag = false;
  }

  switch (current_frontend_mode)
  {
    case 1:
      vio_manager->processFrameOpticalFlow(LidarMeasures.measures.back().img, LidarMeasures.last_lio_update_time - _first_lidar_time);
      break;
    case 2:
      vio_manager->processMultiCameraFrameFake(LidarMeasures.measures.back());
      break;
    case 0:
    default:
      vio_manager->processMultiCameraFrame(LidarMeasures.measures.back(), _pv_list, voxelmap_manager->voxel_map_);
      break;
  }

  if (imu_prop_enable)
  {
    ekf_finish_once = true;
    latest_ekf_state = _state;
    latest_ekf_time = LidarMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  // int size_sub_map = vio_manager->visual_sub_map_cur.size();
  // visual_sub_map->reserve(size_sub_map);
  // for (int i = 0; i < size_sub_map; i++)
  // {
  //   PointType temp_map;
  //   temp_map.x = vio_manager->visual_sub_map_cur[i]->pos_[0];
  //   temp_map.y = vio_manager->visual_sub_map_cur[i]->pos_[1];
  //   temp_map.z = vio_manager->visual_sub_map_cur[i]->pos_[2];
  //   temp_map.intensity = 0.;
  //   visual_sub_map->push_back(temp_map);
  // }

  publish_frame_world(pubLaserCloudFullRes, vio_manager);
  publish_img_rgb(vio_manager);
  if (current_frontend_mode == 1)
  {
    publish_optical_flow_image(pubOpticalFlowImage, vio_manager);
    publish_triangulated_points(pubTriangulatedPoints, vio_manager->optical_flow_triangulated_points);
  }

  euler_cur = RotMtoEuler(_state.rot_end);
  fout_out << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
            << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
            << _state.bias_a.transpose() << " " << _state.inv_expo_time.transpose() << " " << feats_undistort->points.size() << std::endl;
}

void LIVMapper::handleLIO()
{
  euler_cur = RotMtoEuler(_state.rot_end);
  fout_pre << setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
           << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
           << _state.bias_a.transpose() << " " << _state.inv_expo_time.transpose() << endl;

  if (feats_undistort->empty() || (feats_undistort == nullptr))
  {
    std::cout << "[ LIO ]: No point!!!" << std::endl;
    return;
  }

  double t0 = omp_get_wtime();

  downSizeFilterSurf.setInputCloud(feats_undistort);
  downSizeFilterSurf.filter(*feats_down_body);

  double t_down = omp_get_wtime();

  feats_down_size = feats_down_body->points.size();
  voxelmap_manager->feats_down_body_ = feats_down_body;
  transformLidar(_state.rot_end, _state.pos_end, feats_down_body, feats_down_world);
  voxelmap_manager->feats_down_world_ = feats_down_world;
  voxelmap_manager->feats_down_size_ = feats_down_size;

  if (!lidar_map_inited)
  {
    lidar_map_inited = true;
    voxelmap_manager->BuildVoxelMap();
  }

  double t1 = omp_get_wtime();

  voxelmap_manager->StateEstimation(state_propagat);
  _state = voxelmap_manager->state_;
  _pv_list = voxelmap_manager->pv_list_;

  double t2 = omp_get_wtime();

  if (imu_prop_enable)
  {
    ekf_finish_once = true;
    latest_ekf_state = _state;
    latest_ekf_time = LidarMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  if (pose_output_en)
  {
    static bool pos_opend = false;
    static int ocount = 0;
    std::ofstream outFile, evoFile;
    const std::string result_dir = std::string(ROOT_DIR) + "Log/result";
    const std::string result_path = result_dir + "/" + seq_name + ".txt";
    if (!pos_opend)
    {
      std::filesystem::create_directories(result_dir);
      evoFile.open(result_path, std::ios::out);
      pos_opend = true;
      if (!evoFile.is_open()) ROS_ERROR("open fail\n");
    }
    else
    {
      evoFile.open(result_path, std::ios::app);
      if (!evoFile.is_open()) ROS_ERROR("open fail\n");
    }
    Eigen::Matrix4d outT;
    Eigen::Quaterniond q(_state.rot_end);
    evoFile << std::fixed;
    evoFile << LidarMeasures.last_lio_update_time << " " << _state.pos_end[0] << " " << _state.pos_end[1] << " " << _state.pos_end[2] << " "
            << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
  }

  euler_cur = RotMtoEuler(_state.rot_end);
  geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));
  publish_odometry(pubOdomAftMapped);

  double t3 = omp_get_wtime();

  PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI());
  transformLidar(_state.rot_end, _state.pos_end, feats_down_body, world_lidar);
  for (size_t i = 0; i < world_lidar->points.size(); i++)
  {
    voxelmap_manager->pv_list_[i].point_w << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
    M3D point_crossmat = voxelmap_manager->cross_mat_list_[i];
    M3D var = voxelmap_manager->body_cov_list_[i];
    var = (_state.rot_end * extR) * var * (_state.rot_end * extR).transpose() +
          (-point_crossmat) * _state.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose() + _state.cov.block<3, 3>(3, 3);
    voxelmap_manager->pv_list_[i].var = var;
  }
  voxelmap_manager->UpdateVoxelMap(voxelmap_manager->pv_list_);
  std::cout << "[ LIO ] Update Voxel Map" << std::endl;
  _pv_list = voxelmap_manager->pv_list_;

  double t4 = omp_get_wtime();

  if(voxelmap_manager->config_setting_.map_sliding_en)
  {
    voxelmap_manager->mapSliding();
  }

  PointCloudXYZI::Ptr laserCloudFullRes(dense_map_en ? feats_undistort : feats_down_body);
  int size = laserCloudFullRes->points.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

  for (int i = 0; i < size; i++)
  {
    RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
  }
  *pcl_w_wait_pub = *laserCloudWorld;

  if (!img_en) publish_frame_world(pubLaserCloudFullRes, vio_manager);
  if (pub_effect_point_en) publish_effect_world(pubLaserCloudEffect, voxelmap_manager->ptpl_list_);
  if (voxelmap_manager->config_setting_.is_pub_plane_map_) voxelmap_manager->pubVoxelMap();
  publish_path(pubPath);
  publish_mavros(mavros_pose_publisher);

  frame_num++;
  aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t4 - t0) / frame_num;

  // aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + (t2 - t1) / frame_num;
  // aver_time_map_inre = aver_time_map_inre * (frame_num - 1) / frame_num + (t4 - t3) / frame_num;
  // aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + (solve_time) / frame_num;
  // aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) / frame_num + solve_const_H_time / frame_num;
  // printf("[ mapping time ]: per scan: propagation %0.6f downsample: %0.6f match: %0.6f solve: %0.6f  ICP: %0.6f  map incre: %0.6f total: %0.6f \n"
  //         "[ mapping time ]: average: icp: %0.6f construct H: %0.6f, total: %0.6f \n",
  //         t_prop - t0, t1 - t_prop, match_time, solve_time, t3 - t1, t5 - t3, t5 - t0, aver_time_icp, aver_time_const_H_time, aver_time_consu);

  // printf("\033[1;36m[ LIO mapping time ]: current scan: icp: %0.6f secs, map incre: %0.6f secs, total: %0.6f secs.\033[0m\n"
  //         "\033[1;36m[ LIO mapping time ]: average: icp: %0.6f secs, map incre: %0.6f secs, total: %0.6f secs.\033[0m\n",
  //         t2 - t1, t4 - t3, t4 - t0, aver_time_icp, aver_time_map_inre, aver_time_consu);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m|                         LIO Mapping Time                    |\033[0m\n");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage", "Time (secs)");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "DownSample", t_down - t0);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "ICP", t2 - t1);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "updateVoxelMap", t4 - t3);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Current Total Time", t4 - t0);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Average Total Time", aver_time_consu);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");

  euler_cur = RotMtoEuler(_state.rot_end);
  fout_out << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
            << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
            << _state.bias_a.transpose() << " " << _state.inv_expo_time.transpose() << " " << feats_undistort->points.size() << std::endl;
}

void LIVMapper::savePCD()
{
  if (pcd_save_en && (pcl_wait_save->points.size() > 0 || pcl_wait_save_intensity->points.size() > 0) && pcd_save_interval < 0)
  {
    const bool has_rgb = !pcl_wait_save->empty();
    const bool has_intensity = !pcl_wait_save_intensity->empty();
    const std::string pcd_dir = std::string(ROOT_DIR) + "Log/PCD/";
    const std::string raw_rgb_path = pcd_dir + (has_intensity ? "all_raw_points_rgb.pcd" : "all_raw_points.pcd");
    const std::string downsampled_rgb_path =
        pcd_dir + (has_intensity ? "all_downsampled_points_rgb.pcd" : "all_downsampled_points.pcd");
    const std::string intensity_path = pcd_dir + (has_rgb ? "all_raw_points_intensity.pcd" : "all_raw_points.pcd");
    pcl::PCDWriter pcd_writer;

    if (has_rgb)
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
      voxel_filter.setInputCloud(pcl_wait_save);
      voxel_filter.setLeafSize(filter_size_pcd, filter_size_pcd, filter_size_pcd);
      voxel_filter.filter(*downsampled_cloud);

      pcd_writer.writeBinary(raw_rgb_path, *pcl_wait_save);
      std::cout << GREEN << "RGB point cloud data saved to: " << raw_rgb_path
                << " with point count: " << pcl_wait_save->points.size() << RESET << std::endl;

      pcd_writer.writeBinary(downsampled_rgb_path, *downsampled_cloud);
      std::cout << GREEN << "Downsampled RGB point cloud data saved to: " << downsampled_rgb_path
                << " with point count after filtering: " << downsampled_cloud->points.size() << RESET << std::endl;

      if(colmap_output_en)
      {
        fout_points << "# 3D point list with one line of data per point\n";
        fout_points << "#  POINT_ID, X, Y, Z, R, G, B, ERROR\n";
        for (size_t i = 0; i < downsampled_cloud->size(); ++i)
        {
            const auto& point = downsampled_cloud->points[i];
            fout_points << i << " "
                        << std::fixed << std::setprecision(6)
                        << point.x << " " << point.y << " " << point.z << " "
                        << static_cast<int>(point.r) << " "
                        << static_cast<int>(point.g) << " "
                        << static_cast<int>(point.b) << " "
                        << 0 << std::endl;
        }
      }
    }
    if (has_intensity)
    {
      pcd_writer.writeBinary(intensity_path, *pcl_wait_save_intensity);
      std::cout << GREEN << "Intensity point cloud data saved to: " << intensity_path
                << " with point count: " << pcl_wait_save_intensity->points.size() << RESET << std::endl;
    }
  }
}

void LIVMapper::run(ros::NodeHandle &nh)
{
  ros::Rate rate(5000);
  while (ros::ok())
  {
    ros::spinOnce();
    if (!sync_packages(LidarMeasures))
    {
      rate.sleep();
      continue;
    }
    handleFirstFrame();

    processImu();

    // if (!p_imu->imu_time_init) continue;

    stateEstimationAndMapping();
  }
  savePCD();
}

void LIVMapper::prop_imu_once(StatesGroup &imu_prop_state, const double dt, V3D acc_avr, V3D angvel_avr)
{
  double mean_acc_norm = p_imu->IMU_mean_acc_norm;
  acc_avr = acc_avr * G_m_s2 / mean_acc_norm - imu_prop_state.bias_a;
  angvel_avr -= imu_prop_state.bias_g;

  M3D Exp_f = Exp(angvel_avr, dt);
  /* propogation of IMU attitude */
  imu_prop_state.rot_end = imu_prop_state.rot_end * Exp_f;

  /* Specific acceleration (global frame) of IMU */
  V3D acc_imu = imu_prop_state.rot_end * acc_avr + V3D(imu_prop_state.gravity[0], imu_prop_state.gravity[1], imu_prop_state.gravity[2]);

  /* propogation of IMU */
  imu_prop_state.pos_end = imu_prop_state.pos_end + imu_prop_state.vel_end * dt + 0.5 * acc_imu * dt * dt;

  /* velocity of IMU */
  imu_prop_state.vel_end = imu_prop_state.vel_end + acc_imu * dt;
}

void LIVMapper::imu_prop_callback()
{
  if (p_imu->imu_need_init || !new_imu || !ekf_finish_once) { return; }
  mtx_buffer_imu_prop.lock();
  new_imu = false; // 闂佺鐭囬崘銊у幀 propagate 婵☆偆澧楅崹鐟邦啅婵傚憡鏅?IMU 婵☆偆澧楅崹鐟邦啅閸忚偐鈻旈柍褜鍓熼弫?
  if (imu_prop_enable && !prop_imu_buffer.empty())
  {
    static double last_t_from_lidar_end_time = 0;
    if (state_update_flg)
    {
      imu_propagate = latest_ekf_state;
      // drop all useless imu pkg
      while ((!prop_imu_buffer.empty() && stamp2Sec(prop_imu_buffer.front().header.stamp) < latest_ekf_time))
      {
        prop_imu_buffer.pop_front();
      }
      last_t_from_lidar_end_time = 0;
      for (int i = 0; i < prop_imu_buffer.size(); i++)
      {
        double t_from_lidar_end_time = stamp2Sec(prop_imu_buffer[i].header.stamp) - latest_ekf_time;
        double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
        // cout << "prop dt" << dt << ", " << t_from_lidar_end_time << ", " << last_t_from_lidar_end_time << endl;
        V3D acc_imu(prop_imu_buffer[i].linear_acceleration.x, prop_imu_buffer[i].linear_acceleration.y, prop_imu_buffer[i].linear_acceleration.z);
        V3D omg_imu(prop_imu_buffer[i].angular_velocity.x, prop_imu_buffer[i].angular_velocity.y, prop_imu_buffer[i].angular_velocity.z);
        prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
        last_t_from_lidar_end_time = t_from_lidar_end_time;
      }
      state_update_flg = false;
    }
    else
    {
      V3D acc_imu(newest_imu.linear_acceleration.x, newest_imu.linear_acceleration.y, newest_imu.linear_acceleration.z);
      V3D omg_imu(newest_imu.angular_velocity.x, newest_imu.angular_velocity.y, newest_imu.angular_velocity.z);
      double t_from_lidar_end_time = stamp2Sec(newest_imu.header.stamp) - latest_ekf_time;
      double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
      prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
      last_t_from_lidar_end_time = t_from_lidar_end_time;
    }

    V3D posi, vel_i;
    Eigen::Quaterniond q;
    posi = imu_propagate.pos_end;
    vel_i = imu_propagate.vel_end;
    q = Eigen::Quaterniond(imu_propagate.rot_end);
    imu_prop_odom.header.frame_id = "world";
    imu_prop_odom.header.stamp = newest_imu.header.stamp;
    imu_prop_odom.pose.pose.position.x = posi.x();
    imu_prop_odom.pose.pose.position.y = posi.y();
    imu_prop_odom.pose.pose.position.z = posi.z();
    imu_prop_odom.pose.pose.orientation.w = q.w();
    imu_prop_odom.pose.pose.orientation.x = q.x();
    imu_prop_odom.pose.pose.orientation.y = q.y();
    imu_prop_odom.pose.pose.orientation.z = q.z();
    imu_prop_odom.twist.twist.linear.x = vel_i.x();
    imu_prop_odom.twist.twist.linear.y = vel_i.y();
    imu_prop_odom.twist.twist.linear.z = vel_i.z();
    pubImuPropOdom.publish(imu_prop_odom);
  }
  mtx_buffer_imu_prop.unlock();
}

void LIVMapper::transformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud)
{
  PointCloudXYZI().swap(*trans_cloud);
  trans_cloud->reserve(input_cloud->size());
  for (size_t i = 0; i < input_cloud->size(); i++)
  {
    pcl::PointXYZINormal p_c = input_cloud->points[i];
    Eigen::Vector3d p(p_c.x, p_c.y, p_c.z);
    p = (rot * (extR * p + extT) + t);
    PointType pi;
    pi.x = p(0);
    pi.y = p(1);
    pi.z = p(2);
    pi.intensity = p_c.intensity;
    trans_cloud->points.push_back(pi);
  }
}

void LIVMapper::pointBodyToWorld(const PointType &pi, PointType &po)
{
  V3D p_body(pi.x, pi.y, pi.z);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po.x = p_global(0);
  po.y = p_global(1);
  po.z = p_global(2);
  po.intensity = pi.intensity;
}

template <typename T> void LIVMapper::pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
  V3D p_body(pi[0], pi[1], pi[2]);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po[0] = p_global(0);
  po[1] = p_global(1);
  po[2] = p_global(2);
}

template <typename T> Matrix<T, 3, 1> LIVMapper::pointBodyToWorld(const Matrix<T, 3, 1> &pi)
{
  V3D p(pi[0], pi[1], pi[2]);
  p = (_state.rot_end * (extR * p + extT) + _state.pos_end);
  Eigen::Matrix<T, 3, 1> po(p[0], p[1], p[2]);
  return po;
}

void LIVMapper::RGBpointBodyToWorld(PointType const *const pi, PointType *const po)
{
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void LIVMapper::standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
  if (!lidar_en) return;
  ROS_INFO_ONCE("Get standard PointCloud2 LiDAR, first header time: %.6f", stamp2Sec(msg->header.stamp));
  static size_t lidar_cb_count = 0;
  mtx_buffer.lock();
  // cout<<"got feature"<<endl;
  if (stamp2Sec(msg->header.stamp) < last_timestamp_lidar)
  {
    ROS_ERROR("lidar loop back, clear buffer");
    lid_raw_data_buffer.clear();
  }
  // ROS_INFO("get point cloud at time: %.6f", stamp2Sec(msg->header.stamp));
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  ++lidar_cb_count;
  if (!ptr || ptr->empty())
  {
    ROS_WARN("Standard LiDAR callback #%zu got an empty processed cloud. raw width=%u height=%u fields=%zu",
                lidar_cb_count, msg->width, msg->height, msg->fields.size());
  }
  lid_raw_data_buffer.push_back(ptr);
  lid_header_time_buffer.push_back(stamp2Sec(msg->header.stamp));
  last_timestamp_lidar = stamp2Sec(msg->header.stamp);
  if (lidar_cb_count == 1 || lidar_cb_count % 20 == 0)
  {
    ROS_INFO("Standard LiDAR callback #%zu: stamp=%.6f raw_points=%u processed_points=%zu lidar_buffer=%zu",
                lidar_cb_count, stamp2Sec(msg->header.stamp), msg->width * msg->height, ptr ? ptr->size() : 0, lid_raw_data_buffer.size());
  }

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void LIVMapper::livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg_in)
{
  if (!lidar_en) return;
  mtx_buffer.lock();
  livox_ros_driver::CustomMsg::Ptr msg(new livox_ros_driver::CustomMsg(*msg_in));
  // if ((abs(stamp2Sec(msg->header.stamp) - last_timestamp_lidar) > 0.2 && last_timestamp_lidar > 0) || sync_jump_flag)
  // {
  //   ROS_WARN("lidar jumps %.3f\n", stamp2Sec(msg->header.stamp) - last_timestamp_lidar);
  //   sync_jump_flag = true;
  //   msg->header.stamp = ros::Time().fromSec(last_timestamp_lidar + 0.1);
  // }
  if (abs(last_timestamp_imu - stamp2Sec(msg->header.stamp)) > 1.0 && !imu_buffer.empty())
  {
    double timediff_imu_wrt_lidar = last_timestamp_imu - stamp2Sec(msg->header.stamp);
    ROS_INFO("\033[95mSelf sync IMU and LiDAR, HARD time lag is %.10lf \n\033[0m", timediff_imu_wrt_lidar - 0.100);
    // imu_time_offset = timediff_imu_wrt_lidar;
  }

  double cur_head_time = stamp2Sec(msg->header.stamp);
  ROS_INFO("Get LiDAR, its header time: %.6f", cur_head_time);
  if (cur_head_time < last_timestamp_lidar)
  {
    ROS_ERROR("lidar loop back, clear buffer");
    lid_raw_data_buffer.clear();
  }
  ROS_INFO("get point cloud at time: %.6f", stamp2Sec(msg->header.stamp));
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);

  if (!ptr || ptr->empty()) {
    ROS_ERROR("Received an empty point cloud");
    mtx_buffer.unlock();
    return;
  }

  lid_raw_data_buffer.push_back(ptr);
  lid_header_time_buffer.push_back(cur_head_time);
  last_timestamp_lidar = cur_head_time;

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void LIVMapper::imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
  if (!imu_en) return;

  if (last_timestamp_lidar < 0.0) return;
  ROS_INFO("get imu at time: %.6f", stamp2Sec(msg_in->header.stamp));
  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
  msg->header.stamp = sec2Stamp(stamp2Sec(msg->header.stamp) - imu_time_offset);
  double timestamp = stamp2Sec(msg->header.stamp);

  if (fabs(last_timestamp_lidar - timestamp) > 0.5 && (!ros_driver_fix_en))
  {
    ROS_WARN("IMU and LiDAR not synced! delta time: %lf .\n", last_timestamp_lidar - timestamp);
  }

  if (ros_driver_fix_en) timestamp += std::round(last_timestamp_lidar - timestamp);
  msg->header.stamp = sec2Stamp(timestamp);

  mtx_buffer.lock();

  if (last_timestamp_imu > 0.0 && timestamp < last_timestamp_imu)
  {
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    ROS_ERROR("imu loop back, offset: %lf \n", last_timestamp_imu - timestamp);
    return;
  }

  if (last_timestamp_imu > 0.0 && timestamp > last_timestamp_imu + 0.2)
  {
    ROS_WARN("imu time stamp Jumps %0.4lf seconds \n", timestamp - last_timestamp_imu);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    return;
  }

  last_timestamp_imu = timestamp;

  imu_buffer.push_back(msg);
  // cout<<"got imu: "<<timestamp<<" imu size "<<imu_buffer.size()<<endl;
  mtx_buffer.unlock();
  if (imu_prop_enable)
  {
    mtx_buffer_imu_prop.lock();
    if (imu_prop_enable && !p_imu->imu_need_init) { prop_imu_buffer.push_back(*msg); }
    newest_imu = *msg;
    new_imu = true;
    mtx_buffer_imu_prop.unlock();
  }
  sig_buffer.notify_all();
}

cv::Mat LIVMapper::getImageFromMsg(const sensor_msgs::Image::ConstPtr &img_msg)
{
  const cv::Mat img = cv_bridge::toCvShare(img_msg, "bgr8")->image;
  return img.clone();
}

void LIVMapper::flushCompletedImageGroupsLocked()
{
  while (!pending_images.empty())
  {
    auto oldest = pending_images.begin();
    if (oldest->second.isComplete())
    {
      if (image_downclock_en &&
          (++multi_cam_downclock_counter % static_cast<uint64_t>(image_downclock_factor) != 0))
      {
        pending_images.erase(oldest);
        continue;
      }

      MultiCameraFrame frame;
      frame.stamp_ns = oldest->second.stamp_ns;
      frame.timestamp = oldest->second.timestamp;
      frame.frame_id = next_vio_frame_id++;
      frame.images = std::move(oldest->second.images);
      frame.image_stamp_ns = std::move(oldest->second.image_stamp_ns);
      frame.raw_timestamps = std::move(oldest->second.raw_timestamps);
      frame.corrected_timestamps = std::move(oldest->second.corrected_timestamps);
      frame.capture_timestamps = std::move(oldest->second.capture_timestamps);
      frame.td_used = std::move(oldest->second.td_used);
      frame.time_offset_group = std::move(oldest->second.time_offset_group);
      multi_cam_frame_buffer.push_back(std::move(frame));
      pending_images.erase(oldest);
      continue;
    }

    if (hasLaterCompleteImageGroupLocked(pending_images, oldest->first))
    {
      const uint64_t dropped_stamp = oldest->first;
      const std::string missing_camera_ids = formatMissingCameraIdsLocked(oldest->second);
      ROS_WARN("Drop stale incomplete multi-camera group stamp_ns=%llu, missing cameras=%s, because a later complete group exists",
                  static_cast<unsigned long long>(dropped_stamp), missing_camera_ids.c_str());
      pending_images.erase(oldest);
      continue;
    }

    break;
  }
}

void LIVMapper::handleImageFrame(int camera_id, const ros::Time &stamp, const cv::Mat &img_cur)
{
  if (!img_en) return;
  if (camera_id < 0 || camera_id >= num_cameras || img_cur.empty()) return;
  const uint64_t stamp_ns = static_cast<uint64_t>(stamp.sec) * 1000000000ULL + static_cast<uint64_t>(stamp.nanosec);
  const double raw_time = static_cast<double>(stamp_ns) * 1.0e-9;
  const int time_group = camera_configs[camera_id].time_offset_group;
  const double td_used = (time_group >= 0 && time_group < _state.num_time_offset_groups)
                             ? _state.time_offset[time_group]
                             : img_time_offset;
  const double image_time = raw_time + td_used;
  const double capture_time = image_time + exposure_time_init;
  if (last_timestamp_lidar < 0) return;
  if (std::fabs(image_time - last_timestamp_img_by_camera[camera_id]) < 1.0e-9) return;
  if (image_time < last_timestamp_img_by_camera[camera_id])
  {
    ROS_ERROR("camera_id=%d image loop back", camera_id);
    return;
  }

  const CameraInputConfig &camera_config = camera_configs[camera_id];
  cv::Mat image_for_sync = img_cur;
  if (camera_config.image_undistort_en)
  {
    if (camera_config.undistort_map_x.empty() || camera_config.undistort_map_y.empty())
    {
      ROS_ERROR("camera_id=%d input undistort map is not initialized", camera_id);
      return;
    }
    cv::Mat image_undistorted;
    cv::remap(img_cur, image_undistorted, camera_config.undistort_map_x, camera_config.undistort_map_y,
              cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
    image_for_sync = image_undistorted;
  }
  const uint64_t sync_tolerance_ns =
      static_cast<uint64_t>(std::llround(multi_cam_sync_tolerance_ms * 1.0e6));

  {
    std::lock_guard<std::mutex> lock(mtx_buffer);

    const auto initializeGroup = [this](PendingImageGroup &group, uint64_t group_stamp_ns,
                                        double group_timestamp) {
      if (!group.arrived.empty()) return;
      group.stamp_ns = group_stamp_ns;
      group.timestamp = group_timestamp;
      group.images.resize(num_cameras);
      group.arrived.assign(num_cameras, 0);
      group.image_stamp_ns.assign(num_cameras, 0);
      group.raw_timestamps.assign(num_cameras, 0.0);
      group.corrected_timestamps.assign(num_cameras, 0.0);
      group.capture_timestamps.assign(num_cameras, 0.0);
      group.td_used.assign(num_cameras, 0.0);
      group.time_offset_group.assign(num_cameras, 0);
    };
    const auto groupHasImages = [](const PendingImageGroup &group) {
      for (uint8_t arrived : group.arrived)
        if (arrived) return true;
      return false;
    };

    if (camera_id == 0)
    {
      auto anchor_it = pending_images.find(stamp_ns);
      if (anchor_it == pending_images.end())
        anchor_it = pending_images.emplace(stamp_ns, PendingImageGroup()).first;

      PendingImageGroup &anchor = anchor_it->second;
      initializeGroup(anchor, stamp_ns, image_time);
      if (anchor.arrived[0]) return;

      anchor.stamp_ns = stamp_ns;
      anchor.timestamp = std::max(anchor.timestamp, capture_time);
      anchor.images[0] = image_for_sync.clone();
      anchor.arrived[0] = 1;
      anchor.image_stamp_ns[0] = stamp_ns;
      anchor.raw_timestamps[0] = raw_time;
      anchor.corrected_timestamps[0] = image_time;
      anchor.capture_timestamps[0] = capture_time;
      anchor.td_used[0] = td_used;
      anchor.time_offset_group[0] = time_group;

      for (int matched_camera_id = 1; matched_camera_id < num_cameras; ++matched_camera_id)
      {
        if (anchor.arrived[matched_camera_id]) continue;

        auto best_it = pending_images.end();
        uint64_t best_delta_ns = std::numeric_limits<uint64_t>::max();
        for (auto it = pending_images.begin(); it != pending_images.end(); ++it)
        {
          if (it == anchor_it || it->second.arrived.empty() || it->second.arrived[0] ||
              !it->second.arrived[matched_camera_id])
            continue;

          const double candidate_time = it->second.corrected_timestamps[matched_camera_id];
          const uint64_t delta_ns = static_cast<uint64_t>(std::llround(std::fabs(image_time - candidate_time) * 1.0e9));
          if (delta_ns <= sync_tolerance_ns &&
              (delta_ns < best_delta_ns ||
               (delta_ns == best_delta_ns &&
                (best_it == pending_images.end() || candidate_time < best_it->second.corrected_timestamps[matched_camera_id]))))
          {
            best_it = it;
            best_delta_ns = delta_ns;
          }
        }

        if (best_it == pending_images.end()) continue;
        anchor.images[matched_camera_id] = std::move(best_it->second.images[matched_camera_id]);
        anchor.arrived[matched_camera_id] = 1;
        anchor.image_stamp_ns[matched_camera_id] = best_it->second.image_stamp_ns[matched_camera_id];
        anchor.raw_timestamps[matched_camera_id] = best_it->second.raw_timestamps[matched_camera_id];
        anchor.corrected_timestamps[matched_camera_id] = best_it->second.corrected_timestamps[matched_camera_id];
        anchor.capture_timestamps[matched_camera_id] = best_it->second.capture_timestamps[matched_camera_id];
        anchor.td_used[matched_camera_id] = best_it->second.td_used[matched_camera_id];
        anchor.time_offset_group[matched_camera_id] = best_it->second.time_offset_group[matched_camera_id];
        anchor.timestamp = std::max(anchor.timestamp, anchor.capture_timestamps[matched_camera_id]);
        best_it->second.arrived[matched_camera_id] = 0;
        best_it->second.image_stamp_ns[matched_camera_id] = 0;
      }

      for (auto it = pending_images.begin(); it != pending_images.end();)
      {
        if (it != anchor_it && !groupHasImages(it->second))
          it = pending_images.erase(it);
        else
          ++it;
      }
    }
    else
    {
      const auto findClosestGroup = [&](bool require_camera0) {
        auto best_it = pending_images.end();
        uint64_t best_delta_ns = std::numeric_limits<uint64_t>::max();
        double best_reference_time = 0.0;
        for (auto it = pending_images.begin(); it != pending_images.end(); ++it)
        {
          PendingImageGroup &candidate = it->second;
          if (candidate.arrived.empty() ||
              static_cast<bool>(candidate.arrived[0]) != require_camera0 ||
              candidate.arrived[camera_id])
            continue;

          double reference_time = 0.0;
          uint64_t delta_ns = std::numeric_limits<uint64_t>::max();
          if (require_camera0)
          {
            reference_time = candidate.corrected_timestamps[0];
            delta_ns = static_cast<uint64_t>(std::llround(std::fabs(image_time - reference_time) * 1.0e9));
          }
          else
          {
            for (int ref_camera_id = 1; ref_camera_id < num_cameras; ++ref_camera_id)
            {
              if (!candidate.arrived[ref_camera_id]) continue;
              const double ref_time = candidate.corrected_timestamps[ref_camera_id];
              const uint64_t ref_delta_ns =
                  static_cast<uint64_t>(std::llround(std::fabs(image_time - ref_time) * 1.0e9));
              if (ref_delta_ns < delta_ns ||
                  (ref_delta_ns == delta_ns && ref_time < reference_time))
              {
                delta_ns = ref_delta_ns;
                reference_time = ref_time;
              }
            }
          }
          if (delta_ns <= sync_tolerance_ns &&
              (delta_ns < best_delta_ns ||
               (delta_ns == best_delta_ns &&
                (best_it == pending_images.end() || reference_time < best_reference_time))))
          {
            best_it = it;
            best_delta_ns = delta_ns;
            best_reference_time = reference_time;
          }
        }
        return best_it;
      };

      auto group_it = findClosestGroup(true);
      if (group_it == pending_images.end()) group_it = findClosestGroup(false);
      if (group_it == pending_images.end())
      {
        group_it = pending_images.find(stamp_ns);
        if (group_it == pending_images.end())
          group_it = pending_images.emplace(stamp_ns, PendingImageGroup()).first;
      }

      PendingImageGroup &group = group_it->second;
      initializeGroup(group, group_it->first, image_time);
      if (group.arrived[camera_id]) return;
      group.images[camera_id] = image_for_sync.clone();
      group.arrived[camera_id] = 1;
      group.image_stamp_ns[camera_id] = stamp_ns;
      group.raw_timestamps[camera_id] = raw_time;
      group.corrected_timestamps[camera_id] = image_time;
      group.capture_timestamps[camera_id] = capture_time;
      group.td_used[camera_id] = td_used;
      group.time_offset_group[camera_id] = time_group;
      group.timestamp = std::max(group.timestamp, capture_time);
    }

    last_timestamp_img_by_camera[camera_id] = image_time;
    flushCompletedImageGroupsLocked();
    while (static_cast<int>(pending_images.size()) > multi_cam_sync_queue_size)
    {
      auto oldest = pending_images.begin();
      const uint64_t dropped_stamp = oldest->first;
      const std::string missing_camera_ids = formatMissingCameraIdsLocked(oldest->second);
      pending_images.erase(oldest);
      ROS_WARN("Drop incomplete multi-camera group stamp_ns=%llu due to pending queue overflow, missing cameras=%s",
                  static_cast<unsigned long long>(dropped_stamp), missing_camera_ids.c_str());
      flushCompletedImageGroupsLocked();
    }
  }
  ROS_INFO("Get camera_id=%d image, header time %.6f", camera_id, image_time);
  sig_buffer.notify_all();
}
void LIVMapper::img_cbk(int camera_id, const sensor_msgs::Image::ConstPtr &msg_in)
{
  cv::Mat img_cur = getImageFromMsg(msg_in);
  handleImageFrame(camera_id, msg_in->header.stamp, img_cur);
}

void LIVMapper::compressed_img_cbk(int camera_id, const sensor_msgs::CompressedImage::ConstPtr &msg_in)
{
  if (!img_en) return;
  const cv::Mat encoded(1, static_cast<int>(msg_in->data.size()), CV_8UC1, const_cast<uint8_t *>(msg_in->data.data()));
  cv::Mat img_cur = cv::imdecode(encoded, cv::IMREAD_COLOR);
  if (img_cur.empty())
  {
    ROS_WARN("Failed to decode camera_id=%d compressed image at %.6f, format=%s, bytes=%zu",
                camera_id, stamp2Sec(msg_in->header.stamp), msg_in->format.c_str(), msg_in->data.size());
    return;
  }
  handleImageFrame(camera_id, msg_in->header.stamp, img_cur);
}

bool LIVMapper::sync_packages(LidarMeasureGroup &meas)
{
  if ((lid_raw_data_buffer.empty() && lidar_en) || (multi_cam_frame_buffer.empty() && img_en) || (imu_buffer.empty() && imu_en))
  {
    static size_t wait_log_count = 0;
    if (++wait_log_count % 5000 == 0)
    {
      ROS_INFO("Waiting sync buffers: lidar=%zu image=%zu imu=%zu (enabled lidar=%d image=%d imu=%d)",
                  lid_raw_data_buffer.size(), multi_cam_frame_buffer.size(), imu_buffer.size(), lidar_en, img_en, imu_en);
    }
    return false;
  }

  switch (slam_mode_)
  {
  case ONLY_LIO:
  {
    if (meas.last_lio_update_time < 0.0) meas.last_lio_update_time = lid_header_time_buffer.front();
    if (!lidar_pushed)
    {
      // If not push the lidar into measurement data buffer
      meas.lidar = lid_raw_data_buffer.front(); // push the first lidar topic
      if (meas.lidar->points.size() <= 1) return false;

      meas.lidar_frame_beg_time = lid_header_time_buffer.front();                                                // generate lidar_frame_beg_time
      meas.lidar_frame_end_time = meas.lidar_frame_beg_time + meas.lidar->points.back().curvature / double(1000); // calc lidar scan end time
      meas.pcl_proc_cur = meas.lidar;
      lidar_pushed = true;                                                                                       // flag
    }

    if (imu_en && last_timestamp_imu < meas.lidar_frame_end_time)
    { // waiting imu message needs to be
      // larger than _lidar_frame_end_time,
      // make sure complete propagate.
      // ROS_ERROR("out sync");
      return false;
    }

    struct MeasureGroup m; // standard method to keep imu message.

    m.imu.clear();
    m.lio_time = meas.lidar_frame_end_time;
    mtx_buffer.lock();
    while (!imu_buffer.empty())
    {
      if (stamp2Sec(imu_buffer.front()->header.stamp) > meas.lidar_frame_end_time) break;
      m.imu.push_back(imu_buffer.front());
      imu_buffer.pop_front();
    }
    lid_raw_data_buffer.pop_front();
    lid_header_time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();

    meas.lio_vio_flg = LIO; // process lidar topic, so timestamp should be lidar scan end.
    meas.measures.push_back(m);
    // ROS_INFO("ONlY HAS LiDAR and IMU, NO IMAGE!");
    lidar_pushed = false; // sync one whole lidar scan.
    return true;

    break;
  }

  case LIVO:
  {
    /*** For LIVO mode, the time of LIO update is set to be the same as VIO, LIO
     * first than VIO imediatly ***/
    EKF_STATE last_lio_vio_flg = meas.lio_vio_flg;
    // double t0 = omp_get_wtime();
    switch (last_lio_vio_flg)
    {
    // double img_capture_time = meas.lidar_frame_beg_time + exposure_time_init;
    case WAIT:
    case VIO:
    {
      // printf("!!! meas.lio_vio_flg: %d \n", meas.lio_vio_flg);
      double img_capture_time = multi_cam_frame_buffer.front().timestamp;
      /*** has img topic, but img topic timestamp larger than lidar end time,
       * process lidar topic. After LIO update, the meas.lidar_frame_end_time
       * will be refresh. ***/
      if (meas.last_lio_update_time < 0.0) meas.last_lio_update_time = lid_header_time_buffer.front();
      // printf("[ Data Cut ] wait \n");
      // printf("[ Data Cut ] last_lio_update_time: %lf \n",
      // meas.last_lio_update_time);

      double lid_newest_time = lid_header_time_buffer.back() + lid_raw_data_buffer.back()->points.back().curvature / double(1000);
      double imu_newest_time = stamp2Sec(imu_buffer.back()->header.stamp);

      if (img_capture_time < meas.last_lio_update_time + 0.00001)
      {
        multi_cam_frame_buffer.pop_front();
        ROS_ERROR("[ Data Cut ] Throw one image frame! \n");
        return false;
      }

      if (img_capture_time > lid_newest_time || img_capture_time > imu_newest_time)
      {
        // ROS_ERROR("lost first camera frame");
        // printf("img_capture_time, lid_newest_time, imu_newest_time: %lf , %lf
        // , %lf \n", img_capture_time, lid_newest_time, imu_newest_time);
        return false;
      }

      struct MeasureGroup m;

      // printf("[ Data Cut ] LIO \n");
      // printf("[ Data Cut ] img_capture_time: %lf \n", img_capture_time);
      m.imu.clear();
      m.lio_time = img_capture_time;
      mtx_buffer.lock();
      while (!imu_buffer.empty())
      {
        if (stamp2Sec(imu_buffer.front()->header.stamp) > m.lio_time) break;

        if (stamp2Sec(imu_buffer.front()->header.stamp) > meas.last_lio_update_time) m.imu.push_back(imu_buffer.front());

        imu_buffer.pop_front();
        // printf("[ Data Cut ] imu time: %lf \n",
        // stamp2Sec(imu_buffer.front()->header.stamp));
      }
      mtx_buffer.unlock();
      sig_buffer.notify_all();

      *(meas.pcl_proc_cur) = *(meas.pcl_proc_next);
      PointCloudXYZI().swap(*meas.pcl_proc_next);

      int lid_frame_num = lid_raw_data_buffer.size();
      int max_size = meas.pcl_proc_cur->size() + 24000 * lid_frame_num;
      meas.pcl_proc_cur->reserve(max_size);
      meas.pcl_proc_next->reserve(max_size);
      // deque<PointCloudXYZI::Ptr> lidar_buffer_tmp;

      while (!lid_raw_data_buffer.empty())
      {
        if (lid_header_time_buffer.front() > img_capture_time) break;
        auto pcl(lid_raw_data_buffer.front()->points);
        double frame_header_time(lid_header_time_buffer.front());
        float max_offs_time_ms = (m.lio_time - frame_header_time) * 1000.0f;

        for (int i = 0; i < pcl.size(); i++)
        {
          auto pt = pcl[i];
          if (pcl[i].curvature < max_offs_time_ms)
          {
            pt.curvature += (frame_header_time - meas.last_lio_update_time) * 1000.0f;
            meas.pcl_proc_cur->points.push_back(pt);
          }
          else
          {
            pt.curvature += (frame_header_time - m.lio_time) * 1000.0f;
            meas.pcl_proc_next->points.push_back(pt);
          }
        }
        lid_raw_data_buffer.pop_front();
        lid_header_time_buffer.pop_front();
      }

      meas.measures.push_back(m);
      meas.lio_vio_flg = LIO;
      // meas.last_lio_update_time = m.lio_time;
      // printf("!!! meas.lio_vio_flg: %d \n", meas.lio_vio_flg);
      // printf("[ Data Cut ] pcl_proc_cur number: %d \n", meas.pcl_proc_cur
      // ->points.size()); printf("[ Data Cut ] LIO process time: %lf \n",
      // omp_get_wtime() - t0);
      return true;
    }

    case LIO:
    {
      double img_capture_time = multi_cam_frame_buffer.front().timestamp;
      meas.lio_vio_flg = VIO;
      // printf("[ Data Cut ] VIO \n");
      meas.measures.clear();
      double imu_time = stamp2Sec(imu_buffer.front()->header.stamp);

      struct MeasureGroup m;
      m.vio_time = img_capture_time;
      m.lio_time = meas.last_lio_update_time;
      m.multi_cam_frame = multi_cam_frame_buffer.front();
      m.has_multi_cam_frame = true;
      m.img = m.multi_cam_frame.images[0];
      m.img_time = m.multi_cam_frame.timestamp;
      mtx_buffer.lock();
      // while ((!imu_buffer.empty() && (imu_time < img_capture_time)))
      // {
      //   imu_time = stamp2Sec(imu_buffer.front()->header.stamp);
      //   if (imu_time > img_capture_time) break;
      //   m.imu.push_back(imu_buffer.front());
      //   imu_buffer.pop_front();
      //   printf("[ Data Cut ] imu time: %lf \n",
      //   stamp2Sec(imu_buffer.front()->header.stamp));
      // }
      multi_cam_frame_buffer.pop_front();
      mtx_buffer.unlock();
      sig_buffer.notify_all();
      meas.measures.push_back(m);
      lidar_pushed = false; // after VIO update, the _lidar_frame_end_time will be refresh.
      // printf("[ Data Cut ] VIO process time: %lf \n", omp_get_wtime() - t0);
      return true;
    }

    default:
    {
      // printf("!! WRONG EKF STATE !!");
      return false;
    }
      // return false;
    }
    break;
  }

  case ONLY_LO:
  {
    if (!lidar_pushed)
    {
      // If not in lidar scan, need to generate new meas
      if (lid_raw_data_buffer.empty())  return false;
      meas.lidar = lid_raw_data_buffer.front(); // push the first lidar topic
      meas.lidar_frame_beg_time = lid_header_time_buffer.front(); // generate lidar_beg_time
      meas.lidar_frame_end_time  = meas.lidar_frame_beg_time + meas.lidar->points.back().curvature / double(1000); // calc lidar scan end time
      lidar_pushed = true;
    }
    struct MeasureGroup m; // standard method to keep imu message.
    m.lio_time = meas.lidar_frame_end_time;
    mtx_buffer.lock();
    lid_raw_data_buffer.pop_front();
    lid_header_time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    lidar_pushed = false; // sync one whole lidar scan.
    meas.lio_vio_flg = LO; // process lidar topic, so timestamp should be lidar scan end.
    meas.measures.push_back(m);
    return true;
    break;
  }

  default:
  {
    printf("!! WRONG SLAM TYPE !!");
    return false;
  }
  }
  ROS_ERROR("out sync");
}

void LIVMapper::publish_img_rgb(VIOManagerPtr vio_manager)
{
  const int count = std::min(static_cast<int>(pubImages.size()), vio_manager->numCameras());
  for (int camera_id = 0; camera_id < count; ++camera_id)
  {
    const cv::Mat &image = vio_manager->cameras_[camera_id].img_cp;
    if (image.empty()) continue;
    cv_bridge::CvImage out_msg;
    out_msg.header.stamp = ros::Time::now();
    out_msg.header.frame_id = "camera_" + std::to_string(camera_id);
    out_msg.encoding = sensor_msgs::image_encodings::BGR8;
    out_msg.image = image;
    pubImages[camera_id].publish(out_msg.toImageMsg());
  }
}

void LIVMapper::publish_optical_flow_image(const image_transport::Publisher &pubImage, VIOManagerPtr vio_manager)
{
  cv::Mat img_debug = vio_manager->optical_flow_debug_img;
  if (img_debug.empty()) return;
  cv_bridge::CvImage out_msg;
  out_msg.header.stamp = ros::Time::now();
  out_msg.header.frame_id = "camera";
  out_msg.encoding = sensor_msgs::image_encodings::BGR8;
  out_msg.image = img_debug;
  pubImage.publish(out_msg.toImageMsg());
}

void LIVMapper::publish_triangulated_points(const ros::Publisher &pubCloud,
                                            const PointCloudXYZI::Ptr &cloud)
{
  if (!pubCloud || cloud == nullptr || cloud->empty()) return;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(*cloud, cloud_msg);
  cloud_msg.header.stamp = ros::Time::now();
  cloud_msg.header.frame_id = "camera_init";
  pubCloud.publish(cloud_msg);
}

void LIVMapper::publish_frame_world(const ros::Publisher &pubLaserCloudFullRes, VIOManagerPtr vio_manager)
{
  if (pcl_w_wait_pub->empty()) return;
  PointCloudXYZRGB::Ptr laserCloudWorldRGB(new PointCloudXYZRGB());
  PointCloudXYZI::Ptr intensity_to_publish(new PointCloudXYZI());
  static int pub_num = 1;
  if (img_en)
  {
    *pcl_wait_pub += *pcl_w_wait_pub;
    PointCloudXYZI().swap(*pcl_w_wait_pub);
    if (pub_num < pub_scan_num)
    {
      ++pub_num;
      return;
    }
    pub_num = 1;

    const int camera_count = vio_manager->numCameras();
    const long long point_count = static_cast<long long>(pcl_wait_pub->size());
    const int color_worker_count = std::max(1, std::min(MP_PROC_NUM, omp_get_max_threads()));
    int active_color_workers = 1;

    std::vector<uint8_t> colored(pcl_wait_pub->size(), 0);
    std::vector<PointTypeRGB> rgb_points(pcl_wait_pub->size());
    std::vector<size_t> projection_rejects(camera_count, 0);
    std::vector<size_t> blind_rejects(camera_count, 0);
    std::vector<size_t> color_hits(camera_count, 0);
    std::vector<double> sampled_r_sum(camera_count, 0.0);
    std::vector<double> sampled_g_sum(camera_count, 0.0);
    std::vector<double> sampled_b_sum(camera_count, 0.0);

    std::vector<std::vector<size_t>> thread_projection_rejects(
        color_worker_count, std::vector<size_t>(camera_count, 0));
    std::vector<std::vector<size_t>> thread_blind_rejects(
        color_worker_count, std::vector<size_t>(camera_count, 0));
    std::vector<std::vector<size_t>> thread_color_hits(
        color_worker_count, std::vector<size_t>(camera_count, 0));
    std::vector<std::vector<double>> thread_r_sum(
        color_worker_count, std::vector<double>(camera_count, 0.0));
    std::vector<std::vector<double>> thread_g_sum(
        color_worker_count, std::vector<double>(camera_count, 0.0));
    std::vector<std::vector<double>> thread_b_sum(
        color_worker_count, std::vector<double>(camera_count, 0.0));

    // Each point is independent; camera order stays serial per point to preserve first-hit priority.
    const double color_start = omp_get_wtime();
#pragma omp parallel num_threads(color_worker_count) if(point_count >= 1024)
    {
#pragma omp single
      active_color_workers = omp_get_num_threads();

      const int thread_id = omp_get_thread_num();
      std::vector<size_t> &local_projection_rejects = thread_projection_rejects[thread_id];
      std::vector<size_t> &local_blind_rejects = thread_blind_rejects[thread_id];
      std::vector<size_t> &local_color_hits = thread_color_hits[thread_id];
      std::vector<double> &local_r_sum = thread_r_sum[thread_id];
      std::vector<double> &local_g_sum = thread_g_sum[thread_id];
      std::vector<double> &local_b_sum = thread_b_sum[thread_id];

#pragma omp for schedule(static)
      for (long long point_index = 0; point_index < point_count; ++point_index)
      {
        const size_t i = static_cast<size_t>(point_index);
        const PointType &point = pcl_wait_pub->points[i];

        for (int camera_id = 0; camera_id < camera_count; ++camera_id)
        {
          V3F bgr;
          double camera_range = 0.0;
          if (!vio_manager->getColorFromCamera(
                  camera_id, V3D(point.x, point.y, point.z), bgr, &camera_range))
          {
            ++local_projection_rejects[camera_id];
            continue;
          }
          if (camera_range < blind_rgb_points)
          {
            ++local_blind_rejects[camera_id];
            continue;
          }

          PointTypeRGB output;
          output.x = point.x;
          output.y = point.y;
          output.z = point.z;
          output.r = static_cast<uint8_t>(std::clamp(bgr[2], 0.0f, 255.0f));
          output.g = static_cast<uint8_t>(std::clamp(bgr[1], 0.0f, 255.0f));
          output.b = static_cast<uint8_t>(std::clamp(bgr[0], 0.0f, 255.0f));
          rgb_points[i] = output;
          colored[i] = 1;
          ++local_color_hits[camera_id];
          local_r_sum[camera_id] += output.r;
          local_g_sum[camera_id] += output.g;
          local_b_sum[camera_id] += output.b;
          break;
        }
      }
    }

    for (int thread_id = 0; thread_id < color_worker_count; ++thread_id)
    {
      for (int camera_id = 0; camera_id < camera_count; ++camera_id)
      {
        projection_rejects[camera_id] += thread_projection_rejects[thread_id][camera_id];
        blind_rejects[camera_id] += thread_blind_rejects[thread_id][camera_id];
        color_hits[camera_id] += thread_color_hits[thread_id][camera_id];
        sampled_r_sum[camera_id] += thread_r_sum[thread_id][camera_id];
        sampled_g_sum[camera_id] += thread_g_sum[thread_id][camera_id];
        sampled_b_sum[camera_id] += thread_b_sum[thread_id][camera_id];
      }
    }
    const double color_elapsed = omp_get_wtime() - color_start;
    laserCloudWorldRGB->reserve(pcl_wait_pub->size());
    for (size_t i = 0; i < rgb_points.size(); ++i)
      if (colored[i]) laserCloudWorldRGB->push_back(rgb_points[i]);
    const bool has_camera_color = !laserCloudWorldRGB->empty();
    printf("[ VIO Color ] input=%zu colored=%zu retained=%zu output_schema=rgb mode=%s\n",
           pcl_wait_pub->size(),
           static_cast<size_t>(std::count(colored.begin(), colored.end(), static_cast<uint8_t>(1))),
           has_camera_color ? 0UL : pcl_wait_pub->size(),
           has_camera_color ? "camera_color" : "empty_rgb_retry");
    printf("[ VIO Color Timing ] elapsed=%.6f s workers=%d points=%lld cameras=%d\n",
           color_elapsed, active_color_workers, point_count, camera_count);
    for (int camera_id = 0; camera_id < vio_manager->numCameras(); ++camera_id)
    {
      const double hit_count = static_cast<double>(color_hits[camera_id]);
      const double mean_r = hit_count > 0.0 ? sampled_r_sum[camera_id] / hit_count : 0.0;
      const double mean_g = hit_count > 0.0 ? sampled_g_sum[camera_id] / hit_count : 0.0;
      const double mean_b = hit_count > 0.0 ? sampled_b_sum[camera_id] / hit_count : 0.0;
      printf("[ VIO Color ] camera_id=%d hits=%zu projection_rejects=%zu blind_rejects=%zu sampled_rgb_mean=(%.1f,%.1f,%.1f)\n",
             camera_id, color_hits[camera_id], projection_rejects[camera_id], blind_rejects[camera_id],
             mean_r, mean_g, mean_b);
    }
  }
  else
  {
    *intensity_to_publish = *pcl_w_wait_pub;
  }

  sensor_msgs::PointCloud2 laserCloudmsg;
  const bool has_camera_color = img_en && !laserCloudWorldRGB->empty();
  // Keep the PointCloud2 field layout stable for RViz's RGB8 transformer on this topic.
  if (img_en) pcl::toROSMsg(*laserCloudWorldRGB, laserCloudmsg);
  else pcl::toROSMsg(*intensity_to_publish, laserCloudmsg);
  laserCloudmsg.header.stamp = ros::Time::now();
  laserCloudmsg.header.frame_id = "camera_init";
  pubLaserCloudFullRes.publish(laserCloudmsg);

  if (pcd_save_en)
  {
    static int scan_wait_num = 0;
    if (img_en)
    {
      if (has_camera_color) *pcl_wait_save += *laserCloudWorldRGB;
    }
    else
    {
      *pcl_wait_save_intensity += *intensity_to_publish;
    }
    ++scan_wait_num;
    if ((pcl_wait_save->size() > 0 || pcl_wait_save_intensity->size() > 0) && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
    {
      ++pcd_index;
      const string all_points_base = string(ROOT_DIR) + "Log/PCD/" + to_string(pcd_index);
      pcl::PCDWriter pcd_writer;
      const bool has_rgb = !pcl_wait_save->empty();
      const bool has_intensity = !pcl_wait_save_intensity->empty();
      if (has_rgb)
        pcd_writer.writeBinary(all_points_base + (has_intensity ? "_rgb.pcd" : ".pcd"), *pcl_wait_save);
      if (has_intensity)
        pcd_writer.writeBinary(all_points_base + (has_rgb ? "_intensity.pcd" : ".pcd"), *pcl_wait_save_intensity);
      PointCloudXYZRGB().swap(*pcl_wait_save);
      PointCloudXYZI().swap(*pcl_wait_save_intensity);
      Eigen::Quaterniond q(_state.rot_end);
      fout_pcd_pos << _state.pos_end.transpose() << " " << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << endl;
      scan_wait_num = 0;
    }
  }
  if (!img_en || has_camera_color) PointCloudXYZI().swap(*pcl_wait_pub);
  PointCloudXYZI().swap(*pcl_w_wait_pub);
}

void LIVMapper::publish_visual_sub_map(const ros::Publisher &pubSubVisualMap)
{
  PointCloudXYZI::Ptr laserCloudFullRes(visual_sub_map);
  int size = laserCloudFullRes->points.size(); if (size == 0) return;
  PointCloudXYZI::Ptr sub_pcl_visual_map_pub(new PointCloudXYZI());
  *sub_pcl_visual_map_pub = *laserCloudFullRes;
  if (1)
  {
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*sub_pcl_visual_map_pub, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time::now();
    laserCloudmsg.header.frame_id = "camera_init";
    pubSubVisualMap.publish(laserCloudmsg);
  }
}

void LIVMapper::publish_effect_world(const ros::Publisher &pubLaserCloudEffect, const std::vector<PointToPlane> &ptpl_list)
{
  int effect_feat_num = ptpl_list.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effect_feat_num, 1));
  for (int i = 0; i < effect_feat_num; i++)
  {
    laserCloudWorld->points[i].x = ptpl_list[i].point_w_[0];
    laserCloudWorld->points[i].y = ptpl_list[i].point_w_[1];
    laserCloudWorld->points[i].z = ptpl_list[i].point_w_[2];
  }
  sensor_msgs::PointCloud2 laserCloudFullRes3;
  pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
  laserCloudFullRes3.header.stamp = ros::Time::now();
  laserCloudFullRes3.header.frame_id = "camera_init";
  pubLaserCloudEffect.publish(laserCloudFullRes3);
}

template <typename T> void LIVMapper::set_posestamp(T &out)
{
  out.position.x = _state.pos_end(0);
  out.position.y = _state.pos_end(1);
  out.position.z = _state.pos_end(2);
  out.orientation.x = geoQuat.x;
  out.orientation.y = geoQuat.y;
  out.orientation.z = geoQuat.z;
  out.orientation.w = geoQuat.w;
}

void LIVMapper::publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
  odomAftMapped.header.frame_id = "camera_init";
  odomAftMapped.child_frame_id = "aft_mapped";
  odomAftMapped.header.stamp = ros::Time::now(); //.ros::Time()fromSec(last_timestamp_lidar);
  set_posestamp(odomAftMapped.pose.pose);

  static tf2_ros::TransformBroadcaster br;
  tf2::Transform transform;
  tf2::Quaternion q;
  transform.setOrigin(tf2::Vector3(_state.pos_end(0), _state.pos_end(1), _state.pos_end(2)));
  q.setW(geoQuat.w);
  q.setX(geoQuat.x);
  q.setY(geoQuat.y);
  q.setZ(geoQuat.z);
  transform.setRotation(q);
  br.sendTransform(createTransformStamped(transform, odomAftMapped.header.stamp, "camera_init", "aft_mapped"));
  pubOdomAftMapped.publish(odomAftMapped);
}

void LIVMapper::publish_mavros(const ros::Publisher &mavros_pose_publisher)
{
  msg_body_pose.header.stamp = ros::Time::now();
  msg_body_pose.header.frame_id = "camera_init";
  set_posestamp(msg_body_pose.pose);
  mavros_pose_publisher.publish(msg_body_pose);
}

void LIVMapper::publish_path(const ros::Publisher &pubPath)
{
  set_posestamp(msg_body_pose.pose);
  msg_body_pose.header.stamp = ros::Time::now();
  msg_body_pose.header.frame_id = "camera_init";
  path.poses.push_back(msg_body_pose);
  pubPath.publish(path);
}
