#include "LIVMapper_multi_cam.h"
#include "memory_stats.h"

#include <mutex>
#include <string>

namespace
{
using memory_stats::Snapshot;

// These helpers count only dynamic members of embedded values. Their headers,
// fixed-size Eigen storage, and pointer handles belong to the containing object.
void collectState(Snapshot &snapshot, const std::string &key, const StatesGroup &state)
{
  snapshot.eigen(key + ".covariance", state.cov);
  snapshot.eigen(key + ".inverse_exposure", state.inv_expo_time);
  snapshot.eigen(key + ".time_offset", state.time_offset);
  snapshot.eigen(key + ".time_offset_prior", state.time_offset_prior);
  snapshot.vector(key + ".Rcl", state.Rcl);
  snapshot.vector(key + ".Pcl", state.Pcl);
  snapshot.vector(key + ".Rcl_prior", state.Rcl_prior);
  snapshot.vector(key + ".Pcl_prior", state.Pcl_prior);
}

template <typename PointT>
void collectCloudStorage(Snapshot &snapshot, const std::string &key, const pcl::PointCloud<PointT> &cloud)
{
  snapshot.vector(key + ".points", cloud.points);
  snapshot.text(key + ".frame_id", cloud.header.frame_id);
}

template <typename CloudPtr>
void collectCloud(Snapshot &snapshot, const std::string &key, const CloudPtr &cloud)
{
  if (cloud && snapshot.object(key + ".object", cloud.get(), sizeof(*cloud)))
    collectCloudStorage(snapshot, key, *cloud);
}

void collectImuMessage(Snapshot &snapshot, const std::string &key, const sensor_msgs::Imu::ConstPtr &message)
{
  if (message && snapshot.object(key + ".object", message.get(), sizeof(*message)))
    snapshot.text(key + ".frame_id", message->header.frame_id);
}

void collectFrame(Snapshot &snapshot, const std::string &key, const MultiCameraFrame &frame)
{
  snapshot.vector(key + ".image_headers", frame.images);
  for (const cv::Mat &image : frame.images) snapshot.mat(key + ".image_pixels", image);
  snapshot.vector(key + ".image_stamps", frame.image_stamp_ns);
  snapshot.vector(key + ".raw_timestamps", frame.raw_timestamps);
  snapshot.vector(key + ".corrected_timestamps", frame.corrected_timestamps);
  snapshot.vector(key + ".capture_timestamps", frame.capture_timestamps);
  snapshot.vector(key + ".time_offsets_used", frame.td_used);
  snapshot.vector(key + ".time_offset_groups", frame.time_offset_group);
}

void collectPendingFrame(Snapshot &snapshot, const PendingImageGroup &frame)
{
  const std::string key = "input.pending_images";
  snapshot.vector(key + ".image_headers", frame.images);
  for (const cv::Mat &image : frame.images) snapshot.mat(key + ".image_pixels", image);
  snapshot.vector(key + ".arrived", frame.arrived);
  snapshot.vector(key + ".image_stamps", frame.image_stamp_ns);
  snapshot.vector(key + ".raw_timestamps", frame.raw_timestamps);
  snapshot.vector(key + ".corrected_timestamps", frame.corrected_timestamps);
  snapshot.vector(key + ".capture_timestamps", frame.capture_timestamps);
  snapshot.vector(key + ".time_offsets_used", frame.td_used);
  snapshot.vector(key + ".time_offset_groups", frame.time_offset_group);
}

void collectMeasure(Snapshot &snapshot, const MeasureGroup &measure)
{
  snapshot.sequence("input.active_measure.imu_handles", measure.imu);
  for (const auto &message : measure.imu)
    collectImuMessage(snapshot, "input.active_measure.imu_messages", message);
  snapshot.mat("input.active_measure.image_pixels", measure.img);
  collectFrame(snapshot, "input.active_measure.camera_frame", measure.multi_cam_frame);
}

void collectPointUncertainty(Snapshot &snapshot, const std::vector<pointWithVar> &points)
{
  // Iterate size(), never capacity(): spare vector slots do not contain objects.
  for (const pointWithVar &point : points)
    if (point.scan_uncertainty)
      snapshot.object("lidar.scan_uncertainty", point.scan_uncertainty.get(), sizeof(*point.scan_uncertainty));
}

void collectTree(Snapshot &snapshot, const VoxelOctoTree *tree)
{
  if (!tree || !snapshot.object("lidar.map.octree_nodes", tree, sizeof(*tree))) return;
  snapshot.vector("lidar.map.node_layer_thresholds", tree->layer_init_num_);
  snapshot.vector("lidar.map.temp_points", tree->temp_points_);
  collectPointUncertainty(snapshot, tree->temp_points_);
  if (tree->plane_ptr_) snapshot.object("lidar.map.planes", tree->plane_ptr_, sizeof(*tree->plane_ptr_));
  for (const VoxelOctoTree *child : tree->leaves_) collectTree(snapshot, child);
}

void collectVoxelManager(Snapshot &snapshot, const VoxelMapManager &manager)
{
  snapshot.object("lidar.manager.object", &manager, sizeof(manager));
  snapshot.vector("lidar.manager.layer_thresholds", manager.config_setting_.layer_init_num_);
  snapshot.map("lidar.map.index", manager.voxel_map_);
  for (const auto &entry : manager.voxel_map_) collectTree(snapshot, entry.second);
  collectCloud(snapshot, "lidar.cloud.undistorted", manager.feats_undistort_);
  collectCloud(snapshot, "lidar.cloud.down_body", manager.feats_down_body_);
  collectCloud(snapshot, "lidar.cloud.down_world", manager.feats_down_world_);
  collectState(snapshot, "lidar.state", manager.state_);
  snapshot.vector("lidar.workspace.cross_matrices", manager.cross_mat_list_);
  snapshot.vector("lidar.workspace.body_covariances", manager.body_cov_list_);
  snapshot.vector("lidar.workspace.points_with_variance", manager.pv_list_);
  collectPointUncertainty(snapshot, manager.pv_list_);
  snapshot.vector("lidar.workspace.point_to_plane", manager.ptpl_list_);
}

void collectPreprocess(Snapshot &snapshot, const Preprocess &preprocess)
{
  snapshot.object("preprocess.object", &preprocess, sizeof(preprocess));
  collectCloudStorage(snapshot, "preprocess.cloud.full", preprocess.pl_full);
  collectCloudStorage(snapshot, "preprocess.cloud.corner", preprocess.pl_corn);
  collectCloudStorage(snapshot, "preprocess.cloud.surface", preprocess.pl_surf);
  for (const PointCloudXYZI &line : preprocess.pl_buff)
    collectCloudStorage(snapshot, "preprocess.cloud.scan_lines", line);
  for (const auto &types : preprocess.typess) snapshot.vector("preprocess.line_types", types);
}

void collectOdometryStrings(Snapshot &snapshot, const std::string &key, const nav_msgs::Odometry &odometry)
{
  snapshot.text(key + ".frame_id", odometry.header.frame_id);
  snapshot.text(key + ".child_frame_id", odometry.child_frame_id);
}
} // namespace

void LIVMapper::collectMemoryStats(memory_stats::Snapshot &snapshot)
{
  // Attribute shared visual references to the visual map before visiting inputs.
  if (vio_manager) vio_manager->collectMemoryStats(snapshot);
  snapshot.object("mapper.object", this, sizeof(*this));

  collectState(snapshot, "mapper.state", _state);
  collectState(snapshot, "mapper.propagated_state", state_propagat);
  if (voxelmap_manager) collectVoxelManager(snapshot, *voxelmap_manager);
  snapshot.map("mapper.voxel_index", voxel_map);
  for (const auto &entry : voxel_map) collectTree(snapshot, entry.second);
  snapshot.vector("mapper.points_with_variance", _pv_list);
  collectPointUncertainty(snapshot, _pv_list);

  collectCloud(snapshot, "mapper.cloud.visual_sub_map", visual_sub_map);
  collectCloud(snapshot, "mapper.cloud.undistorted", feats_undistort);
  collectCloud(snapshot, "mapper.cloud.down_body", feats_down_body);
  collectCloud(snapshot, "mapper.cloud.down_world", feats_down_world);
  collectCloud(snapshot, "output.cloud.world_wait_publish", pcl_w_wait_pub);
  collectCloud(snapshot, "output.cloud.wait_publish", pcl_wait_pub);
  collectCloud(snapshot, "output.cloud.wait_save_rgb", pcl_wait_save);
  collectCloud(snapshot, "output.cloud.wait_save_intensity", pcl_wait_save_intensity);

  // PCL's public cloud/index getters return shared handles; leaf-layout storage
  // has only a by-value getter and is intentionally not copied for measurement.
  const auto &voxel_filter = downSizeFilterSurf;
  collectCloud(snapshot, "lidar.voxel_filter.input", voxel_filter.getInputCloud());
  const auto filter_indices = voxel_filter.getIndices();
  if (filter_indices && snapshot.object("lidar.voxel_filter.indices.object", filter_indices.get(), sizeof(*filter_indices)))
    snapshot.vector("lidar.voxel_filter.indices", *filter_indices);
  const auto removed_indices = voxel_filter.getRemovedIndices();
  if (removed_indices && snapshot.object("lidar.voxel_filter.removed_indices.object", removed_indices.get(), sizeof(*removed_indices)))
    snapshot.vector("lidar.voxel_filter.removed_indices", *removed_indices);

  snapshot.vector("mapper.config.cameras", camera_configs);
  for (const CameraInputConfig &camera : camera_configs)
  {
    snapshot.text("mapper.config.camera.image_topic", camera.img_topic);
    snapshot.text("mapper.config.camera.namespace", camera.camera_namespace);
    snapshot.text("mapper.config.camera.raw_namespace", camera.raw_camera_namespace);
    snapshot.text("mapper.config.camera.model_type", camera.camera_model_type);
    snapshot.vector("mapper.config.camera.rotation", camera.Rcl);
    snapshot.vector("mapper.config.camera.translation", camera.Pcl);
    snapshot.mat("mapper.calibration.undistort_map_x", camera.undistort_map_x);
    snapshot.mat("mapper.calibration.undistort_map_y", camera.undistort_map_y);
  }
  snapshot.vector("mapper.config.ncc_thresholds", ncc_thre_by_level);
  snapshot.vector("mapper.config.extrinsic_mask", online_extrinsic_camera_mask);
  snapshot.vector("mapper.config.time_offset_mask", online_time_offset_group_mask);
  snapshot.vector("mapper.config.time_offset_groups", img_time_offset_groups);
  snapshot.vector("mapper.config.extrinsic_translation", extrinT);
  snapshot.vector("mapper.config.extrinsic_rotation", extrinR);
  snapshot.text("mapper.config.root_dir", root_dir);
  snapshot.text("mapper.config.lidar_topic", lid_topic);
  snapshot.text("mapper.config.imu_topic", imu_topic);
  snapshot.text("mapper.config.sequence_name", seq_name);
  snapshot.text("mapper.config.imu_noise_model", imu_noise_model);
  snapshot.text("mapper.config.memory_stats_output_dir", memory_stats_output_dir);
  snapshot.text("mapper.config.patch_resampling_mode", virtual_patch_resampling_mode);
  snapshot.text("mapper.config.interpolation_mode", virtual_interp_mode);
  snapshot.text("mapper.config.runtime_dump_folder", runtime_support_dump_folder);
  snapshot.text("mapper.config.optical_flow_image_topic", optical_flow_feature_image_topic);
  snapshot.text("mapper.config.triangulated_points_topic", optical_flow_triangulated_points_topic);
  snapshot.vector("mapper.ros.subscriber_handles", sub_imgs);
  snapshot.vector("mapper.ros.compressed_subscriber_handles", sub_imgs_compressed);
  snapshot.vector("mapper.ros.image_publisher_handles", pubImages);

  snapshot.text("output.path.frame_id", path.header.frame_id);
  snapshot.vector("output.path.poses", path.poses);
  for (const auto &pose : path.poses) snapshot.text("output.path.pose_frame_ids", pose.header.frame_id);
  snapshot.text("output.body_pose.frame_id", msg_body_pose.header.frame_id);
  collectOdometryStrings(snapshot, "output.odometry", odomAftMapped);

  {
    std::lock_guard<std::mutex> lock(mtx_buffer);
    // Preprocess is written by the LiDAR callback while holding mtx_buffer.
    if (p_pre) collectPreprocess(snapshot, *p_pre);
    snapshot.vector("input.last_camera_timestamps", last_timestamp_img_by_camera);
    snapshot.sequence("input.lidar_cloud_handles", lid_raw_data_buffer);
    for (const auto &cloud : lid_raw_data_buffer) collectCloud(snapshot, "input.lidar_clouds", cloud);
    snapshot.sequence("input.lidar_header_timestamps", lid_header_time_buffer);
    snapshot.sequence("input.imu_handles", imu_buffer);
    for (const auto &message : imu_buffer) collectImuMessage(snapshot, "input.imu_messages", message);
    snapshot.map("input.pending_images.index", pending_images);
    for (const auto &entry : pending_images) collectPendingFrame(snapshot, entry.second);
    snapshot.sequence("input.camera_frame_queue", multi_cam_frame_buffer);
    for (const MultiCameraFrame &frame : multi_cam_frame_buffer)
      collectFrame(snapshot, "input.queued_camera_frames", frame);
    collectCloud(snapshot, "input.active_lidar.scan", LidarMeasures.lidar);
    collectCloud(snapshot, "input.active_lidar.current", LidarMeasures.pcl_proc_cur);
    collectCloud(snapshot, "input.active_lidar.next", LidarMeasures.pcl_proc_next);
    snapshot.sequence("input.active_measures", LidarMeasures.measures);
    for (const MeasureGroup &measure : LidarMeasures.measures) collectMeasure(snapshot, measure);
  }
  {
    // Never hold both input locks: imu_cbk acquires them in separate scopes too.
    std::lock_guard<std::mutex> lock(mtx_buffer_imu_prop);
    collectState(snapshot, "imu.live_propagated_state", imu_propagate);
    collectState(snapshot, "imu.latest_ekf_state", latest_ekf_state);
    snapshot.sequence("imu.propagation_queue", prop_imu_buffer);
    for (const sensor_msgs::Imu &message : prop_imu_buffer)
      snapshot.text("imu.propagation_queue.frame_ids", message.header.frame_id);
    snapshot.text("imu.newest_message.frame_id", newest_imu.header.frame_id);
    collectOdometryStrings(snapshot, "imu.propagated_odometry", imu_prop_odom);
  }
  if (p_imu) p_imu->collectMemoryStats(snapshot);
  collectColorGrayMemoryStats(snapshot);
}
