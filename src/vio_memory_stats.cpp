#include "vio_multi_cam.h"
#include "memory_stats.h"

#include <vikit/atan_camera.h>
#include <vikit/equidistant_camera.h>
#include <vikit/mei_camera.h>
#include <vikit/omni_camera.h>
#include <vikit/polynomial_camera.h>

namespace
{

// Only recurse into live elements. Reserved vector storage already includes
// their object headers, so children contribute dynamic allocations only.
template <typename T, typename OuterAllocator, typename InnerAllocator>
void nestedVectors(memory_stats::Snapshot &snapshot, const std::string &key,
                   const std::vector<std::vector<T, InnerAllocator>, OuterAllocator> &values)
{
  snapshot.vector(key, values);
  for (const auto &inner : values) snapshot.vector(key + ".elements", inner);
}

void patchInformation(memory_stats::Snapshot &snapshot, const std::string &key,
                      const split_state_math::PatchInformation &value)
{
  snapshot.eigen(key + ".h_cross", value.h_cross);
  snapshot.eigen(key + ".h_camera", value.h_camera);
  snapshot.eigen(key + ".g_camera", value.g_camera);
  snapshot.vector(key + ".camera_indices", value.camera_indices);
}

void patchWorkspace(memory_stats::Snapshot &snapshot, const std::string &key,
                    const split_state_math::PatchWorkspace &value)
{
  patchInformation(snapshot, key + ".information", value);
  snapshot.eigen(key + ".motion", value.motion);
  snapshot.eigen(key + ".weighted_motion", value.weighted_motion);
  snapshot.eigen(key + ".camera", value.camera);
  snapshot.eigen(key + ".weighted_camera", value.weighted_camera);
  snapshot.eigen(key + ".pixel_split.camera", value.pixel_split.camera);
  snapshot.eigen(key + ".residual", value.residual);
  snapshot.eigen(key + ".weight", value.weight);
  snapshot.eigen(key + ".weighted_residual", value.weighted_residual);
  snapshot.eigen(key + ".pixel", value.pixel);
  snapshot.eigen(key + ".nuisance", value.nuisance);
  snapshot.eigen(key + ".reference_noise", value.reference_noise);
  snapshot.eigen(key + ".compact", value.compact);
  snapshot.eigen(key + ".nuisance_root", value.nuisance_root);
  snapshot.eigen(key + ".innovation_root", value.innovation_root);
  snapshot.vector(key + ".solve_to_patch", value.solve_to_patch);
  snapshot.vector(key + ".patch_to_solve", value.patch_to_solve);
  snapshot.eigen(key + ".nis.system", value.nis.system);
  // rows()/cols() also work before the first successful compute(). Accessing
  // matrixLLT() would assert in that state. This LLT owns its private matrix.
  const std::size_t factor_bytes = static_cast<std::size_t>(value.nis.factor.rows()) *
      static_cast<std::size_t>(value.nis.factor.cols()) * sizeof(double);
  snapshot.add(key + ".nis.factor", factor_bytes, factor_bytes);
  snapshot.eigen(key + ".nis.rhs", value.nis.rhs);
  snapshot.eigen(key + ".nis.latent", value.nis.latent);
  snapshot.eigen(key + ".nis.remaining", value.nis.remaining);
}

class VisualCollector
{
public:
  VisualCollector(memory_stats::Snapshot &snapshot, int patch_size)
      : snapshot_(snapshot), patch_bytes_(static_cast<std::size_t>(std::max(0, patch_size)) * sizeof(float)) {}

  void feature(const std::string &key, const Feature *value)
  {
    if (value == nullptr || !snapshot_.object(key + ".objects", value, sizeof(*value))) return;
    snapshot_.allocation(key + ".core_patches", value->patch_, patch_bytes_, patch_bytes_);
    snapshot_.mat(key + ".images", value->img_);
    snapshot_.mat(key + ".source_rois", value->virtual_source_roi_);
    pyramid(key + ".adaptive_pyramids", value->adaptive_image_pyramid_.get());
  }

  void point(const std::string &key, const VisualPoint *value)
  {
    if (value == nullptr || !snapshot_.object(key + ".objects", value, sizeof(*value))) return;
    snapshot_.sequence(key + ".observations", value->obs_);
    snapshot_.vector(key + ".reference_by_camera", value->ref_patch_by_camera_);
    snapshot_.vector(key + ".has_reference_by_camera", value->has_ref_patch_by_camera_);
    snapshot_.vector(key + ".reuse_camera_observed", value->reuse_camera_observed_);
    snapshot_.vector(key + ".view_samples", value->view_samples_);
    for (const Feature *reference : value->obs_) feature(key + ".references", reference);
    // ref_patch/by_camera are non-owning aliases of obs_. Their pointer-array
    // storage is included above; only the owning list is dereferenced.
  }

  void pyramid(const std::string &key, const virtual_sampling::ImagePyramid *value)
  {
    if (value != nullptr) value->collectMemoryStats(snapshot_, key);
  }

  void support(const std::string &key, const VirtualPatchImage &value)
  {
    snapshot_.mat(key + ".values", value.values);
    snapshot_.mat(key + ".valid_masks", value.valid_mask);
  }

  void supports(const std::string &key, const std::vector<VirtualPatchImage> &values)
  {
    snapshot_.vector(key, values);
    for (const auto &value : values) support(key, value);
  }

  void poseAffine(const std::string &key, const PoseAffineTrack *value)
  {
    if (value != nullptr) snapshot_.object(key, value, sizeof(*value));
  }

  void track(const std::string &key, const VirtualTrackPatch &value)
  {
    support(key + ".current_support", value.cur_support);
    supports(key + ".adaptive_supports", value.adaptive_supports);
    poseAffine(key + ".pose_affine", value.pose_affine.get());
  }

  void submap(const std::string &key, const SubSparseMap *value)
  {
    if (value == nullptr || !snapshot_.object(key + ".objects", value, sizeof(*value))) return;
    snapshot_.vector(key + ".propagated_errors", value->propa_errors);
    snapshot_.vector(key + ".errors", value->errors);
    nestedVectors(snapshot_, key + ".warped_patches", value->warp_patch);
    snapshot_.vector(key + ".search_levels", value->search_levels);
    snapshot_.vector(key + ".warp_affines", value->warp_affines);
    snapshot_.vector(key + ".voxel_points", value->voxel_points);
    snapshot_.vector(key + ".reference_features", value->reference_features);
    snapshot_.vector(key + ".inverse_exposures", value->inv_expo_list);
    snapshot_.vector(key + ".add_from_voxel_map", value->add_from_voxel_map);
    for (const auto &point_value : value->add_from_voxel_map) scanUncertainty(key + ".scan_uncertainty", point_value);
    snapshot_.vector(key + ".virtual_track_patches", value->virtual_track_patches);
    snapshot_.vector(key + ".contributes_to_ekf", value->contributes_to_ekf);
    snapshot_.vector(key + ".pose_information", value->pose_information);
    snapshot_.vector(key + ".observation_nis", value->observation_nis);
    snapshot_.vector(key + ".observation_dof", value->observation_dof);
    nestedVectors(snapshot_, key + ".usage_ncc_levels", value->usage_ncc_levels);
    nestedVectors(snapshot_, key + ".usage_sse_levels", value->usage_sse_levels);
    nestedVectors(snapshot_, key + ".usage_level_valid", value->usage_level_valid);
    nestedVectors(snapshot_, key + ".level_active", value->level_active);
    // These point/reference arrays are non-owning and can retain old addresses
    // after map retirement. Their owners were traversed before current caches.
    for (const auto &track_value : value->virtual_track_patches) track(key + ".track_supports", track_value);
  }

  void frame(const std::string &key, const Frame *value)
  {
    if (value == nullptr || !snapshot_.object(key + ".objects", value, sizeof(*value))) return;
    snapshot_.mat(key + ".image", value->img_);
    snapshot_.sequence(key + ".features", value->fts_);
    for (const Feature *reference : value->fts_) feature(key + ".references", reference);
  }

  void cameraModel(const std::string &key, const vk::AbstractCamera *value)
  {
    if (value == nullptr) return;
    // Dynamic camera type is required for the object size. Private remapping
    // buffers in third-party camera classes are outside this collector's API.
    if (const auto *typed = dynamic_cast<const vk::PinholeCamera *>(value))
      snapshot_.object(key + ".pinhole", typed, sizeof(*typed));
    else if (const auto *typed = dynamic_cast<const vk::EquidistantCamera *>(value))
      snapshot_.object(key + ".equidistant", typed, sizeof(*typed));
    else if (const auto *typed = dynamic_cast<const vk::MEICamera *>(value))
      snapshot_.object(key + ".mei", typed, sizeof(*typed));
    else if (const auto *typed = dynamic_cast<const vk::ATANCamera *>(value))
      snapshot_.object(key + ".atan", typed, sizeof(*typed));
    else if (const auto *typed = dynamic_cast<const vk::PolynomialCamera *>(value))
      snapshot_.object(key + ".polynomial", typed, sizeof(*typed));
    else if (const auto *typed = dynamic_cast<const vk::OmniCamera *>(value))
      snapshot_.object(key + ".omni", typed, sizeof(*typed));
    else
      snapshot_.object(key + ".unknown_base_only", value, sizeof(*value));
  }

  void camera(const std::string &key, const PerCameraData &value)
  {
    snapshot_.text(key + ".topic", value.topic);
    snapshot_.text(key + ".namespace", value.camera_namespace);
    snapshot_.text(key + ".model_type", value.camera_model_type);
    cameraModel(key + ".model", value.cam);
    cameraModel(key + ".model", value.pinhole_cam);
    snapshot_.vector(key + ".grid_num", value.grid_num);
    snapshot_.vector(key + ".map_index", value.map_index);
    snapshot_.vector(key + ".border_flag", value.border_flag);
    snapshot_.vector(key + ".update_flag", value.update_flag);
    snapshot_.vector(key + ".map_dist", value.map_dist);
    snapshot_.vector(key + ".scan_value", value.scan_value);
    snapshot_.vector(key + ".retrieved_points", value.retrieve_voxel_points);
    nestedVectors(snapshot_, key + ".retrieved_candidates", value.retrieve_voxel_point_candidates);
    snapshot_.vector(key + ".appended_points", value.append_voxel_points);
    for (const auto &point_value : value.append_voxel_points) scanUncertainty(key + ".scan_uncertainty", point_value);
    snapshot_.vector(key + ".appended_source_types", value.append_voxel_source_type);
    snapshot_.vector(key + ".appended_source_indices", value.append_voxel_source_index);
    nestedVectors(snapshot_, key + ".ray_samples", value.rays_with_sample_points);
    snapshot_.vector(key + ".raw_ray_lut", value.raw_pixel_to_unit_ray_lut);
    snapshot_.vector(key + ".raw_ray_valid", value.raw_pixel_unit_ray_valid_mask);
    snapshot_.map(key + ".feature_index", value.sub_feat_map);
    snapshot_.map(key + ".warp_index", value.warp_map);
    for (const auto &entry : value.warp_map)
    {
      if (entry.second != nullptr) snapshot_.object(key + ".warps", entry.second, sizeof(*entry.second));
    }
    snapshot_.vector(key + ".pending_observations", value.pending_new_points);
    for (const auto &pending : value.pending_new_points)
    {
      scanUncertainty(key + ".scan_uncertainty", pending.pt_var);
      snapshot_.vector(key + ".pending_patches", pending.patch);
      snapshot_.mat(key + ".pending_images", pending.img);
    }
    submap(key + ".submap", value.visual_submap);
    frame(key + ".frame", value.new_frame.get());
    snapshot_.mat(key + ".images.gray", value.img_cp);
    snapshot_.mat(key + ".images.rgb", value.img_rgb);
    snapshot_.mat(key + ".images.test", value.img_test);
    pyramid(key + ".adaptive_pyramid", value.adaptive_image_pyramid.get());
    snapshot_.vector(key + ".rejected_draw_points", value.rejected_visual_points_for_draw);
    snapshot_.eigen(key + ".reference_gradient_cache", value.fixed_template_cache.photometric_gradients);
    snapshot_.vector(key + ".reference_gradient_valid", value.fixed_template_cache.valid);
  }

private:
  void scanUncertainty(const std::string &key, const pointWithVar &value)
  {
    const ScanPoseUncertainty *uncertainty = value.scan_uncertainty.get();
    if (uncertainty != nullptr) snapshot_.object(key, uncertainty, sizeof(*uncertainty));
  }

  memory_stats::Snapshot &snapshot_;
  std::size_t patch_bytes_;
};

template <typename Cells>
void usageCells(memory_stats::Snapshot &snapshot, const std::string &key, const Cells &values)
{
  for (const auto &cell : values) snapshot.vector(key + ".levels", cell.levels);
}

} // namespace

void virtual_sampling::ImagePyramid::collectMemoryStats(memory_stats::Snapshot &snapshot,
                                                       const std::string &key) const
{
  if (!snapshot.object(key + ".objects", this, sizeof(*this))) return;
  std::lock_guard<std::mutex> lock(mutex_);
  for (std::size_t level = 0; level < levels_.size(); ++level)
    snapshot.mat(key + ".level_" + std::to_string(level), levels_[level]);
}

void VIOManager::collectMemoryStats(memory_stats::Snapshot &snapshot) const
{
  if (!snapshot.object("vio.manager", this, sizeof(*this))) return;
  VisualCollector collector(snapshot, patch_size_total);

  // First-owner attribution: map references retain image allocations before
  // current-frame aliases are examined. Snapshot deduplicates image backing
  // storage independently from object pointers and container allocations.
  snapshot.map("vio.map.index", feat_map);
  for (const auto &entry : feat_map)
  {
    const VOXEL_POINTS *voxel = entry.second;
    if (voxel == nullptr || !snapshot.object("vio.map.voxels", voxel, sizeof(*voxel))) continue;
    snapshot.vector("vio.map.voxel_point_arrays", voxel->voxel_points);
    for (const VisualPoint *point : voxel->voxel_points) collector.point("vio.map.points", point);
  }
  snapshot.vector("vio.retirement.point_index", retired_visual_points_);
  snapshot.vector("vio.retirement.reference_index", retired_visual_refs_);
  for (const VisualPoint *point : retired_visual_points_) collector.point("vio.retirement.points", point);
  for (const auto &entry : retired_visual_refs_)
  {
    collector.point("vio.retirement.points", entry.first);
    collector.feature("vio.retirement.references", entry.second);
  }
  snapshot.map("vio.retirement.shadow_points", shadow_retired_point_suggestions_);
  snapshot.map("vio.retirement.shadow_references", shadow_retired_ref_suggestions_);

  snapshot.vector("vio.cameras", cameras_);
  for (std::size_t camera = 0; camera < cameras_.size(); ++camera)
    collector.camera("vio.camera_" + std::to_string(camera), cameras_[camera]);

  snapshot.vector("vio.patch_buffer", patch_buffer);
  snapshot.vector("vio.virtual_ray_lut", virtual_support_ray_lut_);
  snapshot.vector("vio.core_patch_offsets", core_patch_offsets_);
  snapshot.vector("vio.ncc_thresholds", ncc_thre_by_level);
  snapshot.text("vio.config.resampling_mode", virtual_patch_resampling_mode);
  snapshot.text("vio.config.interpolation_mode", virtual_interp_mode);
  snapshot.text("vio.dump.folder", runtime_support_dump_folder);
  snapshot.text("vio.dump.effective_folder", runtime_support_dump_effective_folder_);
  snapshot.vector("vio.dump.anchor_patch", ref_patch_dump_probe_.anchor_virtual_patch);
  snapshot.mat("vio.dump.range_image", ref_patch_dump_range_img_);

  snapshot.vector("vio.calibration.extrinsic_mask", online_extrinsic_camera_mask);
  snapshot.vector("vio.calibration.time_groups", camera_time_offset_groups);
  snapshot.vector("vio.calibration.time_mask", online_time_offset_group_mask);
  snapshot.vector("vio.calibration.camera_accept_counts", online_extrinsic_camera_accept_count_);
  snapshot.vector("vio.calibration.group_accept_counts", online_time_offset_group_accept_count_);
  snapshot.vector("vio.calibration.active_rotations", online_calib_last_active_extrinsic_rot_);
  snapshot.vector("vio.calibration.active_translations", online_calib_last_active_extrinsic_trans_);
  snapshot.vector("vio.calibration.active_time_groups", online_calib_last_active_time_groups_);
  snapshot.vector("vio.calibration.last_time_tracks", online_calib_last_time_tracks_);
  snapshot.vector("vio.calibration.last_pixel_velocities", online_calib_last_time_avg_pixel_velocity_);
  snapshot.vector("vio.calibration.last_time_updates", online_calib_last_time_update_ms_);
  snapshot.vector("vio.calibration.last_rotation_updates", online_calib_last_extrinsic_rot_update_deg_);
  snapshot.vector("vio.calibration.last_translation_updates", online_calib_last_extrinsic_trans_update_cm_);

  snapshot.vector("vio.cross_camera.pairs", current_cross_camera_pairs_);
  for (const auto &pair : current_cross_camera_pairs_)
  {
    collector.poseAffine("vio.cross_camera.pose_affine", pair.source_pose_affine.get());
    collector.poseAffine("vio.cross_camera.pose_affine", pair.target_pose_affine.get());
    collector.supports("vio.cross_camera.source_supports", pair.adaptive_source_supports);
    collector.supports("vio.cross_camera.target_supports", pair.adaptive_target_supports);
    snapshot.vector("vio.cross_camera.sse_levels", pair.sse_levels);
    snapshot.vector("vio.cross_camera.ncc_levels", pair.ncc_levels);
    snapshot.vector("vio.cross_camera.level_valid", pair.level_valid);
    snapshot.vector("vio.cross_camera.level_active", pair.level_active);
  }
  nestedVectors(snapshot, "vio.ekf.pose_affine_evidence", pose_affine_evidence_valid_);
  snapshot.eigen("vio.ekf.information.cross", visual_information_workspace_.cross);
  snapshot.eigen("vio.ekf.information.camera", visual_information_workspace_.camera);
  snapshot.eigen("vio.ekf.information.camera_gradient", visual_information_workspace_.camera_gradient);
  snapshot.vector("vio.ekf.worker_workspaces", visual_patch_workspaces_);
  for (const auto &workspace : visual_patch_workspaces_)
    patchWorkspace(snapshot, "vio.ekf.worker", workspace);
  nestedVectors(snapshot, "vio.ekf.results", visual_patch_results_);
  for (const auto &camera_results : visual_patch_results_)
    for (const auto &result : camera_results)
    {
      patchInformation(snapshot, "vio.ekf.result.information", result.information);
      snapshot.eigen("vio.ekf.result.reference_jacobian", result.reference_jacobian);
      snapshot.text("vio.ekf.result.covariance_error", result.covariance_error);
    }
  snapshot.eigen("vio.ekf.G", G);
  snapshot.eigen("vio.ekf.H_T_H", H_T_H);
  snapshot.vector("vio.color.fixed_gray_luts", fixed_gray_luts_);

  snapshot.vector("vio.usage.camera_pairs", usage_camera_pairs_);
  snapshot.vector("vio.usage.total_camera_pairs", usage_total_camera_pairs_);
  snapshot.vector("vio.usage.current_camera_pairs", usage_current_camera_pairs_);
  snapshot.vector("vio.usage.total_current_camera_pairs", usage_total_current_camera_pairs_);
  usageCells(snapshot, "vio.usage.camera_pairs", usage_camera_pairs_);
  usageCells(snapshot, "vio.usage.total_camera_pairs", usage_total_camera_pairs_);
  usageCells(snapshot, "vio.usage.current_camera_pairs", usage_current_camera_pairs_);
  usageCells(snapshot, "vio.usage.total_current_camera_pairs", usage_total_current_camera_pairs_);
  usageCells(snapshot, "vio.usage.current_view_angles", usage_current_view_angle_bins_);
  usageCells(snapshot, "vio.usage.total_current_view_angles", usage_total_current_view_angle_bins_);
  usageCells(snapshot, "vio.usage.current_footprints", usage_current_footprint_bins_);
  usageCells(snapshot, "vio.usage.total_current_footprints", usage_total_current_footprint_bins_);
  usageCells(snapshot, "vio.usage.current_anisotropy", usage_current_anisotropy_bins_);
  usageCells(snapshot, "vio.usage.total_current_anisotropy", usage_total_current_anisotropy_bins_);
  usageCells(snapshot, "vio.usage.cross_regions", usage_cross_region_pairs_);
  usageCells(snapshot, "vio.usage.total_cross_regions", usage_total_cross_region_pairs_);
  usageCells(snapshot, "vio.usage.view_angles", usage_view_angle_bins_);
  usageCells(snapshot, "vio.usage.total_view_angles", usage_total_view_angle_bins_);
  usageCells(snapshot, "vio.usage.footprints", usage_footprint_bins_);
  usageCells(snapshot, "vio.usage.total_footprints", usage_total_footprint_bins_);
  usageCells(snapshot, "vio.usage.anisotropy", usage_anisotropy_bins_);
  usageCells(snapshot, "vio.usage.total_anisotropy", usage_total_anisotropy_bins_);
  for (const auto &stats : usage_warp_approx_)
    for (const auto &hist : stats.shape_hist) snapshot.vector("vio.usage.warp_histograms", hist);
  for (const auto &stats : usage_total_warp_approx_)
    for (const auto &hist : stats.shape_hist) snapshot.vector("vio.usage.total_warp_histograms", hist);
  snapshot.vector("vio.usage.pose_all", usage_pose_all_.active_information_gains);
  snapshot.vector("vio.usage.pose_same", usage_pose_same_.active_information_gains);
  snapshot.vector("vio.usage.pose_cross", usage_pose_cross_.active_information_gains);
  snapshot.vector("vio.usage.pose_current_cross", usage_pose_current_cross_.active_information_gains);
  snapshot.vector("vio.usage.total_pose_all", usage_total_pose_all_.active_information_gains);
  snapshot.vector("vio.usage.total_pose_same", usage_total_pose_same_.active_information_gains);
  snapshot.vector("vio.usage.total_pose_cross", usage_total_pose_cross_.active_information_gains);
  snapshot.vector("vio.usage.total_pose_current_cross", usage_total_pose_current_cross_.active_information_gains);

  snapshot.mat("vio.optical_flow.debug_image", optical_flow_debug_img);
  snapshot.mat("vio.optical_flow.previous_image", optical_flow_prev_img);
  snapshot.vector("vio.optical_flow.previous_points", optical_flow_prev_pts);
  snapshot.vector("vio.optical_flow.current_points", optical_flow_cur_pts);
  snapshot.vector("vio.optical_flow.ids", optical_flow_ids);
  snapshot.vector("vio.optical_flow.track_counts", optical_flow_track_cnt);
  snapshot.map("vio.optical_flow.track_index", optical_flow_tracks);
  for (const auto &entry : optical_flow_tracks)
  {
    snapshot.sequence("vio.optical_flow.track_history", entry.second.history);
    snapshot.vector("vio.optical_flow.track_observations", entry.second.observations);
  }
  const auto *cloud = optical_flow_triangulated_points.get();
  if (cloud != nullptr && snapshot.object("vio.optical_flow.cloud", cloud, sizeof(*cloud)))
  {
    snapshot.vector("vio.optical_flow.cloud_points", cloud->points);
    snapshot.text("vio.optical_flow.cloud_frame_id", cloud->header.frame_id);
  }
}
