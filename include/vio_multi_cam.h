/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef VIO_MULTI_CAM_H_
#define VIO_MULTI_CAM_H_

#include "voxel_map_multi_cam.h"
#include "feature_multi_cam.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/video/tracking.hpp>
#include <pcl/filters/voxel_grid.h>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vikit/math_utils.h>
#include <vikit/robust_cost.h>
#include <vikit/vision.h>
#include <vikit/pinhole_camera.h>

struct VirtualPatchImage
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  cv::Mat values;        //!< CV_32FC1 local virtual support image.
  cv::Mat valid_mask;    //!< CV_8UC1 validity mask for values.
  M3D R_v_from_c = M3D::Identity();  //!< {}^V R_C.
  M3D R_c_from_v = M3D::Identity();  //!< {}^C R_V.
  SE3<double> T_v_w_seed;            //!< {}^V T_W at submap construction.
  bool valid = false;
};

struct VirtualTrackPatch
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VirtualPatchImage cur_support;
  M3D R_vcur_from_ccur_seed = M3D::Identity();
  M3D R_ccur_from_vcur_seed = M3D::Identity();
  SE3<double> T_vcur_w_seed;
  Matrix2d A_cur_ref = Matrix2d::Identity();
  int search_level = 0;
  bool valid = false;
};

enum class VirtualPatchResamplingMode
{
  FORWARD_SPLAT = 0,
  PULL_EXACT
};

enum class VirtualInterpMode
{
  BILINEAR = 0,
  BICUBIC,
  LANCZOS
};

struct SubSparseMap
{
  vector<float> propa_errors;
  vector<float> errors;
  vector<vector<float>> warp_patch;
  vector<int> search_levels;
  vector<Matrix2d, Eigen::aligned_allocator<Matrix2d>> warp_affines;
  vector<VisualPoint *> voxel_points;
  vector<Feature *> reference_features;
  vector<double> inv_expo_list;
  vector<pointWithVar> add_from_voxel_map;
  vector<VirtualTrackPatch> virtual_track_patches;
  vector<uint8_t> contributes_to_ekf;
  vector<Eigen::Matrix<double, 6, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>> pose_information;
  vector<double> observation_nis;
  vector<int> observation_dof;

  SubSparseMap()
  {
    propa_errors.reserve(SIZE_LARGE);
    errors.reserve(SIZE_LARGE);
    warp_patch.reserve(SIZE_LARGE);
    search_levels.reserve(SIZE_LARGE);
    warp_affines.reserve(SIZE_LARGE);
    voxel_points.reserve(SIZE_LARGE);
    reference_features.reserve(SIZE_LARGE);
    inv_expo_list.reserve(SIZE_LARGE);
    add_from_voxel_map.reserve(SIZE_SMALL);
    virtual_track_patches.reserve(SIZE_LARGE);
    contributes_to_ekf.reserve(SIZE_LARGE);
    pose_information.reserve(SIZE_LARGE);
    observation_nis.reserve(SIZE_LARGE);
    observation_dof.reserve(SIZE_LARGE);
  };

  void reset()
  {
    propa_errors.clear();
    errors.clear();
    warp_patch.clear();
    search_levels.clear();
    warp_affines.clear();
    voxel_points.clear();
    reference_features.clear();
    inv_expo_list.clear();
    add_from_voxel_map.clear();
    virtual_track_patches.clear();
    contributes_to_ekf.clear();
    pose_information.clear();
    observation_nis.clear();
    observation_dof.clear();
  }
};

class Warp
{
public:
  Matrix2d A_cur_ref;
  int search_level;
  Warp(int level, Matrix2d warp_matrix) : search_level(level), A_cur_ref(warp_matrix) {}
  ~Warp() {}
};

class VOXEL_POINTS
{
public:
  std::vector<VisualPoint *> voxel_points;
  int count;
  VOXEL_POINTS(int num) : count(num) {}
  ~VOXEL_POINTS() 
  { 
    for (VisualPoint* vp : voxel_points) 
    {
      if (vp != nullptr) { delete vp; vp = nullptr; }
    }
  }
};

struct OpticalFlowObservation
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int frame_id = -1;
  double timestamp = 0.0;
  cv::Point2f px;
  V3D bearing = V3D::Zero();
  SE3<double> T_f_w;
};

struct OpticalFlowTrack
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int id = -1;
  int age = 0;
  bool triangulated = false;
  bool rejected = false;
  V3D point_w = V3D::Zero();
  std::deque<cv::Point2f> history;
  std::vector<OpticalFlowObservation, Eigen::aligned_allocator<OpticalFlowObservation>> observations;
};

enum CandidateSourceType
{
  SOURCE_PG = 0,
  SOURCE_RAYCAST_PLANE = 1,
  SOURCE_UNKNOWN = 2
};

struct PendingNewPointObservation
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int camera_id = -1;
  int source_type = SOURCE_UNKNOWN;
  int source_index = -1;
  pointWithVar pt_var;
  V2D px = V2D::Zero();
  V3D bearing = V3D::Zero();
  std::vector<float> patch;
  cv::Mat img;
  cv::Point virtual_source_origin;
  SE3<double> T_f_w;
  SE3<double> T_v_w;
  M3D R_v_from_c = M3D::Identity();
  M3D R_c_from_v = M3D::Identity();
  bool virtual_patch_valid = false;
  int level = 0;
  double inv_expo_time = 1.0;
  float texture_score = 0.0f;
  int64_t surface_voxel_x = 0;
  int64_t surface_voxel_y = 0;
  int64_t surface_voxel_z = 0;
  int surface_plane_id = -1;
  uint64_t surface_revision = 0;
  bool surface_valid = false;
  double geometry_chi2 = 0.0;
  std::array<V3D, 4> footprint_corners_w;
  bool footprint_valid = false;
};

struct FixedTemplateGradientCache
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Matrix<double, Eigen::Dynamic, 2> photometric_gradients;
  std::vector<uint8_t> valid;
  int level = -1;
  int point_count = 0;

  void reset(int new_level, int new_point_count, int patch_size_total)
  {
    level = new_level;
    point_count = new_point_count;
    const int row_count = new_point_count * patch_size_total;
    photometric_gradients = Eigen::Matrix<double, Eigen::Dynamic, 2>::Zero(row_count, 2);
    valid.assign(row_count, 0);
  }
};

struct PerCameraData
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int camera_id = -1;
  std::string topic;
  std::string camera_namespace;
  vk::AbstractCamera *cam = nullptr;
  vk::PinholeCamera *pinhole_cam = nullptr;
  std::string camera_model_type = "Pinhole";
  double k1 = 0.0;
  double k2 = 0.0;
  double k3 = 0.0;
  double k4 = 0.0;
  double xi = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;

  M3D Rcl = M3D::Identity();
  M3D Rci = M3D::Identity();
  M3D Rcw = M3D::Identity();
  V3D Pcl = V3D::Zero();
  V3D Pci = V3D::Zero();
  V3D Pcw = V3D::Zero();
  M3D Jdphi_dR = M3D::Identity();
  M3D Jdp_dR = M3D::Zero();
  M3D Jdp_dt = M3D::Identity();

  int width = 0;
  int height = 0;
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  double image_resize_factor = 1.0;
  int grid_size = 0;
  int grid_n_width = 0;
  int grid_n_height = 0;
  int length = 0;
  int total_points = 0;

  std::vector<int> grid_num;
  std::vector<int> map_index;
  std::vector<int> border_flag;
  std::vector<int> update_flag;
  std::vector<float> map_dist;
  std::vector<float> scan_value;
  std::vector<VisualPoint *> retrieve_voxel_points;
  std::vector<pointWithVar> append_voxel_points;
  std::vector<int> append_voxel_source_type;
  std::vector<int> append_voxel_source_index;
  std::vector<std::vector<V3D>> rays_with_sample_points;
  std::vector<V3F> raw_pixel_to_unit_ray_lut;
  std::vector<uint8_t> raw_pixel_unit_ray_valid_mask;
  std::unordered_map<VOXEL_LOCATION, int> sub_feat_map;
  std::unordered_map<Feature *, Warp *> warp_map;
  std::vector<PendingNewPointObservation> pending_new_points;

  FramePtr new_frame;
  SubSparseMap *visual_submap = nullptr;
  cv::Mat img_cp;
  cv::Mat img_rgb;
  cv::Mat img_test;
  std::vector<std::pair<cv::Point2f, int>> rejected_visual_points_for_draw;
  FixedTemplateGradientCache fixed_template_cache;
};

class VIOManager
{
public:
  int grid_size;
  StatesGroup *state;
  StatesGroup *state_propagat;
  M3D Rli;
  V3D Pli;
  std::vector<PerCameraData> cameras_;
  vector<float> patch_buffer;
  bool normal_en, inverse_composition_en, exposure_estimate_en, raycast_en;
  bool ncc_en = false, colmap_output_en = false;
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

  int grid_n_width, grid_n_height;
  int patch_pyrimid_level, patch_size, patch_size_total, patch_size_half, border, warp_len;
  int max_iterations, total_points;

  double img_point_cov, outlier_threshold, ncc_thre = 0.8;

  double virtual_focal_length = 300.0;
  int virtual_patch_margin = 4;
  int virtual_max_search_level = 1;
  double virtual_min_z = 1.0e-6;
  std::string virtual_patch_resampling_mode = "forward_splat";
  VirtualPatchResamplingMode virtual_patch_resampling_mode_enum = VirtualPatchResamplingMode::FORWARD_SPLAT;
  std::string virtual_interp_mode = "bilinear";
  VirtualInterpMode virtual_interp_mode_enum = VirtualInterpMode::BILINEAR;
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
  bool draw_rejected_points_en = false;
  bool usage_stats_en = false;
  int usage_stats_window = 100;
  bool ref_patch_dump_en = false;
  int ref_patch_dump_random_seed = -1;
  int ref_patch_dump_max_candidate_skip = 50;
  double ref_patch_dump_ncc_threshold = 0.6;
  bool runtime_support_dump_en = false;
  std::string runtime_support_dump_folder = "runtime_support";
  bool runtime_support_dump_initialized_ = false;
  int runtime_support_dump_next_point_id_ = 0;
  int runtime_support_dump_best_track_count_ = 0;
  const VisualPoint *runtime_support_dump_best_point_ = nullptr;
  std::string runtime_support_dump_effective_folder_;
  int virtual_support_radius = 0;
  int virtual_support_size = 0;
  int virtual_ref_support_materialized_count_ = 0;
  int virtual_ref_support_materialize_fail_count_ = 0;
  double virtual_ref_support_materialize_time_ = 0.0;
  std::vector<V3F> virtual_support_ray_lut_;
  std::vector<V2F> core_patch_offsets_;
  uint64_t next_visual_point_id_ = 1;
  uint64_t next_visual_ref_id_ = 1;
  std::vector<VisualPoint *> retired_visual_points_;
  std::vector<std::pair<VisualPoint *, Feature *>> retired_visual_refs_;
  std::set<uint64_t> shadow_retired_point_suggestions_;
  std::set<uint64_t> shadow_retired_ref_suggestions_;

  struct VisualMapManageStats
  {
    long long dynamic_selected = 0;
    long long fallback_attempted = 0;
    long long fallback_accepted = 0;
    long long seed_tested = 0;
    long long seed_confirmed = 0;
    long long ref_validated = 0;
    long long ref_retired = 0;
    long long footprint_compared = 0;
    long long redundant_rejected = 0;
    long long information_admitted = 0;
    long long replacement_suggested = 0;
    long long point_retired = 0;
  } visual_map_manage_stats_;

  struct RefPatchDumpProbeState
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    bool active = false;
    int point_id = -1;
    int saved_refs = 0;
    int selected_frame_id = -1;
    int last_saved_frame_id = -1;
    int missed_frames = 0;
    double last_saved_incidence_deg = 0.0;
    V3D point_w = V3D::Zero();
    V3D normal_w = V3D::Zero();
    bool anchor_valid = false;
    SE3<double> anchor_T_c_w;
    SE3<double> anchor_T_v_w;
    std::vector<float> anchor_virtual_patch;
  };
  RefPatchDumpProbeState ref_patch_dump_probe_;
  int ref_patch_dump_next_point_id_ = 0;
  bool ref_patch_dump_initialized_ = false;

  std::mt19937 ref_patch_dump_rng_;
  unsigned int ref_patch_dump_effective_seed_ = 0;
  int ref_patch_dump_candidates_to_skip_ = -1;
  cv::Mat ref_patch_dump_range_img_;
  SE3<double> ref_patch_dump_range_T_f_w_;
  bool ref_patch_dump_range_pose_valid_ = false;
  int rejected_virtual_support_oob_ = 0;
  int rejected_virtual_projection_invalid_ = 0;
  int rejected_virtual_z_ = 0;
  int rejected_virtual_affine_oob_ = 0;
  double build_virtual_support_time_ = 0.0;
  double virtual_affine_time_ = 0.0;
  double virtual_candidate_select_time_ = 0.0;
  double virtual_parallel_track_time_ = 0.0;
  double virtual_result_collect_time_ = 0.0;
  double virtual_warp_time_ = 0.0;
  double virtual_current_core_time_ = 0.0;

  static constexpr int kUsageNccBinCount = 11;
  static constexpr int kUsagePoseDim = 6;

  struct UsageLevelStats
  {
    long long candidate_ncc_count = 0;
    long long accepted_ncc_count = 0;
    long long accepted_sse_samples = 0;
    double candidate_ncc_sum = 0.0;
    double accepted_ncc_sum = 0.0;
    double accepted_sse = 0.0;
    std::array<long long, kUsageNccBinCount> candidate_ncc_hist = {};
    std::array<long long, kUsageNccBinCount> accepted_ncc_hist = {};

    void reset()
    {
      candidate_ncc_count = 0;
      accepted_ncc_count = 0;
      accepted_sse_samples = 0;
      candidate_ncc_sum = 0.0;
      accepted_ncc_sum = 0.0;
      accepted_sse = 0.0;
      candidate_ncc_hist.fill(0);
      accepted_ncc_hist.fill(0);
    }
  };

  struct UsageStatsCell
  {
    long long candidates = 0;
    long long accepted = 0;
    long long theoretical_residuals = 0;
    long long ekf_patches = 0;
    long long ekf_residuals = 0;
    long long candidate_all_level_count = 0;
    long long accepted_all_level_count = 0;
    double candidate_ncc_min_sum = 0.0;
    double candidate_ncc_macro_sum = 0.0;
    double accepted_ncc_min_sum = 0.0;
    double accepted_ncc_macro_sum = 0.0;
    double accepted_rms_macro_sum = 0.0;
    std::vector<UsageLevelStats> levels;

    void ensureLevels(int level_count)
    {
      if (level_count < 0) level_count = 0;
      if (static_cast<int>(levels.size()) == level_count) return;
      levels.assign(level_count, UsageLevelStats());
    }

    void reset()
    {
      candidates = 0;
      accepted = 0;
      theoretical_residuals = 0;
      ekf_patches = 0;
      ekf_residuals = 0;
      candidate_all_level_count = 0;
      accepted_all_level_count = 0;
      candidate_ncc_min_sum = 0.0;
      candidate_ncc_macro_sum = 0.0;
      accepted_ncc_min_sum = 0.0;
      accepted_ncc_macro_sum = 0.0;
      accepted_rms_macro_sum = 0.0;
      for (UsageLevelStats &level : levels) level.reset();
    }
  };

  struct PoseInfoStatsCell
  {
    long long frames = 0;
    long long active_frames = 0;
    long long patches = 0;
    long long residuals = 0;
    double information_gain_sum = 0.0;
    double worst_direction_gain_sum = 0.0;

    void reset()
    {
      frames = 0;
      active_frames = 0;
      patches = 0;
      residuals = 0;
      information_gain_sum = 0.0;
      worst_direction_gain_sum = 0.0;
    }
  };

  long long usage_stats_frames_ = 0;
  long long usage_stats_total_frames_ = 0;
  std::vector<UsageStatsCell> usage_camera_pairs_;
  std::vector<UsageStatsCell> usage_total_camera_pairs_;
  std::array<UsageStatsCell, 16> usage_region_pairs_;
  std::array<UsageStatsCell, 16> usage_total_region_pairs_;
  std::array<UsageStatsCell, 32> usage_cross_region_pairs_;
  std::array<UsageStatsCell, 32> usage_total_cross_region_pairs_;
  std::array<UsageStatsCell, 5> usage_view_angle_bins_;
  std::array<UsageStatsCell, 5> usage_total_view_angle_bins_;
  std::array<UsageStatsCell, 5> usage_footprint_bins_;
  std::array<UsageStatsCell, 5> usage_total_footprint_bins_;
  std::array<UsageStatsCell, 4> usage_anisotropy_bins_;
  std::array<UsageStatsCell, 4> usage_total_anisotropy_bins_;
  PoseInfoStatsCell usage_pose_all_;
  PoseInfoStatsCell usage_pose_same_;
  PoseInfoStatsCell usage_pose_cross_;
  PoseInfoStatsCell usage_total_pose_all_;
  PoseInfoStatsCell usage_total_pose_same_;
  PoseInfoStatsCell usage_total_pose_cross_;
  std::vector<PoseInfoStatsCell> usage_pose_camera_pairs_;
  std::vector<PoseInfoStatsCell> usage_total_pose_camera_pairs_;
  std::array<PoseInfoStatsCell, 16> usage_pose_region_pairs_;
  std::array<PoseInfoStatsCell, 16> usage_total_pose_region_pairs_;
  std::array<PoseInfoStatsCell, 32> usage_pose_cross_region_pairs_;
  std::array<PoseInfoStatsCell, 32> usage_total_pose_cross_region_pairs_;
  int virtual_map_grid_count_ = 0;
  int virtual_candidate_null_count_ = 0;
  int virtual_candidate_normal_uninit_count_ = 0;
  int virtual_candidate_projection_fail_count_ = 0;
  int virtual_candidate_range_reject_count_ = 0;
  int virtual_candidate_close_view_fail_count_ = 0;
  int virtual_candidate_ref_missing_count_ = 0;
  int virtual_candidate_ref_invalid_count_ = 0;
  int virtual_candidate_count_ = 0;
  int virtual_valid_track_count_ = 0;
  int virtual_track_rotation_fail_count_ = 0;
  int virtual_track_support_fail_count_ = 0;
  int virtual_track_affine_fail_count_ = 0;
  int virtual_track_warp_fail_count_ = 0;
  int virtual_track_current_z_fail_count_ = 0;
  int virtual_track_current_core_fail_count_ = 0;
  int virtual_track_ncc_reject_count_ = 0;
  int virtual_track_photometric_reject_count_ = 0;
  enum RejectedVisualPointReason
  {
    REJECT_DRAW_NORMAL_UNINIT = 0,
    REJECT_DRAW_RANGE,
    REJECT_DRAW_CLOSE_VIEW,
    REJECT_DRAW_REF_MISSING,
    REJECT_DRAW_REF_INVALID,
    REJECT_DRAW_ROTATION,
    REJECT_DRAW_SUPPORT_BUILD,
    REJECT_DRAW_AFFINE,
    REJECT_DRAW_WARP_REF,
    REJECT_DRAW_CURRENT_Z,
    REJECT_DRAW_CURRENT_CORE,
    REJECT_DRAW_NCC,
    REJECT_DRAW_PHOTOMETRIC
  };
  struct RejectedVisualPointForDraw
  {
    cv::Point2f px;
    int reason = REJECT_DRAW_RANGE;
  };
  double compute_jacobian_time, update_ekf_time;
  double ave_total = 0;
  // double ave_build_residual_time = 0;
  // double ave_ekf_time = 0;

  int frame_count = 0;
  bool plot_flag;

  Eigen::MatrixXd G, H_T_H;

  ofstream fout_camera, fout_colmap;
  unordered_map<VOXEL_LOCATION, VOXEL_POINTS *> feat_map;
  cv::Mat optical_flow_debug_img;

  int frontend_mode = 0;
  int optical_flow_max_cnt = 250;
  int optical_flow_min_dist = 20;
  int optical_flow_min_track_len_for_triangulation = 3;
  int optical_flow_track_history_size = 20;
  double optical_flow_quality_level = 0.01;
  double optical_flow_f_threshold = 0.5;
  bool optical_flow_flow_back = true;
  int optical_flow_next_id = 0;
  int optical_flow_frame_id = 0;
  double optical_flow_prev_time = -1.0;
  cv::Mat optical_flow_prev_img;
  std::vector<cv::Point2f> optical_flow_prev_pts;
  std::vector<cv::Point2f> optical_flow_cur_pts;
  std::vector<int> optical_flow_ids;
  std::vector<int> optical_flow_track_cnt;
  std::map<int, OpticalFlowTrack> optical_flow_tracks;
  PointCloudXYZI::Ptr optical_flow_triangulated_points;

  enum CellType
  {
    TYPE_MAP = 1,
    TYPE_POINTCLOUD,
    TYPE_UNKNOWN
  };

  enum VisualGeomRejectReason
  {
    VISUAL_GEOM_REJECT_NONE = 0,
    VISUAL_GEOM_REJECT_BAD_POINT,
    VISUAL_GEOM_REJECT_NO_PLANE,
    VISUAL_GEOM_REJECT_NOT_PLANE,
    VISUAL_GEOM_REJECT_PLANE_SIZE,
    VISUAL_GEOM_REJECT_RANGE,
    VISUAL_GEOM_REJECT_NORMAL,
    VISUAL_GEOM_REJECT_COV,
    VISUAL_GEOM_REJECT_SIGMA,
    VISUAL_GEOM_REJECT_CHI2,
    VISUAL_GEOM_REJECT_COUNT
  };

  VIOManager();
  ~VIOManager();
  void configureCameras(int num_cameras);
  void setCameraCalibration(int camera_id, const std::string &topic, const std::string &camera_namespace,
                            const std::vector<double> &R, const std::vector<double> &P);
  void setCameraModelJacobianParameters(int camera_id, const std::string &model_type,
                                        double k1, double k2, double k3, double k4,
                                        double xi, double p1, double p2);
  int numCameras() const { return static_cast<int>(cameras_.size()); }
  void processMultiCameraFrame(const MeasureGroup &meas, vector<pointWithVar> &pg,
                               const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void processFrameOpticalFlow(cv::Mat &img, double img_time);
  void processFrameFake(cv::Mat &img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map, double img_time);
  void processMultiCameraFrameFake(const MeasureGroup &meas);
  bool inOpticalFlowBorder(const PerCameraData &ctx, const cv::Point2f &pt) const;
  V3D getOpticalFlowBearing(const PerCameraData &ctx, const cv::Point2f &px) const;
  void setOpticalFlowMask(const PerCameraData &ctx, cv::Mat &mask);
  void addOpticalFlowObservation(OpticalFlowTrack &track, const cv::Point2f &px, double img_time);
  bool triangulateOpticalFlowTrack(OpticalFlowTrack &track);
  void updateOpticalFlowPointClouds();
  void drawOpticalFlowDebugImage(const std::vector<cv::Point2f> &rejected_pts, int prev, int tracked, int flow_back_pass,
                                 int border_pass, int mask_reject, int new_points, int final_points, int triangulated);
  void retrieveFromVisualSparseMap(PerCameraData &ctx, const cv::Mat &img, vector<pointWithVar> &pg,
                                   const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void generateVisualMapPoints(PerCameraData &ctx, const cv::Mat &img, vector<pointWithVar> &pg,
                               const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void setImuToLidarExtrinsic(const V3D &transl, const M3D &rot);
  void syncCameraExtrinsicsFromState(const StatesGroup &state_value);
  void updateCameraExtrinsicDerived(PerCameraData &ctx);
  bool isOnlineExtrinsicEnabledForCamera(int camera_id) const;
  void applyOnlineExtrinsicPriors(Eigen::MatrixXd &hessian, Eigen::VectorXd &gradient, bool allow_rotation, bool allow_translation) const;
  void limitOnlineExtrinsicUpdate(Eigen::VectorXd &solution, bool allow_rotation, bool allow_translation) const;
  void initializeVIO();
  void getImagePatch(const PerCameraData &ctx, const cv::Mat &img, V2D pc, float *patch_tmp, int level);
  void computeProjectionJacobian(const PerCameraData &ctx, V3D p, MD(2, 3) & J);
  void computeVirtualProjectionJacobian(const V3D &p_v, MD(2, 3) &J) const;
  void computeJacobianAndUpdateEKF();
  bool interpolateReferenceFeature(const Feature &reference, const V2D &px, float &value) const;
  bool computeWarpedReferenceGradient(const Feature &reference, const Matrix2d &A_cur_ref,
                                      int search_level, int level, int patch_index, V2D &gradient) const;
  void buildFixedTemplateGradientCache(int level);
  void resetGrid(PerCameraData &ctx);
  void updateVisualMapPoints(PerCameraData &ctx, const cv::Mat &img);
  void getWarpMatrixAffine(const PerCameraData &ref_ctx, const PerCameraData &cur_ctx, const Vector2d &px_ref,
                           const Vector3d &f_ref, const double depth_ref, const SE3<double> &T_cur_ref,
                           const int level_ref, 
                           const int pyramid_level, const int halfpatch_size, Matrix2d &A_cur_ref);
  void getWarpMatrixAffineHomography(const PerCameraData &ref_ctx, const PerCameraData &cur_ctx, const V2D &px_ref,
                                     const V3D &xyz_ref, const V3D &normal_ref, const SE3<double> &T_cur_ref, const int level_ref, Matrix2d &A_cur_ref);
  bool warpAffine(const Matrix2d &A_cur_ref, const cv::Mat &img_ref, const Vector2d &px_ref, const int level_ref, const int search_level,
                  const int pyramid_level, const int halfpatch_size, float *patch);
  bool buildVirtualFrameRotation(const PerCameraData &ctx, const V3D &point_in_raw_camera, const V2D &raw_center_px,
                                 M3D &R_v_from_c, M3D &R_c_from_v) const;
  bool projectRawFisheyeIfValid(const PerCameraData &ctx, const V3D &ray_or_point_in_raw_camera, int required_border, V2D &raw_px) const;
  bool computeS2SamplePointAndJacobian(const V3D &point_c, const M3D &R_v_from_c, const M3D &R_c_from_v,
                                       const V2D &offset, V3D &p_sample_c, M3D &J_sample_pc) const;
  bool projectEquidistantFisheyeWithJacobian(const PerCameraData &ctx, const V3D &p_c, int required_border,
                                             V2D &raw_px, MD(2, 3) &J_raw_pc) const;
  bool projectRawCameraWithJacobian(const PerCameraData &ctx, const V3D &p_c, int required_border,
                                    V2D &raw_px, MD(2, 3) &J_raw_pc) const;
  bool sampleRawImageValueAndGradient(const cv::Mat &raw_img, const V2D &raw_px, int scale,
                                      float &value, V2D &gradient) const;
  bool linearizeVirtualS2Sample(const PerCameraData &ctx, const cv::Mat &raw_img, const V3D &point_c,
                                const VirtualTrackPatch &track, const V2D &offset, int scale,
                                double current_exposure, float &current_value,
                                MD(1, 3) &J_photo_center) const;
  bool hasRangeDiscontinuity(const PerCameraData &ctx, const cv::Mat &range_img,
                             const V2D &raw_px, double point_range) const;
  bool buildVirtualSupportPatchPullExact(const PerCameraData &ctx, const cv::Mat &raw_img, const M3D &R_c_from_v,
                                         VirtualPatchImage &output, const cv::Point &raw_origin = cv::Point()) const;
  void splatRawPixelToVirtualPatch(const PerCameraData &ctx, const cv::Mat &raw_img, int raw_x, int raw_y, const M3D &R_v_from_c, cv::Mat &value_sum,
                                   cv::Mat &weight_sum) const;
  bool hasFullVirtualCoreCoverage(const cv::Mat &valid_mask) const;
  bool buildVirtualSupportPatchForwardSplat(const PerCameraData &ctx, const cv::Mat &raw_img, const V2D &raw_center_px,
                                            const M3D &R_v_from_c, VirtualPatchImage &output,
                                            const cv::Point &raw_origin = cv::Point()) const;
  bool buildVirtualSupportPatch(const PerCameraData &ctx, const cv::Mat &raw_img, const V2D &raw_center_px,
                                const M3D &R_v_from_c, const M3D &R_c_from_v, VirtualPatchImage &output,
                                const cv::Point &raw_origin = cv::Point()) const;
  bool captureVirtualReferenceSource(const PerCameraData &ctx, const cv::Mat &raw_img, const V2D &raw_center_px,
                                     const M3D &R_c_from_v, cv::Mat &raw_roi, cv::Point &raw_origin) const;
  bool materializeVirtualReferenceSupport(Feature &feature, bool &materialized_now);
  bool interpolateRawVirtualImage(const cv::Mat &img, double u, double v, float &value) const;
  bool interpolateFloatVirtualImage(const cv::Mat &img, const cv::Mat *valid_mask, double u, double v, float &value) const;
  bool interpolateVirtualFloat(const cv::Mat &img, const cv::Mat &valid_mask, float u, float v, float &value) const;
  bool interpolateStoredVirtualImage(const cv::Mat &img, float u, float v, float &value) const;
  V2D virtualProject(const V3D &p_v) const;
  V3D virtualCam2World(const V2D &px_v) const;
  bool createVirtualFeaturePatch(const PerCameraData &ctx, const cv::Mat &raw_img, const SE3<double> &T_c_w,
                                 const V3D &point_w, float *core_patch, cv::Mat &virtual_support_img,
                                 cv::Point &virtual_source_origin, SE3<double> &T_v_w,
                                 M3D &R_v_from_c, M3D &R_c_from_v) const;
  bool extractRefPatchDumpRawRoi(const PerCameraData &ctx, const cv::Mat &raw_img, const V2D &raw_px,
                                 const M3D &R_c_from_v, cv::Mat &raw_roi, M3D &raw_to_roi) const;
  void maybeInitializeRefPatchDumpProbe(const PerCameraData &ctx, const V3D &point_w, const V3D &normal_w,
                                        const V2D &raw_px, const V3D &bearing, const float *core_patch);
  void initializeRefPatchDump();
  void processRefPatchDumpProbe(PerCameraData &ctx, const cv::Mat &raw_img);
  bool buildRefPatchDumpWarpPatches(const PerCameraData &ctx, const cv::Mat &raw_img, const V2D &raw_px,
                                    const V3D &point_c, const SE3<double> &T_v_w, const cv::Mat &virtual_support_img,
                                    cv::Mat &raw_patch_display, cv::Mat &virtual_patch_display,
                                    std::vector<V2D> &raw_sample_pixels, std::vector<V2D> &virtual_sample_pixels,
                                    double &virtual_ncc);
  void dumpRefPatchProbeObservation(const PerCameraData &ctx, const V2D &raw_px, const V2D &virtual_px,
                                    double incidence_deg, const cv::Mat &raw_roi, const M3D &raw_to_roi,
                                    const cv::Mat &virtual_support_img, const std::vector<V2D> &raw_sample_pixels,
                                    const std::vector<V2D> &virtual_sample_pixels,
                                    const cv::Mat &raw_patch_display, const cv::Mat &virtual_patch_display);
  void initializeRuntimeSupportDump();
  void dumpRuntimeSupportObservation(const PerCameraData &ctx, const VisualPoint &point, const Feature &ref_ftr,
                                     const V2D &current_raw_center_px, const VirtualTrackPatch &track,
                                     const std::vector<float> &warped_reference, const std::vector<float> &current_core,
                                     int track_count, int submap_index, double error, double ncc);
  void dumpRuntimeSupportRawObservation(const PerCameraData &ctx, const VisualPoint &point, const Feature &ref_ftr,
                                        const V2D &current_raw_center_px, int search_level,
                                        const std::vector<float> &warped_reference,
                                        const std::vector<float> &current_core, int track_count,
                                        int submap_index, double error, double ncc);
  bool sampleRawCorePatchForDump(const PerCameraData &ctx, const cv::Mat &img, const V2D &pc,
                                 int scale, std::vector<float> &patch) const;
  bool getWarpMatrixAffineVirtual(const V3D &xyz_ref, const SE3<double> &T_vcur_vref, int level_ref, int pyramid_level,
                                  int halfpatch_size, Matrix2d &A_cur_ref) const;
  bool getWarpMatrixAffineHomographyVirtual(const V3D &xyz_ref, const V3D &normal_ref, const SE3<double> &T_vcur_vref,
                                            int level_ref, Matrix2d &A_cur_ref) const;
  bool warpAffineVirtual(const Matrix2d &A_cur_ref, const cv::Mat &virtual_ref_img, int level_ref, int search_level,
                         int pyramid_level, int halfpatch_size, float *patch) const;
  bool sampleVirtualCorePatch(const VirtualPatchImage &support, const V2D &center, int scale, float *patch) const;
  bool sampleSparseVirtualValue(const PerCameraData &ctx, const cv::Mat &raw_img, const M3D &R_c_from_v,
                                const V2D &px, float &value) const;
  bool sampleSparseVirtualCorePatch(const PerCameraData &ctx, const cv::Mat &raw_img, const M3D &R_c_from_v,
                                    const V2D &center, int scale, float *patch) const;
  bool buildSparseVirtualReferenceCorePatch(const PerCameraData &ctx, const cv::Mat &raw_img,
                                            const V2D &raw_center_px, const M3D &R_v_from_c,
                                            const M3D &R_c_from_v, float *patch) const;
  bool sampleSparseVirtualValueAndGradient(const PerCameraData &ctx, const cv::Mat &raw_img,
                                           const M3D &R_c_from_v, const V2D &px, int scale,
                                           float &value, V2D &gradient) const;
  bool sampleVirtualValueAndGradient(const VirtualPatchImage &support, const V2D &px, int scale, float &value, V2D &gradient) const;
  bool sampleStoredVirtualValueAndGradient(const cv::Mat &img, const V2D &px, int scale, float &value, V2D &gradient) const;
  SE3<double> composeVirtualPose(const M3D &R_v_from_c, const SE3<double> &T_c_w) const;
  bool passVisualGeometryFilter(const pointWithVar &candidate,
                                const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map,
                                VisualGeomRejectReason &reject_reason) const;
  void retrieveFromVisualSparseMapVirtual(PerCameraData &ctx, const cv::Mat &img, vector<pointWithVar> &pg,
                                          const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void generateVisualMapPointsVirtual(PerCameraData &ctx, const cv::Mat &img, vector<pointWithVar> &pg,
                                      const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void updateVisualMapPointsVirtual(PerCameraData &ctx, const cv::Mat &img);
  void updateStateVirtualS2(cv::Mat img, int level);
  void insertPointIntoVoxelMap(VisualPoint *pt_new);
  void materializePendingNewPointObservations(PerCameraData &ctx, const cv::Mat &img,
                                               const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void commitPendingNewPoints(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void validateVisualMapManageConfig() const;
  std::vector<Feature *> selectManagedReferenceCandidates(const PerCameraData &ctx, VisualPoint &point,
                                                           int max_candidates) const;
  bool shouldReferenceContributeToEkf(const VisualPoint &point, const Feature &reference) const;
  void appendManagedSubmapMetadata(PerCameraData &ctx, VisualPoint &point, Feature &reference);
  void updateManagedObservationEvidence(PerCameraData &ctx);
  void recordManagedReferenceRejection(VisualPoint &point, Feature &reference,
                                       int frame_id, int camera_id);
  bool shouldCreateManagedReference(const PerCameraData &ctx, const VisualPoint &point,
                                    const V2D &current_px, const Matrix2d &current_affine) const;
  void initializeManagedReference(Feature &feature, VisualPoint &point, const PerCameraData &ctx,
                                  bool initial_point_reference);
  bool associateVisualPointSurface(const V3D &point_w,
                                   const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map,
                                   int64_t &voxel_x, int64_t &voxel_y, int64_t &voxel_z,
                                   const VoxelPlane *&plane) const;
  bool computeManagedFootprint(const PerCameraData &ctx, const SE3<double> &T_c_w, const V3D &point_w,
                               const V3D &normal_w, const Feature *feature, const VoxelPlane &plane,
                               std::array<V3D, 4> &corners_w) const;
  double managedFootprintIoU(const std::array<V3D, 4> &a, const std::array<V3D, 4> &b,
                             const VoxelPlane &plane) const;
  VisualPoint *findRedundantVisualPoint(const PendingNewPointObservation &pending,
                                        const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map,
                                        double &best_iou);
  bool shouldReplaceRedundantPoint(const PendingNewPointObservation &pending, const VisualPoint &existing) const;
  void manageReferenceBank(VisualPoint &point);
  void queueReferenceRetirement(VisualPoint &point, Feature &feature);
  void queuePointRetirement(VisualPoint &point);
  void flushVisualMapRetirements();
  void printVisualMapManageStats(int frame_id) const;
  void plotTrackedPoints(PerCameraData &ctx);
  void updateFrameState(PerCameraData &ctx, const StatesGroup &state_value);
  void projectPatchFromRefToCur(PerCameraData &ctx, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void updateReferencePatch(PerCameraData &ctx, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void dumpDataForColmap();
  int usageRegionBin(const PerCameraData &ctx, const V2D &px) const;
  int usageViewAngleBin(const PerCameraData &ctx, const Feature &ref_ftr, const VisualPoint &pt) const;
  int usageFootprintBin(const Matrix2d &A_cur_ref) const;
  int usageAnisotropyBin(const Matrix2d &A_cur_ref) const;
  int usageNccBin(double ncc) const;
  void resetUsageStatsWindow();
  void resetUsageStatsTotals();
  void recordUsageObservation(const PerCameraData &ctx, const Feature &ref_ftr, const VisualPoint &pt,
                              const V2D &cur_px, const Matrix2d &A_cur_ref, bool accepted,
                              const std::vector<double> &sse_levels = std::vector<double>(),
                              const std::vector<double> &ncc_levels = std::vector<double>(),
                              const std::vector<uint8_t> &level_valid = std::vector<uint8_t>());
  void recordUsageCandidateNcc(const PerCameraData &ctx, const Feature &ref_ftr, const VisualPoint &pt,
                               const V2D &cur_px, const Matrix2d &A_cur_ref,
                               const std::vector<double> &ncc_levels,
                               const std::vector<uint8_t> &level_valid);
  void recordUsageEkfContribution(const PerCameraData &ctx, const Feature &ref_ftr, const V2D &cur_px,
                                  int level_patches, int residuals);
  void recordUsagePoseFrameInfo(const Eigen::MatrixXd &prior_cov, const Eigen::MatrixXd &h_all,
                                const Eigen::MatrixXd &h_same, const Eigen::MatrixXd &h_cross,
                                long long patches_all, long long residuals_all,
                                long long patches_same, long long residuals_same,
                                long long patches_cross, long long residuals_cross);
  void printUsageStatsTable(int frame_id);
  void maybePrintUsageStatsTable(int frame_id);
  double calculateNCC(float *ref_patch, float *cur_patch, int patch_size);
  int getBestSearchLevel(const Matrix2d &A_cur_ref, const int max_level);
  V3F getInterpolatedPixel(const cv::Mat &img, V2D pc) const;
  bool getColorFromCamera(int camera_id, const V3D &p_w, V3F &bgr, double *cam_range = nullptr) const;
  
  // void resetRvizDisplay();
  // deque<VisualPoint *> map_cur_frame;
  // deque<VisualPoint *> sub_map_ray;
  // deque<VisualPoint *> sub_map_ray_fov;
  // deque<VisualPoint *> visual_sub_map_cur;
  // deque<VisualPoint *> visual_converged_point;
  // std::vector<std::vector<V3D>> sample_points;

  // PointCloudXYZI::Ptr pg_down;
  // pcl::VoxelGrid<PointType> downSizeFilter;
};
typedef std::shared_ptr<VIOManager> VIOManagerPtr;

#endif // VIO_FISHEYE_H_
