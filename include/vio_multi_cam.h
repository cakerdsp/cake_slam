/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef VIO_FISHEYE_H_
#define VIO_FISHEYE_H_

#include "voxel_map.h"
#include "feature_fisheye.h"
#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/video/tracking.hpp>
#include <pcl/filters/voxel_grid.h>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
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

struct SubSparseMap
{
  vector<float> propa_errors;
  vector<float> errors;
  vector<vector<float>> warp_patch;
  vector<int> search_levels;
  vector<VisualPoint *> voxel_points;
  vector<double> inv_expo_list;
  vector<pointWithVar> add_from_voxel_map;
  vector<VirtualTrackPatch> virtual_track_patches;

  SubSparseMap()
  {
    propa_errors.reserve(SIZE_LARGE);
    errors.reserve(SIZE_LARGE);
    warp_patch.reserve(SIZE_LARGE);
    search_levels.reserve(SIZE_LARGE);
    voxel_points.reserve(SIZE_LARGE);
    inv_expo_list.reserve(SIZE_LARGE);
    add_from_voxel_map.reserve(SIZE_SMALL);
    virtual_track_patches.reserve(SIZE_LARGE);
  };

  void reset()
  {
    propa_errors.clear();
    errors.clear();
    warp_patch.clear();
    search_levels.clear();
    voxel_points.clear();
    inv_expo_list.clear();
    add_from_voxel_map.clear();
    virtual_track_patches.clear();
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

class VIOManager
{
public:
  int grid_size;
  vk::AbstractCamera *cam;
  vk::PinholeCamera *pinhole_cam;
  StatesGroup *state;
  StatesGroup *state_propagat;
  M3D Rli, Rci, Rcl, Rcw, Jdphi_dR, Jdp_dt, Jdp_dR;
  V3D Pli, Pci, Pcl, Pcw;
  vector<int> grid_num;
  vector<int> map_index;
  vector<int> border_flag;
  vector<int> update_flag;
  vector<float> map_dist;
  vector<float> scan_value;
  vector<float> patch_buffer;
  bool normal_en, inverse_composition_en, exposure_estimate_en, raycast_en, has_ref_patch_cache;
  bool ncc_en = false, colmap_output_en = false;
  bool virtual_fisheye_patch_en = false;

  int width, height, grid_n_width, grid_n_height, length;
  double image_resize_factor;
  double fx, fy, cx, cy;
  int patch_pyrimid_level, patch_size, patch_size_total, patch_size_half, border, warp_len;
  int max_iterations, total_points;

  double img_point_cov, outlier_threshold, ncc_thre;

  double virtual_focal_length = 300.0;
  int virtual_patch_margin = 4;
  int virtual_max_search_level = 1;
  double virtual_min_z = 1.0e-6;
  std::string virtual_patch_resampling_mode = "forward_splat";
  VirtualPatchResamplingMode virtual_patch_resampling_mode_enum = VirtualPatchResamplingMode::FORWARD_SPLAT;
  int virtual_raw_window_half_size = 48;
  double virtual_splat_min_weight = 1.0e-6;
  bool virtual_splat_require_full_core_coverage = true;
  bool virtual_splat_debug_compare_pull_exact = false;
  bool draw_rejected_points_en = false;
  int virtual_support_radius = 0;
  int virtual_support_size = 0;
  std::vector<V3F> virtual_support_ray_lut_;
  std::vector<V2F> core_patch_offsets_;
  std::vector<V3F> raw_pixel_to_unit_ray_lut_;
  std::vector<uint8_t> raw_pixel_unit_ray_valid_mask_;

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
  std::vector<RejectedVisualPointForDraw> rejected_visual_points_for_draw_;
  
  SubSparseMap *visual_submap;
  std::vector<std::vector<V3D>> rays_with_sample_points;

  double compute_jacobian_time, update_ekf_time;
  double ave_total = 0;
  // double ave_build_residual_time = 0;
  // double ave_ekf_time = 0;

  int frame_count = 0;
  bool plot_flag;

  Eigen::Matrix<double, DIM_STATE, DIM_STATE> G, H_T_H;
  Eigen::MatrixXd K, H_sub_inv;

  ofstream fout_camera, fout_colmap;
  unordered_map<VOXEL_LOCATION, VOXEL_POINTS *> feat_map;
  unordered_map<VOXEL_LOCATION, int> sub_feat_map; 
  unordered_map<int, Warp *> warp_map;
  vector<VisualPoint *> retrieve_voxel_points;
  vector<pointWithVar> append_voxel_points;
  FramePtr new_frame_;
  cv::Mat img_cp, img_rgb, img_test;
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

  VIOManager();
  ~VIOManager();
  void updateStateInverse(cv::Mat img, int level);
  void updateState(cv::Mat img, int level);
  void processFrameOpticalFlow(cv::Mat &img, double img_time);
  void processFrame(cv::Mat &img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map, double img_time);
  void processFrameFake(cv::Mat &img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map, double img_time);
  bool inOpticalFlowBorder(const cv::Point2f &pt) const;
  V3D getOpticalFlowBearing(const cv::Point2f &px) const;
  void setOpticalFlowMask(cv::Mat &mask);
  void addOpticalFlowObservation(OpticalFlowTrack &track, const cv::Point2f &px, double img_time);
  bool triangulateOpticalFlowTrack(OpticalFlowTrack &track);
  void updateOpticalFlowPointClouds();
  void drawOpticalFlowDebugImage(const std::vector<cv::Point2f> &rejected_pts, int prev, int tracked, int flow_back_pass,
                                 int border_pass, int mask_reject, int new_points, int final_points, int triangulated);
  void retrieveFromVisualSparseMap(cv::Mat img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void generateVisualMapPoints(cv::Mat img, vector<pointWithVar> &pg);
  void setImuToLidarExtrinsic(const V3D &transl, const M3D &rot);
  void setLidarToCameraExtrinsic(vector<double> &R, vector<double> &P);
  void initializeVIO();
  void getImagePatch(cv::Mat img, V2D pc, float *patch_tmp, int level);
  void computeProjectionJacobian(V3D p, MD(2, 3) & J);
  void computeVirtualProjectionJacobian(const V3D &p_v, MD(2, 3) &J) const;
  void computeJacobianAndUpdateEKF(cv::Mat img);
  void resetGrid();
  void updateVisualMapPoints(cv::Mat img);
  void getWarpMatrixAffine(const vk::AbstractCamera &cam, const Vector2d &px_ref, const Vector3d &f_ref, const double depth_ref, const SE3<double> &T_cur_ref,
                           const int level_ref, 
                           const int pyramid_level, const int halfpatch_size, Matrix2d &A_cur_ref);
  void getWarpMatrixAffineHomography(const vk::AbstractCamera &cam, const V2D &px_ref,
                                     const V3D &xyz_ref, const V3D &normal_ref, const SE3<double> &T_cur_ref, const int level_ref, Matrix2d &A_cur_ref);
  void warpAffine(const Matrix2d &A_cur_ref, const cv::Mat &img_ref, const Vector2d &px_ref, const int level_ref, const int search_level,
                  const int pyramid_level, const int halfpatch_size, float *patch);
  bool buildVirtualFrameRotation(const V3D &point_in_raw_camera, M3D &R_v_from_c, M3D &R_c_from_v) const;
  bool projectRawFisheyeIfValid(const V3D &ray_or_point_in_raw_camera, int required_border, V2D &raw_px) const;
  bool buildVirtualSupportPatchPullExact(const cv::Mat &raw_img, const M3D &R_c_from_v, VirtualPatchImage &output) const;
  void splatRawPixelToVirtualPatch(const cv::Mat &raw_img, int raw_x, int raw_y, const M3D &R_v_from_c, cv::Mat &value_sum,
                                   cv::Mat &weight_sum) const;
  bool hasFullVirtualCoreCoverage(const cv::Mat &valid_mask) const;
  bool buildVirtualSupportPatchForwardSplat(const cv::Mat &raw_img, const V2D &raw_center_px, const M3D &R_v_from_c,
                                            VirtualPatchImage &output) const;
  bool buildVirtualSupportPatch(const cv::Mat &raw_img, const V2D &raw_center_px, const M3D &R_v_from_c, const M3D &R_c_from_v,
                                VirtualPatchImage &output) const;
  bool interpolateVirtualFloat(const cv::Mat &img, const cv::Mat &valid_mask, float u, float v, float &value) const;
  bool interpolateStoredVirtualImage(const cv::Mat &img, float u, float v, float &value) const;
  V2D virtualProject(const V3D &p_v) const;
  V3D virtualCam2World(const V2D &px_v) const;
  bool createVirtualFeaturePatch(const cv::Mat &raw_img, const SE3<double> &T_c_w, const V3D &point_w, float *core_patch,
                                 cv::Mat &virtual_support_img, SE3<double> &T_v_w, M3D &R_v_from_c, M3D &R_c_from_v) const;
  bool getWarpMatrixAffineVirtual(const V3D &xyz_ref, const SE3<double> &T_vcur_vref, int level_ref, int pyramid_level,
                                  int halfpatch_size, Matrix2d &A_cur_ref) const;
  bool getWarpMatrixAffineHomographyVirtual(const V3D &xyz_ref, const V3D &normal_ref, const SE3<double> &T_vcur_vref,
                                            int level_ref, Matrix2d &A_cur_ref) const;
  bool warpAffineVirtual(const Matrix2d &A_cur_ref, const cv::Mat &virtual_ref_img, int level_ref, int search_level,
                         int pyramid_level, int halfpatch_size, float *patch) const;
  bool sampleVirtualCorePatch(const VirtualPatchImage &support, const V2D &center, int scale, float *patch) const;
  bool sampleVirtualValueAndGradient(const VirtualPatchImage &support, const V2D &px, int scale, float &value, V2D &gradient) const;
  bool sampleStoredVirtualValueAndGradient(const cv::Mat &img, const V2D &px, int scale, float &value, V2D &gradient) const;
  SE3<double> composeVirtualPose(const M3D &R_v_from_c, const SE3<double> &T_c_w) const;
  void retrieveFromVisualSparseMapVirtual(cv::Mat img, vector<pointWithVar> &pg,
                                          const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void generateVisualMapPointsVirtual(cv::Mat img, vector<pointWithVar> &pg);
  void updateVisualMapPointsVirtual(cv::Mat img);
  void precomputeReferencePatchesVirtual(int level);
  void updateStateVirtual(cv::Mat img, int level);
  void updateStateInverseVirtual(cv::Mat img, int level);
  void insertPointIntoVoxelMap(VisualPoint *pt_new);
  void plotTrackedPoints();
  void updateFrameState(StatesGroup state);
  void projectPatchFromRefToCur(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void updateReferencePatch(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void precomputeReferencePatches(int level);
  void dumpDataForColmap();
  double calculateNCC(float *ref_patch, float *cur_patch, int patch_size);
  int getBestSearchLevel(const Matrix2d &A_cur_ref, const int max_level);
  V3F getInterpolatedPixel(cv::Mat img, V2D pc);
  
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
