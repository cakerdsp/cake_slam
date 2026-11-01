// Baseline from commit 23e710b89eea774c401c609f0e6f7cf6d4c140ac.
// Included only by vio_multi_cam.cpp. Bodies are checked against frozen hashes
// by check_visual_patch_budget_baseline.py; do not replace this with the split
// preparation/tracking path when patch_budget_en is false.

void VIOManager::retrieveFromVisualSparseMapVirtualBaseline(PerCameraData &ctx, const cv::Mat &img, vector<pointWithVar> &pg,
                                                    const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map,
                                                    int raw_score_point_quota)
{
  // One cache per current camera frame, shared by historical/current pairs.
  // No allocation or filter operation in the disabled path.
  if (virtual_adaptive_sampling_en)
    ctx.adaptive_image_pyramid = std::make_shared<virtual_sampling::ImagePyramid>(img);
  ctx.visual_submap->reset();
  ctx.total_points = 0;
  if (ref_patch_dump_en && ctx.camera_id == kRefPatchDumpCameraId)
  {
    ref_patch_dump_range_img_.release();
    ref_patch_dump_range_pose_valid_ = false;
  }
  rejected_virtual_support_oob_ = 0;
  rejected_virtual_projection_invalid_ = 0;
  rejected_virtual_z_ = 0;
  rejected_virtual_affine_oob_ = 0;
  build_virtual_support_time_ = 0.0;
  virtual_affine_time_ = 0.0;
  virtual_candidate_select_time_ = 0.0;
  virtual_parallel_track_time_ = 0.0;
  virtual_result_collect_time_ = 0.0;
  virtual_warp_time_ = 0.0;
  virtual_current_core_time_ = 0.0;
  virtual_ref_support_materialized_count_ = 0;
  virtual_ref_support_materialize_fail_count_ = 0;
  virtual_ref_support_materialize_time_ = 0.0;
  virtual_map_grid_count_ = 0;
  virtual_candidate_null_count_ = 0;
  virtual_candidate_normal_uninit_count_ = 0;
  virtual_candidate_projection_fail_count_ = 0;
  virtual_candidate_range_reject_count_ = 0;
  virtual_candidate_close_view_fail_count_ = 0;
  virtual_candidate_ref_missing_count_ = 0;
  virtual_candidate_ref_invalid_count_ = 0;
  virtual_candidate_count_ = 0;
  virtual_valid_track_count_ = 0;
  virtual_track_rotation_fail_count_ = 0;
  virtual_track_support_fail_count_ = 0;
  virtual_track_affine_fail_count_ = 0;
  virtual_track_warp_fail_count_ = 0;
  virtual_track_current_z_fail_count_ = 0;
  virtual_track_current_core_fail_count_ = 0;
  virtual_track_ncc_reject_count_ = 0;
  virtual_track_photometric_reject_count_ = 0;
  if (draw_rejected_points_en && virtual_fisheye_patch_en)
  {
    ctx.rejected_visual_points_for_draw.clear();
    ctx.rejected_visual_points_for_draw.reserve(ctx.length);
  }
  if (feat_map.empty()) return;

  const double candidate_select_start = omp_get_wtime();
  ctx.sub_feat_map.clear();

  const float voxel_size = 0.5f;
  cv::Mat range_img = cv::Mat::zeros(ctx.height, ctx.width, CV_32FC1);
  int loc_xyz[3];

  for (const auto &point : pg)
  {
    const V3D &pt_w = point.point_w;
    for (int j = 0; j < 3; ++j) loc_xyz[j] = static_cast<int>(std::floor(pt_w[j] / voxel_size));
    ctx.sub_feat_map[VOXEL_LOCATION(loc_xyz[0], loc_xyz[1], loc_xyz[2])] = 0;

    const V3D pt_c = ctx.new_frame->w2f(pt_w);
    V2D px;
    if (!projectRawFisheyeIfValid(ctx, pt_c, 1, px)) continue;
    const int col = static_cast<int>(px[0]);
    const int row = static_cast<int>(px[1]);
    float &stored_range = range_img.at<float>(row, col);
    const float range = static_cast<float>(pt_c.norm());
    if (stored_range == 0.0f || range < stored_range) stored_range = range;
  }

  if (ref_patch_dump_en && ctx.camera_id == kRefPatchDumpCameraId)
  {
    ref_patch_dump_range_img_ = range_img;
    ref_patch_dump_range_T_f_w_ = ctx.new_frame->T_f_w_;
    ref_patch_dump_range_pose_valid_ = true;
  }

  auto addRetrieveCandidate = [&](int index, VisualPoint *pt, float cur_dist) {
    if (index < 0 || index >= ctx.length || pt == nullptr) return;
    ctx.grid_num[index] = TYPE_MAP;
    if (virtual_raw_score_select_en)
    {
      std::vector<VisualPoint *> &grid_candidates = ctx.retrieve_voxel_point_candidates[index];
      if (std::find(grid_candidates.begin(), grid_candidates.end(), pt) == grid_candidates.end())
        grid_candidates.push_back(pt);
    }
    if (cur_dist <= ctx.map_dist[index])
    {
      ctx.map_dist[index] = cur_dist;
      ctx.retrieve_voxel_points[index] = pt;
    }
  };

  vector<VOXEL_LOCATION> delete_keys;
  for (auto &sub_voxel : ctx.sub_feat_map)
  {
    auto map_voxel = feat_map.find(sub_voxel.first);
    if (map_voxel == feat_map.end()) continue;

    bool voxel_in_fov = false;
    for (VisualPoint *pt : map_voxel->second->voxel_points)
    {
      if (pt == nullptr || pt->obs_.empty()) continue;
      V2D raw_px;
      if (!projectRawFisheyeIfValid(ctx, ctx.new_frame->w2f(pt->pos_), 1, raw_px)) continue;
      const int grid_col = static_cast<int>(raw_px[0] / ctx.grid_size);
      const int grid_row = static_cast<int>(raw_px[1] / ctx.grid_size);
      if (grid_col < 0 || grid_col >= ctx.grid_n_width || grid_row < 0 || grid_row >= ctx.grid_n_height) continue;
      const int index = grid_row * ctx.grid_n_width + grid_col;
      voxel_in_fov = true;
      const float cur_dist = static_cast<float>((ctx.new_frame->pos() - pt->pos_).norm());
      addRetrieveCandidate(index, pt, cur_dist);
    }
    if (!voxel_in_fov) delete_keys.push_back(sub_voxel.first);
  }

  if (raycast_en)
  {
    for (int i = 0; i < ctx.length; ++i)
    {
      if (ctx.grid_num[i] == TYPE_MAP || ctx.border_flag[i] == 1) continue;
      for (const V3D &sample_c : ctx.rays_with_sample_points[i])
      {
        const V3D sample_w = ctx.new_frame->f2w(sample_c);
        for (int j = 0; j < 3; ++j) loc_xyz[j] = static_cast<int>(std::floor(sample_w[j] / voxel_size));
        const VOXEL_LOCATION sample_pos(loc_xyz[0], loc_xyz[1], loc_xyz[2]);
        if (ctx.sub_feat_map.find(sample_pos) != ctx.sub_feat_map.end()) break;

        auto visual_voxel = feat_map.find(sample_pos);
        if (visual_voxel != feat_map.end())
        {
          bool found = false;
          for (VisualPoint *pt : visual_voxel->second->voxel_points)
          {
            if (pt == nullptr || pt->obs_.empty()) continue;
            V2D raw_px;
            if (!projectRawFisheyeIfValid(ctx, ctx.new_frame->w2f(pt->pos_), 1, raw_px)) continue;
            const int grid_col = static_cast<int>(raw_px[0] / ctx.grid_size);
            const int grid_row = static_cast<int>(raw_px[1] / ctx.grid_size);
            if (grid_col < 0 || grid_col >= ctx.grid_n_width || grid_row < 0 || grid_row >= ctx.grid_n_height) continue;
            const int index = grid_row * ctx.grid_n_width + grid_col;
            const float cur_dist = static_cast<float>((ctx.new_frame->pos() - pt->pos_).norm());
            addRetrieveCandidate(index, pt, cur_dist);
            found = true;
          }
          if (found) ctx.sub_feat_map[sample_pos] = 0;
          break;
        }

        auto plane_voxel = plane_map.find(sample_pos);
        if (plane_voxel != plane_map.end())
        {
          VoxelOctoTree *octo = plane_voxel->second->find_correspond(sample_w);
          if (octo->plane_ptr_->is_plane_)
          {
            pointWithVar plane_center;
            plane_center.point_w = octo->plane_ptr_->center_;
            plane_center.normal = octo->plane_ptr_->normal_;
            ctx.visual_submap->add_from_voxel_map.push_back(plane_center);
            break;
          }
        }
      }
    }
  }
  for (const auto &key : delete_keys) ctx.sub_feat_map.erase(key);

  struct VirtualCandidate
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VisualPoint *point = nullptr;
    Feature *reference = nullptr;
    V3D point_c_seed = V3D::Zero();
    V2D current_raw_center_px = V2D::Zero();
    double shi_tomasi_score = 0.0;
    int reference_rank = 0;
  };
  enum VirtualRejectReason
  {
    VIRTUAL_REJECT_NONE = 0,
    VIRTUAL_REJECT_ROTATION,
    VIRTUAL_REJECT_SUPPORT_BUILD,
    VIRTUAL_REJECT_AFFINE_MATRIX,
    VIRTUAL_REJECT_REFERENCE_SUPPORT,
    VIRTUAL_REJECT_REFERENCE_WARP,
    VIRTUAL_REJECT_CURRENT_Z,
    VIRTUAL_REJECT_CURRENT_CORE,
    VIRTUAL_REJECT_NCC,
    VIRTUAL_REJECT_PHOTOMETRIC
  };
  struct VirtualCandidateResult
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VirtualTrackPatch track;
    vector<float> warped_reference;
    vector<float> current_core;
    std::vector<double> level_sse;
    std::vector<double> level_ncc;
    std::vector<uint8_t> level_valid;
    std::vector<uint8_t> level_active;
    float error = 0.0f;
    double inverse_reference_exposure = 0.0;
    double ncc = std::numeric_limits<double>::quiet_NaN();
    double build_time = 0.0;
    double affine_time = 0.0;
    double ref_materialize_time = 0.0;
    double warp_time = 0.0;
    double current_core_time = 0.0;
    double adaptive_time = 0.0;
    bool adaptive_attempted = false;
    int search_level = -1;
    int warp_fail_pyramid_level = -1;
    int rejection = 0;
    bool ref_materialized = false;
    bool valid = false;
  };
  auto recordRejectedPoint = [&](const V2D &raw_px, int reason) {
    if (draw_rejected_points_en)
    {
      ctx.rejected_visual_points_for_draw.emplace_back(
          cv::Point2f(static_cast<float>(raw_px[0]), static_cast<float>(raw_px[1])), reason);
    }
  };
  auto drawReasonFromVirtualReject = [](int rejection) {
    switch (rejection)
    {
    case VIRTUAL_REJECT_ROTATION:
      return VIOManager::REJECT_DRAW_ROTATION;
    case VIRTUAL_REJECT_SUPPORT_BUILD:
      return VIOManager::REJECT_DRAW_SUPPORT_BUILD;
    case VIRTUAL_REJECT_AFFINE_MATRIX:
      return VIOManager::REJECT_DRAW_AFFINE;
    case VIRTUAL_REJECT_REFERENCE_SUPPORT:
      return VIOManager::REJECT_DRAW_REF_INVALID;
    case VIRTUAL_REJECT_REFERENCE_WARP:
      return VIOManager::REJECT_DRAW_WARP_REF;
    case VIRTUAL_REJECT_CURRENT_Z:
      return VIOManager::REJECT_DRAW_CURRENT_Z;
    case VIRTUAL_REJECT_CURRENT_CORE:
      return VIOManager::REJECT_DRAW_CURRENT_CORE;
    case VIRTUAL_REJECT_NCC:
      return VIOManager::REJECT_DRAW_NCC;
    case VIRTUAL_REJECT_PHOTOMETRIC:
      return VIOManager::REJECT_DRAW_PHOTOMETRIC;
    default:
      return VIOManager::REJECT_DRAW_RANGE;
    }
  };
  auto computeShiTomasiCandidateScore = [&](const V2D &raw_px) {
    const float score = vk::shiTomasiScore(img, static_cast<int>(raw_px[0]), static_cast<int>(raw_px[1]));
    return std::isfinite(score) ? static_cast<double>(score) : 0.0;
  };

  vector<VirtualCandidate> candidates;
  candidates.reserve(ctx.length);
  for (int i = 0; i < ctx.length; ++i)
  {
    if (ctx.grid_num[i] != TYPE_MAP) continue;
    ++virtual_map_grid_count_;
    const std::vector<VisualPoint *> &grid_candidates = ctx.retrieve_voxel_point_candidates[i];
    const int grid_candidate_count = virtual_raw_score_select_en && !grid_candidates.empty()
                                         ? static_cast<int>(grid_candidates.size())
                                         : 1;
    for (int grid_candidate_index = 0; grid_candidate_index < grid_candidate_count; ++grid_candidate_index)
    {
    VisualPoint *pt = virtual_raw_score_select_en && !grid_candidates.empty()
                          ? grid_candidates[grid_candidate_index]
                          : ctx.retrieve_voxel_points[i];
    if (pt == nullptr)
    {
      ++virtual_candidate_null_count_;
      continue;
    }
    if (!pt->is_normal_initialized_)
    {
      ++virtual_candidate_normal_uninit_count_;
      if (draw_rejected_points_en)
      {
        V2D raw_px;
        if (projectRawFisheyeIfValid(ctx, ctx.new_frame->w2f(pt->pos_), 1, raw_px))
          recordRejectedPoint(raw_px, REJECT_DRAW_NORMAL_UNINIT);
      }
      continue;
    }

    V2D raw_px;
    const V3D pt_c_seed = ctx.new_frame->w2f(pt->pos_);
    if (!projectRawFisheyeIfValid(ctx, pt_c_seed, 1, raw_px))
    {
      ++virtual_candidate_projection_fail_count_;
      continue;
    }

    bool range_discontinuous = false;
    const double point_range = pt_c_seed.norm();
    const int center_col = static_cast<int>(raw_px[0]);
    const int center_row = static_cast<int>(raw_px[1]);
    for (int dv = -patch_size_half; dv <= patch_size_half && !range_discontinuous; ++dv)
    {
      for (int du = -patch_size_half; du <= patch_size_half; ++du)
      {
        if (du == 0 && dv == 0) continue;
        const int col = center_col + du;
        const int row = center_row + dv;
        if (col < 0 || col >= ctx.width || row < 0 || row >= ctx.height) continue;
        const float range = range_img.at<float>(row, col);
        if (range > 0.0f && std::fabs(point_range - range) > 0.5)
        {
          range_discontinuous = true;
          break;
        }
      }
    }
    if (range_discontinuous)
    {
      ++virtual_candidate_range_reject_count_;
      recordRejectedPoint(raw_px, REJECT_DRAW_RANGE);
      continue;
    }

    if (visual_map_manage_en && visual_ref_current_select_en && visual_map_manage_shadow_en)
    {
      const int candidate_limit = visual_ref_fallback_en ? visual_ref_max_candidates : 1;
      if (!selectManagedReferenceCandidates(ctx, *pt, candidate_limit).empty())
        ++visual_map_manage_stats_.dynamic_selected;
    }
    if (visual_map_manage_en && visual_ref_current_select_en && !visual_map_manage_shadow_en)
    {
      const int candidate_limit = visual_ref_fallback_en ? visual_ref_max_candidates : 1;
      std::vector<Feature *> managed_refs = selectManagedReferenceCandidates(ctx, *pt, candidate_limit);
      if (managed_refs.empty())
      {
        ++virtual_candidate_ref_missing_count_;
        recordRejectedPoint(raw_px, REJECT_DRAW_REF_MISSING);
        continue;
      }
      for (int rank = 0; rank < static_cast<int>(managed_refs.size()); ++rank)
      {
        Feature *managed_ref = managed_refs[rank];
        if (managed_ref == nullptr || !managed_ref->virtual_patch_valid_) continue;
        VirtualCandidate candidate;
        candidate.point = pt;
        candidate.reference = managed_ref;
        candidate.point_c_seed = pt_c_seed;
        candidate.current_raw_center_px = raw_px;
        candidate.shi_tomasi_score = computeShiTomasiCandidateScore(raw_px);
        candidate.reference_rank = rank;
        candidates.push_back(candidate);
      }
      ++visual_map_manage_stats_.dynamic_selected;
      continue;
    }

    Feature *ref_ftr = nullptr;
    if (normal_en)
    {
      pt->ensureCameraCount(numCameras());
      Feature *cached_ref = pt->referencePatch(ctx.camera_id, cross_camera_reference_en);
      if (cached_ref != nullptr && cached_ref->virtual_patch_valid_)
      {
        ref_ftr = cached_ref;
      }
      else
      {
        float minimum_error = std::numeric_limits<float>::max();
        for (Feature *candidate_ref : pt->obs_)
        {
          if (candidate_ref == nullptr || !candidate_ref->virtual_patch_valid_) continue;
          if (!cross_camera_reference_en && candidate_ref->camera_id_ != ctx.camera_id) continue;
          float error = 0.0f;
          int comparisons = 0;
          for (Feature *other : pt->obs_)
          {
            if (other == nullptr || other == candidate_ref || !other->virtual_patch_valid_) continue;
            if (!cross_camera_reference_en && other->camera_id_ != ctx.camera_id) continue;
            for (int patch_index = 0; patch_index < patch_size_total; ++patch_index)
            {
              const float residual = candidate_ref->patch_[patch_index] - other->patch_[patch_index];
              error += residual * residual;
            }
            ++comparisons;
          }
          if (comparisons > 0) error /= comparisons;
          if (error < minimum_error)
          {
            minimum_error = error;
            ref_ftr = candidate_ref;
          }
        }
        if (ref_ftr != nullptr)
        {
          if (cross_camera_reference_en)
          {
            pt->ref_patch = ref_ftr;
            pt->has_ref_patch_ = true;
          }
          else
          {
            pt->ref_patch_by_camera_[ctx.camera_id] = ref_ftr;
            pt->has_ref_patch_by_camera_[ctx.camera_id] = 1;
          }
        }
      }
    }
    else if (!pt->getCloseViewObs(ctx.new_frame->pos(), ref_ftr, raw_px,
                                  cross_camera_reference_en ? -1 : ctx.camera_id))
    {
      ++virtual_candidate_close_view_fail_count_;
      recordRejectedPoint(raw_px, REJECT_DRAW_CLOSE_VIEW);
      continue;
    }
    if (ref_ftr == nullptr)
    {
      ++virtual_candidate_ref_missing_count_;
      recordRejectedPoint(raw_px, REJECT_DRAW_REF_MISSING);
      continue;
    }
    if (!ref_ftr->virtual_patch_valid_)
    {
      ++virtual_candidate_ref_invalid_count_;
      recordRejectedPoint(raw_px, REJECT_DRAW_REF_INVALID);
      continue;
    }

    VirtualCandidate candidate;
    candidate.point = pt;
    candidate.reference = ref_ftr;
    candidate.point_c_seed = pt_c_seed;
    candidate.current_raw_center_px = raw_px;
    candidate.shi_tomasi_score = computeShiTomasiCandidateScore(raw_px);
    candidates.push_back(candidate);
    }
  }

  if (virtual_raw_score_select_en && raw_score_point_quota >= 0)
  {
    std::vector<int> primary_indices;
    primary_indices.reserve(candidates.size());
    for (int i = 0; i < static_cast<int>(candidates.size()); ++i)
      if (candidates[i].reference_rank == 0) primary_indices.push_back(i);

    const int keep_points = std::min(raw_score_point_quota, static_cast<int>(primary_indices.size()));
    if (keep_points < static_cast<int>(primary_indices.size()))
      std::partial_sort(primary_indices.begin(), primary_indices.begin() + keep_points, primary_indices.end(),
                        [&](int lhs, int rhs) {
                          return candidates[lhs].shi_tomasi_score > candidates[rhs].shi_tomasi_score;
                        });

    std::set<VisualPoint *> selected_points;
    for (int i = 0; i < keep_points; ++i) selected_points.insert(candidates[primary_indices[i]].point);

    vector<VirtualCandidate> selected_candidates;
    selected_candidates.reserve(candidates.size());
    for (const VirtualCandidate &candidate : candidates)
      if (selected_points.count(candidate.point) != 0) selected_candidates.push_back(candidate);
    candidates.swap(selected_candidates);
  }
  virtual_candidate_select_time_ = omp_get_wtime() - candidate_select_start;
  virtual_candidate_count_ = static_cast<int>(candidates.size());

  vector<VirtualCandidateResult> results(candidates.size());
  const double parallel_track_start = omp_get_wtime();
  for (int fallback_pass = 0; fallback_pass < 2; ++fallback_pass)
  {
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int candidate_index = 0; candidate_index < static_cast<int>(candidates.size()); ++candidate_index)
    {
    const VirtualCandidate &candidate = candidates[candidate_index];
    const bool is_primary = candidate.reference_rank == 0;
    if (is_primary != (fallback_pass == 0)) continue;
    if (!is_primary)
    {
      int primary_index = candidate_index - 1;
      while (primary_index >= 0 && candidates[primary_index].point == candidate.point &&
             candidates[primary_index].reference_rank != 0)
        --primary_index;
      if (primary_index < 0 || candidates[primary_index].point != candidate.point) continue;
      const int primary_rejection = results[primary_index].rejection;
      if (primary_rejection != VIRTUAL_REJECT_NCC && primary_rejection != VIRTUAL_REJECT_PHOTOMETRIC)
        continue;
    }
    VirtualCandidateResult &result = results[candidate_index];
    VisualPoint *pt = candidate.point;
    Feature *ref_ftr = candidate.reference;
    if (ref_ftr == nullptr || !refreshReferenceCalibration(*ref_ftr))
    {
      result.rejection = VIRTUAL_REJECT_REFERENCE_SUPPORT;
      continue;
    }

    if (!buildVirtualFrameRotation(ctx, candidate.point_c_seed, candidate.current_raw_center_px,
                                   result.track.R_vcur_from_ccur_seed, result.track.R_ccur_from_vcur_seed))
    {
      result.rejection = VIRTUAL_REJECT_ROTATION;
      continue;
    }
    result.track.T_vcur_w_seed = composeVirtualPose(result.track.R_vcur_from_ccur_seed, ctx.new_frame->T_f_w_);
    result.track.cur_support.T_v_w_seed = result.track.T_vcur_w_seed;

    const SE3d T_vcur_vref = result.track.T_vcur_w_seed * ref_ftr->T_v_w_.inverse();
    const V3D point_vref = ref_ftr->T_v_w_ * pt->pos_;
    const double affine_start = omp_get_wtime();
    bool affine_ok;
    if (normal_en)
    {
      const V3D normal_vref = ref_ftr->T_v_w_.rotationMatrix() * pt->normal_;
      affine_ok = getWarpMatrixAffineHomographyVirtual(point_vref, normal_vref, T_vcur_vref, ref_ftr->level_, result.track.A_cur_ref);
    }
    else
    {
      affine_ok = getWarpMatrixAffineVirtual(point_vref, T_vcur_vref, ref_ftr->level_, 0, patch_size_half, result.track.A_cur_ref);
    }
    result.affine_time = omp_get_wtime() - affine_start;
    if (!affine_ok)
    {
      result.rejection = VIRTUAL_REJECT_AFFINE_MATRIX;
      continue;
    }
    if (visual_map_manage_en && visual_ref_current_select_en)
    {
      const Eigen::JacobiSVD<Matrix2d> affine_svd(result.track.A_cur_ref);
      const V2D singular = affine_svd.singularValues();
      if (!singular.array().isFinite().all() || singular[1] <= 1.0e-9 ||
          singular[0] / singular[1] > visual_ref_max_anisotropy)
      {
        result.rejection = VIRTUAL_REJECT_AFFINE_MATRIX;
        continue;
      }
    }
    result.track.search_level = getBestSearchLevel(result.track.A_cur_ref, virtual_max_search_level);
    result.search_level = result.track.search_level;

    if (virtual_sparse_patch_en || virtual_adaptive_sampling_en)
    {
      const double materialize_start = omp_get_wtime();
      if (!materializeVirtualReferenceSupport(*ref_ftr, result.ref_materialized))
      {
        result.ref_materialize_time = omp_get_wtime() - materialize_start;
        result.rejection = VIRTUAL_REJECT_REFERENCE_SUPPORT;
        continue;
      }
      result.ref_materialize_time = omp_get_wtime() - materialize_start;
    }

    const double warp_start = omp_get_wtime();
    result.warped_reference.assign(warp_len, 0.0f);
    bool adaptive_ready = false;
    if (virtual_adaptive_sampling_en)
    {
      result.adaptive_attempted = true;
      const double adaptive_start = omp_get_wtime();
      adaptive_ready = prepareAdaptiveTrack(ctx, *ref_ftr, result.track, result.warped_reference);
      result.adaptive_time = omp_get_wtime() - adaptive_start;
      result.search_level = result.track.search_level;
    }
    for (int pyramid_level = 0; !adaptive_ready && pyramid_level < patch_pyrimid_level; ++pyramid_level)
    {
      // Reuse the stored reference support instead of regenerating it every frame.
      if (!warpAffineVirtual(result.track.A_cur_ref, ref_ftr->img_, ref_ftr->level_, result.track.search_level, pyramid_level,
                             patch_size_half, result.warped_reference.data()))
      {
        result.rejection = VIRTUAL_REJECT_REFERENCE_WARP;
        result.warp_fail_pyramid_level = pyramid_level;
        break;
      }
    }
    result.warp_time = omp_get_wtime() - warp_start;
    if (result.rejection != 0) continue;

    if (!virtual_sparse_patch_en && (!adaptive_ready || cross_camera_current_residual_en))
    {
      // The affine warp and warped reference depend only on geometry and the
      // immutable reference support. Reject them before constructing the more
      // expensive current virtual support.
      const double build_start = omp_get_wtime();
      const bool supports_ok = buildVirtualSupportPatch(ctx, img, candidate.current_raw_center_px,
                                                        result.track.R_vcur_from_ccur_seed,
                                                        result.track.R_ccur_from_vcur_seed,
                                                        result.track.cur_support);
      result.build_time = omp_get_wtime() - build_start;
      if (!supports_ok)
      {
        result.rejection = VIRTUAL_REJECT_SUPPORT_BUILD;
        continue;
      }
    }

    if (virtual_pose_affine_en)
    {
      // Adaptive sampling uses a level-zero affine; preserve the original
      // reference-level convention on the legacy sampling path.
      result.track.pose_affine = preparePoseAffineTrack(
          ctx, ref_ftr->T_v_w_, point_vref,
          ref_ftr->T_v_w_.rotationMatrix() * pt->normal_,
          adaptive_ready ? 0 : ref_ftr->level_, normal_en, result.track.R_vcur_from_ccur_seed);
    }
    const double current_core_start = omp_get_wtime();
    const int usage_levels = std::max(1, patch_pyrimid_level);
    const bool evaluate_all_levels = zncc_residual_en || ncc_en || usage_stats_en;
    vector<float> current_core_all(evaluate_all_levels ? warp_len : patch_size_total, 0.0f);
    result.level_sse.assign(usage_levels, std::numeric_limits<double>::quiet_NaN());
    result.level_ncc.assign(usage_levels, std::numeric_limits<double>::quiet_NaN());
    result.level_valid.assign(usage_levels, 0);
    result.level_active.assign(usage_levels, 0);
    const V3D point_vcur = result.track.T_vcur_w_seed * pt->pos_;
    if (point_vcur[2] <= virtual_min_z)
    {
      result.current_core_time = omp_get_wtime() - current_core_start;
      result.rejection = VIRTUAL_REJECT_CURRENT_Z;
      continue;
    }
    const V2D virtual_center = virtualProject(point_vcur);
    // Match the legacy warped reference and solver coordinates. Adaptive
    // supports already contain their full per-level grid_step.
    const int legacy_base_scale = 1 << result.track.search_level;
    const bool current_core_ok = adaptive_ready
        ? sampleVirtualCorePatch(result.track.adaptive_supports[0], virtual_center, 1, current_core_all.data())
        : virtual_sparse_patch_en
        ? sampleSparseVirtualCorePatch(ctx, img, result.track.R_ccur_from_vcur_seed,
                                       virtual_center, legacy_base_scale, current_core_all.data())
        : sampleVirtualCorePatch(result.track.cur_support, virtual_center, legacy_base_scale, current_core_all.data());
    result.level_valid[0] = current_core_ok ? 1 : 0;
    if (!current_core_ok)
    {
      result.current_core_time = omp_get_wtime() - current_core_start;
      result.rejection = VIRTUAL_REJECT_CURRENT_CORE;
      continue;
    }
    if (evaluate_all_levels)
    {
      for (int pyramid_level = 1; pyramid_level < patch_pyrimid_level; ++pyramid_level)
      {
        const int scale = 1 << (pyramid_level + result.track.search_level);
        const bool level_ok = adaptive_ready
            ? sampleVirtualCorePatch(result.track.adaptive_supports[pyramid_level], virtual_center, 1,
                                      current_core_all.data() + patch_size_total * pyramid_level)
            : virtual_sparse_patch_en
            ? sampleSparseVirtualCorePatch(ctx, img, result.track.R_ccur_from_vcur_seed,
                                           virtual_center, scale,
                                           current_core_all.data() + patch_size_total * pyramid_level)
            : sampleVirtualCorePatch(result.track.cur_support, virtual_center, scale,
                                     current_core_all.data() + patch_size_total * pyramid_level);
        result.level_valid[pyramid_level] = level_ok ? 1 : 0;
      }
    }

    if (zncc_residual_en)
    {
      for (int pyramid_level = 0; pyramid_level < patch_pyrimid_level; ++pyramid_level)
      {
        if (result.level_valid[pyramid_level] == 0) continue;
        const int offset = patch_size_total * pyramid_level;
        if (!normalizedPatchMetrics(result.warped_reference.data() + offset,
                                    current_core_all.data() + offset,
                                    patch_size_total, zncc_min_std,
                                    result.level_sse[pyramid_level],
                                    result.level_ncc[pyramid_level]))
        {
          result.level_valid[pyramid_level] = 0;
        }
      }
      if (result.level_valid[0] == 0)
      {
        result.current_core_time = omp_get_wtime() - current_core_start;
        result.rejection = VIRTUAL_REJECT_PHOTOMETRIC;
        continue;
      }
      result.error = static_cast<float>(result.level_sse[0]);
    }
    else
    {
      double level0_error = 0.0;
      for (int k = 0; k < patch_size_total; ++k)
      {
        const double residual = ref_ftr->inv_expo_time_ * result.warped_reference[k] -
                                state->inv_expo_time[ctx.camera_id] * current_core_all[k];
        level0_error += residual * residual;
      }
      result.error = static_cast<float>(level0_error);
      result.level_sse[0] = level0_error;
      if (evaluate_all_levels)
      {
        for (int pyramid_level = 1; pyramid_level < patch_pyrimid_level; ++pyramid_level)
        {
          if (result.level_valid[pyramid_level] == 0) continue;
          const int offset = patch_size_total * pyramid_level;
          double level_error = 0.0;
          for (int k = 0; k < patch_size_total; ++k)
          {
            const double residual = ref_ftr->inv_expo_time_ * result.warped_reference[offset + k] -
                                    state->inv_expo_time[ctx.camera_id] * current_core_all[offset + k];
            level_error += residual * residual;
          }
          result.level_sse[pyramid_level] = level_error;
        }
      }
      if (ncc_en || usage_stats_en)
      {
        result.level_ncc[0] = calculateNCC(result.warped_reference.data(), current_core_all.data(), patch_size_total);
        if (evaluate_all_levels)
        {
          for (int pyramid_level = 1; pyramid_level < patch_pyrimid_level; ++pyramid_level)
          {
            if (result.level_valid[pyramid_level] == 0) continue;
            const int offset = patch_size_total * pyramid_level;
            result.level_ncc[pyramid_level] =
                calculateNCC(result.warped_reference.data() + offset, current_core_all.data() + offset, patch_size_total);
          }
        }
      }
    }
    result.ncc = result.level_ncc[0];
    bool any_level_active = false;
    bool any_photometric_reject = false;
    if (zncc_residual_en)
    {
      for (int level = 0; level < usage_levels; ++level)
      {
        if (result.level_valid[level] == 0 || !std::isfinite(result.level_ncc[level]) ||
            !std::isfinite(result.level_sse[level]))
          continue;
        if (ncc_en && result.level_ncc[level] < nccThresholdForLevel(level)) continue;
        result.level_active[level] = 1;
        any_level_active = true;
      }
    }
    else if (ncc_en)
    {
      for (int level = 0; level < usage_levels; ++level)
      {
        if (result.level_valid[level] == 0 || !std::isfinite(result.level_ncc[level]) ||
            !std::isfinite(result.level_sse[level]))
          continue;
        if (result.level_ncc[level] < nccThresholdForLevel(level)) continue;
        if (result.level_sse[level] > outlier_threshold * patch_size_total)
        {
          any_photometric_reject = true;
          continue;
        }
        result.level_active[level] = 1;
        any_level_active = true;
      }
    }
    else if (result.error <= outlier_threshold * patch_size_total)
    {
      std::fill(result.level_active.begin(), result.level_active.end(), 1);
      any_level_active = true;
    }
    else
    {
      any_photometric_reject = true;
    }
    if (!any_level_active)
    {
      result.current_core_time = omp_get_wtime() - current_core_start;
      result.rejection = any_photometric_reject ? VIRTUAL_REJECT_PHOTOMETRIC : VIRTUAL_REJECT_NCC;
      continue;
    }
    if (result.level_active[0] == 0)
      for (int level = 1; level < usage_levels; ++level)
        if (result.level_active[level] != 0)
        {
          result.error = static_cast<float>(result.level_sse[level]);
          break;
        }
    result.current_core_time = omp_get_wtime() - current_core_start;

    if (runtime_support_dump_en)
      result.current_core.assign(current_core_all.begin(), current_core_all.begin() + patch_size_total);
    result.inverse_reference_exposure = ref_ftr->inv_expo_time_;
    result.track.valid = true;
    result.valid = true;
    }
  }
  virtual_parallel_track_time_ = omp_get_wtime() - parallel_track_start;

  const double result_collect_start = omp_get_wtime();
  vector<int> virtual_search_level_count(std::max(virtual_max_search_level + 1, 1), 0);
  vector<int> virtual_warp_fail_search_level_count(std::max(virtual_max_search_level + 1, 1), 0);
  vector<int> virtual_warp_fail_pyramid_level_count(std::max(patch_pyrimid_level, 1), 0);
  std::set<VisualPoint *> accepted_managed_points;
  std::set<VisualPoint *> fallback_allowed_points;
  int adaptive_attempted = 0, adaptive_prepared = 0, adaptive_accepted = 0;
  double adaptive_time_sum = 0.0, adaptive_pitch_sum = 0.0;
  for (int candidate_index = 0; candidate_index < static_cast<int>(candidates.size()); ++candidate_index)
  {
    const VirtualCandidate &collect_candidate = candidates[candidate_index];
    if (collect_candidate.reference_rank > 0)
    {
      if (accepted_managed_points.count(collect_candidate.point) != 0) continue;
      if (fallback_allowed_points.count(collect_candidate.point) == 0) continue;
      ++visual_map_manage_stats_.fallback_attempted;
    }
    VirtualCandidateResult &result = results[candidate_index];
    if (result.adaptive_attempted)
    {
      ++adaptive_attempted;
      adaptive_time_sum += result.adaptive_time;
      if (!result.track.adaptive_supports.empty())
      {
        ++adaptive_prepared;
        if (result.valid)
        {
          ++adaptive_accepted;
          adaptive_pitch_sum += result.track.sampling_pitch;
        }
      }
    }
    if (usage_stats_en)
    {
      const VirtualCandidate &candidate = candidates[candidate_index];
      const Matrix2d usage_affine = result.search_level >= 0 ? result.track.A_cur_ref : Matrix2d::Zero();
      recordUsageObservation(ctx, *candidate.reference, *candidate.point, candidate.current_raw_center_px, usage_affine, false);
      recordUsagePreGateMetrics(ctx, *candidate.reference, *candidate.point, candidate.current_raw_center_px,
                                usage_affine, result.level_sse, result.level_ncc, result.level_valid);
      recordUsageWarpApproximation(ctx, *candidate.reference, *candidate.point, candidate.current_raw_center_px,
                                   result.search_level >= 0 ? &result.track : nullptr, result.valid);
    }
    build_virtual_support_time_ += result.build_time;
    virtual_affine_time_ += result.affine_time;
    virtual_ref_support_materialize_time_ += result.ref_materialize_time;
    if (result.ref_materialized) ++virtual_ref_support_materialized_count_;
    virtual_warp_time_ += result.warp_time;
    virtual_current_core_time_ += result.current_core_time;
    if (result.search_level >= 0 && result.search_level < static_cast<int>(virtual_search_level_count.size()))
      ++virtual_search_level_count[result.search_level];
    if (!result.valid)
    {
      if (collect_candidate.reference_rank == 0 &&
          (result.rejection == VIRTUAL_REJECT_NCC || result.rejection == VIRTUAL_REJECT_PHOTOMETRIC))
        fallback_allowed_points.insert(collect_candidate.point);
      if (result.rejection == VIRTUAL_REJECT_NCC || result.rejection == VIRTUAL_REJECT_PHOTOMETRIC)
        recordManagedReferenceRejection(*collect_candidate.point, *collect_candidate.reference,
                                        ctx.new_frame->id_, ctx.camera_id);
      if (result.rejection != VIRTUAL_REJECT_NONE)
        recordRejectedPoint(candidates[candidate_index].current_raw_center_px, drawReasonFromVirtualReject(result.rejection));
      switch (result.rejection)
      {
      case VIRTUAL_REJECT_ROTATION:
        ++virtual_track_rotation_fail_count_;
        break;
      case VIRTUAL_REJECT_SUPPORT_BUILD:
        ++virtual_track_support_fail_count_;
        break;
      case VIRTUAL_REJECT_AFFINE_MATRIX:
        ++virtual_track_affine_fail_count_;
        break;
      case VIRTUAL_REJECT_REFERENCE_SUPPORT:
        ++virtual_ref_support_materialize_fail_count_;
        break;
      case VIRTUAL_REJECT_REFERENCE_WARP:
        ++virtual_track_warp_fail_count_;
        if (result.search_level >= 0 && result.search_level < static_cast<int>(virtual_warp_fail_search_level_count.size()))
          ++virtual_warp_fail_search_level_count[result.search_level];
        if (result.warp_fail_pyramid_level >= 0 &&
            result.warp_fail_pyramid_level < static_cast<int>(virtual_warp_fail_pyramid_level_count.size()))
          ++virtual_warp_fail_pyramid_level_count[result.warp_fail_pyramid_level];
        break;
      case VIRTUAL_REJECT_CURRENT_Z:
        ++virtual_track_current_z_fail_count_;
        break;
      case VIRTUAL_REJECT_CURRENT_CORE:
        ++virtual_track_current_core_fail_count_;
        break;
      case VIRTUAL_REJECT_NCC:
        ++virtual_track_ncc_reject_count_;
        break;
      case VIRTUAL_REJECT_PHOTOMETRIC:
        ++virtual_track_photometric_reject_count_;
        break;
      default:
        break;
      }
      continue;
    }

    ++virtual_valid_track_count_;
    if (collect_candidate.reference_rank > 0) ++visual_map_manage_stats_.fallback_accepted;
    accepted_managed_points.insert(collect_candidate.point);
    if (usage_stats_en)
    {
      const VirtualCandidate &candidate = candidates[candidate_index];
      recordUsageObservation(ctx, *candidate.reference, *candidate.point, candidate.current_raw_center_px,
                             result.track.A_cur_ref, true, result.level_sse, result.level_ncc, result.level_active);
    }
    const VirtualCandidate &accepted_candidate = candidates[candidate_index];
    if (runtime_support_dump_en && accepted_candidate.point != nullptr)
    {
      if (accepted_candidate.point->runtime_support_dump_id_ < 0)
        accepted_candidate.point->runtime_support_dump_id_ = runtime_support_dump_next_point_id_++;
      const int track_count = ++accepted_candidate.point->runtime_support_track_count_;
      if (track_count > runtime_support_dump_best_track_count_)
      {
        runtime_support_dump_best_track_count_ = track_count;
        runtime_support_dump_best_point_ = accepted_candidate.point;
        const int submap_index = static_cast<int>(ctx.visual_submap->voxel_points.size());
        dumpRuntimeSupportObservation(ctx, *accepted_candidate.point, *accepted_candidate.reference,
                                      accepted_candidate.current_raw_center_px, result.track,
                                      result.warped_reference, result.current_core,
                                      track_count, submap_index, result.error, result.ncc);
      }
    }
    ctx.visual_submap->voxel_points.push_back(candidates[candidate_index].point);
    ctx.visual_submap->reference_features.push_back(candidates[candidate_index].reference);
    ctx.visual_submap->propa_errors.push_back(result.error);
    ctx.visual_submap->search_levels.push_back(result.track.search_level);
    ctx.visual_submap->warp_affines.push_back(result.track.A_cur_ref);
    ctx.visual_submap->errors.push_back(result.error);
    ctx.visual_submap->warp_patch.push_back(std::move(result.warped_reference));
    ctx.visual_submap->inv_expo_list.push_back(result.inverse_reference_exposure);
    ctx.visual_submap->level_active.push_back(result.level_active);
    if (usage_stats_en)
    {
      ctx.visual_submap->usage_ncc_levels.push_back(result.level_ncc);
      ctx.visual_submap->usage_sse_levels.push_back(result.level_sse);
      ctx.visual_submap->usage_level_valid.push_back(result.level_active);
    }
    ctx.visual_submap->virtual_track_patches.push_back(std::move(result.track));
    appendManagedSubmapMetadata(ctx, *candidates[candidate_index].point,
                                *candidates[candidate_index].reference);
  }
  virtual_result_collect_time_ = omp_get_wtime() - result_collect_start;

  ctx.total_points = static_cast<int>(ctx.visual_submap->voxel_points.size());
  rejected_virtual_support_oob_ = virtual_track_support_fail_count_ + virtual_track_current_core_fail_count_ + virtual_ref_support_materialize_fail_count_;
  rejected_virtual_projection_invalid_ = virtual_candidate_projection_fail_count_;
  rejected_virtual_z_ = virtual_track_rotation_fail_count_ + virtual_track_current_z_fail_count_;
  rejected_virtual_affine_oob_ = virtual_track_affine_fail_count_ + virtual_track_warp_fail_count_;

  auto formatCountVector = [](const vector<int> &counts) {
    std::string text;
    for (int i = 0; i < static_cast<int>(counts.size()); ++i)
    {
      if (!text.empty()) text += ",";
      text += std::to_string(i) + ":" + std::to_string(counts[i]);
    }
    return text;
  };
  const std::string search_level_text = formatCountVector(virtual_search_level_count);
  const std::string warp_fail_search_text = formatCountVector(virtual_warp_fail_search_level_count);
  const std::string warp_fail_pyramid_text = formatCountVector(virtual_warp_fail_pyramid_level_count);

  printf("[ VIO Virtual ] Retrieve %d/%d tracked from map_grid=%d candidates=%d\n",
         ctx.total_points, virtual_candidate_count_, virtual_map_grid_count_, virtual_candidate_count_);
  printf("[ VIO Virtual Candidate ] null=%d normal_uninit=%d proj_fail=%d range_reject=%d close_view_fail=%d ref_missing=%d ref_invalid=%d select_wall=%.6f s\n",
         virtual_candidate_null_count_, virtual_candidate_normal_uninit_count_, virtual_candidate_projection_fail_count_,
         virtual_candidate_range_reject_count_, virtual_candidate_close_view_fail_count_, virtual_candidate_ref_missing_count_,
         virtual_candidate_ref_invalid_count_, virtual_candidate_select_time_);
  printf("[ VIO Virtual Reject ] rotation=%d support_build=%d affine_matrix=%d ref_support=%d warp_ref=%d current_z=%d current_core=%d ncc=%d photometric=%d\n",
         virtual_track_rotation_fail_count_, virtual_track_support_fail_count_, virtual_track_affine_fail_count_,
         virtual_ref_support_materialize_fail_count_, virtual_track_warp_fail_count_,
         virtual_track_current_z_fail_count_, virtual_track_current_core_fail_count_,
         virtual_track_ncc_reject_count_, virtual_track_photometric_reject_count_);
  printf("[ VIO Virtual Warp ] search_level{%s} warp_fail_search{%s} warp_fail_pyramid{%s}\n",
         search_level_text.c_str(), warp_fail_search_text.c_str(), warp_fail_pyramid_text.c_str());
  printf("[ VIO Virtual Timing ] retrieve_wall=%.6f s parallel_wall=%.6f s support_sum=%.6f s affine_sum=%.6f s warp_sum=%.6f s current_core_sum=%.6f s result_collect_wall=%.6f s\n",
         virtual_candidate_select_time_ + virtual_parallel_track_time_ + virtual_result_collect_time_,
         virtual_parallel_track_time_, build_virtual_support_time_, virtual_affine_time_, virtual_warp_time_,
         virtual_current_core_time_, virtual_result_collect_time_);
  printf("[ VIO Virtual Timing Note ] *_sum accumulates per-candidate time across OpenMP workers; wall fields are real elapsed time.\n");
  if (virtual_adaptive_sampling_en)
    printf("[ VIO Adaptive Sampling ] cam=%d attempted=%d prepared=%d fallback=%d accepted=%d pitch_mean=%.4f prepare_sum=%.6f s\n",
           ctx.camera_id, adaptive_attempted, adaptive_prepared, adaptive_attempted - adaptive_prepared,
           adaptive_accepted, adaptive_accepted ? adaptive_pitch_sum / adaptive_accepted : 0.0, adaptive_time_sum);
  if (virtual_sparse_patch_en)
    printf("[ VIO Virtual Sparse ] ref_materialized=%d ref_fail=%d ref_materialize_sum=%.6f s\n",
           virtual_ref_support_materialized_count_, virtual_ref_support_materialize_fail_count_,
           virtual_ref_support_materialize_time_);
}

void VIOManager::processMultiCameraFrameBaseline(const MeasureGroup &meas, vector<pointWithVar> &pg,
                                         const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (!meas.has_multi_cam_frame) throw std::runtime_error("direct VIO requires MeasureGroup::multi_cam_frame");
  const MultiCameraFrame &mf = meas.multi_cam_frame;
  if (static_cast<int>(mf.images.size()) != numCameras())
    throw std::runtime_error("MultiCameraFrame image count does not match VIOManager camera count");

  const double frame_start = omp_get_wtime();
  // printf("[ VIO Debug ] processMultiCameraFrame begin frame=%d cameras=%d pg=%zu feat_map=%zu plane_map=%zu virtual=%d cross_ref=%d normal=%d inverse=%d raycast=%d\n",
  //        mf.frame_id, numCameras(), pg.size(), feat_map.size(), plane_map.size(), virtual_fisheye_patch_en ? 1 : 0,
  //        cross_camera_reference_en ? 1 : 0, normal_en ? 1 : 0, inverse_composition_en ? 1 : 0, raycast_en ? 1 : 0);
  // fflush(stdout);
  for (int camera_id = 0; camera_id < numCameras(); ++camera_id)
  {
    PerCameraData &ctx = cameras_[camera_id];
    cv::Mat image = mf.images[camera_id].clone();
    if (image.empty()) throw std::runtime_error("empty image for camera_id=" + std::to_string(camera_id));
    if (ctx.width != image.cols || ctx.height != image.rows)
      cv::resize(image, image, cv::Size(image.cols * ctx.image_resize_factor, image.rows * ctx.image_resize_factor), 0, 0, CV_INTER_LINEAR);
    ctx.img_rgb = image.clone();
    ctx.img_cp = image.clone();
    if (image.channels() == 3)
    {
      const fixed_gray_image::PreparedLut *weights =
          fixed_gray_weights_installed_ && camera_id > 0 ? &fixed_gray_luts_[camera_id] : nullptr;
      if (weights == nullptr || weights->legacy_default)
        cv::cvtColor(image, image, CV_BGR2GRAY);
      else
        image = fixed_gray_image::convertFixedGray(image, weights);
    }
    const double raw_timestamp = camera_id < static_cast<int>(mf.raw_timestamps.size())
                                     ? mf.raw_timestamps[camera_id]
                                     : mf.timestamp;
    const double corrected_timestamp = camera_id < static_cast<int>(mf.corrected_timestamps.size())
                                           ? mf.corrected_timestamps[camera_id]
                                           : mf.timestamp;
    const double capture_timestamp = camera_id < static_cast<int>(mf.capture_timestamps.size())
                                         ? mf.capture_timestamps[camera_id]
                                         : mf.timestamp;
    const double td_used = camera_id < static_cast<int>(mf.td_used.size()) ? mf.td_used[camera_id] : 0.0;
    const int time_group = camera_id < static_cast<int>(mf.time_offset_group.size())
                               ? mf.time_offset_group[camera_id]
                               : (camera_id < static_cast<int>(camera_time_offset_groups.size())
                                      ? camera_time_offset_groups[camera_id]
                                      : 0);
    ctx.new_frame.reset(new Frame(ctx.cam, image, mf.frame_id, camera_id, capture_timestamp,
                                  raw_timestamp, corrected_timestamp, td_used,
                                  capture_timestamp - corrected_timestamp, time_group));
    updateFrameState(ctx, *state);
    resetGrid(ctx);
    // printf("[ VIO Debug ] frame setup camera_id=%d frame=%d image=%dx%d type=%d ns=%s\n",
    //        ctx.camera_id, mf.frame_id, image.cols, image.rows, image.type(), ctx.camera_namespace.c_str());
    // fflush(stdout);
  }
  flushVisualMapRetirements();
  const double frame_setup_end = omp_get_wtime();

  int raw_score_remaining_total = max_total_points;
  for (PerCameraData &ctx : cameras_)
  {
    int raw_score_point_quota = -1;
    if (virtual_raw_score_select_en)
    {
      const int camera_id = std::max(0, ctx.camera_id);
      const int cameras_left = std::max(0, numCameras() - camera_id - 1);
      const int min_per_camera = std::max(0, points_per_camera_min);
      const int max_per_camera = std::max(min_per_camera, points_per_camera_max);
      const int reserved_for_later = std::min(raw_score_remaining_total, min_per_camera * cameras_left);
      raw_score_point_quota = std::max(0, raw_score_remaining_total - reserved_for_later);
      raw_score_point_quota = std::min(raw_score_point_quota, max_per_camera);
    }
    // printf("[ VIO Debug ] retrieve begin camera_id=%d frame=%d\n", ctx.camera_id, mf.frame_id);
    // fflush(stdout);
    retrieveFromVisualSparseMap(ctx, ctx.new_frame->img_, pg, plane_map, raw_score_point_quota);
    if (virtual_raw_score_select_en)
      raw_score_remaining_total = std::max(0, raw_score_remaining_total - ctx.total_points);
    // printf("[ VIO Debug ] retrieve end camera_id=%d frame=%d total_points=%d\n",
    //        ctx.camera_id, mf.frame_id, ctx.total_points);
    // fflush(stdout);
    // printf("[ VIO Debug ] generate begin camera_id=%d frame=%d\n", ctx.camera_id, mf.frame_id);
    // fflush(stdout);
    generateVisualMapPoints(ctx, ctx.new_frame->img_, pg, plane_map);
    if (ctx.visual_submap != nullptr && ctx.new_frame != nullptr)
    {
      for (VisualPoint *point : ctx.visual_submap->voxel_points)
      {
        if (point == nullptr || point->pending_delete_) continue;
        if (point->reuse_last_selected_frame_id_ == ctx.new_frame->id_ &&
            point->reuse_last_selected_camera_id_ == ctx.camera_id)
          continue;
        ++point->reuse_selected_count_;
        point->reuse_last_selected_frame_id_ = ctx.new_frame->id_;
        point->reuse_last_selected_camera_id_ = ctx.camera_id;
        point->ensureCameraCount(numCameras());
        if (ctx.camera_id >= 0 && ctx.camera_id < static_cast<int>(point->reuse_camera_observed_.size()) &&
            point->reuse_camera_observed_[ctx.camera_id] == 0)
        {
          point->reuse_camera_observed_[ctx.camera_id] = 1;
          ++point->reuse_camera_count_;
        }
      }
    }
    // printf("[ VIO Debug ] generate end camera_id=%d frame=%d pending=%zu\n",
    //        ctx.camera_id, mf.frame_id, ctx.pending_new_points.size());
    // fflush(stdout);
  }
  const double retrieve_end = omp_get_wtime();

  // printf("[ VIO Debug ] ekf begin frame=%d\n", mf.frame_id);
  // fflush(stdout);
  computeJacobianAndUpdateEKF();
  // printf("[ VIO Debug ] ekf end frame=%d\n", mf.frame_id);
  // fflush(stdout);
  const double ekf_end = omp_get_wtime();

  if (visual_map_manage_en)
    for (PerCameraData &ctx : cameras_) updateManagedObservationEvidence(ctx);

  if (visual_ref_post_ekf_build_en)
  {
    for (PerCameraData &ctx : cameras_)
      materializePendingNewPointObservations(ctx, ctx.new_frame->img_, plane_map);
  }

  if (ref_patch_dump_en && !cameras_.empty())
    processRefPatchDumpProbe(cameras_[kRefPatchDumpCameraId], cameras_[kRefPatchDumpCameraId].new_frame->img_);

  for (PerCameraData &ctx : cameras_)
  {
    if (!visual_ref_post_ekf_build_en)
    {
      for (PendingNewPointObservation &pending : ctx.pending_new_points)
      {
        pending.T_f_w = ctx.new_frame->T_f_w_;
        fillPendingObservationTiming(pending, ctx);
        if (pending.virtual_patch_valid) pending.T_v_w = composeVirtualPose(pending.R_v_from_c, pending.T_f_w);
      }
    }
    // printf("[ VIO Debug ] updateVisualMap begin camera_id=%d frame=%d pending=%zu\n",
    //        ctx.camera_id, mf.frame_id, ctx.pending_new_points.size());
    // fflush(stdout);
    updateVisualMapPoints(ctx, ctx.new_frame->img_);
    // printf("[ VIO Debug ] updateVisualMap end camera_id=%d frame=%d\n", ctx.camera_id, mf.frame_id);
    // fflush(stdout);
  }
  const double map_update_end = omp_get_wtime();
  // printf("[ VIO Debug ] commitPendingNewPoints begin frame=%d\n", mf.frame_id);
  // fflush(stdout);
  commitPendingNewPoints(plane_map);
  // printf("[ VIO Debug ] commitPendingNewPoints end frame=%d feat_map=%zu\n", mf.frame_id, feat_map.size());
  // fflush(stdout);
  const double commit_end = omp_get_wtime();
  for (PerCameraData &ctx : cameras_)
  {
    // printf("[ VIO Debug ] updateReference begin camera_id=%d frame=%d total_points=%d\n",
    //        ctx.camera_id, mf.frame_id, ctx.total_points);
    // fflush(stdout);
    updateReferencePatch(ctx, plane_map);
    // printf("[ VIO Debug ] updateReference end camera_id=%d frame=%d\n", ctx.camera_id, mf.frame_id);
    // fflush(stdout);
    plotTrackedPoints(ctx);
    if (plot_flag) projectPatchFromRefToCur(ctx, plane_map);
  }
  queueStaleVisualMapPoints(mf.frame_id);
  printVisualMapManageStats(mf.frame_id);
  if (colmap_output_en && numCameras() == 1) dumpDataForColmap();
  const double reference_end = omp_get_wtime();
  ++frame_count;
  const double elapsed = reference_end - frame_start;
  ave_total = ave_total * (frame_count - 1) / frame_count + elapsed / frame_count;
  long long active_visual_point_count = 0;
  long long active_visual_point_reuse_sum = 0;
  long long active_visual_point_camera_reuse_sum = 0;
  for (const auto &voxel : feat_map)
  {
    if (voxel.second == nullptr) continue;
    for (const VisualPoint *point : voxel.second->voxel_points)
    {
      if (point == nullptr || point->pending_delete_) continue;
      ++active_visual_point_count;
      active_visual_point_reuse_sum += point->reuse_selected_count_;
      active_visual_point_camera_reuse_sum += point->reuse_camera_count_;
    }
  }
  const double mean_reuse_count =
      active_visual_point_count > 0
          ? static_cast<double>(active_visual_point_reuse_sum) / static_cast<double>(active_visual_point_count)
          : 0.0;
  const double mean_camera_count =
      active_visual_point_count > 0
          ? static_cast<double>(active_visual_point_camera_reuse_sum) / static_cast<double>(active_visual_point_count)
          : 0.0;
  printf("[ VIO ] multi-camera frame=%d cameras=%d observations=%zu elapsed=%.6f s\n",
         mf.frame_id, numCameras(), feat_map.size(), elapsed);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m|                         VIO Time                            |\033[0m\n");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27zu |\033[0m\n", "Sparse Map Size", feat_map.size());
  printf("\033[1;34m| %-29s | %-27d |\033[0m\n", "Camera Count", numCameras());
  printf("\033[1;34m| %-29s | %-27lld |\033[0m\n", "Active Visual Points", active_visual_point_count);
  printf("\033[1;34m| %-29s | %-27.3f |\033[0m\n", "Mean Reuse Count", mean_reuse_count);
  printf("\033[1;34m| %-29s | %-27.3f |\033[0m\n", "Mean Camera Count", mean_camera_count);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage", "Time (secs)");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Frame Setup", frame_setup_end - frame_start);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Retrieve + Generate", retrieve_end - frame_setup_end);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Joint EKF Update", ekf_end - retrieve_end);
  printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> computeJacobian", compute_jacobian_time);
  printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> updateEKF", update_ekf_time);
  printf("\033[1;36m| %-29s | %-27d |\033[0m\n", "Jacobian Worker Limit", visual_linearization_threads_);
  printf("\033[1;36m| %-29s | %-27lf |\033[0m\n", "Patch Parallel Wall (s)", visual_parallel_time_);
  printf("\033[1;36m| %-29s | %-27lf |\033[0m\n", "Patch Reduce Wall (s)", visual_reduce_time_);
  printf("\033[1;36m| %-29s | %-27lf |\033[0m\n", "NIS Worker Sum (s)", visual_nis_worker_time_);
  printf("\033[1;36m| %-29s | %-27lf |\033[0m\n", "Directional Wall (s)", visual_directional_time_);
  printf("\033[1;32m| %-29s | %-27lld |\033[0m\n", "Linearized Residuals", vio_linearized_residual_count_);
  const double avg_time_per_residual_us =
      vio_linearized_residual_count_ > 0
          ? compute_jacobian_time * 1.0e6 / static_cast<double>(vio_linearized_residual_count_)
          : 0.0;
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Avg Time / Residual (us)", avg_time_per_residual_us);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Update Visual Map", map_update_end - ekf_end);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Commit New Points", commit_end - map_update_end);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Reference + Debug", reference_end - commit_end);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Current Total Time", elapsed);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Average Total Time", ave_total);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printOnlineCalibrationStatsTable(mf.frame_id);
  maybePrintUsageStatsTable(mf.frame_id);
}
