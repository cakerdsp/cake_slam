// Enabled-only geometry collection; included by voxel_map_multi_cam.cpp.
void VoxelMapManager::BuildResidualListWithVisualPatchGeometry(std::vector<pointWithVar> &pv_list, std::vector<PointToPlane> &ptpl_list)
{
  int max_layer = config_setting_.max_layer_;
  double voxel_size = config_setting_.max_voxel_size_;
  double sigma_num = config_setting_.sigma_num_;
  std::mutex mylock;
  ptpl_list.clear();
  std::vector<PointToPlane> all_ptpl_list(pv_list.size());
  std::vector<bool> useful_ptpl(pv_list.size());
  const size_t geometry_stride = std::max<size_t>(1, (pv_list.size() + 1023) / 1024);
  std::vector<visual_patch_score::GeometrySample> geometry_samples;
  geometry_samples.resize((pv_list.size() + geometry_stride - 1) / geometry_stride);
  std::vector<size_t> index(pv_list.size());
  for (size_t i = 0; i < index.size(); ++i)
  {
    index[i] = i;
    useful_ptpl[i] = false;
  }
  #ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
    #pragma omp parallel for
  #endif
  for (int i = 0; i < index.size(); i++)
  {
    pointWithVar &pv = pv_list[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = pv.point_w[j] / voxel_size;
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = voxel_map_.find(position);
    if (iter != voxel_map_.end())
    {
      VoxelOctoTree *current_octo = iter->second;
      PointToPlane single_ptpl;
      bool is_sucess = false;
      double prob = 0;
      auto *geometry_sample = i % geometry_stride == 0
          ? &geometry_samples[i / geometry_stride] : nullptr;
      build_single_residual_with_geometry(pv, current_octo, 0, is_sucess, prob, single_ptpl, geometry_sample);
      if (!is_sucess)
      {
        VOXEL_LOCATION near_position = position;
        if (loc_xyz[0] > (current_octo->voxel_center_[0] + current_octo->quater_length_)) { near_position.x = near_position.x + 1; }
        else if (loc_xyz[0] < (current_octo->voxel_center_[0] - current_octo->quater_length_)) { near_position.x = near_position.x - 1; }
        if (loc_xyz[1] > (current_octo->voxel_center_[1] + current_octo->quater_length_)) { near_position.y = near_position.y + 1; }
        else if (loc_xyz[1] < (current_octo->voxel_center_[1] - current_octo->quater_length_)) { near_position.y = near_position.y - 1; }
        if (loc_xyz[2] > (current_octo->voxel_center_[2] + current_octo->quater_length_)) { near_position.z = near_position.z + 1; }
        else if (loc_xyz[2] < (current_octo->voxel_center_[2] - current_octo->quater_length_)) { near_position.z = near_position.z - 1; }
        auto iter_near = voxel_map_.find(near_position);
        if (iter_near != voxel_map_.end()) { build_single_residual(pv, iter_near->second, 0, is_sucess, prob, single_ptpl); }
      }
      if (is_sucess)
      {
        mylock.lock();
        useful_ptpl[i] = true;
        all_ptpl_list[i] = single_ptpl;
        mylock.unlock();
      }
      else
      {
        mylock.lock();
        useful_ptpl[i] = false;
        mylock.unlock();
      }
    }
  }
  {
    const double aggregate_start = omp_get_wtime();
    visual_patch_geometry = visual_patch_score::summarize(geometry_samples);
    visual_patch_geometry.aggregate_seconds = omp_get_wtime() - aggregate_start;
  }
  for (size_t i = 0; i < useful_ptpl.size(); i++)
  {
    if (useful_ptpl[i]) { ptpl_list.push_back(all_ptpl_list[i]); }
  }
}

void VoxelMapManager::build_single_residual_with_geometry(pointWithVar &pv, const VoxelOctoTree *current_octo, const int current_layer, bool &is_sucess,
                                            double &prob, PointToPlane &single_ptpl, visual_patch_score::GeometrySample *geometry_sample)
{
  int max_layer = config_setting_.max_layer_;
  double sigma_num = config_setting_.sigma_num_;

  double radius_k = 3;
  Eigen::Vector3d p_w = pv.point_w;
  if (current_octo->plane_ptr_->is_plane_)
  {
    VoxelPlane &plane = *current_octo->plane_ptr_;
    Eigen::Vector3d p_world_to_center = p_w - plane.center_;
    float dis_to_plane = fabs(plane.normal_(0) * p_w(0) + plane.normal_(1) * p_w(1) + plane.normal_(2) * p_w(2) + plane.d_);
    float dis_to_center = (plane.center_(0) - p_w(0)) * (plane.center_(0) - p_w(0)) + (plane.center_(1) - p_w(1)) * (plane.center_(1) - p_w(1)) +
                          (plane.center_(2) - p_w(2)) * (plane.center_(2) - p_w(2));
    float range_dis = sqrt(std::max(0.0f, dis_to_center - dis_to_plane * dis_to_plane));

    if (range_dis <= radius_k * plane.radius_)
    {
      visual_patch_score::consider(geometry_sample, plane.normal_, dis_to_plane);
      Eigen::Matrix<double, 1, 6> J_nq;
      J_nq.block<1, 3>(0, 0) = p_w - plane.center_;
      J_nq.block<1, 3>(0, 3) = -plane.normal_;
      double sigma_l = J_nq * plane.plane_var_ * J_nq.transpose();
      sigma_l += plane.normal_.transpose() * pv.var * plane.normal_;
      if (std::isfinite(sigma_l) && sigma_l > 0.0 && dis_to_plane < sigma_num * sqrt(sigma_l))
      {
        is_sucess = true;
        double this_prob = 1.0 / (sqrt(sigma_l)) * exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);
        if (this_prob > prob)
        {
          prob = this_prob;
          pv.normal = plane.normal_;
          single_ptpl.body_cov_ = pv.body_var;
          single_ptpl.point_b_ = pv.point_b;
          single_ptpl.point_w_ = pv.point_w;
          single_ptpl.plane_var_ = plane.plane_var_;
          single_ptpl.normal_ = plane.normal_;
          single_ptpl.center_ = plane.center_;
          single_ptpl.d_ = plane.d_;
          single_ptpl.layer_ = current_layer;
          single_ptpl.plane_id_ = plane.id_;
          single_ptpl.dis_to_plane_ = plane.normal_(0) * p_w(0) + plane.normal_(1) * p_w(1) + plane.normal_(2) * p_w(2) + plane.d_;
        }
        return;
      }
      else
      {
        // is_sucess = false;
        return;
      }
    }
    else
    {
      // is_sucess = false;
      return;
    }
  }
  else
  {
    if (current_layer < max_layer)
    {
      for (size_t leafnum = 0; leafnum < 8; leafnum++)
      {
        if (current_octo->leaves_[leafnum] != nullptr)
        {

          VoxelOctoTree *leaf_octo = current_octo->leaves_[leafnum];
          build_single_residual_with_geometry(pv, leaf_octo, current_layer + 1, is_sucess, prob, single_ptpl, geometry_sample);
        }
      }
      return;
    }
    else { return; }
  }
}

