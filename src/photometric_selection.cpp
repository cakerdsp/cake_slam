#include "vio_multi_cam.h"
#include "estimator_covariance.h"
#include <cstring>

namespace {
using Footprint = std::map<int, double>;

// These kernels match the frontend's interpolation convention (a=-0.5,
// Lanczos radius 4). Coordinates are in the actual stored image grid.
double cubic(double x) {
  x = std::abs(x);
  if (x <= 1.0) return (1.5 * x - 2.5) * x * x + 1.0;
  if (x < 2.0) return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
  return 0.0;
}
double lanczos(double x) {
  x = std::abs(x);
  if (x < 1.e-12) return 1.0;
  if (x >= 4.0) return 0.0;
  const double a = 3.14159265358979323846 * x;
  return std::sin(a) * std::sin(a / 4.0) / (a * a / 4.0);
}
bool footprint(double u, double v, int width, int height, int radius, Footprint &out) {
  if (!std::isfinite(u) || !std::isfinite(v)) return false;
  const int x = static_cast<int>(std::floor(u)), y = static_cast<int>(std::floor(v));
  if (x - radius + 1 < 0 || y - radius + 1 < 0 || x + radius >= width || y + radius >= height) return false;
  out.clear();
  auto weight = [radius](double d) {
    if (radius == 1) return std::max(0.0, 1.0 - std::abs(d));
    return radius == 2 ? cubic(d) : lanczos(d);
  };
  double sum = 0.0;
  for (int yy = y - radius + 1; yy <= y + radius; ++yy)
    for (int xx = x - radius + 1; xx <= x + radius; ++xx) {
      const double w = weight(u - xx) * weight(v - yy);
      if (w != 0.0) out[yy * width + xx] = w;
      sum += w;
    }
  if (std::abs(sum) < 1.e-12) return false;
  for (auto &entry : out) entry.second /= sum;
  return true;
}
}  // namespace

bool VIOManager::buildPhotometricSelectionCandidate(
    const PerCameraData &ctx, int slot, int level, const Eigen::MatrixXd &full_jacobian,
    const Eigen::MatrixXd &point_jacobian, const Eigen::VectorXd &weights,
    const std::vector<int> &patch_indices,
    photometric_selection::Candidate &candidate) const
{
  const auto &submap = *ctx.visual_submap;
  const VisualPoint &point = *submap.voxel_points[slot];
  const Feature &reference = *submap.reference_features[slot];
  const int rows = full_jacobian.rows();
  const double stddev = std::sqrt(photometricNoiseCovariance());
  if (rows == 0 || !std::isfinite(stddev) || stddev <= 0.0 ||
      static_cast<int>(patch_indices.size()) != rows || weights.size() != rows ||
      point_jacobian.rows() != rows || point_jacobian.cols() != 3) return false;
  candidate.camera = ctx.camera_id;
  candidate.slot = slot;
  candidate.point = reinterpret_cast<uintptr_t>(&point);
  V2D current_pixel;
  if (ctx.grid_size <= 0 || ctx.grid_n_width <= 0 ||
      !projectRawFisheyeIfValid(ctx, ctx.Rcw * point.pos_ + ctx.Pcw, 1, current_pixel)) return false;
  candidate.cell = static_cast<int>(current_pixel.y()) / ctx.grid_size * ctx.grid_n_width +
                   static_cast<int>(current_pixel.x()) / ctx.grid_size;
  // A selected track can be sampled at all pyramid levels. Charge the full
  // patch, not the rank of its score sketch or partially surviving pixels.
  candidate.cost = patch_size_total;
  candidate.h = weights.asDiagonal() * full_jacobian / stddev;

  // Geometry is tied to its actual creation source. Scan landmarks use their
  // local sensor covariance. The frontend's raycast additions are plane
  // centers (not ray-plane intersections), so their derivative is [0 I] in
  // [normal, center] coordinates. Never substitute the current plane fit for
  // an old point's birth geometry, or count point and plane noise twice.
  if (!point.local_geometry_covariance_valid_) return false;
  const photometric_selection::SourceKey geometry_key = point.local_geometry_plane_id_ >= 0
      ? photometric_selection::SourceKey{0, static_cast<uint64_t>(point.local_geometry_plane_id_),
                                          point.local_geometry_plane_revision_, 0}
      : photometric_selection::SourceKey{2, candidate.point, 0, 0};
  candidate.sources[geometry_key] = weights.asDiagonal() * point_jacobian *
      estimator_covariance::positiveSemidefiniteRoot(point.local_geometry_covariance_) / stddev;

  if (photometric_selection_reference_pixel_std > 0.0) {
    if (reference.camera_id_ < 0 || reference.camera_id_ >= numCameras()) return false;
    const auto &source = cameras_[reference.camera_id_];
    const bool virt = reference.virtual_patch_valid_;
    if (virt && (reference.img_.rows != virtual_support_size || reference.img_.cols != virtual_support_size ||
                 virtual_support_ray_lut_.size() != static_cast<size_t>(virtual_support_size * virtual_support_size)))
      return false;
    if (virt && virtual_patch_resampling_mode_enum == VirtualPatchResamplingMode::FORWARD_SPLAT &&
        (source.raw_pixel_unit_ray_valid_mask.size() != static_cast<size_t>(source.width * source.height) ||
         source.raw_pixel_to_unit_ray_lut.size() != static_cast<size_t>(source.width * source.height))) return false;
    const int radius = !virt || virtual_interp_mode_enum == VirtualInterpMode::BILINEAR ? 1 :
                       virtual_interp_mode_enum == VirtualInterpMode::BICUBIC ? 2 : 4;
    const int scale = (1 << submap.search_levels[slot]) * (1 << level);
    const Matrix2f inverse_warp = submap.warp_affines[slot].inverse().cast<float>();
    if (!inverse_warp.allFinite()) return false;
    V2F center = reference.px_.cast<float>();
    if (virt) center = V2F(virtual_support_radius, virtual_support_radius);
    std::vector<Footprint> samples(patch_size_total);
    std::set<int> support_pixels;
    for (int p = 0; p < patch_size_total; ++p) {
      const V2F offset(p % patch_size - patch_size_half, p / patch_size - patch_size_half);
      const V2F xy = inverse_warp * (offset * static_cast<float>(scale)) + center;
      if (!footprint(xy.x(), xy.y(), reference.img_.cols, reference.img_.rows, radius, samples[p])) return false;
      if (virt) {
        // Stored interpolation clamps to [0,255]. At a saturated output the
        // local derivative is defined as zero (the clipping kink is ignored).
        const float value = submap.warp_patch[slot][level * patch_size_total + p];
        if (value <= 0.f || value >= 255.f) samples[p].clear();
        for (const auto &entry : samples[p]) support_pixels.insert(entry.first);
      }
    }
    std::map<int, Footprint> ancestry;
    if (virt && virtual_patch_resampling_mode_enum == VirtualPatchResamplingMode::PULL_EXACT) {
      for (int index : support_pixels) {
        const float value = reference.img_.ptr<float>(index / reference.img_.cols)[index % reference.img_.cols];
        if (value <= 0.f || value >= 255.f) { ancestry[index] = {}; continue; }
        const V3D ray = reference.R_c_from_v_ * virtual_support_ray_lut_[index].cast<double>();
        V2D raw;
        if (!projectRawFisheyeIfValid(source, ray, radius, raw) ||
            !footprint(raw.x(), raw.y(), source.width, source.height, radius, ancestry[index])) return false;
      }
    } else if (virt) {
      // Reconstruct normalized forward-splat weights for the requested support
      // texels. No raw intensities are needed, and different virtual supports
      // share a source whenever they reuse the same original pixel.
      std::map<int, double> sums;
      const int cx = static_cast<int>(std::lround(reference.px_.x()));
      const int cy = static_cast<int>(std::lround(reference.px_.y()));
      for (int y = std::max(0, cy - virtual_raw_window_half_size);
           y <= std::min(source.height - 1, cy + virtual_raw_window_half_size); ++y) {
        for (int x = std::max(0, cx - virtual_raw_window_half_size);
             x <= std::min(source.width - 1, cx + virtual_raw_window_half_size); ++x) {
          const int raw = y * source.width + x;
          if (!source.raw_pixel_unit_ray_valid_mask[raw]) continue;
          const V3D ray = reference.R_v_from_c_ * source.raw_pixel_to_unit_ray_lut[raw].cast<double>();
          if (!ray.allFinite() || ray.z() <= virtual_min_z) continue;
          const float u = static_cast<float>(virtual_focal_length * ray.x() / ray.z() + virtual_support_radius);
          const float v = static_cast<float>(virtual_focal_length * ray.y() / ray.z() + virtual_support_radius);
          Footprint splat;
          if (!footprint(u, v, virtual_support_size, virtual_support_size, 1, splat)) continue;
          for (const auto &entry : splat) {
            if (support_pixels.count(entry.first) == 0) continue;
            ancestry[entry.first][raw] += entry.second;
            sums[entry.first] += entry.second;
          }
        }
      }
      for (int index : support_pixels) {
        if (sums[index] <= virtual_splat_min_weight) return false;
        for (auto &entry : ancestry[index]) entry.second /= sums[index];
      }
    }
    std::map<int, Eigen::VectorXd> raw_columns;
    for (int p = 0; p < patch_size_total; ++p) {
      auto add = [&](int pixel, double weight) {
        auto inserted = raw_columns.emplace(pixel, Eigen::VectorXd());
        if (inserted.second) inserted.first->second = Eigen::VectorXd::Zero(patch_size_total);
        inserted.first->second[p] += weight;
      };
      for (const auto &entry : samples[p]) {
        if (!virt) add(entry.first, entry.second);
        else for (const auto &raw : ancestry[entry.first]) add(raw.first, entry.second * raw.second);
      }
    }
    Eigen::VectorXd centered;
    double sigma = 1.0;
    if (zncc_residual_en) {
      centered.resize(patch_size_total);
      for (int p = 0; p < patch_size_total; ++p) centered[p] = submap.warp_patch[slot][level * patch_size_total + p];
      centered.array() -= centered.mean();
      sigma = std::sqrt(centered.squaredNorm() / patch_size_total);
      if (!std::isfinite(sigma) || sigma < zncc_min_std) return false;
    }
    uint64_t image_stamp = 0;
    const double stamp = reference.raw_timestamp_ != 0.0 ? reference.raw_timestamp_ : reference.capture_timestamp_;
    if (!std::isfinite(stamp)) return false;
    std::memcpy(&image_stamp, &stamp, sizeof(image_stamp));
    // A zero timestamp carries no image identity; do not invent cross-feature
    // correlation in that compatibility path.
    if (stamp == 0.0) image_stamp = reinterpret_cast<uintptr_t>(&reference);
    for (const auto &entry : raw_columns) {
      Eigen::VectorXd normalized;
      if (zncc_residual_en)
        normalized = (entry.second.array() - entry.second.mean()).matrix() / sigma -
            centered * centered.dot(entry.second) / (patch_size_total * sigma * sigma * sigma);
      else normalized = reference.inv_expo_time_ * entry.second;
      Eigen::MatrixXd b(rows, 1);
      for (int row = 0; row < rows; ++row)
        b(row, 0) = -weights[row] * normalized[patch_indices[row]] * photometric_selection_reference_pixel_std / stddev;
      if (b.squaredNorm() > 0.0)
        candidate.sources[{1, static_cast<uint64_t>(reference.camera_id_), image_stamp, static_cast<uint64_t>(entry.first)}] = std::move(b);
    }
  }
  return photometric_selection::project(candidate);
}
