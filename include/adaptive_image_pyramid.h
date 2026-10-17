#ifndef CAKE_SLAM_ADAPTIVE_IMAGE_PYRAMID_H
#define CAKE_SLAM_ADAPTIVE_IMAGE_PYRAMID_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <cmath>
#include <mutex>

namespace virtual_sampling
{

class ImagePyramid;

// Read-only layer views. Coordinates are always expressed in the original
// image (or raw ROI) pixels, and intensities retain the original 0..255 scale.
struct FilteredImage
{
  bool sample(double u, double v, float &value) const
  {
    if (!std::isfinite(u) || !std::isfinite(v)) return false;

    float lower_value = 0.0f;
    if (!sampleLayer(lower_, lower_level_, u, v, lower_value)) return false;
    if (upper_weight_ == 0.0)
    {
      value = lower_value;
      return true;
    }

    float upper_value = 0.0f;
    if (!sampleLayer(upper_, upper_level_, u, v, upper_value)) return false;
    const double mixed = (1.0 - upper_weight_) * lower_value + upper_weight_ * upper_value;
    if (!std::isfinite(mixed)) return false;
    value = static_cast<float>(mixed);
    return true;
  }

private:
  friend class ImagePyramid;

  cv::Mat lower_;
  cv::Mat upper_;
  int lower_level_ = 0;
  int upper_level_ = 0;
  int source_width_ = 0;
  int source_height_ = 0;
  double upper_weight_ = 0.0;

  bool sampleLayer(const cv::Mat &image, int level, double u, double v, float &value) const
  {
    if (image.empty()) return false;

    const int scale = 1 << level;
    const double x = u / scale;
    const double y = v / scale;
    // Check before converting to int, including for very large finite input.
    if (x < 0.0 || y < 0.0 || x >= image.cols - 1.0 || y >= image.rows - 1.0) return false;
    const int ix = static_cast<int>(std::floor(x));
    const int iy = static_cast<int>(std::floor(y));

    // A level-l pyrDown pixel depends on original pixels within radius
    // 2*(2^l-1). Require the complete support of all four bilinear taps so
    // border extrapolation never contributes an artificial observation.
    const int radius = 2 * (scale - 1);
    if (static_cast<double>(ix) * scale - radius < 0.0 ||
        static_cast<double>(iy) * scale - radius < 0.0 ||
        (static_cast<double>(ix) + 1.0) * scale + radius > source_width_ - 1.0 ||
        (static_cast<double>(iy) + 1.0) * scale + radius > source_height_ - 1.0)
      return false;

    double p00, p01, p10, p11;
    if (image.type() == CV_8UC1)
    {
      const unsigned char *row0 = image.ptr<unsigned char>(iy);
      const unsigned char *row1 = image.ptr<unsigned char>(iy + 1);
      p00 = row0[ix];
      p01 = row0[ix + 1];
      p10 = row1[ix];
      p11 = row1[ix + 1];
    }
    else if (image.type() == CV_32FC1)
    {
      const float *row0 = image.ptr<float>(iy);
      const float *row1 = image.ptr<float>(iy + 1);
      p00 = row0[ix];
      p01 = row0[ix + 1];
      p10 = row1[ix];
      p11 = row1[ix + 1];
    }
    else
    {
      return false;
    }

    const double dx = x - ix;
    const double dy = y - iy;
    const double interpolated = (1.0 - dy) * ((1.0 - dx) * p00 + dx * p01) +
                                dy * ((1.0 - dx) * p10 + dx * p11);
    if (!std::isfinite(interpolated)) return false;
    value = static_cast<float>(interpolated);
    return true;
  }
};

// Share this object with std::shared_ptr. The base Mat is retained by shallow
// copy; callers must keep its pixels immutable while the pyramid is in use.
class ImagePyramid
{
public:
  explicit ImagePyramid(const cv::Mat &base)
  {
    levels_[0] = base;
  }

  bool prepare(double extra_sigma, FilteredImage &out)
  {
    out = FilteredImage{};
    if (!std::isfinite(extra_sigma) || extra_sigma < 0.0) return false;
    if (extra_sigma > std::sqrt(levelVariance(kMaxLevel))) return false;
    const double variance = extra_sigma * extra_sigma;

    std::lock_guard<std::mutex> lock(mutex_);
    const cv::Mat &base = levels_[0];
    if (base.empty() || base.dims != 2 ||
        (base.type() != CV_8UC1 && base.type() != CV_32FC1))
      return false;

    int upper_level = 0;
    while (upper_level < kMaxLevel && std::sqrt(levelVariance(upper_level)) < extra_sigma) ++upper_level;
    int lower_level = upper_level > 0 ? upper_level - 1 : 0;
    double upper_weight = 0.0;
    // Compare in sigma coordinates: squaring sqrt(V_l) can round slightly
    // above V_l and would otherwise introduce a spurious higher-level halo.
    if (extra_sigma == std::sqrt(levelVariance(upper_level)))
    {
      lower_level = upper_level;
    }
    else
    {
      upper_weight = (variance - levelVariance(lower_level)) /
                     (levelVariance(upper_level) - levelVariance(lower_level));
    }

    for (int level = built_level_ + 1; level <= upper_level; ++level)
    {
      // Destination layers are assigned once and never modified afterwards.
      cv::Mat next;
      cv::pyrDown(levels_[level - 1], next);
      levels_[level] = next;
      built_level_ = level;
    }

    out.lower_ = levels_[lower_level];
    out.upper_ = levels_[upper_level];
    out.lower_level_ = lower_level;
    out.upper_level_ = upper_level;
    out.source_width_ = base.cols;
    out.source_height_ = base.rows;
    out.upper_weight_ = upper_weight;
    return true;
  }

private:
  static constexpr int kMaxLevel = 5;
  std::array<cv::Mat, kMaxLevel + 1> levels_;
  int built_level_ = 0;
  std::mutex mutex_;

  static double levelVariance(int level)
  {
    // pyrDown's separable [1,4,6,4,1]/16 kernel adds one pixel^2 per
    // level in that level's input coordinates: sum_{j=0}^{l-1} 4^j.
    return (static_cast<double>(1 << (2 * level)) - 1.0) / 3.0;
  }
};

} // namespace virtual_sampling

#endif // CAKE_SLAM_ADAPTIVE_IMAGE_PYRAMID_H
