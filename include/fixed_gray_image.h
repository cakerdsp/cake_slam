#ifndef CAKE_SLAM_FIXED_GRAY_IMAGE_H
#define CAKE_SLAM_FIXED_GRAY_IMAGE_H

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace fixed_gray_image
{

struct PreparedLut
{
  // RGB order, regardless of OpenCV's BGR byte order. Prepared once at install.
  std::array<std::array<double, 256>, 3> rgb{};
  bool legacy_default = false;
};

inline PreparedLut prepareFixedGrayLut(const Eigen::Vector3d &weights_rgb)
{
  static_assert(sizeof(double) == sizeof(std::uint64_t) && std::numeric_limits<double>::is_iec559,
                "fixed gray weights require IEEE-754 binary64 doubles");
  double sum = 0.0;
  for (int channel = 0; channel < 3; ++channel)
  {
    const double weight = weights_rgb[channel];
    std::uint64_t bits = 0;
    std::memcpy(&bits, &weight, sizeof(bits));
    // Keep the boundary check effective with the project's ARM -ffast-math flags.
    if ((bits & UINT64_C(0x7ff0000000000000)) == UINT64_C(0x7ff0000000000000) ||
        weight < 0.0 || weight > 1.0)
      throw std::invalid_argument("fixed gray weights must be finite and in [0, 1]");
    sum += weight;
  }
  if (std::abs(sum - 1.0) > 1.0e-6)
    throw std::invalid_argument("fixed gray weights must sum to one");

  PreparedLut result;
  result.legacy_default = weights_rgb[0] == 0.299 && weights_rgb[1] == 0.587 && weights_rgb[2] == 0.114;
  for (int channel = 0; channel < 3; ++channel)
    for (int value = 0; value < 256; ++value)
      result.rgb[channel][value] = weights_rgb[channel] * value;
  return result;
}

inline cv::Mat convertFixedGray(const cv::Mat &bgr, const PreparedLut *weights = nullptr)
{
  cv::Mat gray;
  if (weights == nullptr || weights->legacy_default)
  {
    // The disabled path and anchor camera retain OpenCV's exact integer conversion.
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    return gray;
  }
  if (bgr.empty() || bgr.type() != CV_8UC3)
    throw std::invalid_argument("fixed gray conversion requires a nonempty CV_8UC3 BGR image");
  gray.create(bgr.rows, bgr.cols, CV_8UC1);
  for (int row = 0; row < bgr.rows; ++row)
  {
    const cv::Vec3b *source = bgr.ptr<cv::Vec3b>(row);
    std::uint8_t *destination = gray.ptr<std::uint8_t>(row);
    for (int col = 0; col < bgr.cols; ++col)
      destination[col] = cv::saturate_cast<std::uint8_t>(
          weights->rgb[0][source[col][2]] + weights->rgb[1][source[col][1]] +
          weights->rgb[2][source[col][0]]);
  }
  return gray;
}

} // namespace fixed_gray_image

#endif
