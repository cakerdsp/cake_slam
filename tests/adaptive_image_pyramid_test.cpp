#include "adaptive_image_pyramid.h"

#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace vs = virtual_sampling;

namespace
{

void require(bool condition, const char *message)
{
  if (!condition) throw std::runtime_error(message);
}

void near(double actual, double expected, double tolerance, const char *message)
{
  require(std::isfinite(actual) && std::isfinite(expected) &&
              std::abs(actual - expected) <= tolerance,
          message);
}

cv::Mat makeRamp(int rows, int cols)
{
  cv::Mat image(rows, cols, CV_32FC1);
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x)
      image.at<float>(y, x) = 20.0f + 0.25f * x + 0.125f * y;
  return image;
}

void testConstantIntensityScale()
{
  const double sigmas[] = {0.0, 0.5, 1.0, 1.75, std::sqrt(5.0),
                           3.0, 7.0, std::sqrt(85.0), std::sqrt(341.0)};
  for (int type : {CV_8UC1, CV_32FC1})
  {
    for (int intensity : {0, 37, 173, 255})
    {
      vs::ImagePyramid pyramid(cv::Mat(257, 321, type, cv::Scalar(intensity)));
      for (double sigma : sigmas)
      {
        vs::FilteredImage filtered;
        require(pyramid.prepare(sigma, filtered), "prepare constant image");
        float value = -1.0f;
        require(filtered.sample(160.375, 128.625, value), "sample constant image");
        near(value, intensity, 1.0e-5, "filter preserves intensity without 0..1 normalization");
      }
    }
  }
}

void testRampCoordinatesAndGradient()
{
  vs::ImagePyramid pyramid(makeRamp(385, 385));
  const double u = 160.375;
  const double v = 160.625;
  const double step = 0.5;
  for (double sigma : {0.0, 0.7, 1.0, std::sqrt(5.0), 5.0,
                       std::sqrt(85.0), std::sqrt(341.0)})
  {
    vs::FilteredImage filtered;
    require(pyramid.prepare(sigma, filtered), "prepare ramp");
    float center, left, right, top, bottom;
    require(filtered.sample(u, v, center) &&
                filtered.sample(u - step, v, left) &&
                filtered.sample(u + step, v, right) &&
                filtered.sample(u, v - step, top) &&
                filtered.sample(u, v + step, bottom),
            "sample ramp and derivatives");
    near(center, 20.0 + 0.25 * u + 0.125 * v, 2.0e-5,
         "pyramid uses original pixel centers without a half-pixel shift");
    near((right - left) / (2.0 * step), 0.25, 2.0e-5,
         "horizontal gradient remains in original pixel units");
    near((bottom - top) / (2.0 * step), 0.125, 2.0e-5,
         "vertical gradient remains in original pixel units");
  }
}

void testNominalVarianceMixture()
{
  // At a center shared by all layers, a quadratic avoids interpolation phase
  // blur. A unit-mass symmetric filter adds c*(Var[x]+Var[y]) to its value.
  cv::Mat image(513, 513, CV_32FC1);
  constexpr double curvature = 0.001;
  for (int y = 0; y < image.rows; ++y)
    for (int x = 0; x < image.cols; ++x)
    {
      const double dx = x - 256.0;
      const double dy = y - 256.0;
      image.at<float>(y, x) = static_cast<float>(50.0 + curvature * (dx * dx + dy * dy));
    }
  vs::ImagePyramid pyramid(image);
  for (double variance : {0.0, 0.25, 1.0, 2.0, 5.0, 10.0, 40.0, 85.0, 150.0, 341.0})
  {
    vs::FilteredImage filtered;
    require(pyramid.prepare(std::sqrt(variance), filtered), "prepare variance mixture");
    float value;
    require(filtered.sample(256.0, 256.0, value), "sample quadratic center");
    near(value, 50.0 + 2.0 * curvature * variance, 5.0e-5,
         "nominal filter variance is preserved between pyramid levels");
  }
}

void testZeroSigmaBilinearAndStridedInput()
{
  cv::Mat storage(6, 8, CV_8UC1, cv::Scalar(0));
  cv::Mat roi = storage(cv::Rect(2, 1, 3, 3));
  require(!roi.isContinuous(), "test uses a non-contiguous Mat");
  roi.at<unsigned char>(0, 0) = 10;
  roi.at<unsigned char>(0, 1) = 30;
  roi.at<unsigned char>(1, 0) = 50;
  roi.at<unsigned char>(1, 1) = 90;
  vs::ImagePyramid pyramid(roi);
  vs::FilteredImage filtered;
  require(pyramid.prepare(0.0, filtered), "zero sigma accepts raw source");
  float value;
  require(filtered.sample(0.25, 0.5, value), "bilinear sample in strided Mat");
  near(value, 37.5, 1.0e-6, "zero sigma equals ordinary bilinear interpolation");
  require(filtered.sample(0.0, 0.0, value), "zero sigma accepts top-left pixel");
  near(value, 10.0, 1.0e-6, "integer coordinate retains the original pixel");
  require(!filtered.sample(-0.001, 0.5, value), "reject negative raw coordinate");
  require(!filtered.sample(2.0, 0.5, value), "reject missing right bilinear tap");
  require(!filtered.sample(0.5, 2.0, value), "reject missing bottom bilinear tap");
}

void testFilterHalo()
{
  vs::ImagePyramid pyramid(cv::Mat(257, 257, CV_8UC1, cv::Scalar(100)));
  vs::FilteredImage filtered;
  float value;

  require(pyramid.prepare(1.0, filtered), "prepare level one");
  require(!filtered.sample(1.99, 128.0, value), "reject level-one left reflection support");
  require(filtered.sample(2.0, 128.0, value), "accept complete level-one left support");
  require(filtered.sample(253.99, 128.0, value), "accept complete level-one right support");
  require(!filtered.sample(254.0, 128.0, value), "reject level-one right reflection support");
  require(!filtered.sample(128.0, 1.99, value), "reject level-one top reflection support");
  require(!filtered.sample(128.0, 254.0, value), "reject level-one bottom reflection support");

  require(pyramid.prepare(std::sqrt(5.0), filtered), "prepare level two");
  require(!filtered.sample(6.5, 128.0, value),
          "halo must cover interpolation taps, not merely the query center");
  require(filtered.sample(8.0, 128.0, value), "accept complete level-two left support");
  require(filtered.sample(247.99, 128.0, value), "accept complete level-two right support");
  require(!filtered.sample(248.0, 128.0, value), "reject level-two right reflection support");

  require(pyramid.prepare(std::sqrt(341.0), filtered), "prepare maximum level");
  require(!filtered.sample(63.99, 128.0, value), "reject maximum-level left reflection support");
  require(filtered.sample(64.0, 128.0, value), "accept complete maximum-level left support");
  require(filtered.sample(191.99, 128.0, value), "accept complete maximum-level right support");
  require(!filtered.sample(192.0, 128.0, value), "reject maximum-level right reflection support");

  require(pyramid.prepare(0.5, filtered), "prepare mixed levels");
  require(!filtered.sample(1.5, 128.0, value), "both weighted layers need valid support");
}

void testBoundsAndInvalidInputs()
{
  vs::ImagePyramid pyramid(cv::Mat(257, 257, CV_8UC1, cv::Scalar(100)));
  vs::FilteredImage filtered;
  float value;
  require(!filtered.sample(128.0, 128.0, value), "default filtered image is invalid");
  require(pyramid.prepare(std::sqrt(341.0), filtered), "maximum variance is accepted");
  require(!pyramid.prepare(std::nextafter(std::sqrt(341.0),
                                        std::numeric_limits<double>::infinity()), filtered),
          "variance above the cache limit is rejected, not clamped");
  require(!filtered.sample(128.0, 128.0, value), "failed prepare clears stale layer views");
  require(!pyramid.prepare(-0.1, filtered), "negative sigma is invalid");
  require(!pyramid.prepare(std::numeric_limits<double>::infinity(), filtered), "infinite sigma is invalid");
  require(!pyramid.prepare(std::numeric_limits<double>::quiet_NaN(), filtered), "NaN sigma is invalid");
  require(!pyramid.prepare(std::numeric_limits<double>::max(), filtered), "overflowing variance is invalid");

  vs::ImagePyramid empty(cv::Mat{});
  require(!empty.prepare(0.0, filtered), "empty source is invalid");
  vs::ImagePyramid color(cv::Mat(10, 10, CV_8UC3));
  require(!color.prepare(0.0, filtered), "multichannel source is unsupported");
  vs::ImagePyramid unsupported_depth(cv::Mat(10, 10, CV_16UC1));
  require(!unsupported_depth.prepare(0.0, filtered), "unsupported source depth is rejected");
  vs::ImagePyramid tiny(cv::Mat(1, 1, CV_8UC1, cv::Scalar(100)));
  require(tiny.prepare(0.0, filtered), "tiny raw Mat can be retained");
  require(!filtered.sample(0.0, 0.0, value), "tiny Mat cannot supply bilinear support");

  require(pyramid.prepare(0.0, filtered), "valid prepare after rejected input");
  require(!filtered.sample(std::numeric_limits<double>::quiet_NaN(), 128.0, value), "NaN coordinate rejected");
  require(!filtered.sample(128.0, std::numeric_limits<double>::infinity(), value), "infinite coordinate rejected");
  require(!filtered.sample(std::numeric_limits<double>::max(), 128.0, value), "huge coordinate rejected before integer cast");
  cv::Mat nonfinite(3, 3, CV_32FC1, cv::Scalar(100));
  nonfinite.at<float>(0, 0) = std::numeric_limits<float>::quiet_NaN();
  vs::ImagePyramid nan_image(nonfinite);
  require(nan_image.prepare(0.0, filtered), "float image accepted without a full-image scan");
  require(!filtered.sample(0.5, 0.5, value), "nonfinite sampled intensity rejected");
}

void testAlignedRoiMatchesFullImage()
{
  cv::Mat full(512, 640, CV_32FC1);
  for (int y = 0; y < full.rows; ++y)
    for (int x = 0; x < full.cols; ++x)
      full.at<float>(y, x) = static_cast<float>((13 * x + 7 * y + (x * y) % 31) % 256);
  // Origin divisible by 2^5 keeps decimation phase identical at every level.
  const cv::Rect crop(96, 64, 384, 384);
  vs::ImagePyramid global_pyramid(full);
  vs::ImagePyramid roi_pyramid(full(crop).clone());
  for (double sigma : {0.0, 0.7, 1.0, std::sqrt(5.0), 5.0, std::sqrt(341.0)})
  {
    vs::FilteredImage global_filtered, roi_filtered;
    require(global_pyramid.prepare(sigma, global_filtered) &&
                roi_pyramid.prepare(sigma, roi_filtered), "prepare full-image and ROI views");
    float global_value, roi_value;
    require(global_filtered.sample(256.375, 224.625, global_value) &&
                roi_filtered.sample(256.375 - crop.x, 224.625 - crop.y, roi_value),
            "sample common interior footprint");
    near(roi_value, global_value, 2.0e-5,
         "aligned ROI and full-image pyramid have identical interior sampling phase");
  }
}

void testSharedCacheConcurrencyAndLifetime()
{
  auto pyramid = std::make_shared<vs::ImagePyramid>(makeRamp(513, 513));
  vs::FilteredImage retained;
  require(pyramid->prepare(0.0, retained), "retain a raw view before cache growth");
  std::atomic<bool> ok{true};
  std::vector<std::thread> workers;
  const double sigmas[] = {0.0, 0.7, 1.0, std::sqrt(5.0), 3.0, std::sqrt(85.0), 9.0, std::sqrt(341.0)};
  for (int worker = 0; worker < 8; ++worker)
  {
    workers.emplace_back([&, worker]() {
      try
      {
        for (int iteration = 0; iteration < 16; ++iteration)
        {
          vs::FilteredImage filtered;
          const double u = 256.125 + 0.5 * worker;
          const double v = 192.75 + 0.125 * iteration;
          float value;
          if (!pyramid->prepare(sigmas[(worker + iteration) % 8], filtered) ||
              !filtered.sample(u, v, value) ||
              std::abs(value - (20.0 + 0.25 * u + 0.125 * v)) > 2.0e-5)
            ok.store(false);
        }
      }
      catch (...)
      {
        ok.store(false);
      }
    });
  }
  for (auto &worker : workers) worker.join();
  require(ok.load(), "concurrent prepare/sample retains consistent immutable layers");
  pyramid.reset();
  float value;
  require(retained.sample(160.375, 160.625, value), "Mat view outlives its source pyramid");
  near(value, 20.0 + 0.25 * 160.375 + 0.125 * 160.625, 2.0e-5,
       "cache growth and destruction do not invalidate an existing view");
}

} // namespace

int main()
{
  try
  {
    testConstantIntensityScale();
    testRampCoordinatesAndGradient();
    testNominalVarianceMixture();
    testZeroSigmaBilinearAndStridedInput();
    testFilterHalo();
    testBoundsAndInvalidInputs();
    testAlignedRoiMatchesFullImage();
    testSharedCacheConcurrencyAndLifetime();
    std::cout << "adaptive_image_pyramid_test passed\n";
    return 0;
  }
  catch (const std::exception &error)
  {
    std::cerr << "adaptive_image_pyramid_test failed: " << error.what() << '\n';
    return 1;
  }
}
