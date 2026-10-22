#include "fixed_gray_image.h"

#include <iostream>
#include <limits>
#include <stdexcept>

namespace fgi = fixed_gray_image;

namespace
{

void require(bool condition, const char *message)
{
  if (!condition) throw std::runtime_error(message);
}

template <typename Callable>
void requireInvalid(Callable callable, const char *message)
{
  bool rejected = false;
  try { callable(); }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, message);
}

void testRgbWeightsOnBgrStorage()
{
  const auto lut = fgi::prepareFixedGrayLut(Eigen::Vector3d(0.5, 0.25, 0.25));
  cv::Mat bgr(1, 4, CV_8UC3);
  bgr.at<cv::Vec3b>(0, 0) = cv::Vec3b(0, 0, 200);
  bgr.at<cv::Vec3b>(0, 1) = cv::Vec3b(0, 200, 0);
  bgr.at<cv::Vec3b>(0, 2) = cv::Vec3b(200, 0, 0);
  bgr.at<cv::Vec3b>(0, 3) = cv::Vec3b(20, 40, 80);
  const cv::Mat gray = fgi::convertFixedGray(bgr, &lut);
  require(gray.type() == CV_8UC1, "runtime patches require CV_8UC1");
  require(gray.at<std::uint8_t>(0, 0) == 100, "RGB red weight must address BGR channel two");
  require(gray.at<std::uint8_t>(0, 1) == 50, "green basis intensity");
  require(gray.at<std::uint8_t>(0, 2) == 50, "RGB blue weight must address BGR channel zero");
  require(gray.at<std::uint8_t>(0, 3) == 55, "mixed-color intensity remains on the 0..255 scale");
}

void testNoncontinuousRoiAndSourceOwnership()
{
  cv::Mat storage(7, 13, CV_8UC3);
  for (int row = 0; row < storage.rows; ++row)
    for (int col = 0; col < storage.cols; ++col)
      storage.at<cv::Vec3b>(row, col) = cv::Vec3b(
          (row * 47 + col * 13) % 256, (row * 19 + col * 37) % 256, (row * 61 + col * 7) % 256);
  const cv::Mat before = storage.clone();
  const cv::Mat roi = storage(cv::Rect(2, 1, 7, 4));
  require(!roi.isContinuous(), "test must exercise row stride beyond ROI width");
  const auto lut = fgi::prepareFixedGrayLut(Eigen::Vector3d(0.16, 0.31, 0.53));
  const cv::Mat gray = fgi::convertFixedGray(roi, &lut);
  const cv::Mat compact = fgi::convertFixedGray(roi.clone(), &lut);
  require(gray.size() == roi.size(), "ROI dimensions must be preserved");
  require(cv::norm(gray, compact, cv::NORM_INF) == 0.0, "row stride must not change gray values");
  require(cv::norm(storage, before, cv::NORM_INF) == 0.0, "shared source image must remain immutable");
}

void testDisabledAndAnchorUseExactOpenCv()
{
  cv::Mat bgr(256, 256, CV_8UC3);
  for (int row = 0; row < bgr.rows; ++row)
    for (int col = 0; col < bgr.cols; ++col)
      bgr.at<cv::Vec3b>(row, col) = cv::Vec3b(row, col, (row * 71 + col * 29) % 256);
  cv::Mat legacy;
  cv::cvtColor(bgr, legacy, cv::COLOR_BGR2GRAY);
  const cv::Mat disabled = fgi::convertFixedGray(bgr);
  const cv::Mat anchor = fgi::convertFixedGray(bgr, nullptr);
  const auto default_lut = fgi::prepareFixedGrayLut(Eigen::Vector3d(0.299, 0.587, 0.114));
  const cv::Mat supported_default = fgi::convertFixedGray(bgr, &default_lut);
  require(default_lut.legacy_default, "exact default weights must preserve OpenCV conversion");
  require(cv::norm(legacy, disabled, cv::NORM_INF) == 0.0, "disabled conversion must be bit-exact OpenCV");
  require(cv::norm(legacy, anchor, cv::NORM_INF) == 0.0, "anchor conversion must be bit-exact OpenCV");
  require(cv::norm(legacy, supported_default, cv::NORM_INF) == 0.0,
          "accepted default weights must be bit-exact OpenCV");
}

void testNeutralRampAndInputRejection()
{
  const auto lut = fgi::prepareFixedGrayLut(Eigen::Vector3d(0.5, 0.25, 0.25));
  cv::Mat bgr(1, 256, CV_8UC3);
  for (int value = 0; value < 256; ++value)
    bgr.at<cv::Vec3b>(0, value) = cv::Vec3b(value, value, value);
  const cv::Mat gray = fgi::convertFixedGray(bgr, &lut);
  for (int value = 0; value < 256; ++value)
    require(gray.at<std::uint8_t>(0, value) == value, "normalized weights must preserve the entire neutral ramp");

  requireInvalid([] { fgi::prepareFixedGrayLut(Eigen::Vector3d(-0.1, 0.7, 0.4)); },
                 "negative weights must be rejected");
  requireInvalid([] { fgi::prepareFixedGrayLut(Eigen::Vector3d(0.2, 0.3, 0.4)); },
                 "unnormalized weights must be rejected");
  requireInvalid([] { fgi::prepareFixedGrayLut(Eigen::Vector3d(
                     std::numeric_limits<double>::quiet_NaN(), 0.5, 0.5)); },
                 "NaN weights must be rejected");
  requireInvalid([] { fgi::prepareFixedGrayLut(Eigen::Vector3d(
                     std::numeric_limits<double>::infinity(), 0.5, 0.5)); },
                 "infinite weights must be rejected");
  requireInvalid([&] { fgi::convertFixedGray(cv::Mat(2, 3, CV_32FC3), &lut); },
                 "float storage must not be read as bytes");
  requireInvalid([&] { fgi::convertFixedGray(cv::Mat(2, 3, CV_8UC1), &lut); },
                 "single-channel images do not contain RGB information");
}

} // namespace

int main()
{
  try
  {
    testRgbWeightsOnBgrStorage();
    testNoncontinuousRoiAndSourceOwnership();
    testDisabledAndAnchorUseExactOpenCv();
    testNeutralRampAndInputRejection();
    std::cout << "fixed_gray_image_test: PASS\n";
    return 0;
  }
  catch (const std::exception &error)
  {
    std::cerr << "fixed_gray_image_test: " << error.what() << '\n';
    return 1;
  }
}
