#include "adaptive_image_pyramid.h"
#include <Eigen/Core>
#include <Eigen/LU>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using V2D = Eigen::Vector2d;
using V3D = Eigen::Vector3d;
using V3F = Eigen::Vector3f;
using Eigen::Matrix2d;
using Eigen::Matrix2f;
using Eigen::Vector2d;
using Eigen::Vector2f;
constexpr double kInterpPi = 3.14159265358979323846;

// Identity projection isolates mask admission from camera-model mathematics.
struct CameraFixture
{
  V2D world2cam(const V3D &p) const { return p.head<2>(); }
  bool isInFrame(const Eigen::Vector2i &p, int border) const
  { return p.x() >= border && p.y() >= border && p.x() < 256-border && p.y() < 256-border; }
};
struct FrameFixture { V3D w2f(const V3D &p) const { return p; } };

#include "camera_mask_impl.inc"

void require(bool ok, const char *message)
{
  if (!ok) throw std::runtime_error(message);
}

void testColorRejectsEveryTapAndUsesOtherCamera()
{
  CameraFixture camera;
  VIOManager manager;
  manager.cameras_.resize(2);
  for (auto &ctx : manager.cameras_)
  {
    ctx.cam = &camera;
    ctx.new_frame = std::make_shared<FrameFixture>();
    ctx.img_rgb = cv::Mat(256, 256, CV_8UC3, cv::Scalar(30, 60, 90));
  }
  cv::Mat mask(256, 256, CV_8UC1, cv::Scalar(255));
  const V3D point(100.25, 110.5, 1.0);
  for (int dy = 0; dy <= 1; ++dy)
    for (int dx = 0; dx <= 1; ++dx)
    {
      mask.setTo(255);
      mask.at<uchar>(110+dy, 100+dx) = 0;
      manager.cameras_[0].setImageMask(mask);
      V3F color = V3F::Constant(-1);
      require(!manager.getColorFromCamera(0, point, color), "any invalid bilinear tap rejects color");
      require((color.array() == -1).all(), "reject occurs before changing the color output");
      int selected = -1;
      for (int i = 0; i < manager.numCameras(); ++i)
        if (manager.getColorFromCamera(i, point, color)) { selected = i; break; }
      require(selected == 1 && (color-V3F(30,60,90)).norm() < 1e-5, "other camera supplies valid color");
      manager.cameras_[1].setImageMask(mask);
      require(!manager.getColorFromCamera(1, point, color), "all masked cameras yield no color");
      manager.cameras_[1].setImageMask(cv::Mat());
    }
  manager.cameras_[0].setImageMask(cv::Mat());
  V3F color;
  require(manager.getColorFromCamera(0, point, color), "missing mask preserves color sampling");
  mask.setTo(255);
  mask.at<uchar>(110,102)=0;
  manager.cameras_[0].setImageMask(mask);
  require(!manager.getColorFromCamera(0,V3D(100.9999999,110.5,1),color), "mask matches float rounding in color sampler");
  manager.cameras_[0].setImageMask(cv::Mat(256,256,CV_8UC1,cv::Scalar(255)));
  manager.cameras_[0].img_rgb.setTo(cv::Scalar(0,0,0));
  require(manager.getColorFromCamera(0, point, color) && color.norm() == 0, "valid black scene is not masked");
}

void testRawSamplingAndRoi()
{
  PerCameraData ctx;
  VIOManager manager;
  cv::Mat raw(256,256,CV_8UC1,cv::Scalar(80));
  cv::Mat mask(256,256,CV_8UC1,cv::Scalar(255));
  mask.at<uchar>(111,101) = 0;
  ctx.setImageMask(mask);
  float value;
  require(!manager.interpolateRawVirtualImage(ctx,raw,100.25,110.5,value), "bilinear rejects neighboring black mask pixel");
  const cv::Point origin(64,80);
  const cv::Mat roi = raw(cv::Rect(origin.x,origin.y,128,128)).clone();
  require(!manager.interpolateRawVirtualImage(ctx,roi,36.25,30.5,value,origin), "ROI uses full image mask coordinates");
  require(manager.interpolateRawVirtualImage(ctx,roi,20.25,20.5,value,origin), "valid ROI stays usable");
  mask.setTo(255);
  mask.at<uchar>(110,102)=0;
  ctx.setImageMask(mask);
  require(!manager.interpolateRawVirtualImage(ctx,raw,100.9999999,110.5,value), "mask matches vikit float-coordinate taps");
  mask.setTo(255);
  mask.at<uchar>(110,104) = 0;
  ctx.setImageMask(mask);
  V2D gradient;
  require(manager.interpolateRawVirtualImage(ctx,raw,100.25,110.5,value), "center itself is valid");
  require(!manager.sampleRawImageValueAndGradient(ctx,raw,V2D(100.25,110.5),4,value,gradient), "gradient halo is checked");
  for (auto mode : {VirtualInterpMode::BICUBIC,VirtualInterpMode::LANCZOS})
  {
    manager.virtual_interp_mode_enum = mode;
    mask.setTo(255);
    mask.at<uchar>(110,99) = 0;
    ctx.setImageMask(mask);
    require(!manager.interpolateRawVirtualImage(ctx,raw,100.25,110.5,value), "higher order interpolation checks entire kernel");
  }
  ctx.setImageMask(cv::Mat());
  require(manager.interpolateRawVirtualImage(ctx,raw,100.25,110.5,value) && std::abs(value-80)<1e-4,
          "unmasked interpolation retains intensity");
}

void testPatchesAndReferenceCamera()
{
  PerCameraData ctx;
  VIOManager manager;
  cv::Mat img(256,256,CV_8UC1);
  cv::randu(img,0,255);
  std::vector<float> patch(64*4);
  require(manager.getImagePatch(ctx,img,V2D(100.5,100.5),patch.data(),0), "unmasked patch succeeds");
  const auto legacy = patch;
  require(manager.maskedShiTomasiScore(ctx,img,100,100) == vk::shiTomasiScore(img,100,100), "unmasked score exactly matches vikit");
  cv::Mat mask(256,256,CV_8UC1,cv::Scalar(255));
  ctx.setImageMask(mask);
  require(manager.getImagePatch(ctx,img,V2D(100.5,100.5),patch.data(),0) && patch==legacy, "all-white mask preserves raw patch");
  mask.at<uchar>(100,104) = 0;
  ctx.setImageMask(mask);
  require(!manager.getImagePatch(ctx,img,V2D(100.5,100.5),patch.data(),0), "patch edge rejects despite valid center");
  require(manager.maskedShiTomasiScore(ctx,img,100,100)==0, "score gradient window respects mask");
  mask.setTo(255);
  mask.at<uchar>(100,105)=0;
  ctx.setImageMask(mask);
  require(!manager.getImagePatch(ctx,img,V2D(100.9999999,100.5),patch.data(),0), "raw patch mask matches floorf grid");
  mask.setTo(255);
  mask.at<uchar>(100,106)=0;
  ctx.setImageMask(mask);
  require(manager.getImagePatch(ctx,img,V2D(100.5,100.5),patch.data(),0), "small patch remains valid");
  require(!manager.getImagePatch(ctx,img,V2D(100.5,100.5),patch.data(),1), "coarse patch checks wider footprint");
  PerCameraData reference;
  reference.setImageMask(mask);
  require(!manager.warpAffine(reference,Matrix2d::Identity(),img,V2D(104.5,100.5),0,0,0,4,patch.data()),
          "affine reference sampling checks reference camera mask");
  require(manager.warpAffine(PerCameraData{},Matrix2d::Identity(),img,V2D(104.5,100.5),0,0,0,4,patch.data()),
          "other camera without mask is unaffected");
}

void testFilteredSupportAndOpticalFlow()
{
  PerCameraData ctx;
  VIOManager manager;
  cv::Mat mask(256,256,CV_8UC1,cv::Scalar(255));
  virtual_sampling::ImagePyramid pyramid(cv::Mat(128,128,CV_8UC1,cv::Scalar(80)));
  virtual_sampling::FilteredImage filtered;
  require(pyramid.prepare(std::sqrt(5.0),filtered), "prepare level two filter");
  const cv::Point origin(40,50);
  mask.at<uchar>(50+34,40+34)=0; // Gaussian support; outside the raw 2x2 taps at (40,40).
  ctx.setImageMask(mask);
  require(ctx.maskAllowsSample(80.25,90.5), "raw center interpolation remains valid");
  require(!ctx.maskAllowsFilteredSample(filtered,40.25,40.5,origin), "filtered ROI checks Gaussian halo in full image");
  require(ctx.maskAllowsFilteredSample(filtered,70.25,70.5,origin), "unrelated filtered sample remains valid");
  mask.setTo(255);
  ctx.setImageMask(mask);
  require(manager.inOpticalFlowBorder(ctx,cv::Point2f(128,128)), "LK interior remains usable");
  mask.at<uchar>(128,200)=0;
  ctx.setImageMask(mask);
  require(!manager.inOpticalFlowBorder(ctx,cv::Point2f(128,128)), "LK coarse support respects mask");
}

int main()
{
  try
  {
    testColorRejectsEveryTapAndUsesOtherCamera();
    testRawSamplingAndRoi();
    testPatchesAndReferenceCamera();
    testFilteredSupportAndOpticalFlow();
    std::cout << "camera_mask_test passed: production color, sampling, ROI, patch and filter checks\n";
    return 0;
  }
  catch (const std::exception &error)
  {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
