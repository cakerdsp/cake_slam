#include "adaptive_image_pyramid.h"
#include <Eigen/Core>
#include <Eigen/LU>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <omp.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
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
struct FrameFixture
{
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  V3D translation = V3D::Zero();
  V3D w2f(const V3D &p) const { return rotation * p + translation; }
};
struct PointType { float x, y, z; };
struct PointTypeRGB { float x, y, z; uint8_t r, g, b; };
struct ColorInputFixture
{
  std::vector<PointType> points;
  size_t size() const { return points.size(); }
};
constexpr int MP_PROC_NUM = 4;

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
      require(manager.getColorFromCamera(1, point, color) && (color-V3F(30,60,90)).norm() < 1e-5,
              "other camera supplies valid color");
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

void testColorFrontHemispherePerCamera()
{
  CameraFixture camera;
  VIOManager manager;
  manager.cameras_.resize(2);
  for (auto &ctx : manager.cameras_)
  {
    ctx.cam = &camera;
    ctx.new_frame = std::make_shared<FrameFixture>();
    ctx.img_rgb = cv::Mat(256,256,CV_8UC3,cv::Scalar(30,60,90));
  }
  // Opposite-facing camera: a world point behind camera 0 is in front of camera 1.
  auto &opposite = *manager.cameras_[1].new_frame;
  opposite.rotation(0,0) = -1;
  opposite.rotation(2,2) = -1;
  opposite.translation.x() = 200;
  V3F color = V3F::Constant(-1);
  for (double z : {-1.0, 0.0})
  {
    require(!manager.getColorFromCamera(0,V3D(100,110,z),color), "camera z <= 0 is excluded");
    require((color.array() == -1).all(), "hemisphere rejection happens before sampling");
  }
  require(manager.getColorFromCamera(0,V3D(100,110,1e-6),color), "strictly positive camera z is allowed");
  require(!manager.getColorFromCamera(1,V3D(100,110,1),color), "each camera applies its own orientation");
  require(manager.getColorFromCamera(1,V3D(100,110,-1),color), "world negative z may be camera positive z");
  require(!manager.getColorFromCamera(1,V3D(100,110,0),color), "opposite camera also excludes z = 0");
}

void testEveryValidCameraObservationIsEmitted()
{
  CameraFixture camera;
  VIOManager manager;
  manager.cameras_.resize(3);
  for (int id = 0; id < manager.numCameras(); ++id)
  {
    auto &ctx = manager.cameras_[id];
    ctx.cam = &camera;
    ctx.new_frame = std::make_shared<FrameFixture>();
    ctx.img_rgb = cv::Mat(256,256,CV_8UC3,cv::Scalar(10+id,20+id,30+id));
  }
  auto input = std::make_shared<ColorInputFixture>();
  input->points = {{100.25f,110.5f,1.0f},{130.25f,135.5f,1.0f}};
  auto output = colorCloud(&manager,input);
  require(output.size() == 6, "two points visible in three cameras emit six observations");
  for (size_t i = 0; i < output.size(); ++i)
  {
    const auto &point = input->points[i%2];
    require(output[i].x == point.x && output[i].y == point.y && output[i].z == point.z,
            "camera-major output retains each observation's original XYZ");
    require(output[i].r == 30+i/2 && output[i].g == 20+i/2 && output[i].b == 10+i/2,
            "each copy retains its own camera's RGB");
  }
  cv::Mat mask(256,256,CV_8UC1,cv::Scalar(255));
  mask.at<uchar>(110,100) = 0;
  manager.cameras_[1].setImageMask(mask);
  output = colorCloud(&manager,input);
  require(output.size() == 5 && output[2].x == input->points[1].x && output[2].r == 31,
          "one camera's mask removes only its invalid observation, without stale output");
  manager.cameras_[2].new_frame->translation.z() = -2;
  output = colorCloud(&manager,input);
  require(output.size() == 3, "rear-facing observations are excluded per camera in the actual output loop");
  require(colorCloud(&manager,input,1000).empty(), "distance rejection cannot reuse previous camera output");
  for (auto &ctx : manager.cameras_) ctx.setImageMask(cv::Mat::zeros(256,256,CV_8UC1));
  require(colorCloud(&manager,input).empty(), "all masked observations emit nothing");

  for (auto &ctx : manager.cameras_)
  {
    ctx.setImageMask(cv::Mat());
    ctx.new_frame->translation.setZero();
  }
  input->points.clear();
  for (int i = 0; i < 2048; ++i) input->points.push_back({100.25f+i%31,110.5f,1.0f});
  const int old_threads = omp_get_max_threads();
  omp_set_num_threads(1);
  const auto serial = colorCloud(&manager,input);
  omp_set_num_threads(4);
  const auto parallel = colorCloud(&manager,input);
  omp_set_num_threads(old_threads);
  require(serial.size() == 6144 && parallel.size() == serial.size(), "parallel path preserves all camera observations");
  for (size_t i = 0; i < serial.size(); ++i)
  {
    const auto &a = serial[i];
    const auto &b = parallel[i];
    require(a.x==b.x && a.y==b.y && a.z==b.z && a.r==b.r && a.g==b.g && a.b==b.b,
            "parallel emission has deterministic camera and point order");
  }
}

void testColorDistanceRange()
{
  CameraFixture camera;
  VIOManager manager;
  manager.cameras_.resize(2);
  for (int id = 0; id < manager.numCameras(); ++id)
  {
    auto &ctx = manager.cameras_[id];
    ctx.cam = &camera;
    ctx.new_frame = std::make_shared<FrameFixture>();
    ctx.img_rgb = cv::Mat(256,256,CV_8UC3,cv::Scalar(10,20,30+id));
  }
  auto input = std::make_shared<ColorInputFixture>();
  const std::vector<PointType> points = {{1,2,2},{2,3,6},{3,4,12},{1,2,1}};
  input->points = points; // Distances: 3, 7, 13, sqrt(6).
  require(colorCloud(&manager,input,0,false,6,5).size() == 8,
          "disabled range ignores its bounds and preserves every valid camera observation");
  auto output = colorCloud(&manager,input,0,true,3,7);
  require(output.size() == 4 && output[0].z == 2 && output[1].z == 6 && output[2].r == 31,
          "inclusive minimum and maximum accept exact boundaries in every camera");
  output = colorCloud(&manager,input,4,true,3,7);
  require(output.size() == 2 && output[0].z == 6 && output[1].z == 6,
          "legacy blind distance remains active alongside the new range");
  output = colorCloud(&manager,input,0,true,0,5);
  require(output.size() == 4 && output[0].z == 2 && output[1].z == 1,
          "five-meter maximum excludes more distant observations");
  input->points = {{4,4,1}};
  require(colorCloud(&manager,input,0,true,0,5).empty(),
          "range uses three-dimensional distance, not just optical-axis depth");

  input->points = points;
  manager.cameras_[1].new_frame->translation = V3D(-2,-2,-11);
  output = colorCloud(&manager,input,0,true,0,5);
  require(output.size() == 3 && output[2].x == 3 && output[2].y == 4 && output[2].z == 12 && output[2].r == 31,
          "a distant world point is colored only by the nearby camera, using that camera's distance");
  cv::Mat mask(256,256,CV_8UC1,cv::Scalar(255));
  mask.at<uchar>(2,1) = 0;
  manager.cameras_[1].setImageMask(mask);
  require(colorCloud(&manager,input,0,true,0,5).size() == 2,
          "being inside the distance range does not bypass camera masks");
  manager.cameras_[1].setImageMask(cv::Mat());

  input->points.clear();
  for (int i = 0; i < 2048; ++i) input->points.push_back(points[i%4]);
  const int old_threads = omp_get_max_threads();
  omp_set_num_threads(4);
  require(colorCloud(&manager,input,0,true,0,5).size() == 1536,
          "parallel range filtering preserves the correct count for cameras with different positions");
  omp_set_num_threads(old_threads);

  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (const auto &bounds : std::vector<V2D>{V2D(-1,5),V2D(6,5),V2D(nan,5),V2D(0,inf)})
  {
    bool rejected = false;
    try { colorCloud(&manager,input,0,true,bounds.x(),bounds.y()); }
    catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "enabled range rejects negative, reversed or non-finite configuration");
  }
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
    testColorFrontHemispherePerCamera();
    testEveryValidCameraObservationIsEmitted();
    testColorDistanceRange();
    testRawSamplingAndRoi();
    testPatchesAndReferenceCamera();
    testFilteredSupportAndOpticalFlow();
    std::cout << "camera_mask_test passed: production multi-camera emission, distance range, front hemisphere, mask and sampling checks\n";
    return 0;
  }
  catch (const std::exception &error)
  {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
