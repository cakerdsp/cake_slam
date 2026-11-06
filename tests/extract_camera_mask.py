"""Compile production mask/sampling/color methods with small camera/frame fixtures.

This avoids a ROS1 runtime dependency. It tests actual production method bodies,
not the complete ROS node, camera calibration model, or rosbag integration.
"""
import argparse
from pathlib import Path


def definition(path, signature):
    source = path.read_text(encoding="utf-8")
    start = source.index(signature)
    end = source.index("\n}\n", start) + 3
    line = source.count("\n", 0, start) + 1
    return f'#line {line} "{path.as_posix()}"\n' + source[start:end] + "\n"


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
root = args.root.resolve()
header_path = root / "include/vio_multi_cam.h"
header = header_path.read_text(encoding="utf-8")
source = root / "src/vio_multi_cam.cpp"
start = header.index("  cv::Mat image_mask;")
end = header.index("  int grid_size = 0;", start)
mask_methods = header[start:end]
enum_start = header.index("enum class VirtualInterpMode")
enum_end = header.index("};", enum_start) + 2
output = [header[enum_start:enum_end], "namespace vk {",
          definition(root / "third_party/rpg_vikit/vikit_common/include/vikit/vision.h", "inline float\ninterpolateMat_8u("),
          definition(root / "third_party/rpg_vikit/vikit_common/src/vision.cpp", "float\nshiTomasiScore("), "}"]
for signature in ["int interpolationKernelRadius(", "float clampInterpValue(",
                  "double cubicInterpWeight(", "double sinc(", "double lanczosInterpWeight("]:
    output.append(definition(source, signature))
output += ["struct PerCameraData {\n  CameraFixture *cam = nullptr;\n"
           "  std::shared_ptr<FrameFixture> new_frame;\n  cv::Mat img_rgb;\n"
           "  int width = 256, height = 256;\n" + mask_methods + "};",
           "class VIOManager { public:\n  std::vector<PerCameraData> cameras_;\n"
           "  int numCameras() const { return static_cast<int>(cameras_.size()); }\n"
           "  int patch_size = 8, patch_size_half = 4, patch_size_total = 64;\n"
           "  VirtualInterpMode virtual_interp_mode_enum = VirtualInterpMode::BILINEAR;\n"]
methods = ["V3F getInterpolatedPixel", "bool getColorFromCamera", "bool getImagePatch",
           "float maskedShiTomasiScore", "bool interpolateRawVirtualImage",
           "bool sampleRawImageValueAndGradient", "bool warpAffine", "bool inOpticalFlowBorder"]
for signature in methods:
    start = header.index("  " + signature + "(")
    output.append(header[start:header.index(";", start) + 1])
output.append("};")
for signature in methods:
    result, name = signature.split(" ", 1)
    output.append(definition(source, result + " VIOManager::" + name + "("))

# Exercise the actual multi-camera emission loop, including its OpenMP region,
# with simple point containers instead of requiring ROS publishers and PCL.
mapper_path = root / "src/LIVMapper_multi_cam.cpp"
mapper = mapper_path.read_text(encoding="utf-8")
publish_start = mapper.index("void LIVMapper::publish_frame_world(")
color_start = mapper.index("    const int camera_count = vio_manager->numCameras();", publish_start)
color_end = mapper.index("\n  }\n  else\n", color_start)
validation_start = mapper.index("  if (color_range_en &&")
validation_end = mapper.index("\n\n", validation_start)
output += ["std::vector<PointTypeRGB> colorCloud(VIOManager *vio_manager,\n"
           "    const std::shared_ptr<ColorInputFixture> &pcl_wait_pub, double blind_rgb_points = 0.0,\n"
           "    bool color_range_en = false, double color_min_distance = 0.0, double color_max_distance = 5.0) {\n"
           "  auto laserCloudWorldRGB = std::make_shared<std::vector<PointTypeRGB>>();",
           f'#line {mapper.count(chr(10), 0, validation_start) + 1} "{mapper_path.as_posix()}"',
           mapper[validation_start:validation_end],
           f'#line {mapper.count(chr(10), 0, color_start) + 1} "{mapper_path.as_posix()}"',
           mapper[color_start:color_end], "  return *laserCloudWorldRGB;\n}"]
args.output.write_text("\n".join(output), encoding="utf-8")
