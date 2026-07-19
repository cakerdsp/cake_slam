# FAST-LIVO Multi-Camera ROS1 Port

This branch is a ROS1/catkin port of the multi-camera IESKF pipeline. It vendors the matching `rpg_vikit` sources under `third_party/rpg_vikit` so `vikit_common` and `vikit_ros` are built in the same catkin workspace as `fast_livo`.

## Requirements

- Ubuntu 20.04 + ROS Noetic
- `livox_ros_driver` ROS1 package in the same catkin workspace
- A modern Sophus installation that exports CMake package config and provides `sophus/se3.hpp`
- OpenCV, PCL, Eigen, Boost, fmt

The old Sophus API with only `sophus/se3.h` is not compatible with this codebase. Install or source a modern Sophus instead of hardcoding include paths.

## Build On NUC

```bash
sudo apt update
sudo apt install ros-noetic-cv-bridge ros-noetic-image-transport ros-noetic-pcl-ros ros-noetic-pcl-conversions ros-noetic-tf2-ros ros-noetic-tf2-geometry-msgs libfmt-dev

mkdir -p ~/catkin_fast_livo/src
cd ~/catkin_fast_livo/src
git clone -b multi-cam-ieskf-ros1-clean <your-fast-livo-repo-url> fast_livo

# Put the ROS1 livox driver here as livox_ros_driver.
# Example:
# git clone https://github.com/Livox-SDK/livox_ros_driver.git

cd ~/catkin_fast_livo
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

If Sophus is installed from source but CMake cannot find it, check the package config instead of editing this repo:

```bash
find ~/Sophus* /usr/local /usr -path '*SophusConfig.cmake' -o -path '*sophus-config.cmake'
echo $CMAKE_PREFIX_PATH
```

Then source the install prefix or pass it to catkin:

```bash
catkin_make -DCMAKE_BUILD_TYPE=Release -DSophus_DIR=/path/to/sophus/install/share/sophus/cmake
```

## Run M2DGR Multi-Camera

```bash
source ~/catkin_fast_livo/devel/setup.bash
roslaunch fast_livo mapping_m2dgr_multi_cam.launch use_rviz:=false
```

The ROS1 launch loads:

- `config/M2DGR_multi_cam.yaml`
- `config/camera_M2DGR_multi_cam.yaml`

Both files use standard ROS1 nested YAML loaded into the private namespace of `laserMapping`.
