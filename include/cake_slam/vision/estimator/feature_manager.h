/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS and has been adapted for Cake-SLAM.
 *
 * Licensed under the GNU General Public License v3.0.
 *******************************************************/

#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <algorithm>
#include <list>
#include <map>
#include <numeric>
#include <set>
#include <unordered_map>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <rcutils/logging_macros.h>
#include <rcpputils/asserts.hpp>

#include "cake_slam/common_lib.h"
#include "cake_slam/lidar_visual_types.h"
#include "../utility/tic_toc.h"
#include "parameters.h"

using namespace std;
#define ROS_INFO RCUTILS_LOG_INFO
#define ROS_WARN RCUTILS_LOG_WARN
#define ROS_DEBUG RCUTILS_LOG_DEBUG
#define ROS_ERROR RCUTILS_LOG_ERROR

/**
 * @brief One feature observation in one camera frame.
 */
class FeaturePerFrame
{
public:
    /**
     * @param _point [x,y,z,u,v,du,dv], normalized point, pixel, pixel velocity.
     * @param td Image time-delay value [s].
     */
    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td)
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5);
        velocity.y() = _point(6);
        cur_td = td;
        is_stereo = false;
    }

    /** @brief Attach right-camera observation for legacy stereo mode. */
    void rightObservation(const Eigen::Matrix<double, 7, 1> &_point)
    {
        pointRight.x() = _point(0);
        pointRight.y() = _point(1);
        pointRight.z() = _point(2);
        uvRight.x() = _point(3);
        uvRight.y() = _point(4);
        velocityRight.x() = _point(5);
        velocityRight.y() = _point(6);
        is_stereo = true;
    }

    double cur_td = 0.0;              ///< Observation time delay [s].
    Eigen::Vector3d point, pointRight;       ///< Normalized camera coordinates.
    Eigen::Vector2d uv, uvRight;             ///< Pixel coordinates [pixel].
    Eigen::Vector2d velocity, velocityRight; ///< Pixel velocity [pixel/s].
    bool is_stereo = false;
};

/**
 * @brief Track history and inverse-depth state for one feature id.
 */
class FeaturePerId
{
public:
    const int feature_id;
    int start_frame;
    vector<FeaturePerFrame> feature_per_frame;
    int used_num = 0;
    double estimated_depth = -1.0; ///< Host-frame depth [m].
    int solve_flag = 0;            ///< 0 unsolved, 1 solved, 2 failed.
    bool has_lidar_depth_prior = false;
    cake_slam::LidarDepthPrior lidar_depth_prior;

    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame)
    {
    }

    /** @brief Return the last frame index where this feature is observed. */
    int endFrame();

    /** @brief Minimum observations required by the optimizer. */
    int minObservationCountForOptimization() const;

    /** @brief Whether this track is mature enough for optimization. */
    bool isUsableForOptimization() const;
};

/**
 * @brief Sliding-window feature/landmark manager.
 *
 * FeatureManager stores only visual tracks and optional LiDAR inverse-depth
 * priors. Raw LiDAR points and point-to-plane residuals stay in LioCore.
 */
class FeatureManager
{
public:
    FeatureManager();

    /** @brief Set camera-to-IMU rotations R_I_C for all cameras. */
    void setRic(Eigen::Matrix3d _ric[]);

    /** @brief Clear all tracked features and pending priors. */
    void clearState();

    /** @brief Provide priors for new feature ids in the next image. */
    void setPendingLidarDepthPriors(
        const std::unordered_map<int, cake_slam::LidarDepthPrior> &priors);

    int getFeatureCount();
    bool addFeatureCheckParallax(int frame_count,
                                 const std::map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
                                 double td);
    vector<pair<Eigen::Vector3d, Eigen::Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);
    void setDepth(const Eigen::VectorXd &x);
    void removeFailures();
    void clearDepth();
    Eigen::VectorXd getDepthVector();
    void triangulate(int frameCnt, StatesGroup states_[], Eigen::Vector3d tic[], Eigen::Matrix3d ric[]);
    void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0,
                          Eigen::Matrix<double, 3, 4> &Pose1,
                          Eigen::Vector2d &point0,
                          Eigen::Vector2d &point1,
                          Eigen::Vector3d &point_3d);
    void initFramePoseByPnP(int frameCnt, StatesGroup states_[], Eigen::Vector3d tic[], Eigen::Matrix3d ric[]);
    bool solvePoseByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial,
                        vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D);
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P,
                              Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    void removeBack();
    void removeFront(int frame_count);
    void removeOutlier(set<int> &outlierIndex);

    list<FeaturePerId> feature;
    int last_track_num = 0;
    double last_average_parallax = 0.0;
    int new_feature_num = 0;
    int long_track_num = 0;

private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);

    Eigen::Matrix3d ric[2];
    std::unordered_map<int, cake_slam::LidarDepthPrior> pending_lidar_depth_priors_;
};

#endif
