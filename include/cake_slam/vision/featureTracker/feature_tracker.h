/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS and has been adapted for Cake-SLAM.
 *
 * Licensed under the GNU General Public License v3.0.
 *******************************************************/

#pragma once

// 模块功能：视觉前端特征跟踪接口，
// 负责图像特征跟踪、质量筛选与 LiDAR 先验种子注入。

// #define GPU_MODE 1

#include <cstdio>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <rcutils/logging_macros.h>

#ifdef GPU_MODE
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
#endif

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "../estimator/parameters.h"
#include "../utility/tic_toc.h"
#include "cake_slam/lidar_visual_types.h"
#include "cake_slam/utils/logging.h"

using namespace std;
using namespace camodocal;
#define ROS_INFO CAKE_INFO
#define ROS_WARN RCUTILS_LOG_WARN
#define ROS_DEBUG RCUTILS_LOG_DEBUG
#define ROS_ERROR RCUTILS_LOG_ERROR

/** @brief Compact a point vector in place according to a status mask. */
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);

/** @brief Compact an integer vector in place according to a status mask. */
void reduceVector(vector<int> &v, vector<uchar> status);

struct FeatureTrackerTiming
{
    double total_ms = 0.0;
    double lk_forward_ms = 0.0;
    double lk_backward_ms = 0.0;
    double lio_gate_ms = 0.0;
    double reduce_ms = 0.0;
    double set_mask_ms = 0.0;
    double add_lidar_ms = 0.0;
    double good_features_ms = 0.0;
    double console_ms = 0.0;
    double add_points_ms = 0.0;
    double undistort_ms = 0.0;
    double velocity_ms = 0.0;
    double stereo_ms = 0.0;
    double draw_track_ms = 0.0;
    double debug_draw_ms = 0.0;
    double pack_ms = 0.0;
    int rows = 0;
    int cols = 0;
    int type = 0;
    int channels = 0;
    int prev_tracks = 0;
    int after_flow_tracks = 0;
    int final_tracks = 0;
    int pending_lidar = 0;
    int added_lidar = 0;
    int requested_visual = 0;
    int added_visual = 0;
    int flow_back = 0;
    int has_prediction = 0;
};

/**
 * @brief VINS-style sparse optical-flow tracker with LiDAR seed injection.
 *
 * The tracker owns only image-domain operations: LK tracking, forward-backward
 * checks, valid-domain masking, LiDAR-projected seed insertion, and LIO-prior
 * reprojection gating. It does not run backend optimization.
 */
class FeatureTracker
{
public:
    FeatureTracker();

    /**
     * @brief Track one image and output VINS feature observations.
     * @param _cur_time Image timestamp [s], monotonic.
     * @param _img Mono left/cam0 image.
     * @param _img1 Optional right image for legacy stereo mode.
     * @return feature id -> camera id -> [x,y,z,u,v,du,dv].
     */
    std::map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImage(
        double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());

    /** @brief Set LiDAR-projected candidates [pixel, depth m] for this frame. */
    void setLidarDepthCandidates(const vector<cake_slam::LidarVisualCandidate> &candidates);

    /** @brief Set current LIO pose prior gate, replacing legacy F-matrix gating. */
    void setLioPriorGate(const cake_slam::LioPosePrior &prior,
                         const Eigen::Matrix3d &R_I_C,
                         const Eigen::Vector3d &t_I_C);

    /** @brief Take inverse-depth priors generated for newly inserted tracks. */
    std::unordered_map<int, cake_slam::LidarDepthPrior> takeLidarDepthPriors();

    /** @brief Build a VINS-style spatial mask; long tracks reserve space first. */
    void setMask();

    /** @brief Add image-only corner features into free mask regions. */
    void addPoints();

    /** @brief Load camodocal camera models and optional valid-domain mask. */
    void readIntrinsicParameter(const vector<string> &calib_file);

    /** @brief Render undistortion diagnostics. */
    void showUndistortion(const string &name);

    /** @brief Gate LiDAR-seeded tracks by LIO-prior reprojection error [pixel]. */
    void rejectWithLioPrior(vector<uchar> &status);

    /** @brief Convert current pixel tracks to normalized camera coordinates. */
    void undistortedPoints();

    /** @brief Undistort arbitrary pixel points with the selected camera model. */
    vector<cv::Point2f> undistortedPts(vector<cv::Point2f> &pts, int camera_id);

    /** @brief Estimate per-feature pixel velocity [pixel/s]. */
    vector<cv::Point2f> ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts,
                                    std::map<int, cv::Point2f> &cur_id_pts,
                                    std::map<int, cv::Point2f> &prev_id_pts);

    void showTwoImage(const cv::Mat &img1, const cv::Mat &img2,
                      vector<cv::Point2f> pts1, vector<cv::Point2f> pts2);
    void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight,
                   vector<int> &curLeftIds,
                   vector<cv::Point2f> &curLeftPts,
                   vector<cv::Point2f> &curRightPts,
                   std::map<int, cv::Point2f> &prevLeftPtsMap);
    void setPrediction(std::map<int, Eigen::Vector3d> &predictPts);
    double distance(cv::Point2f &pt1, cv::Point2f &pt2);
    void removeOutliers(set<int> &removePtsIds);
    cv::Mat getTrackImage();
    cv::Mat getFeatureDebugImage() const;
    int getLastFeatureCount() const;
    int getLastDepthFeatureCount() const;
    int getLastPrevTrackCount() const { return last_prev_track_count; }
    int getLastTrackedAfterFlowCount() const { return last_tracked_after_flow_count; }
    int getLastPrevLidarTrackCount() const { return last_prev_lidar_track_count; }
    int getLastTrackedLidarCount() const { return last_tracked_lidar_count; }
    int getLastRejectedByLioPriorCount() const { return last_rejected_by_lio_prior_count; }
    int getLastAddedLidarCount() const { return last_added_lidar_count; }
    int getLastAddedVisualCount() const { return last_added_visual_count; }
    int getLastPendingLidarCandidateCount() const { return last_pending_lidar_candidate_count; }
    const FeatureTrackerTiming &getLastTiming() const { return last_timing; }

    /** @brief Check image border and valid-domain mask membership. */
    bool inBorder(const cv::Point2f &pt);

    /** @brief Return the valid-domain mask used by setMask(), CV_8UC1. */
    const cv::Mat &validMask() const;

    /** @brief Insert LiDAR-projected candidates as new tracks. */
    int addLidarCandidatePoints(int max_num);

    /** @brief Remove LiDAR prior metadata for tracks that disappeared. */
    void pruneLidarTracks();

    /** @brief Draw feature/depth association diagnostics for the latest frame. */
    void drawFeatureDebugImage(const std::map<int, cv::Point2f> &previous_points);

    int row = 0;
    int col = 0;
    cv::Mat imTrack;
    cv::Mat feature_debug_image;
    cv::Mat mask;
    cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img;
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> predict_pts;
    vector<cv::Point2f> predict_pts_debug;
    vector<cv::Point2f> prev_pts, cur_pts, cur_right_pts;
    vector<cv::Point2f> prev_un_pts, cur_un_pts, cur_un_right_pts;
    vector<cv::Point2f> pts_velocity, right_pts_velocity;
    vector<int> ids, ids_right;
    vector<int> track_cnt;
    std::map<int, cv::Point2f> cur_un_pts_map, prev_un_pts_map;
    std::map<int, cv::Point2f> cur_un_right_pts_map, prev_un_right_pts_map;
    std::map<int, cv::Point2f> prevLeftPtsMap;
    vector<camodocal::CameraPtr> m_camera;
    vector<int> fast_undistort_model;
    vector<cv::Mat> fast_undistort_K;
    vector<cv::Mat> fast_undistort_D;
    double cur_time = 0.0;
    double prev_time = 0.0;
    bool stereo_cam = false;
    int n_id = 0;
    bool hasPrediction = false;
    int last_feature_count = 0;
    int last_depth_feature_count = 0;
    int last_prev_track_count = 0;
    int last_tracked_after_flow_count = 0;
    int last_prev_lidar_track_count = 0;
    int last_tracked_lidar_count = 0;
    int last_rejected_by_lio_prior_count = 0;
    int last_added_lidar_count = 0;
    int last_added_visual_count = 0;
    int last_pending_lidar_candidate_count = 0;
    FeatureTrackerTiming last_timing;

    vector<cake_slam::LidarVisualCandidate> pending_lidar_candidates;
    std::unordered_map<int, cake_slam::LidarDepthPrior> active_lidar_priors;
    std::unordered_map<int, cake_slam::LidarDepthPrior> current_lidar_priors;
    cake_slam::LioPosePrior lio_prior_gate;
    Eigen::Matrix3d gate_R_I_C = Eigen::Matrix3d::Identity();
    Eigen::Vector3d gate_t_I_C = Eigen::Vector3d::Zero();
};
