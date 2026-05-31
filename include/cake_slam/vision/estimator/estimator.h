/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS and has been adapted for Cake-SLAM.
 *
 * Licensed under the GNU General Public License v3.0.
 *******************************************************/

#pragma once

// 模块功能：视觉后端估计器接口定义，组织滑窗优化、状态更新与
// LiDAR 先验融合，输出 VIO/LIO 结果与诊断信息。

#include <mutex>
#include <map>
#include <queue>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include <rcutils/logging_macros.h>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/header.hpp>

#include <cake_slam/imu_sample.h>

#include "cake_slam/common_lib.h"
#include "cake_slam/imu_processor.h"
#include "cake_slam/lidar_visual_types.h"
#include "cake_slam/utils/logging.h"
#include "../factor/imu_factor.h"
#include "../factor/marginalization_factor.h"
#include "../factor/pose_local_parameterization.h"
#include "../factor/projectionTwoFrameOneCamFactor.h"
#include "../featureTracker/feature_tracker.h"
#include "../initial/initial_alignment.h"
#include "../initial/initial_ex_rotation.h"
#include "../initial/initial_sfm.h"
#include "../initial/solve_5pts.h"
#include "../utility/tic_toc.h"
#include "../utility/utility.h"
#include "feature_manager.h"
#include "parameters.h"

#define ROS_INFO CAKE_INFO
#define ROS_WARN RCUTILS_LOG_WARN
#define ROS_ERROR RCUTILS_LOG_ERROR

/**
 * @brief One visual frame plus the LiDAR/LIO priors bound to that timestamp.
 */
struct VisionFeaturePacket
{
    double timestamp = 0.0; ///< Image timestamp [s].
    std::map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> features;
    std::unordered_map<int, cake_slam::LidarDepthPrior> lidar_depth_priors;
    cake_slam::LioPosePrior lio_pose_prior;
    cake_slam::LioFullStatePrior lio_full_prior;
};

/**
 * @brief VINS-Fusion sliding-window estimator adapted for Cake-SLAM.
 *
 * The estimator receives IMU samples, tracked visual observations, LiDAR
 * inverse-depth priors, and one LIO pose prior per image frame. Raw LiDAR
 * residuals stay inside LioCore; the VIO backend only consumes compressed
 * priors to keep the architecture flat and real-time.
 */
class Estimator
{
public:
    Estimator();
    ~Estimator();

    /** @brief Initialize noise, extrinsics, camera models, and solver options. */
    void setParameter();

    /** @brief Return cam0 projection model for external LiDAR projection/coloring. */
    camodocal::CameraConstPtr getCameraModel(int camera_id = 0) const;

    /** @brief Return R_I_C for P_imu = R_I_C * P_cam + t_I_C. */
    Eigen::Matrix3d getCameraToImuRotation(int camera_id = 0) const;

    /** @brief Return t_I_C [m] for P_imu = R_I_C * P_cam + t_I_C. */
    Eigen::Vector3d getCameraToImuTranslation(int camera_id = 0) const;

    /** @brief Set an external initial body pose T_WB. */
    void initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r);

    /** @brief Input one IMU sample [m/s^2, rad/s]. */
    void inputImuSample(const ImuSample &sample);

    /**
     * @brief Track one image and enqueue its visual measurements.
     * @param t Image timestamp [s].
     * @param _img Mono cam0 image.
     * @param lidar_candidates LiDAR-projected visual seeds [pixel, m].
     * @param lio_pose_prior Whitened LIO pose prior for this frame [rad, m].
     * @param lio_full_prior Whitened LIO full-state prior for this frame.
     * @param _img1 Optional right image for legacy stereo mode.
     */
    void inputImage(double t, const cv::Mat &_img,
                    const std::vector<cake_slam::LidarVisualCandidate> &lidar_candidates,
                    const cake_slam::LidarDepthFrame &lidar_depth_frame,
                    const cake_slam::LioPosePrior &lio_pose_prior,
                    const cake_slam::LioFullStatePrior &lio_full_prior,
                    const cv::Mat &_img1 = cv::Mat());

    /** @brief Integrate one IMU sample into the current preintegration segment. */
    void processIMU(double t, double dt, const Eigen::Vector3d &linear_acceleration,
                    const Eigen::Vector3d &angular_velocity);

    /** @brief Process one visual frame and advance initialization/optimization. */
    void processImage(const VisionFeaturePacket &packet);

    /** @brief Consume queued IMU/features in timestamp order. */
    void processMeasurements();

    void clearState();
    bool initialStructure();
    bool visualInitialAlign();
    bool relativePose(Eigen::Matrix3d &relative_R, Eigen::Vector3d &relative_T, int &l);
    void slideWindow();
    void slideWindowNew();
    void slideWindowOld();
    void optimization();
    void vector2double();
    void double2vector();
    bool failureDetection();
    bool getIMUInterval(double t0, double t1, std::vector<ImuSample> &imuVector);
    void getPoseInWorldFrame(Eigen::Matrix4d &T);
    void getPoseInWorldFrame(int index, Eigen::Matrix4d &T);

    /** @brief Return the latest optimized body state in world frame. */
    const StatesGroup &getLatestState() const;

    /** @brief Return the valid image-domain mask used by the feature tracker. */
    const cv::Mat &getImageValidMask() const;

    /** @brief Return latest feature/depth debug image rendered by FeatureTracker. */
    cv::Mat getFeatureDebugImage() const;

    /** @brief Return latest tracked feature count rendered into the debug image. */
    int getLastTrackedFeatureCount() const;

    /** @brief Return latest LiDAR-depth-associated feature count. */
    int getLastDepthFeatureCount() const;

    int getLastPrevTrackCount() const;
    int getLastTrackedAfterFlowCount() const;
    int getLastPrevLidarTrackCount() const;
    int getLastTrackedLidarCount() const;
    int getLastRejectedByLioPriorCount() const;
    int getLastAddedLidarCount() const;
    int getLastAddedVisualCount() const;
    int getLastPendingLidarCandidateCount() const;
    int getLastOptimizationFeatureCount() const;
    int getLastOptimizationLidarFeatureCount() const;
    int getLastOptimizationVisualResidualCount() const;

    void predictPtsInNextFrame();
    void outliersRejection(set<int> &removeIndex);
    double reprojectionError(Eigen::Matrix3d &Ri, Eigen::Vector3d &Pi, Eigen::Matrix3d &rici, Eigen::Vector3d &tici,
                             Eigen::Matrix3d &Rj, Eigen::Vector3d &Pj, Eigen::Matrix3d &ricj, Eigen::Vector3d &ticj,
                             double depth, Eigen::Vector3d &uvi, Eigen::Vector3d &uvj);
    void updateLatestStates();
    void fastPredictIMU(double t, Eigen::Vector3d linear_acceleration,
                        Eigen::Vector3d angular_velocity);
    bool IMUAvailable(double t);
    void initFirstIMUPose(const std::vector<ImuSample> &imuVector);

    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

    std::mutex mProcess;
    std::mutex mBuf;
    std::mutex mPropagate;
    queue<ImuSample> imuBuf;
    queue<VisionFeaturePacket> featureBuf;
    double prevTime = -1.0;
    double curTime = 0.0;
    bool openExEstimation = false;

    std::thread trackThread;
    std::thread processThread;
    FeatureTracker featureTracker;

    SolverFlag solver_flag = INITIAL;
    MarginalizationFlag marginalization_flag = MARGIN_OLD;
    Eigen::Vector3d g = Eigen::Vector3d::Zero();

    Eigen::Matrix3d ric[2];
    Eigen::Vector3d tic[2];
    StatesGroup states_[WINDOW_SIZE + 1];
    double td = 0.0;

    Eigen::Matrix3d back_R0, last_R, last_R0;
    Eigen::Vector3d back_P0, last_P, last_P0;
    double Headers[WINDOW_SIZE + 1];

    IntegrationBase *pre_integrations[WINDOW_SIZE + 1];
    Eigen::Vector3d acc_0, gyr_0;

    vector<double> dt_buf[WINDOW_SIZE + 1];
    vector<Eigen::Vector3d> linear_acceleration_buf[WINDOW_SIZE + 1];
    vector<Eigen::Vector3d> angular_velocity_buf[WINDOW_SIZE + 1];

    int frame_count = 0;
    int sum_of_outlier = 0;
    int sum_of_back = 0;
    int sum_of_front = 0;
    int sum_of_invalid = 0;
    int inputImageCnt = 0;

    FeatureManager f_manager;
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu = false;
    bool is_valid = false;
    bool is_key = false;
    bool failure_occur = false;

    vector<Eigen::Vector3d> point_cloud;
    vector<Eigen::Vector3d> margin_cloud;
    vector<Eigen::Vector3d> key_poses;
    double initial_timestamp = 0.0;

    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    double para_Ex_Pose[2][SIZE_POSE];
    double para_Retrive_Pose[SIZE_POSE];
    double para_Td[1][1];
    double para_Tr[1][1];

    int loop_window_index = -1;

    MarginalizationInfo *last_marginalization_info = nullptr;
    vector<double *> last_marginalization_parameter_blocks;

    std::map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration = nullptr;
    cake_slam::LioPosePrior lio_pose_priors_[WINDOW_SIZE + 1];
    cake_slam::LioFullStatePrior lio_full_priors_[WINDOW_SIZE + 1];
    bool lio_full_state_init_attempted_ = false;
    int last_optimization_feature_count_ = 0;
    int last_optimization_lidar_feature_count_ = 0;
    int last_optimization_visual_residual_count_ = 0;
    Eigen::Vector3d initP = Eigen::Vector3d::Zero();
    Eigen::Matrix3d initR = Eigen::Matrix3d::Identity();

    double latest_time = 0.0;
    Eigen::Vector3d latest_P, latest_V, latest_Ba, latest_Bg, latest_acc_0, latest_gyr_0;
    Eigen::Quaterniond latest_Q = Eigen::Quaterniond::Identity();

    bool initFirstPoseFlag = false;
    bool initThreadFlag = false;
};
