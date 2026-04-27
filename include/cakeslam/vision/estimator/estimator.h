/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once
 
#include <thread>
#include <mutex>
#include <std_msgs/msg/header.h>
#include <std_msgs/msg/float32.h>
#include <ceres/ceres.h>
#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include <cake_slam/imu_sample.h>
#include <cake_slam/slam_state.h>

#include "parameters.h"
#include "feature_manager.h"
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../initial/solve_5pts.h"
#include "../initial/initial_sfm.h"
#include "../initial/initial_alignment.h"
#include "../initial/initial_ex_rotation.h"
#include "../factor/imu_factor.h"
#include "../factor/pose_local_parameterization.h"
#include "../factor/marginalization_factor.h"
#include "../factor/projectionTwoFrameOneCamFactor.h"
#include "../factor/projectionTwoFrameTwoCamFactor.h"
#include "../factor/projectionOneFrameTwoCamFactor.h"
#include "../featureTracker/feature_tracker.h"

#include "cake_slam/imu_processor.h"
#include "cake_slam/common_lib.h"

#define ROS_INFO RCUTILS_LOG_INFO
#define ROS_WARN RCUTILS_LOG_WARN
#define ROS_ERROR RCUTILS_LOG_ERROR


class Estimator
{
  public:
    Estimator();
    ~Estimator();
    // 根据全局参数初始化外参、噪声、求解器配置等内部状态。
    void setParameter();

    // -------------------- 对外输入接口 --------------------
    // 指定一个显式的初始位姿。适用于外部已经给定系统初值的场景。
    void initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r);
    // 输入一条 IMU 测量。
    // 要求 t 单调递增，单位为秒。
    void inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity);
    // 与 inputIMU 等价的统一接口版本。
    void inputImuSample(const ImuSample &sample);
    // 输入已经整理好的特征观测帧。
    void inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame);
    // 输入原始图像，让前端自行完成跟踪。
    void inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());
    // 对单条 IMU 测量执行传播。
    void processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    // 处理一帧图像对应的特征观测，推进视觉后端。
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double header);
    // 从缓存中取出“可处理的一段图像 + IMU”，按时序驱动整个估计流程。
    void processMeasurements();
    // 动态切换是否使用 IMU、是否使用双目。
    void changeSensorType(int use_imu, int use_stereo);

    // -------------------- 内部流程函数 --------------------
    // 清空状态机与滑窗。
    void clearState();
    // 视觉初始化：恢复结构与粗略位姿。
    bool initialStructure();
    // 视觉-惯导对齐：恢复尺度、重力和速度等量。
    bool visualInitialAlign();
    // 在滑窗中寻找满足初始化条件的一对关键帧。
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    // 执行滑窗边缘化。
    void slideWindow();
    void slideWindowNew();
    void slideWindowOld();
    // 组装 ceres 问题并执行非线性优化。
    void optimization();
    // 在 Eigen 状态表示与优化参数块之间做拷贝。
    void vector2double();
    void double2vector();
    // 检测当前状态是否已经发散或失效。
    bool failureDetection();
    // 取出 [t0, t1] 区间内的 IMU。
    bool getIMUInterval(double t0, double t1, std::vector<ImuSample> &imuVector);
    // 导出当前或指定滑窗帧位姿。
    void getPoseInWorldFrame(Eigen::Matrix4d &T);
    void getPoseInWorldFrame(int index, Eigen::Matrix4d &T);
    // 基于当前运动估计预测下一帧特征位置。
    void predictPtsInNextFrame();
    // 根据重投影误差筛除外点。
    void outliersRejection(set<int> &removeIndex);
    // 计算两帧重投影误差。
    double reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                     Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                     double depth, Vector3d &uvi, Vector3d &uvj);
    // 刷新 latest_* 系列快速传播状态。
    void updateLatestStates();
    // 用最新 IMU 数据做高频前向预测。
    void fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity);
    // 判断缓存中是否已有足够的 IMU 覆盖到时间 t。
    bool IMUAvailable(double t);
    // 利用系统启动阶段的 IMU 初始化重力方向和姿态。
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

    std::mutex mProcess; // 保护后端处理流程。
    std::mutex mBuf; // 保护输入缓存队列。
    std::mutex mPropagate; // 保护 latest_* 高频传播状态。
    queue<ImuSample> imuBuf; // IMU 输入缓存。
    queue<pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > > featureBuf; // 特征观测缓存。
    double prevTime, curTime; // 最近两次图像/处理时刻。
    bool openExEstimation; // 是否开启外参在线估计。

    std::thread trackThread; // 前端跟踪线程。
    std::thread processThread; // 后端处理线程。

    FeatureTracker featureTracker; // 图像特征跟踪器。

    SolverFlag solver_flag; // 当前解算阶段：初始化或非线性优化。
    MarginalizationFlag  marginalization_flag; // 本轮边缘化策略。
    Vector3d g; // 当前估计的重力向量。

    Matrix3d ric[2]; // IMU 到相机的旋转外参。
    Vector3d tic[2]; // IMU 到相机的平移外参。

    // Vector3d        Ps[(WINDOW_SIZE + 1)];
    // Vector3d        Vs[(WINDOW_SIZE + 1)];
    // Matrix3d        Rs[(WINDOW_SIZE + 1)];
    // Vector3d        Bas[(WINDOW_SIZE + 1)];
    // Vector3d        Bgs[(WINDOW_SIZE + 1)];

    StatesGroup states_[(WINDOW_SIZE + 1)]; // 滑窗内每帧的统一状态。
    double td; // 当前时间延迟估计。

    Matrix3d back_R0, last_R, last_R0; // 视觉初始化与滑窗移位用的姿态缓存。
    Vector3d back_P0, last_P, last_P0; // 视觉初始化与滑窗移位用的位置缓存。
    double Headers[(WINDOW_SIZE + 1)]; // 滑窗内各帧时间戳。

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)]; // 每两个关键帧之间的 IMU 预积分。
    Vector3d acc_0, gyr_0; // 当前积分起点的原始 IMU 值。

    vector<double> dt_buf[(WINDOW_SIZE + 1)]; // 组成预积分的时间间隔序列。
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)]; // 对应的加速度序列。
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)]; // 对应的角速度序列。

    int frame_count; // 当前滑窗内最后一帧索引。
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid; // 特征跟踪统计量。
    int inputImageCnt; // 已输入的图像计数。

    FeatureManager f_manager; // 路标与观测管理器。
    MotionEstimator m_estimator; // 五点法相对位姿求解器。
    InitialEXRotation initial_ex_rotation; // 视觉-IMU 旋转外参初始化器。

    bool first_imu; // 是否尚未收到第一条 IMU。
    bool is_valid, is_key; // 当前图像是否有效、是否关键帧。
    bool failure_occur; // 是否触发失败检测。

    vector<Vector3d> point_cloud; // 当前滑窗重建出的稀疏点云。
    vector<Vector3d> margin_cloud; // 被边缘化的点云缓存。
    vector<Vector3d> key_poses; // 历史关键帧位置。
    double initial_timestamp; // 初始化起始时间。


    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE]; // 位姿参数块。
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS]; // 速度和偏置参数块。
    double para_Feature[NUM_OF_F][SIZE_FEATURE]; // 特征逆深度参数块。
    double para_Ex_Pose[2][SIZE_POSE]; // 相机外参参数块。
    double para_Retrive_Pose[SIZE_POSE]; // 回环/检索位姿预留参数块。
    double para_Td[1][1]; // 时间延迟参数块。
    double para_Tr[1][1]; // 滚动快门相关参数块。

    int loop_window_index; // 与回环或特殊处理相关的窗口索引。

    MarginalizationInfo *last_marginalization_info; // 上一轮边缘化得到的先验。
    vector<double *> last_marginalization_parameter_blocks; // 先验关联的参数块地址。

    map<double, ImageFrame> all_image_frame; // 初始化阶段缓存的全部图像帧。
    IntegrationBase *tmp_pre_integration; // 临时预积分对象。

    Eigen::Vector3d initP; // 外部指定的初始位置。
    Eigen::Matrix3d initR; // 外部指定的初始姿态。

    double latest_time; // 高频传播状态对应的最新时间。
    Eigen::Vector3d latest_P, latest_V, latest_Ba, latest_Bg, latest_acc_0, latest_gyr_0; // 高频传播缓存。
    Eigen::Quaterniond latest_Q; // 高频传播姿态。

    bool initFirstPoseFlag; // 是否已显式设置初始位姿。
    bool initThreadFlag; // 线程是否已经初始化。

    // 导出对外统一状态。
    SlamState getSlamState() const;
};
