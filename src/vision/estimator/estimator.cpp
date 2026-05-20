/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

// 模块功能：视觉后端估计器实现，
// 负责滑窗优化、状态更新与 LiDAR 先验融合。

#include "estimator.h"
#include "cake_slam/vision/utility/visualization.h"
#include "cake_slam/vision/factor/inverse_depth_prior_factor.h"
#include "cake_slam/vision/factor/lio_full_state_prior_factor.h"
#include "cake_slam/vision/factor/lio_pose_prior_factor.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

// 视觉-惯导估计器构造函数。
// 这里只初始化容器和默认状态，不进行参数读取。
Estimator::Estimator()
{
    CAKE_INFO("init begins");
    initThreadFlag = false;
    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    for (auto &pre_integration : pre_integrations)
        pre_integration = nullptr;
    clearState();
}

// 析构时释放预积分和边缘化先验等动态内存。
Estimator::~Estimator()
{
    if (processThread.joinable())
    {
        MULTIPLE_THREAD = 0;
        processThread.join();
        printf("join thread \n");
    }
}

// 清空整个估计器状态机，回到系统尚未初始化的状态。
void Estimator::clearState()
{
    mProcess.lock();
    while(!imuBuf.empty())
        imuBuf.pop();
    while(!featureBuf.empty())
        featureBuf.pop();

    prevTime = -1;
    curTime = 0;
    openExEstimation = 0;
    initP = Eigen::Vector3d(0, 0, 0);
    initR = Eigen::Matrix3d::Identity();
    inputImageCnt = 0;
    initFirstPoseFlag = false;

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        // states_[i].rot_end.setIdentity();
        // states_[i].pos_end.setZero();
        // states_[i].vel_end.setZero();
        // states_[i].bias_a.setZero();
        // states_[i].bias_g.setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
        {
            delete pre_integrations[i];
            pre_integrations[i] = nullptr;
        }
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Eigen::Vector3d::Zero();
        ric[i] = Eigen::Matrix3d::Identity();
    }

    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();

    if (tmp_pre_integration != nullptr)
    {
        delete tmp_pre_integration;
        tmp_pre_integration = nullptr;
    }
    if (last_marginalization_info != nullptr)
    {
        delete last_marginalization_info;
        last_marginalization_info = nullptr;
    }

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();
    for (auto &prior : lio_pose_priors_)
        prior = cake_slam::LioPosePrior();
    for (auto &prior : lio_full_priors_)
        prior = cake_slam::LioFullStatePrior();
    lio_full_state_init_attempted_ = false;
    last_optimization_feature_count_ = 0;
    last_optimization_lidar_feature_count_ = 0;
    last_optimization_visual_residual_count_ = 0;

    failure_occur = 0;

    mProcess.unlock();
}

// 把 parameters.cpp 中加载好的全局参数写入当前估计器实例。
void Estimator::setParameter()
{
    mProcess.lock();
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
        cout << " exitrinsic cam " << i << endl  << ric[i] << endl << tic[i].transpose() << endl;
    }
    f_manager.setRic(ric);
    ProjectionTwoFrameOneCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Eigen::Matrix2d::Identity();
    // Cake-SLAM currently exposes the mono ROS2 visual chain. The historical
    // VINS stereo residual classes are not built until the stereo frontend is
    // ported as a first-class module.
    td = TD;
    g = G;
    cout << "set g " << g.transpose() << endl;
    featureTracker.readIntrinsicParameter(CAM_NAMES);

    std::cout << "MULTIPLE_THREAD is " << MULTIPLE_THREAD << '\n';
    if (MULTIPLE_THREAD && !initThreadFlag)
    {
        initThreadFlag = true;
        processThread = std::thread(&Estimator::processMeasurements, this);
    }
    mProcess.unlock();
}

camodocal::CameraConstPtr Estimator::getCameraModel(int camera_id) const
{
    if (camera_id < 0 || camera_id >= static_cast<int>(featureTracker.m_camera.size()))
        return camodocal::CameraConstPtr();
    return featureTracker.m_camera[camera_id];
}

Eigen::Matrix3d Estimator::getCameraToImuRotation(int camera_id) const
{
    if (camera_id < 0 || camera_id >= NUM_OF_CAM)
        return Eigen::Matrix3d::Identity();
    return ric[camera_id];
}

Eigen::Vector3d Estimator::getCameraToImuTranslation(int camera_id) const
{
    // Fixed-extrinsic path: SlamNode colors LiDAR points with the YAML T_C_L.
    // If online extrinsic refinement is enabled later, the refined ric/tic
    // should also be exported to the main node before coloring.
    if (camera_id < 0 || camera_id >= NUM_OF_CAM)
        return Eigen::Vector3d::Zero();
    return tic[camera_id];
}

// 输入原始图像，并由前端先做跟踪，再把结果送入后端缓存。
void Estimator::inputImage(double t, const cv::Mat &_img,
                           const std::vector<cake_slam::LidarVisualCandidate> &lidar_candidates,
                           const cake_slam::LioPosePrior &lio_pose_prior,
                           const cake_slam::LioFullStatePrior &lio_full_prior,
                           const cv::Mat &_img1)
{
    inputImageCnt++;
    std::map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    TicToc featureTrackerTime;

    featureTracker.setLidarDepthCandidates(lidar_candidates);
    featureTracker.setLioPriorGate(lio_pose_prior, ric[0], tic[0]);
    if(_img1.empty())
        featureFrame = featureTracker.trackImage(t, _img);
    else
        featureFrame = featureTracker.trackImage(t, _img, _img1);
    const double feature_tracker_ms = featureTrackerTime.toc();
    const FeatureTrackerTiming &ft_timing = featureTracker.getLastTiming();
    std::printf("FEATURE TRACKER DEBUG stamp=%.6f total=%.3f measured=%.3f image=%dx%d type=%d ch=%d prev=%d flow=%d final=%d pending_lidar=%d add_lidar=%d req_visual=%d add_visual=%d flow_back=%d pred=%d stages_ms={lk_fwd=%.3f lk_back=%.3f lio_gate=%.3f reduce=%.3f mask=%.3f add_lidar=%.3f gft=%.3f cout=%.3f add_pts=%.3f undist=%.3f velocity=%.3f stereo=%.3f draw=%.3f debug=%.3f pack=%.3f}\n",
                t,
                ft_timing.total_ms,
                feature_tracker_ms,
                ft_timing.cols,
                ft_timing.rows,
                ft_timing.type,
                ft_timing.channels,
                ft_timing.prev_tracks,
                ft_timing.after_flow_tracks,
                ft_timing.final_tracks,
                ft_timing.pending_lidar,
                ft_timing.added_lidar,
                ft_timing.requested_visual,
                ft_timing.added_visual,
                ft_timing.flow_back,
                ft_timing.has_prediction,
                ft_timing.lk_forward_ms,
                ft_timing.lk_backward_ms,
                ft_timing.lio_gate_ms,
                ft_timing.reduce_ms,
                ft_timing.set_mask_ms,
                ft_timing.add_lidar_ms,
                ft_timing.good_features_ms,
                ft_timing.console_ms,
                ft_timing.add_points_ms,
                ft_timing.undistort_ms,
                ft_timing.velocity_ms,
                ft_timing.stereo_ms,
                ft_timing.draw_track_ms,
                ft_timing.debug_draw_ms,
                ft_timing.pack_ms);

    if (SHOW_TRACK)
    {
        cv::Mat imgTrack = featureTracker.getTrackImage();
        pubTrackImage(imgTrack, t);
    }

    VisionFeaturePacket packet;
    packet.timestamp = t;
    packet.features = featureFrame;
    packet.lidar_depth_priors = featureTracker.takeLidarDepthPriors();
    packet.lio_pose_prior = lio_pose_prior;
    packet.lio_full_prior = lio_full_prior;

    CAKE_INFO("[VIO frontend time] stamp=%.6f track=%.3f ms features=%zu lidar_depth=%zu lidar_candidates=%zu lio_full_prior=%d threaded=%d",
             t, feature_tracker_ms, featureFrame.size(), packet.lidar_depth_priors.size(),
             lidar_candidates.size(), lio_full_prior.valid ? 1 : 0, MULTIPLE_THREAD);

    if(MULTIPLE_THREAD)
    {
        mBuf.lock();
        featureBuf.push(packet);
        mBuf.unlock();
    }
    else
    {
        mBuf.lock();
        featureBuf.push(packet);
        mBuf.unlock();
        TicToc processTime;
        processMeasurements();
        CAKE_INFO("[VIO process time] stamp=%.6f process=%.3f ms", t, processTime.toc());
    }
}

// inputIMU 的统一数据结构版本。
void Estimator::inputImuSample(const ImuSample &sample)
{
    mBuf.lock();
    imuBuf.push(sample);
    //printf("input imu with time %f \n", sample.stamp);
    mBuf.unlock();

    if (solver_flag == NON_LINEAR)
    {
        mPropagate.lock();
        fastPredictIMU(sample.stamp, sample.acc, sample.gyr);
        pubLatestOdometry(latest_P, latest_Q, latest_V, sample.stamp);
        mPropagate.unlock();
    }
}

// 从 imuBuf 中取出覆盖 [t0, t1] 的 IMU 子序列。
bool Estimator::getIMUInterval(double t0, double t1, std::vector<ImuSample> &imuVector)
{
    if(imuBuf.empty())
    {
        printf("not receive imu\n");
        return false;
    }
    if(t1 <= imuBuf.back().stamp)
    {
        while (!imuBuf.empty() && imuBuf.front().stamp <= t0)
        {
            imuBuf.pop();
        }
        while (!imuBuf.empty() && imuBuf.front().stamp < t1)
        {
            imuVector.push_back(imuBuf.front());
            imuBuf.pop();
        }
        if (!imuBuf.empty()) {
            imuVector.push_back(imuBuf.front());
        }
    }
    else
    {
        printf("wait for imu\n");
        return false;
    }
    return true;
}

// 判断缓冲区里的 IMU 是否已经覆盖到时间 t。
bool Estimator::IMUAvailable(double t)
{
    if(!imuBuf.empty() && t <= imuBuf.back().stamp)
        return true;
    else
        return false;
}

// 后端主循环：
// 反复从图像/特征缓存和 IMU 缓存中取出一组完整测量并执行传播、初始化或优化。
void Estimator::processMeasurements()
{
    while (1)
    {
        // cout << "[processMeasurements]  loop - start" << endl;

        VisionFeaturePacket feature;
        std::vector<ImuSample> imuVector;
        if(!featureBuf.empty())
        {
            // cout << "1" << endl;
            feature = featureBuf.front();
            curTime = feature.timestamp + td;
            // std::cout << "t0: " << std::fixed << curTime << std::endl;
            while(1)
            {
                if ((!USE_IMU  || IMUAvailable(feature.timestamp + td)))
                    break;
                else
                {
                    printf("wait for imu ... \n");
                    if (! MULTIPLE_THREAD)
                        return;
                    std::chrono::milliseconds dura(5);
                    std::this_thread::sleep_for(dura);
                }
            }
            // cout << "2" << endl;
            mBuf.lock();
            if(USE_IMU)
            {
                // cout << "2-1)" << endl;
                getIMUInterval(prevTime, curTime, imuVector);
                // cout << "2-2)" << endl;
            }

            featureBuf.pop();
            mBuf.unlock();

            // cout << "3" << endl;
            if(USE_IMU)
            {
                if(!initFirstPoseFlag)
                    initFirstIMUPose(imuVector);
                for(size_t i = 0; i < imuVector.size(); i++)
                {
                    double dt;
                    if(i == 0)
                        dt = imuVector[i].stamp - prevTime;
                    else if (i == imuVector.size() - 1)
                        dt = curTime - imuVector[i - 1].stamp;
                    else
                        dt = imuVector[i].stamp - imuVector[i - 1].stamp;
                    processIMU(imuVector[i].stamp, dt, imuVector[i].acc, imuVector[i].gyr);
                }
            }
            // cout << "4" << endl;

            mProcess.lock();
            TicToc processImageTime;
            processImage(feature);
            const double process_image_ms = processImageTime.toc();
            prevTime = curTime;

            // cout << "5" << endl;

            printStatistics(*this, 0);

            std_msgs::msg::Header header;
            header.frame_id = WORLD_FRAME_ID;

            int sec_ts = (int)feature.timestamp;
            uint nsec_ts = (uint)((feature.timestamp - sec_ts) * 1e9);
            header.stamp.sec = sec_ts;
            header.stamp.nanosec = nsec_ts;

            pubOdometry(*this, header);
            // cout << "5-1" << endl;
            pubKeyPoses(*this, header);
            // cout << "5-2" << endl;
            pubCameraPose(*this, header);
            // cout << "5-3" << endl;
            pubPointCloud(*this, header);
            // cout << "5-4" << endl;
            pubKeyframe(*this);
            // cout << "5-5" << endl;
            pubTF(*this, header);
            // cout << "5-6" << endl;
            CAKE_INFO("[VIO backend time] stamp=%.6f image_process=%.3f ms imu=%zu solver_flag=%d frame_count=%d",
                     feature.timestamp, process_image_ms, imuVector.size(),
                     static_cast<int>(solver_flag), frame_count);
            mProcess.unlock();


            // cout << "6" << endl;
        }
        // cout << "[processMeasurements]  loop - end" << endl;

        if (! MULTIPLE_THREAD)
            break;

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}


// 利用系统启动阶段的 IMU 平均值估计初始重力方向和姿态。
void Estimator::initFirstIMUPose(const std::vector<ImuSample> &imuVector)
{
    printf("init first imu pose\n");
    initFirstPoseFlag = true;
    //return;
    Eigen::Vector3d averAcc(0, 0, 0);
    int n = (int)imuVector.size();
    for(size_t i = 0; i < imuVector.size(); i++)
    {
        averAcc = averAcc + imuVector[i].acc;
    }
    averAcc = averAcc / n;
    printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());
    Eigen::Matrix3d R0 = Utility::g2R(averAcc);
    double yaw = Utility::R2ypr(R0).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    states_[0].rot_end = R0;
    cout << "init R0 " << endl << states_[0].rot_end << endl;
    //states_[0].vel_end = Eigen::Vector3d(5, 0, 0);
}

// 设置外部给定的初始位姿。
void Estimator::initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r)
{
    // states_[0].pos_end = p;
    // states_[0].rot_end = r;
    states_[0].pos_end= p;
    states_[0].rot_end = r;
    initP = p;
    initR = r;
}


// 对单条 IMU 测量做预积分与名义状态传播。
void Estimator::processIMU(double t, double dt, const Eigen::Vector3d &linear_acceleration, const Eigen::Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, states_[frame_count].bias_a, states_[frame_count].bias_g};
    }
    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;
        Eigen::Vector3d un_acc_0 = states_[j].rot_end * (acc_0 - states_[j].bias_a) - g;
        Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - states_[j].bias_g;
        states_[j].rot_end *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Eigen::Vector3d un_acc_1 = states_[j].rot_end * (linear_acceleration - states_[j].bias_a) - g;
        Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        states_[j].pos_end += dt * states_[j].vel_end + 0.5 * dt * dt * un_acc;
        states_[j].vel_end += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

// 处理一帧图像观测。
// 根据当前阶段不同，它可能触发视觉初始化，也可能直接进入滑窗优化。
void Estimator::processImage(const VisionFeaturePacket &packet)
{
    const auto &image = packet.features;
    const double header = packet.timestamp;

    f_manager.setPendingLidarDepthPriors(packet.lidar_depth_priors);

    cout << std::fixed << header << endl;

    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
    {
        marginalization_flag = MARGIN_OLD;
        //printf("keyframe\n");
    }
    else
    {
        marginalization_flag = MARGIN_SECOND_NEW;
        //printf("non-keyframe\n");
    }

    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;
    lio_pose_priors_[frame_count] = packet.lio_pose_prior;
    lio_full_priors_[frame_count] = packet.lio_full_prior;
    if (lio_pose_priors_[frame_count].valid && lio_pose_priors_[frame_count].timestamp <= 0.0)
        lio_pose_priors_[frame_count].timestamp = header;
    if (lio_full_priors_[frame_count].valid && lio_full_priors_[frame_count].timestamp <= 0.0)
        lio_full_priors_[frame_count].timestamp = header;
    if (lio_pose_priors_[frame_count].valid)
    {
        states_[frame_count].rot_end = lio_pose_priors_[frame_count].R_WB;
        states_[frame_count].pos_end = lio_pose_priors_[frame_count].p_WB;
    }

    ImageFrame imageframe(image, header);
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(header, imageframe));
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, states_[frame_count].bias_a, states_[frame_count].bias_g};

    if(ESTIMATE_EXTRINSIC == 2)
    {
        CAKE_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            vector<pair<Eigen::Vector3d, Eigen::Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Eigen::Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                // ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }


    if (solver_flag == INITIAL)
    {
        if (frame_count == WINDOW_SIZE && !lio_full_state_init_attempted_)
        {
            bool has_lio_full_prior = false;
            for (int i = 0; i <= WINDOW_SIZE; ++i)
            {
                if (lio_full_priors_[i].valid)
                {
                    has_lio_full_prior = true;
                    break;
                }
            }

            if (has_lio_full_prior)
            {
                auto &anchor_prior = lio_full_priors_[0];
                if (!anchor_prior.valid || std::abs(anchor_prior.timestamp - Headers[0]) > 0.01)
                {
                    anchor_prior.valid = false;
                }
                constexpr double kMaxLioFullPriorDt = 0.01;
                bool lio_window_ok = anchor_prior.valid;
                bool imu_preintegration_ok = USE_IMU;
                double max_lio_prior_dt = 0.0;
                for (int i = 0; i <= WINDOW_SIZE; ++i)
                {
                    const auto &prior = lio_full_priors_[i];
                    const double prior_dt = prior.valid ? std::abs(prior.timestamp - Headers[i])
                                                        : std::numeric_limits<double>::infinity();
                    max_lio_prior_dt = std::max(max_lio_prior_dt, prior_dt);
                    if (!prior.valid || prior_dt > kMaxLioFullPriorDt)
                    {
                        lio_window_ok = false;
                        break;
                    }
                }
                for (int i = 1; i <= WINDOW_SIZE; ++i)
                {
                    if (!pre_integrations[i] || pre_integrations[i]->sum_dt <= 0.0)
                    {
                        imu_preintegration_ok = false;
                        break;
                    }
                }

                if (lio_window_ok && imu_preintegration_ok)
                {
                    lio_full_state_init_attempted_ = true;
                    std::printf(BOLDYELLOW "| %-29s | %-27s |" RESET "\n", "VIO Init Path", "LIO full-state window");
                    CAKE_INFO("LIO full-state initialization: anchor=0 stamp=%.6f window_stamp=%.6f max_dt=%.6f",
                              anchor_prior.timestamp, Headers[0], max_lio_prior_dt);
                    std::printf(BOLDYELLOW "| %-29s | ba_z=% .6f bg_z=% .6f |" RESET "\n",
                                "LIO Full Bias",
                                anchor_prior.ba.z(),
                                anchor_prior.bg.z());

                    for (int i = 0; i <= WINDOW_SIZE; ++i)
                    {
                        const auto &prior = lio_full_priors_[i];
                        states_[i].rot_end = prior.R_WB;
                        states_[i].pos_end = prior.p_WB;
                        states_[i].vel_end = prior.v_WB;
                        states_[i].bias_a = prior.ba;
                        states_[i].bias_g = prior.bg;
                    }
                    for (int i = 1; i <= WINDOW_SIZE; ++i)
                    {
                        pre_integrations[i]->repropagate(states_[i - 1].bias_a, states_[i - 1].bias_g);
                    }

                    std::printf(BOLDYELLOW "| %-29s | z0=% .4f zN=% .4f v0z=% .4f vNz=% .4f |" RESET "\n",
                                "VIO Init Prior",
                                states_[0].pos_end.z(),
                                states_[WINDOW_SIZE].pos_end.z(),
                                states_[0].vel_end.z(),
                                states_[WINDOW_SIZE].vel_end.z());
                    std::printf(BOLDYELLOW "| %-29s | g=% .3f % .3f % .3f |" RESET "\n",
                                "VIO Init Gravity",
                                g.x(), g.y(), g.z());

                    if (tmp_pre_integration)
                        tmp_pre_integration->repropagate(states_[frame_count].bias_a, states_[frame_count].bias_g);
                    f_manager.triangulate(frame_count, states_, tic, ric);
                    optimization();
                    std::printf(BOLDYELLOW "| %-29s | z0=% .4f zN=% .4f v0z=% .4f vNz=% .4f |" RESET "\n",
                                "VIO Init Post-Opt",
                                states_[0].pos_end.z(),
                                states_[WINDOW_SIZE].pos_end.z(),
                                states_[0].vel_end.z(),
                                states_[WINDOW_SIZE].vel_end.z());
                    updateLatestStates();
                    solver_flag = NON_LINEAR;
                    slideWindow();
                    CAKE_INFO("LIO full-state initialization finish!");
                    return;
                }

                std::printf("LIO FULL INIT WAIT clean_window=0 anchor_valid=%d imu_preintegration_ok=%d max_lio_dt=%.6f t0=%.6f tN=%.6f force_margin_old=1\n",
                            static_cast<int>(anchor_prior.valid),
                            static_cast<int>(imu_preintegration_ok),
                            max_lio_prior_dt,
                            Headers[0],
                            Headers[WINDOW_SIZE]);
                marginalization_flag = MARGIN_OLD;
                slideWindow();
                return;
            }
        }

        if (frame_count == WINDOW_SIZE && lio_pose_priors_[frame_count].valid)
        {
            std::printf(BOLDYELLOW "| %-29s | %-27s |" RESET "\n", "VIO Init Path", "LIO prior");
            f_manager.triangulate(frame_count, states_, tic, ric);
            optimization();
            updateLatestStates();
            solver_flag = NON_LINEAR;
            slideWindow();
            CAKE_INFO("LIO-prior initialization finish!");
            return;
        }

        // monocular + IMU initilization
        if (!STEREO && USE_IMU)
        {
            if (frame_count == WINDOW_SIZE)
            {
                std::printf(BOLDYELLOW "| %-29s | %-27s |" RESET "\n", "VIO Init Path", "VINS initialStructure");
                bool result = false;
                if(ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1)
                {
                    result = initialStructure();
                    initial_timestamp = header;
                }
                if(result)
                {
                    optimization();
                    updateLatestStates();
                    solver_flag = NON_LINEAR;
                    slideWindow();
                    CAKE_INFO("Initialization finish!");
                }
                else
                    slideWindow();
            }
        }

        // stereo + IMU initilization
        if(STEREO && USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, states_, tic, ric);
            f_manager.triangulate(frame_count, states_, tic, ric);
            if (frame_count == WINDOW_SIZE)
            {
                std::map<double, ImageFrame>::iterator frame_it;
                int i = 0;
                for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++)
                {
                    frame_it->second.R = states_[i].rot_end;
                    frame_it->second.T = states_[i].pos_end;
                    i++;
                }
                solveGyroscopeBias(all_image_frame, states_);
                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    pre_integrations[i]->repropagate(Eigen::Vector3d::Zero(), states_[i].bias_g);
                }
                optimization();
                updateLatestStates();
                solver_flag = NON_LINEAR;
                slideWindow();
                CAKE_INFO("Initialization finish!");
            }
        }

        // stereo only initilization
        if(STEREO && !USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, states_, tic, ric);
            f_manager.triangulate(frame_count, states_, tic, ric);
            optimization();

            if(frame_count == WINDOW_SIZE)
            {
                optimization();
                updateLatestStates();
                solver_flag = NON_LINEAR;
                slideWindow();
                CAKE_INFO("Initialization finish!");
            }
        }

        if(frame_count < WINDOW_SIZE)
        {
            frame_count++;
            int prev_frame = frame_count - 1;
            states_[frame_count].pos_end = states_[prev_frame].pos_end;
            states_[frame_count].vel_end = states_[prev_frame].vel_end;
            states_[frame_count].rot_end = states_[prev_frame].rot_end;
            states_[frame_count].bias_a = states_[prev_frame].bias_a;
            states_[frame_count].bias_g = states_[prev_frame].bias_g;
        }

    }
    else
    {
        if(!USE_IMU)
            f_manager.initFramePoseByPnP(frame_count, states_, tic, ric);
        f_manager.triangulate(frame_count, states_, tic, ric);

        // optimization
        TicToc t_solve;
        optimization();
        CAKE_INFO("solver costs: %f [ms]", t_solve.toc());

        set<int> removeIndex;
        outliersRejection(removeIndex);
        f_manager.removeOutlier(removeIndex);
        if (! MULTIPLE_THREAD)
        {
            featureTracker.removeOutliers(removeIndex);
            predictPtsInNextFrame();
        }


        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }

        slideWindow();
        f_manager.removeFailures();
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(states_[i].pos_end);

        last_R = states_[WINDOW_SIZE].rot_end;
        last_P = states_[WINDOW_SIZE].pos_end;
        last_R0 = states_[0].rot_end;
        last_P0 = states_[0].pos_end;
        updateLatestStates();
    }
}

// 视觉初始化第一阶段：估计相对位姿并执行多视图 SFM。
bool Estimator::initialStructure()
{
    TicToc t_sfm;
    //check imu observibility
    {
        std::map<double, ImageFrame>::iterator frame_it;
        Eigen::Vector3d sum_g = Eigen::Vector3d::Zero();
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Eigen::Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Eigen::Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Eigen::Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //ROS_WARN("IMU variation %f!", var);
        if(var < 0.25)
        {
            CAKE_INFO("IMU excitation not enouth!");
            //return false;
        }
    }
    // global sfm
    Eigen::Quaterniond Q[frame_count + 1];
    Eigen::Vector3d T[frame_count + 1];
    std::map<int, Eigen::Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Eigen::Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    }
    Eigen::Matrix3d relative_R;
    Eigen::Vector3d relative_T;
    int l;
    if (!relativePose(relative_R, relative_T, l))
    {
        CAKE_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    GlobalSFM sfm;
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame
    std::map<double, ImageFrame>::iterator frame_it;
    std::map<int, Eigen::Vector3d>::iterator it;
    frame_it = all_image_frame.begin( );
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if((frame_it->first) == Headers[i])
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if((frame_it->first) > Headers[i])
        {
            i++;
        }
        Eigen::Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Eigen::Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end())
                {
                    Eigen::Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Eigen::Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        Eigen::MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        Eigen::MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
    if (visualInitialAlign())
        return true;
    else
    {
        CAKE_INFO("misalign visual structure with IMU");
        return false;
    }

}

// 视觉初始化第二阶段：把纯视觉结果和 IMU 预积分对齐，恢复尺度/重力/速度。
bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    Eigen::VectorXd x;
    //solve scale
    bool result = VisualIMUAlignment(all_image_frame, states_, g, x);
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
    for (int i = 0; i <= frame_count; i++)
    {
        Eigen::Matrix3d Ri = all_image_frame[Headers[i]].R;
        Eigen::Vector3d Pi = all_image_frame[Headers[i]].T;
        states_[i].pos_end = Pi;
        states_[i].rot_end = Ri;
        all_image_frame[Headers[i]].is_key_frame = true;
    }

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Eigen::Vector3d::Zero(), states_[i].bias_g);
    }
    for (int i = frame_count; i >= 0; i--)
        states_[i].pos_end = s * states_[i].pos_end - states_[i].rot_end * TIC[0] - (s * states_[0].pos_end - states_[0].rot_end * TIC[0]);
    int kv = -1;
    std::map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            states_[kv].vel_end = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    Eigen::Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * states_[0].rot_end).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    //Eigen::Matrix3d rot_diff = R0 * states_[0].rot_end.transpose();
    Eigen::Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        states_[i].pos_end = rot_diff * states_[i].pos_end;
        states_[i].rot_end = rot_diff * states_[i].rot_end;
        states_[i].vel_end = rot_diff * states_[i].vel_end;
    }
    // ROS_DEBUG_STREAM("g0     " << g.transpose());
    // ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(states_[0].rot_end).transpose());

    f_manager.clearDepth();
    f_manager.triangulate(frame_count, states_, tic, ric);

    return true;
}

// 在滑窗内搜索一对拥有足够视差的帧，用作初始化基线。
bool Estimator::relativePose(Eigen::Matrix3d &relative_R, Eigen::Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Eigen::Vector3d, Eigen::Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Eigen::Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Eigen::Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

// 把当前 Eigen 状态拷贝到 ceres 使用的裸数组参数块中。
void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        // cout << states_[i].pos_end.x() << " " << states_[i].pos_end.y() << " " << states_[i].pos_end.z() << endl;
        // cout << "--------" << endl;

        para_Pose[i][0] = states_[i].pos_end.x();
        para_Pose[i][1] = states_[i].pos_end.y();
        para_Pose[i][2] = states_[i].pos_end.z();
        Eigen::Quaterniond q{states_[i].rot_end};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        if(USE_IMU)
        {
            para_SpeedBias[i][0] = states_[i].vel_end.x();
            para_SpeedBias[i][1] = states_[i].vel_end.y();
            para_SpeedBias[i][2] = states_[i].vel_end.z();

            para_SpeedBias[i][3] = states_[i].bias_a.x();
            para_SpeedBias[i][4] = states_[i].bias_a.y();
            para_SpeedBias[i][5] = states_[i].bias_a.z();

            para_SpeedBias[i][6] = states_[i].bias_g.x();
            para_SpeedBias[i][7] = states_[i].bias_g.y();
            para_SpeedBias[i][8] = states_[i].bias_g.z();
        }
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Eigen::Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }


    Eigen::VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);

    para_Td[0][0] = td;
}

// 把优化后的参数块写回 Eigen 状态表示。
void Estimator::double2vector()
{
    Eigen::Vector3d origin_R0 = Utility::R2ypr(states_[0].rot_end);
    Eigen::Vector3d origin_P0 = states_[0].pos_end;

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }

    bool has_absolute_lio_prior = false;
    for (int i = 0; i <= frame_count && i <= WINDOW_SIZE; ++i)
    {
        if (lio_pose_priors_[i].valid || lio_full_priors_[i].valid)
        {
            has_absolute_lio_prior = true;
            break;
        }
    }

    if(USE_IMU && has_absolute_lio_prior)
    {
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            states_[i].rot_end = Eigen::Quaterniond(para_Pose[i][6],
                                                    para_Pose[i][3],
                                                    para_Pose[i][4],
                                                    para_Pose[i][5]).normalized().toRotationMatrix();
            states_[i].pos_end = Eigen::Vector3d(para_Pose[i][0],
                                                 para_Pose[i][1],
                                                 para_Pose[i][2]);
            states_[i].vel_end = Eigen::Vector3d(para_SpeedBias[i][0],
                                                 para_SpeedBias[i][1],
                                                 para_SpeedBias[i][2]);
            states_[i].bias_a = Eigen::Vector3d(para_SpeedBias[i][3],
                                                para_SpeedBias[i][4],
                                                para_SpeedBias[i][5]);
            states_[i].bias_g = Eigen::Vector3d(para_SpeedBias[i][6],
                                                para_SpeedBias[i][7],
                                                para_SpeedBias[i][8]);
        }
    }
    else if(USE_IMU)
    {
        Eigen::Vector3d origin_R00 = Utility::R2ypr(Eigen::Quaterniond(para_Pose[0][6],
                                                          para_Pose[0][3],
                                                          para_Pose[0][4],
                                                          para_Pose[0][5]).toRotationMatrix());
        double y_diff = origin_R0.x() - origin_R00.x();
        // Keep the original VINS yaw-gauge handling after optimization.
        Eigen::Matrix3d rot_diff = Utility::ypr2R(Eigen::Vector3d(y_diff, 0, 0));
        if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
        {
            ROS_DEBUG("euler singular point!");
            rot_diff = states_[0].rot_end * Eigen::Quaterniond(para_Pose[0][6],
                                           para_Pose[0][3],
                                           para_Pose[0][4],
                                           para_Pose[0][5]).toRotationMatrix().transpose();
        }

        for (int i = 0; i <= WINDOW_SIZE; i++)
        {

            states_[i].rot_end = rot_diff * Eigen::Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

            states_[i].pos_end = rot_diff * Eigen::Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                                    para_Pose[i][1] - para_Pose[0][1],
                                                    para_Pose[i][2] - para_Pose[0][2]) + origin_P0;


                states_[i].vel_end = rot_diff * Eigen::Vector3d(para_SpeedBias[i][0],
                                                        para_SpeedBias[i][1],
                                                        para_SpeedBias[i][2]);

                states_[i].bias_a = Eigen::Vector3d(para_SpeedBias[i][3],
                                               para_SpeedBias[i][4],
                                               para_SpeedBias[i][5]);

                states_[i].bias_g = Eigen::Vector3d(para_SpeedBias[i][6],
                                               para_SpeedBias[i][7],
                                               para_SpeedBias[i][8]);

        }
    }
    else
    {
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            states_[i].rot_end = Eigen::Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

            states_[i].pos_end = Eigen::Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
        }
    }

    if(USE_IMU)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            tic[i] = Eigen::Vector3d(para_Ex_Pose[i][0],
                              para_Ex_Pose[i][1],
                              para_Ex_Pose[i][2]);
            ric[i] = Eigen::Quaterniond(para_Ex_Pose[i][6],
                                 para_Ex_Pose[i][3],
                                 para_Ex_Pose[i][4],
                                 para_Ex_Pose[i][5]).normalized().toRotationMatrix();
        }
    }

    Eigen::VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);

    if(USE_IMU)
        td = para_Td[0][0];

}

// 对偏置、位姿跳变等指标做启发式发散检测。
bool Estimator::failureDetection()
{
    return false;
    if (f_manager.last_track_num < 2)
    {
        CAKE_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    if (states_[WINDOW_SIZE].bias_a.norm() > 2.5)
    {
        CAKE_INFO(" big IMU acc bias estimation %f", states_[WINDOW_SIZE].bias_a.norm());
        return true;
    }
    if (states_[WINDOW_SIZE].bias_g.norm() > 1.0)
    {
        CAKE_INFO(" big IMU gyr bias estimation %f", states_[WINDOW_SIZE].bias_g.norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        CAKE_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Eigen::Vector3d tmp_P = states_[WINDOW_SIZE].pos_end;
    if ((tmp_P - last_P).norm() > 5)
    {
        // CAKE_INFO(" big translation");
        //return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        // CAKE_INFO(" big z translation");
        //return true;
    }
    Eigen::Matrix3d tmp_R = states_[WINDOW_SIZE].rot_end;
    Eigen::Matrix3d delta_R = tmp_R.transpose() * last_R;
    Eigen::Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        CAKE_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}

// 组装所有视觉、IMU 和先验残差，然后执行一次滑窗非线性优化。
void Estimator::optimization()
{
    TicToc t_whole, t_prepare;
    vector2double();

    const bool use_lio_full_prior_factor = USE_IMU && LIO_FULL_STATE_PRIOR_ENABLE;
    const bool has_lio_full_prior_for_debug = [&]() {
        if (!USE_IMU)
            return false;
        for (int i = 0; i < frame_count + 1; ++i)
        {
            if (lio_full_priors_[i].valid)
                return true;
        }
        return false;
    }();
    const bool debug_lio_full_prior = VIO_DEBUG_FACTOR_COSTS && has_lio_full_prior_for_debug;
    auto rotationDeltaDeg = [](const Eigen::Quaterniond &q_ref, const Eigen::Quaterniond &q) {
        const Eigen::Quaterniond dq = q_ref.conjugate() * q.normalized();
        return 2.0 * std::atan2(dq.vec().norm(), std::abs(dq.w())) * 57.29577951308232;
    };
    auto print_lio_full_prior_para_delta = [&](const char *stage) {
        if (!debug_lio_full_prior)
            return;
        double dz0 = std::numeric_limits<double>::quiet_NaN();
        double dzN = std::numeric_limits<double>::quiet_NaN();
        double max_abs_dz = 0.0;
        double sum_abs_dz = 0.0;
        double max_abs_vz = 0.0;
        double max_rot_deg = 0.0;
        int max_dz_idx = -1;
        int valid_count = 0;
        for (int i = 0; i < frame_count + 1; ++i)
        {
            const auto &prior = lio_full_priors_[i];
            if (!prior.valid)
                continue;
            const double dz = para_Pose[i][2] - prior.p_WB.z();
            const double abs_dz = std::abs(dz);
            const double dvz = para_SpeedBias[i][2] - prior.v_WB.z();
            const Eigen::Quaterniond q_prior(prior.R_WB);
            const Eigen::Quaterniond q_para(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]);
            const double rot_deg = rotationDeltaDeg(q_prior, q_para);
            if (i == 0)
                dz0 = dz;
            if (i == frame_count)
                dzN = dz;
            if (abs_dz > max_abs_dz)
            {
                max_abs_dz = abs_dz;
                max_dz_idx = i;
            }
            max_abs_vz = std::max(max_abs_vz, std::abs(dvz));
            max_rot_deg = std::max(max_rot_deg, rot_deg);
            sum_abs_dz += abs_dz;
            ++valid_count;
        }
        const double avg_abs_dz = valid_count > 0 ? sum_abs_dz / valid_count : 0.0;
        const int last = frame_count;
        std::printf("VIO OPT DEBUG %s para_delta: n=%d z0=% .6f zN=% .6f dz0=% .6f dzN=% .6f max_abs_dz=% .6f max_dz_idx=%d avg_abs_dz=% .6f max_abs_vz=% .6f max_rot_deg=% .6f prior_z0=% .6f prior_zN=% .6f\n",
                    stage,
                    valid_count,
                    para_Pose[0][2],
                    para_Pose[last][2],
                    dz0,
                    dzN,
                    max_abs_dz,
                    max_dz_idx,
                    avg_abs_dz,
                    max_abs_vz,
                    max_rot_deg,
                    lio_full_priors_[0].valid ? lio_full_priors_[0].p_WB.z() : std::numeric_limits<double>::quiet_NaN(),
                    lio_full_priors_[last].valid ? lio_full_priors_[last].p_WB.z() : std::numeric_limits<double>::quiet_NaN());
    };
    auto print_lio_full_prior_state_delta = [&](const char *stage) {
        if (!debug_lio_full_prior)
            return;
        double dz0 = std::numeric_limits<double>::quiet_NaN();
        double dzN = std::numeric_limits<double>::quiet_NaN();
        double max_abs_dz = 0.0;
        double sum_abs_dz = 0.0;
        double max_abs_vz = 0.0;
        double max_rot_deg = 0.0;
        int max_dz_idx = -1;
        int valid_count = 0;
        for (int i = 0; i < frame_count + 1; ++i)
        {
            const auto &prior = lio_full_priors_[i];
            if (!prior.valid)
                continue;
            const double dz = states_[i].pos_end.z() - prior.p_WB.z();
            const double abs_dz = std::abs(dz);
            const double dvz = states_[i].vel_end.z() - prior.v_WB.z();
            const Eigen::Quaterniond q_prior(prior.R_WB);
            const Eigen::Quaterniond q_state(states_[i].rot_end);
            const double rot_deg = rotationDeltaDeg(q_prior, q_state);
            if (i == 0)
                dz0 = dz;
            if (i == frame_count)
                dzN = dz;
            if (abs_dz > max_abs_dz)
            {
                max_abs_dz = abs_dz;
                max_dz_idx = i;
            }
            max_abs_vz = std::max(max_abs_vz, std::abs(dvz));
            max_rot_deg = std::max(max_rot_deg, rot_deg);
            sum_abs_dz += abs_dz;
            ++valid_count;
        }
        const double avg_abs_dz = valid_count > 0 ? sum_abs_dz / valid_count : 0.0;
        const int last = frame_count;
        std::printf("VIO OPT DEBUG %s state_delta: n=%d z0=% .6f zN=% .6f dz0=% .6f dzN=% .6f max_abs_dz=% .6f max_dz_idx=%d avg_abs_dz=% .6f max_abs_vz=% .6f max_rot_deg=% .6f prior_z0=% .6f prior_zN=% .6f\n",
                    stage,
                    valid_count,
                    states_[0].pos_end.z(),
                    states_[last].pos_end.z(),
                    dz0,
                    dzN,
                    max_abs_dz,
                    max_dz_idx,
                    avg_abs_dz,
                    max_abs_vz,
                    max_rot_deg,
                    lio_full_priors_[0].valid ? lio_full_priors_[0].p_WB.z() : std::numeric_limits<double>::quiet_NaN(),
                    lio_full_priors_[last].valid ? lio_full_priors_[last].p_WB.z() : std::numeric_limits<double>::quiet_NaN());
    };
    struct DebugCostStats
    {
        int blocks = 0;
        double sum_sq = 0.0;
        double robust_cost = 0.0;
        double max_norm = 0.0;
    };
    struct DebugWorstVisual
    {
        int feature_id = -1;
        int host_frame = -1;
        int target_frame = -1;
        bool has_lidar_depth = false;
        double norm = 0.0;
        double inv_depth = std::numeric_limits<double>::quiet_NaN();
        double depth = std::numeric_limits<double>::quiet_NaN();
        double prior_inv_depth = std::numeric_limits<double>::quiet_NaN();
        double prior_inv_depth_var = std::numeric_limits<double>::quiet_NaN();
    };
    struct DebugWorstImu
    {
        int i = -1;
        int j = -1;
        double sum_dt = 0.0;
        double raw_norm = 0.0;
        double white_norm = 0.0;
        Eigen::Matrix<double, 15, 1> raw = Eigen::Matrix<double, 15, 1>::Zero();
        Eigen::Vector3d gravity_body_i = Eigen::Vector3d::Zero();
        Eigen::Vector3d delta_v_avg = Eigen::Vector3d::Zero();
        Eigen::Vector3d implied_acc_body = Eigen::Vector3d::Zero();
        Eigen::Vector3d delta_p_avg = Eigen::Vector3d::Zero();
    };
    auto huber_cost_1 = [](double squared_norm) {
        if (squared_norm <= 1.0)
            return 0.5 * squared_norm;
        return std::sqrt(squared_norm) - 0.5;
    };
    auto add_cost_block = [](DebugCostStats &stats, double squared_norm) {
        stats.blocks++;
        stats.sum_sq += squared_norm;
        stats.max_norm = std::max(stats.max_norm, std::sqrt(std::max(0.0, squared_norm)));
    };
    auto print_debug_factor_costs = [&](const char *stage) {
        if (!debug_lio_full_prior)
            return;

        DebugCostStats imu_cost;
        DebugCostStats lio_pose_cost;
        DebugCostStats lio_full_cost;
        DebugCostStats depth_prior_cost;
        DebugCostStats visual_cost;
        DebugWorstImu worst_imu;
        DebugWorstVisual worst_visual;
        double lio_full_pos_sq = 0.0;
        double lio_full_rot_sq = 0.0;
        double lio_full_vel_sq = 0.0;
        double lio_full_ba_sq = 0.0;
        double lio_full_bg_sq = 0.0;
        double min_lidar_depth = std::numeric_limits<double>::infinity();
        double max_lidar_depth = 0.0;
        double min_inv_depth_var = std::numeric_limits<double>::infinity();
        double max_inv_depth_var = 0.0;

        if (USE_IMU)
        {
            for (int i = 0; i < frame_count; ++i)
            {
                const int j = i + 1;
                if (!pre_integrations[j] || pre_integrations[j]->sum_dt > 10.0)
                    continue;
                const Eigen::Vector3d Pi(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
                const Eigen::Quaterniond Qi(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]);
                const Eigen::Vector3d Vi(para_SpeedBias[i][0], para_SpeedBias[i][1], para_SpeedBias[i][2]);
                const Eigen::Vector3d Bai(para_SpeedBias[i][3], para_SpeedBias[i][4], para_SpeedBias[i][5]);
                const Eigen::Vector3d Bgi(para_SpeedBias[i][6], para_SpeedBias[i][7], para_SpeedBias[i][8]);
                const Eigen::Vector3d Pj(para_Pose[j][0], para_Pose[j][1], para_Pose[j][2]);
                const Eigen::Quaterniond Qj(para_Pose[j][6], para_Pose[j][3], para_Pose[j][4], para_Pose[j][5]);
                const Eigen::Vector3d Vj(para_SpeedBias[j][0], para_SpeedBias[j][1], para_SpeedBias[j][2]);
                const Eigen::Vector3d Baj(para_SpeedBias[j][3], para_SpeedBias[j][4], para_SpeedBias[j][5]);
                const Eigen::Vector3d Bgj(para_SpeedBias[j][6], para_SpeedBias[j][7], para_SpeedBias[j][8]);
                const Eigen::Matrix<double, 15, 1> raw = pre_integrations[j]->evaluate(
                    Pi, Qi, Vi, Bai, Bgi, Pj, Qj, Vj, Baj, Bgj);
                IMUFactor imu_factor(pre_integrations[j]);
                double residuals[15] = {0.0};
                double const *params[4] = {para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]};
                imu_factor.Evaluate(params, residuals, nullptr);
                double sq = 0.0;
                for (double residual : residuals)
                    sq += residual * residual;
                add_cost_block(imu_cost, sq);
                const double raw_norm = raw.norm();
                const double white_norm = std::sqrt(std::max(0.0, sq));
                if (white_norm > worst_imu.white_norm)
                {
                    worst_imu.i = i;
                    worst_imu.j = j;
                    worst_imu.sum_dt = pre_integrations[j]->sum_dt;
                    worst_imu.raw_norm = raw_norm;
                    worst_imu.white_norm = white_norm;
                    worst_imu.raw = raw;
                    worst_imu.gravity_body_i = Qi.inverse() * G;
                    worst_imu.delta_v_avg =
                        pre_integrations[j]->sum_dt > 0.0
                            ? (pre_integrations[j]->delta_v / pre_integrations[j]->sum_dt).eval()
                            : Eigen::Vector3d::Zero();
                    worst_imu.implied_acc_body =
                        pre_integrations[j]->sum_dt > 0.0
                            ? Qi.inverse() * (G + (Vj - Vi) / pre_integrations[j]->sum_dt)
                            : Eigen::Vector3d::Zero();
                    worst_imu.delta_p_avg =
                        pre_integrations[j]->sum_dt > 0.0
                            ? (2.0 * pre_integrations[j]->delta_p /
                               (pre_integrations[j]->sum_dt * pre_integrations[j]->sum_dt))
                                  .eval()
                            : Eigen::Vector3d::Zero();
                }
            }
        }

        for (int i = 0; i < frame_count + 1; ++i)
        {
            if (lio_pose_priors_[i].valid)
            {
                const auto &prior = lio_pose_priors_[i];
                const Eigen::Vector3d p(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
                const Eigen::Quaterniond q(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]);
                const Eigen::Quaterniond q_prior(prior.R_WB);
                const Eigen::Quaterniond dq = q_prior.conjugate() * q;
                Eigen::Matrix<double, 6, 1> raw;
                raw.segment<3>(0) = 2.0 * Eigen::Vector3d(dq.x(), dq.y(), dq.z());
                raw.segment<3>(3) = prior.R_WB.transpose() * (p - prior.p_WB);
                const Eigen::Matrix<double, 6, 1> whitened = prior.sqrt_information * raw;
                add_cost_block(lio_pose_cost, whitened.squaredNorm());
            }
            if (USE_IMU && lio_full_priors_[i].valid)
            {
                const auto &prior = lio_full_priors_[i];
                const Eigen::Vector3d p(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
                const Eigen::Quaterniond q(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]);
                const Eigen::Quaterniond q_prior(prior.R_WB);
                const Eigen::Quaterniond dq = q_prior.conjugate() * q;
                Eigen::Matrix<double, 15, 1> raw;
                raw.segment<3>(0) = p - prior.p_WB;
                raw.segment<3>(3) = 2.0 * Eigen::Vector3d(dq.x(), dq.y(), dq.z());
                raw.segment<3>(6) =
                    Eigen::Vector3d(para_SpeedBias[i][0], para_SpeedBias[i][1], para_SpeedBias[i][2]) - prior.v_WB;
                raw.segment<3>(9) =
                    Eigen::Vector3d(para_SpeedBias[i][3], para_SpeedBias[i][4], para_SpeedBias[i][5]) - prior.ba;
                raw.segment<3>(12) =
                    Eigen::Vector3d(para_SpeedBias[i][6], para_SpeedBias[i][7], para_SpeedBias[i][8]) - prior.bg;
                const Eigen::Matrix<double, 15, 1> whitened = prior.sqrt_information * raw;
                add_cost_block(lio_full_cost, whitened.squaredNorm());
                lio_full_pos_sq += whitened.segment<3>(0).squaredNorm();
                lio_full_rot_sq += whitened.segment<3>(3).squaredNorm();
                lio_full_vel_sq += whitened.segment<3>(6).squaredNorm();
                lio_full_ba_sq += whitened.segment<3>(9).squaredNorm();
                lio_full_bg_sq += whitened.segment<3>(12).squaredNorm();
            }
        }

        int feature_index = -1;
        for (auto &it_per_id : f_manager.feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!it_per_id.isUsableForOptimization())
                continue;

            ++feature_index;
            if (it_per_id.has_lidar_depth_prior && it_per_id.lidar_depth_prior.valid)
            {
                const double var = std::max(it_per_id.lidar_depth_prior.inv_depth_var, MIN_INV_DEPTH_VAR);
                const double residual = (para_Feature[feature_index][0] -
                                         it_per_id.lidar_depth_prior.inv_depth) /
                                        std::sqrt(var);
                add_cost_block(depth_prior_cost, residual * residual);
                min_lidar_depth = std::min(min_lidar_depth, it_per_id.lidar_depth_prior.depth);
                max_lidar_depth = std::max(max_lidar_depth, it_per_id.lidar_depth_prior.depth);
                min_inv_depth_var = std::min(min_inv_depth_var, it_per_id.lidar_depth_prior.inv_depth_var);
                max_inv_depth_var = std::max(max_inv_depth_var, it_per_id.lidar_depth_prior.inv_depth_var);
            }

            const int imu_i = it_per_id.start_frame;
            int imu_j = imu_i - 1;
            const Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point;
            for (auto &it_per_frame : it_per_id.feature_per_frame)
            {
                imu_j++;
                if (imu_i == imu_j)
                    continue;

                ProjectionTwoFrameOneCamFactor factor(
                    pts_i, it_per_frame.point,
                    it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                    it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                double residuals[2] = {0.0, 0.0};
                double const *params[5] = {
                    para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0],
                    para_Feature[feature_index], para_Td[0]};
                factor.Evaluate(params, residuals, nullptr);
                const double sq = residuals[0] * residuals[0] + residuals[1] * residuals[1];
                add_cost_block(visual_cost, sq);
                visual_cost.robust_cost += huber_cost_1(sq);
                const double norm = std::sqrt(std::max(0.0, sq));
                if (norm > worst_visual.norm)
                {
                    worst_visual.feature_id = it_per_id.feature_id;
                    worst_visual.host_frame = imu_i;
                    worst_visual.target_frame = imu_j;
                    worst_visual.has_lidar_depth = it_per_id.has_lidar_depth_prior;
                    worst_visual.norm = norm;
                    worst_visual.inv_depth = para_Feature[feature_index][0];
                    worst_visual.depth = it_per_id.estimated_depth;
                    if (it_per_id.lidar_depth_prior.valid)
                    {
                        worst_visual.prior_inv_depth = it_per_id.lidar_depth_prior.inv_depth;
                        worst_visual.prior_inv_depth_var = it_per_id.lidar_depth_prior.inv_depth_var;
                    }
                }
            }
        }

        if (depth_prior_cost.blocks == 0)
        {
            min_lidar_depth = std::numeric_limits<double>::quiet_NaN();
            max_lidar_depth = std::numeric_limits<double>::quiet_NaN();
            min_inv_depth_var = std::numeric_limits<double>::quiet_NaN();
            max_inv_depth_var = std::numeric_limits<double>::quiet_NaN();
        }

        std::printf("VIO OPT DEBUG cost %s: imu=%d/%.9e lio_pose=%d/%.9e lio_full=%d/%.9e depth_prior=%d/%.9e visual=%d raw=%.9e huber=%.9e\n",
                    stage,
                    imu_cost.blocks, 0.5 * imu_cost.sum_sq,
                    lio_pose_cost.blocks, 0.5 * lio_pose_cost.sum_sq,
                    lio_full_cost.blocks, 0.5 * lio_full_cost.sum_sq,
                    depth_prior_cost.blocks, 0.5 * depth_prior_cost.sum_sq,
                    visual_cost.blocks, 0.5 * visual_cost.sum_sq, visual_cost.robust_cost);
        std::printf("VIO OPT DEBUG lio_full_cost %s: pos=%.9e rot=%.9e vel=%.9e ba=%.9e bg=%.9e max_norm=%.6f\n",
                    stage,
                    0.5 * lio_full_pos_sq,
                    0.5 * lio_full_rot_sq,
                    0.5 * lio_full_vel_sq,
                    0.5 * lio_full_ba_sq,
                    0.5 * lio_full_bg_sq,
                    lio_full_cost.max_norm);
        std::printf("VIO OPT DEBUG imu_worst %s: factor=%d->%d dt=%.6f raw_norm=%.6f white_norm=%.6f raw_p=(% .6f % .6f % .6f) raw_r=(% .6f % .6f % .6f) raw_v=(% .6f % .6f % .6f) raw_ba=(% .6f % .6f % .6f) raw_bg=(% .6f % .6f % .6f)\n",
                    stage,
                    worst_imu.i,
                    worst_imu.j,
                    worst_imu.sum_dt,
                    worst_imu.raw_norm,
                    worst_imu.white_norm,
                    worst_imu.raw(0), worst_imu.raw(1), worst_imu.raw(2),
                    worst_imu.raw(3), worst_imu.raw(4), worst_imu.raw(5),
                    worst_imu.raw(6), worst_imu.raw(7), worst_imu.raw(8),
                    worst_imu.raw(9), worst_imu.raw(10), worst_imu.raw(11),
                    worst_imu.raw(12), worst_imu.raw(13), worst_imu.raw(14));
        std::printf("VIO OPT DEBUG imu_gravity %s: factor=%d->%d QiT_G=(% .6f % .6f % .6f) implied_acc_body=(% .6f % .6f % .6f) delta_v_avg=(% .6f % .6f % .6f) delta_p_avg=(% .6f % .6f % .6f)\n",
                    stage,
                    worst_imu.i,
                    worst_imu.j,
                    worst_imu.gravity_body_i.x(), worst_imu.gravity_body_i.y(), worst_imu.gravity_body_i.z(),
                    worst_imu.implied_acc_body.x(), worst_imu.implied_acc_body.y(), worst_imu.implied_acc_body.z(),
                    worst_imu.delta_v_avg.x(), worst_imu.delta_v_avg.y(), worst_imu.delta_v_avg.z(),
                    worst_imu.delta_p_avg.x(), worst_imu.delta_p_avg.y(), worst_imu.delta_p_avg.z());
        std::printf("VIO OPT DEBUG visual_worst %s: norm=%.6f feature=%d frames=%d->%d lidar=%d inv_depth=% .9e depth=% .6f prior_inv=% .9e prior_var=% .9e depth_range=[% .6f,% .6f] inv_var_range=[% .9e,% .9e]\n",
                    stage,
                    worst_visual.norm,
                    worst_visual.feature_id,
                    worst_visual.host_frame,
                    worst_visual.target_frame,
                    static_cast<int>(worst_visual.has_lidar_depth),
                    worst_visual.inv_depth,
                    worst_visual.depth,
                    worst_visual.prior_inv_depth,
                    worst_visual.prior_inv_depth_var,
                    min_lidar_depth,
                    max_lidar_depth,
                    min_inv_depth_var,
                    max_inv_depth_var);
    };
    print_lio_full_prior_para_delta("before_build");

    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = NULL;
    loss_function = new ceres::HuberLoss(1.0);
    //loss_function = new ceres::CauchyLoss(1.0 / FOCAL_LENGTH);
    //ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
    for (int i = 0; i < frame_count + 1; i++)
    {
        ceres::Manifold *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        if(USE_IMU)
            problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    if(!USE_IMU)
        problem.SetParameterBlockConstant(para_Pose[0]);

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::Manifold *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if ((ESTIMATE_EXTRINSIC && frame_count == WINDOW_SIZE && states_[0].vel_end.norm() > 0.2) || openExEstimation)
        {
            // CAKE_INFO("estimate extinsic param");
            openExEstimation = 1;
        }
        else
        {
            // CAKE_INFO("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
    }
    problem.AddParameterBlock(para_Td[0], 1);

    if (!ESTIMATE_TD || states_[0].vel_end.norm() < 0.2)
        problem.SetParameterBlockConstant(para_Td[0]);

    int marginalization_factor_count = 0;
    int imu_factor_count = 0;
    int lio_pose_prior_count = 0;
    int lio_full_prior_count = 0;
    int optimized_feature_count = 0;
    int lidar_depth_feature_count = 0;
    int lidar_depth_prior_count = 0;
    int visual_residual_count = 0;

    if (last_marginalization_info && last_marginalization_info->valid)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
        ++marginalization_factor_count;
    }
    if(USE_IMU)
    {
        for (int i = 0; i < frame_count; i++)
        {
            int j = i + 1;
            if (pre_integrations[j]->sum_dt > 10.0)
                continue;
            IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
            problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
            ++imu_factor_count;
        }
    }
    for (int i = 0; i < frame_count + 1; i++)
    {
        if (lio_pose_priors_[i].valid)
        {
            problem.AddResidualBlock(cake_slam::LioPosePriorFactor::Create(lio_pose_priors_[i]), NULL, para_Pose[i]);
            ++lio_pose_prior_count;
        }
        if (use_lio_full_prior_factor && lio_full_priors_[i].valid)
        {
            problem.AddResidualBlock(cake_slam::LioFullStatePriorFactor::Create(lio_full_priors_[i]), NULL,
                                     para_Pose[i], para_SpeedBias[i]);
            ++lio_full_prior_count;
        }
    }

    int f_m_cnt = 0;
    int feature_index = -1;
    int total_feature_track_count = 0;
    int total_lidar_depth_track_count = 0;
    int lidar_prior_valid_count = 0;
    int lidar_prior_invalid_count = 0;
    int max_feature_track_length = 0;
    int feature_len1 = 0;
    int feature_len2 = 0;
    int feature_len3 = 0;
    int feature_len4p = 0;
    int rejected_short_visual = 0;
    int rejected_short_lidar = 0;
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        ++total_feature_track_count;
        max_feature_track_length = std::max(max_feature_track_length, it_per_id.used_num);
        if (it_per_id.used_num <= 1)
            ++feature_len1;
        else if (it_per_id.used_num == 2)
            ++feature_len2;
        else if (it_per_id.used_num == 3)
            ++feature_len3;
        else
            ++feature_len4p;
        if (it_per_id.has_lidar_depth_prior)
        {
            ++total_lidar_depth_track_count;
            if (it_per_id.lidar_depth_prior.valid)
                ++lidar_prior_valid_count;
            else
                ++lidar_prior_invalid_count;
        }
        if (!it_per_id.isUsableForOptimization())
        {
            if (it_per_id.used_num < it_per_id.minObservationCountForOptimization())
            {
                if (it_per_id.has_lidar_depth_prior)
                    ++rejected_short_lidar;
                else
                    ++rejected_short_visual;
            }
            continue;
        }

        ++feature_index;
        ++optimized_feature_count;
        if (it_per_id.has_lidar_depth_prior)
            ++lidar_depth_feature_count;
        problem.AddParameterBlock(para_Feature[feature_index], SIZE_FEATURE);
        if (it_per_id.has_lidar_depth_prior && !LIDAR_INV_DEPTH_OPTIMIZE)
        {
            problem.SetParameterBlockConstant(para_Feature[feature_index]);
        }
        if (it_per_id.has_lidar_depth_prior && it_per_id.lidar_depth_prior.valid)
        {
            auto *prior_factor = new cake_slam::InverseDepthPriorFactor(
                it_per_id.lidar_depth_prior.inv_depth,
                it_per_id.lidar_depth_prior.inv_depth_var);
            problem.AddResidualBlock(prior_factor, NULL, para_Feature[feature_index]);
            ++lidar_depth_prior_count;
        }

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Eigen::Vector3d pts_j = it_per_frame.point;
                ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
                ++visual_residual_count;
            }

            // Stereo residuals are intentionally omitted in this ROS2 mono
            // implementation; STEREO must remain false unless the missing
            // two-camera factors are ported and registered in CMake.
            f_m_cnt++;
        }
    }
    last_optimization_feature_count_ = optimized_feature_count;
    last_optimization_lidar_feature_count_ = lidar_depth_feature_count;
    last_optimization_visual_residual_count_ = visual_residual_count;

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    //printf("prepare for ceres: %f \n", t_prepare.toc());

    static double last_light_opt_debug_stamp = -1.0;
    const bool weak_visual_constraints = optimized_feature_count < 10 || visual_residual_count < 20;
    const bool print_light_opt_debug =
        weak_visual_constraints &&
        (last_light_opt_debug_stamp < 0.0 || Headers[frame_count] - last_light_opt_debug_stamp >= 1.0);
    if (debug_lio_full_prior || print_light_opt_debug)
    {
        std::printf("VIO OPT DEBUG factors: solver_flag=%d frame_count=%d t0=%.6f tN=%.6f marg=%d imu=%d lio_pose=%d lio_full=%d features=%d lidar_features=%d depth_priors=%d visual_residuals=%d visual_measurements=%d residual_blocks=%d parameter_blocks=%d lidar_inv_depth_opt=%d\n",
                    solver_flag,
                    frame_count,
                    Headers[0],
                    Headers[frame_count],
                    marginalization_factor_count,
                    imu_factor_count,
                    lio_pose_prior_count,
                    lio_full_prior_count,
                    optimized_feature_count,
                    lidar_depth_feature_count,
                    lidar_depth_prior_count,
                    visual_residual_count,
                    f_m_cnt,
                    problem.NumResidualBlocks(),
                    problem.NumParameterBlocks(),
                    LIDAR_INV_DEPTH_OPTIMIZE);
        std::printf("VIO OPT DEBUG feature_health: total_tracks=%d lidar_tracks=%d max_track_len=%d usable=%d usable_lidar=%d min_lidar_obs=%d full_prior_factor=%d expensive_debug=%d\n",
                    total_feature_track_count,
                    total_lidar_depth_track_count,
                    max_feature_track_length,
                    optimized_feature_count,
                    lidar_depth_feature_count,
                    2,
                    static_cast<int>(use_lio_full_prior_factor),
                    static_cast<int>(debug_lio_full_prior));
        std::printf("VIO OPT DEBUG reject_reason: len1=%d len2=%d len3=%d len4p=%d short_visual=%d short_lidar=%d lidar_prior_valid=%d lidar_prior_invalid=%d usable=%d visual_residuals=%d weak=%d\n",
                    feature_len1,
                    feature_len2,
                    feature_len3,
                    feature_len4p,
                    rejected_short_visual,
                    rejected_short_lidar,
                    lidar_prior_valid_count,
                    lidar_prior_invalid_count,
                    optimized_feature_count,
                    visual_residual_count,
                    static_cast<int>(weak_visual_constraints));
        last_light_opt_debug_stamp = Headers[frame_count];
        if (debug_lio_full_prior)
        {
            print_lio_full_prior_para_delta("before_solve");
            print_debug_factor_costs("before_solve");
        }
    }

    ceres::Solver::Options options;

    if (USE_GPU_CERES)
        // std::cout << "1" << endl;
        options.dense_linear_algebra_library_type = ceres::CUDA;
    else
        // std::cout << "2" << endl;
        options.linear_solver_type = ceres::DENSE_SCHUR;

    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;


    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    //printf("solver costs: %f \n", t_solver.toc());
    if (debug_lio_full_prior)
    {
        std::printf("VIO OPT DEBUG solve: initial_cost=%.9e final_cost=%.9e fixed_cost=%.9e iterations=%d usable=%d term=%d message=%s\n",
                    summary.initial_cost,
                    summary.final_cost,
                    summary.fixed_cost,
                    static_cast<int>(summary.iterations.size()),
                    static_cast<int>(summary.IsSolutionUsable()),
                    static_cast<int>(summary.termination_type),
                    summary.message.c_str());
        print_lio_full_prior_para_delta("after_solve");
        print_debug_factor_costs("after_solve");
    }

    double2vector();
    print_lio_full_prior_state_delta("after_double2vector");
    //printf("frame_count: %d \n", frame_count);

    if(frame_count < WINDOW_SIZE)
        return;

    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        if (last_marginalization_info && last_marginalization_info->valid)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        if(USE_IMU)
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }
        if (lio_pose_priors_[0].valid)
        {
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
                cake_slam::LioPosePriorFactor::Create(lio_pose_priors_[0]), NULL,
                vector<double *>{para_Pose[0]},
                vector<int>{0});
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }
        if (use_lio_full_prior_factor && lio_full_priors_[0].valid)
        {
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
                cake_slam::LioFullStatePriorFactor::Create(lio_full_priors_[0]), NULL,
                vector<double *>{para_Pose[0], para_SpeedBias[0]},
                vector<int>{0, 1});
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (!it_per_id.isUsableForOptimization())
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point;
                if (it_per_id.has_lidar_depth_prior && it_per_id.lidar_depth_prior.valid)
                {
                    auto *prior_factor = new cake_slam::InverseDepthPriorFactor(
                        it_per_id.lidar_depth_prior.inv_depth,
                        it_per_id.lidar_depth_prior.inv_depth_var);
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
                        prior_factor, NULL,
                        vector<double *>{para_Feature[feature_index]},
                        vector<int>{0});
                    marginalization_info->addResidualBlockInfo(residual_block_info);
                }

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if(imu_i != imu_j)
                    {
                        Eigen::Vector3d pts_j = it_per_frame.point;
                        ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    // Stereo marginalization follows the same policy as the
                    // active solve path: disabled until the two-camera factor
                    // port exists in this ROS2 codebase.
                }
            }
        }

        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());

        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            if(USE_IMU)
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

        addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
        {
            delete last_marginalization_info;
            last_marginalization_info = nullptr;
        }
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;

    }
    else
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info && last_marginalization_info->valid)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    assert(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
            if (lio_pose_priors_[WINDOW_SIZE - 1].valid)
            {
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
                    cake_slam::LioPosePriorFactor::Create(lio_pose_priors_[WINDOW_SIZE - 1]), NULL,
                    vector<double *>{para_Pose[WINDOW_SIZE - 1]},
                    vector<int>{0});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
            if (use_lio_full_prior_factor && lio_full_priors_[WINDOW_SIZE - 1].valid)
            {
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
                    cake_slam::LioFullStatePriorFactor::Create(lio_full_priors_[WINDOW_SIZE - 1]), NULL,
                    vector<double *>{para_Pose[WINDOW_SIZE - 1], para_SpeedBias[WINDOW_SIZE - 1]},
                    vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());

            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];


            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
            {
                delete last_marginalization_info;
                last_marginalization_info = nullptr;
            }
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;

        }
    }
    //printf("whole marginalization costs: %f \n", t_whole_marginalization.toc());
    //printf("whole time for ceres: %f \n", t_whole.toc());
}

// 根据 marginalization_flag 选择边缘化策略。
void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0];
        back_R0 = states_[0].rot_end;
        back_P0 = states_[0].pos_end;
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Headers[i] = Headers[i + 1];
                lio_pose_priors_[i] = lio_pose_priors_[i + 1];
                lio_full_priors_[i] = lio_full_priors_[i + 1];
                states_[i].rot_end.swap(states_[i + 1].rot_end);
                states_[i].pos_end.swap(states_[i + 1].pos_end);
                if(USE_IMU)
                {
                    std::swap(pre_integrations[i], pre_integrations[i + 1]);

                    dt_buf[i].swap(dt_buf[i + 1]);
                    linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                    angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                    states_[i].vel_end.swap(states_[i + 1].vel_end);
                    states_[i].bias_a.swap(states_[i + 1].bias_a);
                    states_[i].bias_g.swap(states_[i + 1].bias_g);
                }
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            lio_pose_priors_[WINDOW_SIZE] = lio_pose_priors_[WINDOW_SIZE - 1];
            lio_full_priors_[WINDOW_SIZE] = lio_full_priors_[WINDOW_SIZE - 1];
            states_[WINDOW_SIZE].pos_end = states_[WINDOW_SIZE - 1].pos_end;
            states_[WINDOW_SIZE].rot_end = states_[WINDOW_SIZE - 1].rot_end;

            if(USE_IMU)
            {
                states_[WINDOW_SIZE].vel_end = states_[WINDOW_SIZE - 1].vel_end;
                states_[WINDOW_SIZE].bias_a = states_[WINDOW_SIZE - 1].bias_a;
                states_[WINDOW_SIZE].bias_g = states_[WINDOW_SIZE - 1].bias_g;

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = nullptr;
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, states_[WINDOW_SIZE].bias_a, states_[WINDOW_SIZE].bias_g};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }

            if (true || solver_flag == INITIAL)
            {
                std::map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                it_0->second.pre_integration = nullptr;
                all_image_frame.erase(all_image_frame.begin(), it_0);
            }
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            Headers[frame_count - 1] = Headers[frame_count];
            lio_pose_priors_[frame_count - 1] = lio_pose_priors_[frame_count];
            lio_full_priors_[frame_count - 1] = lio_full_priors_[frame_count];
            states_[frame_count - 1].pos_end = states_[frame_count].pos_end;
            states_[frame_count - 1].rot_end = states_[frame_count].rot_end;

            if(USE_IMU)
            {
                for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
                {
                    double tmp_dt = dt_buf[frame_count][i];
                    Eigen::Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                    Eigen::Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                    pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                    dt_buf[frame_count - 1].push_back(tmp_dt);
                    linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                    angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
                }

                states_[frame_count - 1].vel_end = states_[frame_count].vel_end;
                states_[frame_count - 1].bias_a = states_[frame_count].bias_a;
                states_[frame_count - 1].bias_g = states_[frame_count].bias_g;

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = nullptr;
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, states_[WINDOW_SIZE].bias_a, states_[WINDOW_SIZE].bias_g};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
            slideWindowNew();
        }
    }
}

// 边缘化最新次新帧的相关观测。
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}

// 边缘化最老帧，并同步维护特征管理器和预积分缓存。
void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Eigen::Matrix3d R0, R1;
        Eigen::Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = states_[0].rot_end * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = states_[0].pos_end + states_[0].rot_end * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}


// 导出当前帧位姿。
void Estimator::getPoseInWorldFrame(Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = states_[frame_count].rot_end;
    T.block<3, 1>(0, 3) = states_[frame_count].pos_end;
}

// 导出滑窗中指定索引帧的位姿。
void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = states_[index].rot_end;
    T.block<3, 1>(0, 3) = states_[index].pos_end;
}

const StatesGroup &Estimator::getLatestState() const
{
    return states_[frame_count];
}

const cv::Mat &Estimator::getUndistortedValidMask() const
{
    return featureTracker.validMask();
}

cv::Mat Estimator::getFeatureDebugImage() const
{
    return featureTracker.getFeatureDebugImage();
}

int Estimator::getLastTrackedFeatureCount() const
{
    return featureTracker.getLastFeatureCount();
}

int Estimator::getLastDepthFeatureCount() const
{
    return featureTracker.getLastDepthFeatureCount();
}

int Estimator::getLastPrevTrackCount() const
{
    return featureTracker.getLastPrevTrackCount();
}

int Estimator::getLastTrackedAfterFlowCount() const
{
    return featureTracker.getLastTrackedAfterFlowCount();
}

int Estimator::getLastPrevLidarTrackCount() const
{
    return featureTracker.getLastPrevLidarTrackCount();
}

int Estimator::getLastTrackedLidarCount() const
{
    return featureTracker.getLastTrackedLidarCount();
}

int Estimator::getLastRejectedByLioPriorCount() const
{
    return featureTracker.getLastRejectedByLioPriorCount();
}

int Estimator::getLastAddedLidarCount() const
{
    return featureTracker.getLastAddedLidarCount();
}

int Estimator::getLastAddedVisualCount() const
{
    return featureTracker.getLastAddedVisualCount();
}

int Estimator::getLastPendingLidarCandidateCount() const
{
    return featureTracker.getLastPendingLidarCandidateCount();
}

int Estimator::getLastOptimizationFeatureCount() const
{
    return last_optimization_feature_count_;
}

int Estimator::getLastOptimizationLidarFeatureCount() const
{
    return last_optimization_lidar_feature_count_;
}

int Estimator::getLastOptimizationVisualResidualCount() const
{
    return last_optimization_visual_residual_count_;
}

bool Estimator::buildVinsFallbackInitialLandmarksDeadCode(std::map<int, Eigen::Vector3d> &sfm_tracked_points)
{
    // Dead-code fallback hook: keep the original VINS-Fusion GlobalSFM landmark
    // bootstrap available for a future LiDAR-degeneration policy, without
    // changing the active LIO-prior initialization path.
    sfm_tracked_points.clear();
    if (frame_count < WINDOW_SIZE)
        return false;

    std::vector<Eigen::Quaterniond> Q(frame_count + 1);
    std::vector<Eigen::Vector3d> T(frame_count + 1);
    std::vector<SFMFeature> sfm_f;
    sfm_f.reserve(f_manager.feature.size());

    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Eigen::Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(
                make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    }

    Eigen::Matrix3d relative_R;
    Eigen::Vector3d relative_T;
    int l = 0;
    if (!relativePose(relative_R, relative_T, l))
        return false;

    GlobalSFM sfm;
    if (!sfm.construct(frame_count + 1, Q.data(), T.data(), l,
                       relative_R, relative_T, sfm_f, sfm_tracked_points))
    {
        return false;
    }
    return !sfm_tracked_points.empty();
}

// 基于当前滑窗中的运动趋势，为前端预测下一帧特征位置。
void Estimator::predictPtsInNextFrame()
{
    //printf("predict pts in next frame\n");
    if(frame_count < 2)
        return;
    // predict next pose. Assume constant velocity motion
    Eigen::Matrix4d curT, prevT, nextT;
    getPoseInWorldFrame(curT);
    getPoseInWorldFrame(frame_count - 1, prevT);
    nextT = curT * (prevT.inverse() * curT);
    std::map<int, Eigen::Vector3d> predictPts;

    for (auto &it_per_id : f_manager.feature)
    {
        if(it_per_id.estimated_depth > 0)
        {
            int firstIndex = it_per_id.start_frame;
            int lastIndex = it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1;
            //printf("cur frame index  %d last frame index %d\n", frame_count, lastIndex);
            if((int)it_per_id.feature_per_frame.size() >= 2 && lastIndex == frame_count)
            {
                double depth = it_per_id.estimated_depth;
                Eigen::Vector3d pts_j = ric[0] * (depth * it_per_id.feature_per_frame[0].point) + tic[0];
                Eigen::Vector3d pts_w = states_[firstIndex].rot_end * pts_j + states_[firstIndex].pos_end;
                Eigen::Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3));
                Eigen::Vector3d pts_cam = ric[0].transpose() * (pts_local - tic[0]);
                int ptsIndex = it_per_id.feature_id;
                predictPts[ptsIndex] = pts_cam;
            }
        }
    }
    featureTracker.setPrediction(predictPts);
    //printf("estimator output %d predict pts\n",(int)predictPts.size());
}

// 计算一个双视图观测的重投影误差大小。
double Estimator::reprojectionError(Eigen::Matrix3d &Ri, Eigen::Vector3d &Pi, Eigen::Matrix3d &rici, Eigen::Vector3d &tici,
                                 Eigen::Matrix3d &Rj, Eigen::Vector3d &Pj, Eigen::Matrix3d &ricj, Eigen::Vector3d &ticj,
                                 double depth, Eigen::Vector3d &uvi, Eigen::Vector3d &uvj)
{
    Eigen::Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Eigen::Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Eigen::Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}

// 根据重投影误差从特征集合中筛出外点。
void Estimator::outliersRejection(set<int> &removeIndex)
{
    //return;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        double err = 0;
        int errCnt = 0;
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!it_per_id.isUsableForOptimization())
            continue;
        feature_index ++;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        double depth = it_per_id.estimated_depth;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Eigen::Vector3d pts_j = it_per_frame.point;
                double tmp_error = reprojectionError(states_[imu_i].rot_end, states_[imu_i].pos_end, ric[0], tic[0],
                                                    states_[imu_j].rot_end, states_[imu_j].pos_end, ric[0], tic[0],
                                                    depth, pts_i, pts_j);
                err += tmp_error;
                errCnt++;
                //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
            }
            // need to rewrite projecton factor.........
            if(STEREO && it_per_frame.is_stereo)
            {

                Eigen::Vector3d pts_j_right = it_per_frame.pointRight;
                if(imu_i != imu_j)
                {
                    double tmp_error = reprojectionError(states_[imu_i].rot_end, states_[imu_i].pos_end, ric[0], tic[0],
                                                        states_[imu_j].rot_end, states_[imu_j].pos_end, ric[1], tic[1],
                                                        depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
                else
                {
                    double tmp_error = reprojectionError(states_[imu_i].rot_end, states_[imu_i].pos_end, ric[0], tic[0],
                                                        states_[imu_j].rot_end, states_[imu_j].pos_end, ric[1], tic[1],
                                                        depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
            }
        }
        double ave_err = err / errCnt;
        if(ave_err * FOCAL_LENGTH > 3)
            removeIndex.insert(it_per_id.feature_id);

    }
}

// 使用最新 IMU 做高频前向预测，主要服务于实时输出而不是滑窗优化。
void Estimator::fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity)
{
    double dt = t - latest_time;
    latest_time = t;
    Eigen::Vector3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - g;
    Eigen::Vector3d un_gyr = 0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;
    latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);
    Eigen::Vector3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - g;
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    latest_P = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
    latest_V = latest_V + dt * un_acc;
    latest_acc_0 = linear_acceleration;
    latest_gyr_0 = angular_velocity;
}

// 把当前滑窗尾部状态同步到 latest_* 高频传播缓存。
void Estimator::updateLatestStates()
{
    mPropagate.lock();
    latest_time = Headers[frame_count] + td;
    latest_P = states_[frame_count].pos_end;
    latest_Q = states_[frame_count].rot_end;
    latest_V = states_[frame_count].vel_end;
    latest_Ba = states_[frame_count].bias_a;
    latest_Bg = states_[frame_count].bias_g;
    latest_acc_0 = acc_0;
    latest_gyr_0 = gyr_0;
    mBuf.lock();
    queue<ImuSample> tmp_imuBuf = imuBuf;
    mBuf.unlock();
    while(!tmp_imuBuf.empty())
    {
        double t = tmp_imuBuf.front().stamp;
        Eigen::Vector3d acc = tmp_imuBuf.front().acc;
        Eigen::Vector3d gyr = tmp_imuBuf.front().gyr;
        fastPredictIMU(t, acc, gyr);
        tmp_imuBuf.pop();
    }
    mPropagate.unlock();
}
