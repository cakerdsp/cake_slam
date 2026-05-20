/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

// 模块功能：视觉前端特征跟踪实现，
// 完成光流跟踪、特征筛选与 LiDAR 先验注入。

#include "feature_tracker.h"

#include <algorithm>
#include <cmath>

// 判断点是否落在图像有效区域内。
bool FeatureTracker::inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    if (!(BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE &&
          BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE))
        return false;
    return fisheye_mask.empty() ||
           (fisheye_mask.size() == cv::Size(col, row) && fisheye_mask.at<uchar>(img_y, img_x) != 0);
}

// 全局辅助函数：计算两像素点欧氏距离。
double distance(cv::Point2f pt1, cv::Point2f pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

// 按 status 原地压缩点向量。
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

// 按 status 原地压缩整型向量。
void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

// 前端跟踪器初始状态。
FeatureTracker::FeatureTracker()
{
    stereo_cam = 0;
    n_id = 0;
    hasPrediction = false;
    gate_R_I_C.setIdentity();
    gate_t_I_C.setZero();
}

void FeatureTracker::setLidarDepthCandidates(const vector<cake_slam::LidarVisualCandidate> &candidates)
{
    pending_lidar_candidates = candidates;
}

void FeatureTracker::setLioPriorGate(const cake_slam::LioPosePrior &prior,
                                     const Eigen::Matrix3d &R_I_C,
                                     const Eigen::Vector3d &t_I_C)
{
    lio_prior_gate = prior;
    gate_R_I_C = R_I_C;
    gate_t_I_C = t_I_C;
}

std::unordered_map<int, cake_slam::LidarDepthPrior> FeatureTracker::takeLidarDepthPriors()
{
    std::unordered_map<int, cake_slam::LidarDepthPrior> out;
    out.swap(current_lidar_priors);
    return out;
}

void FeatureTracker::setMask()
{
    // 用“长轨迹优先”策略构造新的检测掩膜：
    // 连续跟踪时间更长的点优先保留，附近区域则不再重复检测新点。
    if (!fisheye_mask.empty() && fisheye_mask.size() == cv::Size(col, row))
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < cur_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(cur_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    cur_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            cur_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

// 把新检测到的点加入当前特征集合，并分配新的全局 id。
void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        cur_pts.push_back(p);
        ids.push_back(n_id++);
        track_cnt.push_back(1);
    }
}

// 类内版本的距离函数，和上面的自由函数语义相同。
double FeatureTracker::distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

std::map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> FeatureTracker::trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1)
{
    // 这是视觉前端的主流程：
    // 1. 用 LK 光流跟踪上一帧特征；
    // 2. 做边界检查、反向校验和 LIO 先验重投影门控；
    // 3. 补充新特征；
    // 4. 输出统一的“id -> 观测”结构。
    TicToc t_r;
    last_timing = FeatureTrackerTiming();
    cur_time = _cur_time;
    cur_img = _img;
    row = cur_img.rows;
    col = cur_img.cols;
    cv::Mat rightImg = _img1;
    last_timing.rows = row;
    last_timing.cols = col;
    last_timing.type = cur_img.type();
    last_timing.channels = cur_img.channels();
    last_timing.prev_tracks = static_cast<int>(prev_pts.size());
    last_timing.pending_lidar = static_cast<int>(pending_lidar_candidates.size());
    last_timing.flow_back = FLOW_BACK;
    last_timing.has_prediction = hasPrediction ? 1 : 0;
    current_lidar_priors.clear();
    auto countStatus = [](const vector<uchar> &status) {
        int count = 0;
        for (const auto s : status)
        {
            if (s)
                ++count;
        }
        return count;
    };
    int survival_lk_forward = 0;
    int survival_lk_backward = 0;
    int survival_lio_gate = 0;
    int survival_border = 0;
    last_prev_track_count = static_cast<int>(prev_pts.size());
    last_tracked_after_flow_count = 0;
    last_prev_lidar_track_count = 0;
    last_tracked_lidar_count = 0;
    last_rejected_by_lio_prior_count = 0;
    last_added_lidar_count = 0;
    last_added_visual_count = 0;
    last_pending_lidar_candidate_count = static_cast<int>(pending_lidar_candidates.size());
    for (const int id : ids)
    {
        if (active_lidar_priors.find(id) != active_lidar_priors.end())
            last_prev_lidar_track_count++;
    }
    /*
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(cur_img, cur_img);
        if(!rightImg.empty())
            clahe->apply(rightImg, rightImg);
    }
    */
    cur_pts.clear();

    // 若存在上一帧，则尝试把上一帧特征跟踪到当前帧。
    if (prev_pts.size() > 0)
    {
        vector<uchar> status;
        if(!USE_GPU_ACC_FLOW)
        {
            TicToc t_lk_forward;
            
            vector<float> err;
            // 有预测时，把预测位置作为光流初值，可减少快速运动下的跟踪失败。
            if(hasPrediction)
            {
                cur_pts = predict_pts;
                cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 1, 
                cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
                
                int succ_num = 0;
                for (size_t i = 0; i < status.size(); i++)
                {
                    if (status[i])
                        succ_num++;
                }
                if (succ_num < 10)
                cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);
            }
            else
            {
                cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);
            }
            survival_lk_forward = countStatus(status);
            last_timing.lk_forward_ms = t_lk_forward.toc();
            // reverse check
            if(FLOW_BACK)
            {
                TicToc t_lk_backward;
                vector<uchar> reverse_status;
                vector<cv::Point2f> reverse_pts = prev_pts;
                cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 1, 
                cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
                //cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 3); 
                for(size_t i = 0; i < status.size(); i++)
                {
                    if(status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 0.5)
                    {
                        status[i] = 1;
                    }
                    else
                        status[i] = 0;
                }
                last_timing.lk_backward_ms = t_lk_backward.toc();
            }
            survival_lk_backward = FLOW_BACK ? countStatus(status) : survival_lk_forward;
            TicToc t_lio_gate;
            rejectWithLioPrior(status);
            survival_lio_gate = countStatus(status);
            last_timing.lio_gate_ms = t_lio_gate.toc();
            // printf("temporal optical flow costs: %fms\n", t_o.toc());
        }
#ifdef GPU_MODE
        else
        {
            TicToc t_og;
            cv::cuda::GpuMat prev_gpu_img(prev_img);
            cv::cuda::GpuMat cur_gpu_img(cur_img);
            cv::cuda::GpuMat prev_gpu_pts(prev_pts);
            cv::cuda::GpuMat cur_gpu_pts(cur_pts);
            cv::cuda::GpuMat gpu_status;
            if(hasPrediction)
            {
                cur_gpu_pts = cv::cuda::GpuMat(predict_pts);
                cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                cv::Size(21, 21), 1, 30, true);
                d_pyrLK_sparse->calc(prev_gpu_img, cur_gpu_img, prev_gpu_pts, cur_gpu_pts, gpu_status);
                
                vector<cv::Point2f> tmp_cur_pts(cur_gpu_pts.cols);
                cur_gpu_pts.download(tmp_cur_pts);
                cur_pts = tmp_cur_pts;

                vector<uchar> tmp_status(gpu_status.cols);
                gpu_status.download(tmp_status);
                status = tmp_status;

                int succ_num = 0;
                for (size_t i = 0; i < tmp_status.size(); i++)
                {
                    if (tmp_status[i])
                        succ_num++;
                }
                if (succ_num < 10)
                {
                    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                    cv::Size(21, 21), 3, 30, false);
                    d_pyrLK_sparse->calc(prev_gpu_img, cur_gpu_img, prev_gpu_pts, cur_gpu_pts, gpu_status);

                    vector<cv::Point2f> tmp1_cur_pts(cur_gpu_pts.cols);
                    cur_gpu_pts.download(tmp1_cur_pts);
                    cur_pts = tmp1_cur_pts;

                    vector<uchar> tmp1_status(gpu_status.cols);
                    gpu_status.download(tmp1_status);
                    status = tmp1_status;
                }
            }
            else
            {
                cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                cv::Size(21, 21), 3, 30, false);
                d_pyrLK_sparse->calc(prev_gpu_img, cur_gpu_img, prev_gpu_pts, cur_gpu_pts, gpu_status);

                vector<cv::Point2f> tmp1_cur_pts(cur_gpu_pts.cols);
                cur_gpu_pts.download(tmp1_cur_pts);
                cur_pts = tmp1_cur_pts;

                vector<uchar> tmp1_status(gpu_status.cols);
                gpu_status.download(tmp1_status);
                status = tmp1_status;
            }
            survival_lk_forward = countStatus(status);
            if(FLOW_BACK)
            {
                cv::cuda::GpuMat reverse_gpu_status;
                cv::cuda::GpuMat reverse_gpu_pts = prev_gpu_pts;
                cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                cv::Size(21, 21), 1, 30, true);
                d_pyrLK_sparse->calc(cur_gpu_img, prev_gpu_img, cur_gpu_pts, reverse_gpu_pts, reverse_gpu_status);

                vector<cv::Point2f> reverse_pts(reverse_gpu_pts.cols);
                reverse_gpu_pts.download(reverse_pts);

                vector<uchar> reverse_status(reverse_gpu_status.cols);
                reverse_gpu_status.download(reverse_status);

                for(size_t i = 0; i < status.size(); i++)
                {
                    if(status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 0.5)
                    {
                        status[i] = 1;
                    }
                    else
                        status[i] = 0;
                }
            }
            survival_lk_backward = FLOW_BACK ? countStatus(status) : survival_lk_forward;
            survival_lio_gate = survival_lk_backward;
            // printf("gpu temporal optical flow costs: %f ms\n",t_og.toc());
        }
#endif
    
        TicToc t_reduce;
        for (int i = 0; i < int(cur_pts.size()); i++)
            if (status[i] && !inBorder(cur_pts[i]))
                status[i] = 0;
        survival_border = countStatus(status);
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        pruneLidarTracks();
        last_timing.reduce_ms = t_reduce.toc();
        last_tracked_after_flow_count = static_cast<int>(cur_pts.size());
        last_timing.after_flow_tracks = last_tracked_after_flow_count;
        for (const int id : ids)
        {
            if (active_lidar_priors.find(id) != active_lidar_priors.end())
                last_tracked_lidar_count++;
        }
        // ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
        
        //printf("track cnt %d\n", (int)ids.size());
    }

    for (auto &n : track_cnt)
        n++;

    if (1)
    {
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();
        last_timing.set_mask_ms = t_m.toc();
        // ROS_DEBUG("set mask costs %fms", t_m.toc());
        // printf("set mask costs %fms\n", t_m.toc());
        ROS_DEBUG("detect feature begins");
        
        int n_max_cnt = MAX_CNT - static_cast<int>(cur_pts.size());
        if (LIDAR_DEPTH_ENABLE && LIDAR_PRIOR_FEATURE_ENABLE &&
            n_max_cnt > 0 && !pending_lidar_candidates.empty())
        {
            TicToc t_lidar_add;
            const int added_lidar = addLidarCandidatePoints(n_max_cnt);
            last_timing.add_lidar_ms = t_lidar_add.toc();
            last_added_lidar_count = added_lidar;
            last_timing.added_lidar = added_lidar;
            n_max_cnt = MAX_CNT - static_cast<int>(cur_pts.size());
            ROS_DEBUG("add LiDAR visual candidates: %d", added_lidar);
        }
        last_timing.requested_visual = std::max(0, n_max_cnt);
        if(!USE_GPU)
        {
            if (n_max_cnt > 0)
            {
                TicToc t_t;
                if(mask.empty())
                    cout << "mask is empty " << endl;
                if (mask.type() != CV_8UC1)
                    cout << "mask type wrong " << endl;
                cv::goodFeaturesToTrack(cur_img, n_pts, MAX_CNT - cur_pts.size(), 0.01, MIN_DIST, mask);
                last_timing.good_features_ms = t_t.toc();
                // printf("good feature to track costs: %fms\n", t_t.toc());
                TicToc t_console;
                std::cout << "n_pts size: "<< n_pts.size()<<std::endl;
                last_timing.console_ms = t_console.toc();
            }
            else
                n_pts.clear();
            // sum_n += n_pts.size();
            // printf("total point from non-gpu: %d\n",sum_n);
        }
#ifdef GPU_MODE
        // ROS_DEBUG("detect feature costs: %fms", t_t.toc());
        // printf("good feature to track costs: %fms\n", t_t.toc());
        else
        {
            if (n_max_cnt > 0)
            {
                if(mask.empty())
                    cout << "mask is empty " << endl;
                if (mask.type() != CV_8UC1)
                    cout << "mask type wrong " << endl;
                TicToc t_g;
                cv::cuda::GpuMat cur_gpu_img(cur_img);
                cv::cuda::GpuMat d_prevPts;
                TicToc t_gg;
                cv::cuda::GpuMat gpu_mask(mask);
                // printf("gpumat cost: %fms\n",t_gg.toc());
                cv::Ptr<cv::cuda::CornersDetector> detector = cv::cuda::createGoodFeaturesToTrackDetector(cur_gpu_img.type(), MAX_CNT - cur_pts.size(), 0.01, MIN_DIST);
                // cout << "new gpu points: "<< MAX_CNT - cur_pts.size()<<endl;
                detector->detect(cur_gpu_img, d_prevPts, gpu_mask);
                // std::cout << "d_prevPts size: "<< d_prevPts.size()<<std::endl;
                if(!d_prevPts.empty())
                    n_pts = cv::Mat_<cv::Point2f>(cv::Mat(d_prevPts));
                else
                    n_pts.clear();
                // sum_n += n_pts.size();
                // printf("total point from gpu: %d\n",sum_n);
                // printf("gpu good feature to track cost: %fms\n", t_g.toc());
            }
            else 
                n_pts.clear();
        }
#endif

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        last_added_visual_count = static_cast<int>(n_pts.size());
        addPoints();
        last_timing.added_visual = last_added_visual_count;
        last_timing.add_points_ms = t_a.toc();
        // ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
        // printf("selectFeature costs: %fms\n", t_a.toc());
    }

    TicToc t_undistort;
    cur_un_pts = undistortedPts(cur_pts, 0);
    last_timing.undistort_ms = t_undistort.toc();
    TicToc t_velocity;
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);
    last_timing.velocity_ms = t_velocity.toc();

    if(!_img1.empty() && stereo_cam)
    {
        TicToc t_stereo;
        ids_right.clear();
        cur_right_pts.clear();
        cur_un_right_pts.clear();
        right_pts_velocity.clear();
        cur_un_right_pts_map.clear();
        if(!cur_pts.empty())
        {
            //printf("stereo image; track feature on right image\n");
            
            vector<cv::Point2f> reverseLeftPts;
            vector<uchar> status, statusRightLeft;
            if(!USE_GPU_ACC_FLOW)
            {
                TicToc t_check;
                vector<float> err;
                // cur left ---- cur right
                cv::calcOpticalFlowPyrLK(cur_img, rightImg, cur_pts, cur_right_pts, status, err, cv::Size(21, 21), 3);
                // reverse check cur right ---- cur left
                if(FLOW_BACK)
                {
                    cv::calcOpticalFlowPyrLK(rightImg, cur_img, cur_right_pts, reverseLeftPts, statusRightLeft, err, cv::Size(21, 21), 3);
                    for(size_t i = 0; i < status.size(); i++)
                    {
                        if(status[i] && statusRightLeft[i] && inBorder(cur_right_pts[i]) && distance(cur_pts[i], reverseLeftPts[i]) <= 0.5)
                            status[i] = 1;
                        else
                            status[i] = 0;
                    }
                }
                // printf("left right optical flow cost %fms\n",t_check.toc());
            }
#ifdef GPU_MODE
            else
            {
                TicToc t_og1;
                cv::cuda::GpuMat cur_gpu_img(cur_img);
                cv::cuda::GpuMat right_gpu_Img(rightImg);
                cv::cuda::GpuMat cur_gpu_pts(cur_pts);
                cv::cuda::GpuMat cur_right_gpu_pts;
                cv::cuda::GpuMat gpu_status;
                cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                cv::Size(21, 21), 3, 30, false);
                d_pyrLK_sparse->calc(cur_gpu_img, right_gpu_Img, cur_gpu_pts, cur_right_gpu_pts, gpu_status);

                vector<cv::Point2f> tmp_cur_right_pts(cur_right_gpu_pts.cols);
                cur_right_gpu_pts.download(tmp_cur_right_pts);
                cur_right_pts = tmp_cur_right_pts;

                vector<uchar> tmp_status(gpu_status.cols);
                gpu_status.download(tmp_status);
                status = tmp_status;

                if(FLOW_BACK)
                {   
                    cv::cuda::GpuMat reverseLeft_gpu_Pts;
                    cv::cuda::GpuMat status_gpu_RightLeft;
                    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
                    cv::Size(21, 21), 3, 30, false);
                    d_pyrLK_sparse->calc(right_gpu_Img, cur_gpu_img, cur_right_gpu_pts, reverseLeft_gpu_Pts, status_gpu_RightLeft);

                    vector<cv::Point2f> tmp_reverseLeft_Pts(reverseLeft_gpu_Pts.cols);
                    reverseLeft_gpu_Pts.download(tmp_reverseLeft_Pts);
                    reverseLeftPts = tmp_reverseLeft_Pts;

                    vector<uchar> tmp1_status(status_gpu_RightLeft.cols);
                    status_gpu_RightLeft.download(tmp1_status);
                    statusRightLeft = tmp1_status;
                    for(size_t i = 0; i < status.size(); i++)
                    {
                        if(status[i] && statusRightLeft[i] && inBorder(cur_right_pts[i]) && distance(cur_pts[i], reverseLeftPts[i]) <= 0.5)
                            status[i] = 1;
                        else
                            status[i] = 0;
                    }
                }
                // printf("gpu left right optical flow cost %fms\n",t_og1.toc());
            }
#endif
            ids_right = ids;
            reduceVector(cur_right_pts, status);
            reduceVector(ids_right, status);
            // only keep left-right pts
            /*
            reduceVector(cur_pts, status);
            reduceVector(ids, status);
            reduceVector(track_cnt, status);
            reduceVector(cur_un_pts, status);
            reduceVector(pts_velocity, status);
            */
            cur_un_right_pts = undistortedPts(cur_right_pts, 1);
            right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);
            
        }
        prev_un_right_pts_map = cur_un_right_pts_map;
        last_timing.stereo_ms = t_stereo.toc();
    }
    if(SHOW_TRACK)
    {
        TicToc t_draw_track;
        drawTrack(cur_img, rightImg, ids, cur_pts, cur_right_pts, prevLeftPtsMap);
        last_timing.draw_track_ms = t_draw_track.toc();
    }

    TicToc t_debug_draw;
    drawFeatureDebugImage(prevLeftPtsMap);
    last_timing.debug_draw_ms = t_debug_draw.toc();

    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    prev_un_pts_map = cur_un_pts_map;
    prev_time = cur_time;
    hasPrediction = false;
    pending_lidar_candidates.clear();

    prevLeftPtsMap.clear();
    for(size_t i = 0; i < cur_pts.size(); i++)
        prevLeftPtsMap[ids[i]] = cur_pts[i];

    TicToc t_pack;
    std::map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (size_t i = 0; i < ids.size(); i++)
    {
        int feature_id = ids[i];
        double x, y ,z;
        x = cur_un_pts[i].x;
        y = cur_un_pts[i].y;
        z = 1;
        double p_u, p_v;
        p_u = cur_pts[i].x;
        p_v = cur_pts[i].y;
        int camera_id = 0;
        double velocity_x, velocity_y;
        velocity_x = pts_velocity[i].x;
        velocity_y = pts_velocity[i].y;

        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }

    if (!_img1.empty() && stereo_cam)
    {
        for (size_t i = 0; i < ids_right.size(); i++)
        {
            int feature_id = ids_right[i];
            double x, y ,z;
            x = cur_un_right_pts[i].x;
            y = cur_un_right_pts[i].y;
            z = 1;
            double p_u, p_v;
            p_u = cur_right_pts[i].x;
            p_v = cur_right_pts[i].y;
            int camera_id = 1;
            double velocity_x, velocity_y;
            velocity_x = right_pts_velocity[i].x;
            velocity_y = right_pts_velocity[i].y;

            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
        }
    }

    //printf("feature track whole time %f\n", t_r.toc());
    last_timing.pack_ms = t_pack.toc();
    last_timing.final_tracks = static_cast<int>(cur_pts.size());
    last_timing.total_ms = t_r.toc();
    int cur_len1 = 0;
    int cur_len2 = 0;
    int cur_len3 = 0;
    int cur_len4p = 0;
    for (const int cnt : track_cnt)
    {
        if (cnt <= 1)
            ++cur_len1;
        else if (cnt == 2)
            ++cur_len2;
        else if (cnt == 3)
            ++cur_len3;
        else
            ++cur_len4p;
    }
    std::printf("FEATURE TRACKER SURVIVAL stamp=%.6f prev=%d lk_fwd=%d lk_back=%d lio_gate=%d border=%d old_survive=%d add_lidar=%d add_visual=%d new_total=%d final=%d len1=%d len2=%d len3=%d len4p=%d prev_lidar=%d tracked_lidar=%d lio_reject=%d\n",
                cur_time,
                last_prev_track_count,
                survival_lk_forward,
                survival_lk_backward,
                survival_lio_gate,
                survival_border,
                last_tracked_after_flow_count,
                last_added_lidar_count,
                last_added_visual_count,
                last_added_lidar_count + last_added_visual_count,
                last_timing.final_tracks,
                cur_len1,
                cur_len2,
                cur_len3,
                cur_len4p,
                last_prev_lidar_track_count,
                last_tracked_lidar_count,
                last_rejected_by_lio_prior_count);
    return featureFrame;
}

void FeatureTracker::rejectWithLioPrior(vector<uchar> &status)
{
    // LiDAR-seeded tracks have a world anchor from the LIO update. Reproject
    // that anchor through the current LIO pose prior and reject tracks whose
    // LK result moved too far in pixels. This replaces the old F-matrix gate.
    if (!lio_prior_gate.valid || active_lidar_priors.empty() || m_camera.empty() ||
        LIDAR_PRIOR_REPROJ_THRESHOLD <= 0.0)
    {
        return;
    }

    const double threshold_sq = LIDAR_PRIOR_REPROJ_THRESHOLD * LIDAR_PRIOR_REPROJ_THRESHOLD;
    for (size_t i = 0; i < ids.size() && i < cur_pts.size() && i < status.size(); ++i)
    {
        if (!status[i])
            continue;

        const auto it = active_lidar_priors.find(ids[i]);
        if (it == active_lidar_priors.end() || !it->second.valid)
            continue;

        const Eigen::Vector3d p_body =
            lio_prior_gate.R_WB.transpose() * (it->second.P_W_init - lio_prior_gate.p_WB);
        const Eigen::Vector3d p_cam = gate_R_I_C.transpose() * (p_body - gate_t_I_C);
        if (p_cam.z() <= 0.05)
        {
            status[i] = 0;
            last_rejected_by_lio_prior_count++;
            continue;
        }

        Eigen::Vector2d uv;
        m_camera[0]->spaceToPlane(p_cam, uv);
        const double dx = static_cast<double>(cur_pts[i].x) - uv.x();
        const double dy = static_cast<double>(cur_pts[i].y) - uv.y();
        if (dx * dx + dy * dy > threshold_sq) {
            status[i] = 0;
            last_rejected_by_lio_prior_count++;
        }
    }
}

int FeatureTracker::addLidarCandidatePoints(int max_num)
{
    // Insert LiDAR candidates as normal VINS feature ids while attaching their
    // inverse-depth priors. The mask check preserves VINS spatial uniformity.
    if (max_num <= 0 || mask.empty())
        return 0;

    int added = 0;
    for (auto candidate : pending_lidar_candidates)
    {
        if (added >= max_num)
            break;
        if (!inBorder(candidate.pixel))
            continue;

        const int u = cvRound(candidate.pixel.x);
        const int v = cvRound(candidate.pixel.y);
        if (u < 0 || v < 0 || u >= mask.cols || v >= mask.rows || mask.at<uchar>(v, u) != 255)
            continue;

        const int id = n_id++;
        cake_slam::LidarDepthPrior prior = cake_slam::MakeDepthPrior(candidate);

        cur_pts.push_back(candidate.pixel);
        ids.push_back(id);
        track_cnt.push_back(1);
        active_lidar_priors[id] = prior;
        current_lidar_priors[id] = prior;

        const int radius = std::max(1, static_cast<int>(std::round(candidate.mask_radius)));
        cv::circle(mask, candidate.pixel, radius, 0, -1);
        added++;
    }
    return added;
}

void FeatureTracker::pruneLidarTracks()
{
    // Keep prior metadata only for still-alive feature ids; otherwise old
    // LiDAR anchors would be accidentally reused by a future id.
    if (active_lidar_priors.empty())
        return;

    std::set<int> alive(ids.begin(), ids.end());
    for (auto it = active_lidar_priors.begin(); it != active_lidar_priors.end();)
    {
        if (alive.find(it->first) == alive.end())
            it = active_lidar_priors.erase(it);
        else
            ++it;
    }
}

void FeatureTracker::drawFeatureDebugImage(const std::map<int, cv::Point2f> &previous_points)
{
    if (cur_img.empty())
    {
        feature_debug_image.release();
        last_feature_count = 0;
        last_depth_feature_count = 0;
        return;
    }

    if (cur_img.channels() == 1)
        cv::cvtColor(cur_img, feature_debug_image, cv::COLOR_GRAY2BGR);
    else if (cur_img.channels() == 3)
        feature_debug_image = cur_img.clone();
    else
        cv::cvtColor(cur_img, feature_debug_image, cv::COLOR_GRAY2BGR);

    last_feature_count = static_cast<int>(cur_pts.size());
    last_depth_feature_count = 0;

    for (size_t i = 0; i < cur_pts.size() && i < ids.size(); ++i)
    {
        const int id = ids[i];
        const auto prior_it = active_lidar_priors.find(id);
        const bool has_lidar_depth =
            prior_it != active_lidar_priors.end() && prior_it->second.valid;
        if (has_lidar_depth)
            last_depth_feature_count++;

        const auto prev_it = previous_points.find(id);
        if (prev_it != previous_points.end())
        {
            cv::line(feature_debug_image, prev_it->second, cur_pts[i],
                     cv::Scalar(255, 200, 0), 1, cv::LINE_AA);
        }

        if (has_lidar_depth)
            cv::circle(feature_debug_image, cur_pts[i], 3, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
        else
            cv::circle(feature_debug_image, cur_pts[i], 3, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    }

    const double ratio = last_feature_count > 0
                             ? static_cast<double>(last_depth_feature_count) / last_feature_count
                             : 0.0;
    char text[128];
    std::snprintf(text, sizeof(text), "features: %d  lidar-depth: %d (%.1f%%)",
                  last_feature_count, last_depth_feature_count, ratio * 100.0);
    cv::rectangle(feature_debug_image, cv::Point(0, 0), cv::Point(430, 28),
                  cv::Scalar(0, 0, 0), -1);
    cv::putText(feature_debug_image, text, cv::Point(8, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1,
                cv::LINE_AA);
}

// 读取所有相机模型文件。
void FeatureTracker::readIntrinsicParameter(const vector<string> &calib_file)
{
    if (calib_file.empty())
        throw std::runtime_error("FeatureTracker requires at least one camera calibration file");

    m_camera.clear();
    fast_undistort_model.clear();
    fast_undistort_K.clear();
    fast_undistort_D.clear();
    stereo_cam = false;
    for (size_t i = 0; i < calib_file.size(); i++)
    {
        if (calib_file[i].empty())
            throw std::runtime_error("FeatureTracker camera calibration path is empty");

        CAKE_INFO("reading paramerter of camera %s", calib_file[i].c_str());
        camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
        if (!camera)
            throw std::runtime_error("FeatureTracker failed to load camera calibration: " + calib_file[i]);
        m_camera.push_back(camera);

        fast_undistort_model.push_back(0);
        fast_undistort_K.emplace_back();
        fast_undistort_D.emplace_back();

        cv::FileStorage fs(calib_file[i], cv::FileStorage::READ);
        std::string model_type;
        if (fs.isOpened())
            fs["model_type"] >> model_type;

        if (model_type == "KANNALA_BRANDT")
        {
            const cv::FileNode projection = fs["projection_parameters"];
            if (!projection.empty() &&
                !projection["mu"].empty() && !projection["mv"].empty() &&
                !projection["u0"].empty() && !projection["v0"].empty() &&
                !projection["k2"].empty() && !projection["k3"].empty() &&
                !projection["k4"].empty() && !projection["k5"].empty())
            {
                const double mu = static_cast<double>(projection["mu"]);
                const double mv = static_cast<double>(projection["mv"]);
                const double u0 = static_cast<double>(projection["u0"]);
                const double v0 = static_cast<double>(projection["v0"]);
                const double k2 = static_cast<double>(projection["k2"]);
                const double k3 = static_cast<double>(projection["k3"]);
                const double k4 = static_cast<double>(projection["k4"]);
                const double k5 = static_cast<double>(projection["k5"]);

                fast_undistort_model.back() = 1;
                fast_undistort_K.back() = (cv::Mat_<double>(3, 3) <<
                    mu, 0.0, u0,
                    0.0, mv, v0,
                    0.0, 0.0, 1.0);
                fast_undistort_D.back() = (cv::Mat_<double>(4, 1) << k2, k3, k4, k5);
                CAKE_INFO("FeatureTracker fast undistort: camera=%zu model=KANNALA_BRANDT backend=opencv_fisheye",
                          i);
            }
            else
            {
                ROS_WARN("KANNALA_BRANDT camera calibration is missing projection parameters, falling back to camodocal liftProjective: %s",
                         calib_file[i].c_str());
            }
        }
    }
    if (calib_file.size() == 2)
        stereo_cam = 1;
    if (!FISHEYE_MASK.empty())
    {
        fisheye_mask = cv::imread(FISHEYE_MASK, cv::IMREAD_GRAYSCALE);
        if (fisheye_mask.empty())
            ROS_WARN("fisheye/valid-domain mask path is set but failed to load: %s", FISHEYE_MASK.c_str());
    }
}

// 把像素网格经相机模型反投影后再投影回来，用于观察去畸变效果。
void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(row + 600, col + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < col; i++)
        for (int j = 0; j < row; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera[0]->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + col / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + row / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < col + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    // turn the following code on if you need
    // cv::imshow(name, undistortedImg);
    // cv::waitKey(0);
}

// 对输入像素点逐个执行去畸变，并转换为归一化平面坐标。
vector<cv::Point2f> FeatureTracker::undistortedPts(vector<cv::Point2f> &pts, int camera_id)
{
    vector<cv::Point2f> un_pts;
    un_pts.reserve(pts.size());

    if (camera_id >= 0 && camera_id < static_cast<int>(fast_undistort_model.size()) &&
        fast_undistort_model[camera_id] == 1 && !pts.empty())
    {
        cv::fisheye::undistortPoints(pts, un_pts,
                                      fast_undistort_K[camera_id],
                                      fast_undistort_D[camera_id]);
        return un_pts;
    }

    if (camera_id < 0 || camera_id >= static_cast<int>(m_camera.size()) || !m_camera[camera_id])
        return un_pts;

    for (const auto &pt : pts)
    {
        Eigen::Vector2d a(pt.x, pt.y);
        Eigen::Vector3d b;
        m_camera[camera_id]->liftProjective(a, b);
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    }
    return un_pts;
}

// 计算特征的像素速度，用于时间延迟补偿与后端残差构建。
vector<cv::Point2f> FeatureTracker::ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, 
                                            std::map<int, cv::Point2f> &cur_id_pts, std::map<int, cv::Point2f> &prev_id_pts)
{
    vector<cv::Point2f> pts_velocity;
    cur_id_pts.clear();
    for (unsigned int i = 0; i < ids.size(); i++)
    {
        cur_id_pts.insert(make_pair(ids[i], pts[i]));
    }

    // caculate points velocity
    if (!prev_id_pts.empty())
    {
        double dt = cur_time - prev_time;
        
        for (unsigned int i = 0; i < pts.size(); i++)
        {
            std::map<int, cv::Point2f>::iterator it;
            it = prev_id_pts.find(ids[i]);
            if (it != prev_id_pts.end())
            {
                double v_x = (pts[i].x - it->second.x) / dt;
                double v_y = (pts[i].y - it->second.y) / dt;
                pts_velocity.push_back(cv::Point2f(v_x, v_y));
            }
            else
                pts_velocity.push_back(cv::Point2f(0, 0));

        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    return pts_velocity;
}

// 绘制左右图跟踪结果，供调试可视化使用。
void FeatureTracker::drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                               vector<int> &curLeftIds,
                               vector<cv::Point2f> &curLeftPts, 
                               vector<cv::Point2f> &curRightPts,
                               std::map<int, cv::Point2f> &prevLeftPtsMap)
{
    //int rows = imLeft.rows;
    int cols = imLeft.cols;
    if (!imRight.empty() && stereo_cam)
        cv::hconcat(imLeft, imRight, imTrack);
    else
        imTrack = imLeft.clone();
    cv::cvtColor(imTrack, imTrack, cv::COLOR_GRAY2RGB);

    for (size_t j = 0; j < curLeftPts.size(); j++)
    {
        double len = std::min(1.0, 1.0 * track_cnt[j] / 20);
        cv::circle(imTrack, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }
    if (!imRight.empty() && stereo_cam)
    {
        for (size_t i = 0; i < curRightPts.size(); i++)
        {
            cv::Point2f rightPt = curRightPts[i];
            rightPt.x += cols;
            cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
            //cv::Point2f leftPt = curLeftPtsTrackRight[i];
            //cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);
        }
    }
    
    std::map<int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < curLeftIds.size(); i++)
    {
        int id = curLeftIds[i];
        mapIt = prevLeftPtsMap.find(id);
        if(mapIt != prevLeftPtsMap.end())
        {
            cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }

    //draw prediction
    /*
    for(size_t i = 0; i < predict_pts_debug.size(); i++)
    {
        cv::circle(imTrack, predict_pts_debug[i], 2, cv::Scalar(0, 170, 255), 2);
    }
    */
    //printf("predict pts size %d \n", (int)predict_pts_debug.size());

    //cv::Mat imCur2Compress;
    //cv::resize(imCur2, imCur2Compress, cv::Size(cols, rows / 2));
}


// 接收来自后端的三维点预测，并投影成下一帧初值。
void FeatureTracker::setPrediction(std::map<int, Eigen::Vector3d> &predictPts)
{
    hasPrediction = true;
    predict_pts.clear();
    predict_pts_debug.clear();
    std::map<int, Eigen::Vector3d>::iterator itPredict;
    for (size_t i = 0; i < ids.size(); i++)
    {
        //printf("prevLeftId size %d prevLeftPts size %d\n",(int)prevLeftIds.size(), (int)prevLeftPts.size());
        int id = ids[i];
        itPredict = predictPts.find(id);
        if (itPredict != predictPts.end())
        {
            Eigen::Vector2d tmp_uv;
            m_camera[0]->spaceToPlane(itPredict->second, tmp_uv);
            predict_pts.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
            predict_pts_debug.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
        }
        else
            predict_pts.push_back(prev_pts[i]);
    }
}


// 根据外点 id 集合移除当前跟踪点。
void FeatureTracker::removeOutliers(set<int> &removePtsIds)
{
    std::set<int>::iterator itSet;
    vector<uchar> status;
    for (size_t i = 0; i < ids.size(); i++)
    {
        itSet = removePtsIds.find(ids[i]);
        if(itSet != removePtsIds.end())
            status.push_back(0);
        else
            status.push_back(1);
    }

    reduceVector(prev_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
}


// 返回最近一次绘制的跟踪图像。
cv::Mat FeatureTracker::getTrackImage()
{
    return imTrack;
}

cv::Mat FeatureTracker::getFeatureDebugImage() const
{
    return feature_debug_image.clone();
}

int FeatureTracker::getLastFeatureCount() const
{
    return last_feature_count;
}

int FeatureTracker::getLastDepthFeatureCount() const
{
    return last_depth_feature_count;
}

const cv::Mat &FeatureTracker::validMask() const
{
    return fisheye_mask;
}
