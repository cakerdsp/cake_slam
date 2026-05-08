/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "parameters.h"

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

int USE_GPU;
int USE_GPU_ACC_FLOW;
int USE_GPU_CERES;

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string OUTPUT_FOLDER;
std::string IMU_TOPIC;
int ROW, COL;
double TD;
int NUM_OF_CAM;
int STEREO;
int USE_IMU;
int MULTIPLE_THREAD;
std::map<int, Eigen::Vector3d> pts_gt;
std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
std::string FISHEYE_MASK;
std::vector<std::string> CAM_NAMES;
int MAX_CNT;
int MIN_DIST;
int SHOW_TRACK;
int FLOW_BACK;
int LIDAR_DEPTH_ENABLE;
int LIDAR_INV_DEPTH_OPTIMIZE;
double LIDAR_PRIOR_REPROJ_THRESHOLD;
double MIN_INV_DEPTH_VAR;
double MIN_LIO_POSE_PRIOR_VAR;

std::string WORLD_FRAME_ID;
std::string BODY_FRAME_ID;
std::string CAMERA_FRAME_ID;

namespace {

cv::FileNode nodeForKey(const cv::FileStorage &fs, const std::string &key)
{
    cv::FileNode direct = fs[key];
    if (!direct.empty())
        return direct;

    cv::FileNode node = fs.root();
    size_t begin = 0;
    while (begin < key.size())
    {
        const size_t end = key.find('.', begin);
        const std::string part = key.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        node = node[part];
        if (node.empty())
            return node;
        if (end == std::string::npos)
            break;
        begin = end + 1;
    }
    return node;
}

template <typename T>
void readIfPresent(const cv::FileStorage &fs, const std::string &key, T &out)
{
    cv::FileNode node = nodeForKey(fs, key);
    if (!node.empty())
        node >> out;
}

template <typename T>
T readOrDefault(const cv::FileStorage &fs, const std::string &key, const T &fallback)
{
    T out = fallback;
    readIfPresent(fs, key, out);
    return out;
}

bool readVectorIfPresent(const cv::FileStorage &fs, const std::string &key, std::vector<double> &out)
{
    cv::FileNode node = nodeForKey(fs, key);
    if (node.empty())
        return false;

    out.clear();
    if (node.isSeq())
    {
        for (auto it = node.begin(); it != node.end(); ++it)
            out.push_back((double)*it);
        return true;
    }

    if (node.isMap())
    {
        cv::Mat mat;
        node >> mat;
        out.assign((double *)mat.datastart, (double *)mat.dataend);
        return true;
    }

    return false;
}

bool readVectorFromAnyKey(const cv::FileStorage &fs, const std::vector<std::string> &keys, std::vector<double> &out)
{
    for (const auto &key : keys)
    {
        if (readVectorIfPresent(fs, key, out))
            return true;
    }
    return false;
}

Eigen::Matrix3d matrix3dFromRowMajor(const std::vector<double> &data)
{
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    if (data.size() >= 9)
    {
        R << data[0], data[1], data[2],
             data[3], data[4], data[5],
             data[6], data[7], data[8];
    }
    return R;
}

Eigen::Vector3d vector3dFromArray(const std::vector<double> &data)
{
    if (data.size() >= 3)
        return Eigen::Vector3d(data[0], data[1], data[2]);
    return Eigen::Vector3d::Zero();
}

bool readBodyToCameraExtrinsic(const cv::FileStorage &fs, Eigen::Matrix3d &R_ic, Eigen::Vector3d &t_ic)
{
    cv::Mat cv_T;
    nodeForKey(fs, "extrinsic.body_T_cam0") >> cv_T;
    if (cv_T.empty())
        return false;

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    cv::cv2eigen(cv_T, T);
    R_ic = T.block<3, 3>(0, 0);
    t_ic = T.block<3, 1>(0, 3);
    return true;
}

bool readFastLivoExtrinsic(const cv::FileStorage &fs, Eigen::Matrix3d &R_ic, Eigen::Vector3d &t_ic)
{
    std::vector<double> lidar_R, lidar_T, camera_R, camera_T;
    const bool has_lidar_R = readVectorFromAnyKey(fs, {"extrinsic.lidar_R", "extrin_calib.extrinsic_R"}, lidar_R);
    const bool has_lidar_T = readVectorFromAnyKey(fs, {"extrinsic.lidar_T", "extrin_calib.extrinsic_T"}, lidar_T);
    const bool has_camera_R = readVectorFromAnyKey(fs, {"extrinsic.camera_R", "extrinsic.Rcl", "extrin_calib.Rcl"}, camera_R);
    const bool has_camera_T = readVectorFromAnyKey(fs, {"extrinsic.camera_T", "extrinsic.Pcl", "extrin_calib.Pcl"}, camera_T);
    if (!has_lidar_R || !has_lidar_T || !has_camera_R || !has_camera_T)
        return false;

    // FAST-LIVO2 约定：
    // P_imu = R_I_L * P_lidar + t_I_L
    // P_cam = R_C_L * P_lidar + t_C_L
    // VINS 约定：
    // P_imu = R_I_C * P_cam + t_I_C
    const Eigen::Matrix3d R_i_l = matrix3dFromRowMajor(lidar_R);
    const Eigen::Vector3d t_i_l = vector3dFromArray(lidar_T);
    const Eigen::Matrix3d R_c_l = matrix3dFromRowMajor(camera_R);
    const Eigen::Vector3d t_c_l = vector3dFromArray(camera_T);

    const Eigen::Matrix3d R_l_i = R_i_l.transpose();
    const Eigen::Vector3d t_l_i = -R_l_i * t_i_l;
    const Eigen::Matrix3d R_c_i = R_c_l * R_l_i;
    const Eigen::Vector3d t_c_i = R_c_l * t_l_i + t_c_l;

    R_ic = R_c_i.transpose();
    t_ic = -R_ic * t_c_i;
    return true;
}

} // namespace

template <typename T>
// 从 ROS 参数服务器读取参数的辅助函数。
T readParam(rclcpp::Node::SharedPtr n, std::string name)
{
    T ans;
    if (n->get_parameter(name, ans))
    {
        ROS_INFO("Loaded %s: ", name);
        std::cout << ans << std::endl;
    }
    else
    {
        ROS_ERROR("Failed to load %s", name);
        rclcpp::shutdown();
    }
    return ans;
}

void readParameters(std::string config_file)
{
    // 视觉模块统一参数加载入口。
    // 该函数会把配置文件中的值写入 parameters.h 中声明的全局变量，
    // 后续前端、初始化和后端都会直接读取这些全局量。
    FILE *fh = fopen(config_file.c_str(),"r");
    if(fh == NULL){
        ROS_WARN("config_file dosen't exist; wrong config_file path");
        // ROS_BREAK();
        return;          
    }
    fclose(fh);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    RIC.clear();
    TIC.clear();
    CAM_NAMES.clear();
    FISHEYE_MASK.clear();

    // -------------------- 前端跟踪参数 --------------------
    readIfPresent(fsSettings, "vision.image_topic", IMAGE0_TOPIC);
    IMAGE1_TOPIC.clear();
    MAX_CNT = readOrDefault<int>(fsSettings, "vision.max_cnt", 150);
    MIN_DIST = readOrDefault<int>(fsSettings, "vision.min_dist", 30);
    SHOW_TRACK = readOrDefault<int>(fsSettings, "vision.show_track", 0);
    FLOW_BACK = readOrDefault<int>(fsSettings, "vision.flow_back", 0);
    LIDAR_DEPTH_ENABLE = readOrDefault<int>(fsSettings, "vision.lidar_depth_enable", 1);
    LIDAR_INV_DEPTH_OPTIMIZE = readOrDefault<int>(fsSettings, "vision.optimize_lidar_inv_depth", 1);
    LIDAR_PRIOR_REPROJ_THRESHOLD = readOrDefault<double>(fsSettings, "vision.lio_prior_reproj_threshold", 3.0);
    MIN_INV_DEPTH_VAR = readOrDefault<double>(fsSettings, "vision.min_inv_depth_var", 1e-6);
    MIN_LIO_POSE_PRIOR_VAR = readOrDefault<double>(fsSettings, "vision.min_lio_pose_prior_var", 1e-6);

    MULTIPLE_THREAD = readOrDefault<int>(fsSettings, "vision.multiple_thread", 0);

    // 当前实现默认关闭 GPU 路径。
    USE_GPU = 0;
    USE_GPU_ACC_FLOW = 0;
    USE_GPU_CERES = 0;

    // -------------------- IMU 参数 --------------------
    USE_IMU = readOrDefault<int>(fsSettings, "imu.enable", 1);
    if(USE_IMU)
    {
        readIfPresent(fsSettings, "imu.topic", IMU_TOPIC);
        ACC_N = readOrDefault<double>(fsSettings, "imu.acc_n", 0.02);
        ACC_W = readOrDefault<double>(fsSettings, "imu.acc_w", 0.04);
        GYR_N = readOrDefault<double>(fsSettings, "imu.gyr_n", 0.01);
        GYR_W = readOrDefault<double>(fsSettings, "imu.gyr_w", 0.001);
        G.z() = readOrDefault<double>(fsSettings, "imu.g_norm", 9.81);
    }

    // -------------------- 优化与关键帧参数 --------------------
    SOLVER_TIME = readOrDefault<double>(fsSettings, "vision.max_solver_time", 0.04);
    NUM_ITERATIONS = readOrDefault<int>(fsSettings, "vision.max_num_iterations", 8);
    MIN_PARALLAX = readOrDefault<double>(fsSettings, "vision.keyframe_parallax", 10.0);
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    // -------------------- 输出路径 --------------------
    readIfPresent(fsSettings, "output.path", OUTPUT_FOLDER);
    if (OUTPUT_FOLDER.empty())
        OUTPUT_FOLDER = ".";
    VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv";
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    // -------------------- 外参处理策略 --------------------
    ESTIMATE_EXTRINSIC = readOrDefault<int>(fsSettings, "vision.estimate_extrinsic", 0);
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    }
    else 
    {
        if ( ESTIMATE_EXTRINSIC == 1)
        {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0)
            ROS_WARN(" fix extrinsic param ");

        Eigen::Matrix3d R_ic = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t_ic = Eigen::Vector3d::Zero();
        if (readBodyToCameraExtrinsic(fsSettings, R_ic, t_ic))
        {
            ROS_INFO("load VINS camera-to-IMU extrinsic from extrinsic.body_T_cam0");
        }
        else if (readFastLivoExtrinsic(fsSettings, R_ic, t_ic))
        {
            ROS_INFO("compose VINS camera-to-IMU extrinsic from FAST-LIVO2 lidar/camera extrinsics");
        }
        else
        {
            ROS_WARN("camera extrinsic missing, use identity camera-to-IMU extrinsic");
        }
        RIC.push_back(R_ic);
        TIC.push_back(t_ic);
    } 

    // 当前配置默认单目。
    NUM_OF_CAM = 1;
    STEREO = 0;

    std::string cam0Calib;
    readIfPresent(fsSettings, "vision.cam0_calib", cam0Calib);
    if (!cam0Calib.empty())
    {
        const bool absolute_path =
            (cam0Calib.size() > 1 && cam0Calib[1] == ':') ||
            (!cam0Calib.empty() && (cam0Calib[0] == '/' || cam0Calib[0] == '\\'));
        if (!absolute_path)
        {
            const size_t slash = config_file.find_last_of("/\\");
            if (slash != std::string::npos)
                cam0Calib = config_file.substr(0, slash + 1) + cam0Calib;
        }
    }
    CAM_NAMES.push_back(cam0Calib);
    readIfPresent(fsSettings, "vision.fisheye_mask", FISHEYE_MASK);
    if (!FISHEYE_MASK.empty())
    {
        const bool absolute_path =
            (FISHEYE_MASK.size() > 1 && FISHEYE_MASK[1] == ':') ||
            (!FISHEYE_MASK.empty() && (FISHEYE_MASK[0] == '/' || FISHEYE_MASK[0] == '\\'));
        if (!absolute_path)
        {
            const size_t slash = config_file.find_last_of("/\\");
            if (slash != std::string::npos)
                FISHEYE_MASK = config_file.substr(0, slash + 1) + FISHEYE_MASK;
        }
    }

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    // -------------------- 时间偏移与图像尺寸 --------------------
    TD = readOrDefault<double>(fsSettings, "time_offset.td", 0.0);
    ESTIMATE_TD = readOrDefault<int>(fsSettings, "time_offset.estimate_td", 0);
    if (ESTIMATE_TD)
        ROS_INFO("Unsynchronized sensors, online estimate time offset, initial td: %f", TD);
    else
        ROS_INFO("Synchronized sensors, fix time offset: %f", TD);

    ROW = readOrDefault<int>(fsSettings, "vision.image_height", 480);
    COL = readOrDefault<int>(fsSettings, "vision.image_width", 640);
    ROS_INFO("ROW: %d COL: %d ", ROW, COL);

    // 没有 IMU 时，不允许再估计相机-IMU 外参和时间延迟。
    if(!USE_IMU)
    {
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        printf("no imu, fix extrinsic param; no time offset calibration\n");
    }

    // -------------------- 坐标系命名 --------------------
    readIfPresent(fsSettings, "frame.world", WORLD_FRAME_ID);
    WORLD_FRAME_ID.empty()? WORLD_FRAME_ID = "world" : WORLD_FRAME_ID;
    readIfPresent(fsSettings, "frame.body", BODY_FRAME_ID);
    BODY_FRAME_ID.empty()? BODY_FRAME_ID = "body" : BODY_FRAME_ID;
    readIfPresent(fsSettings, "frame.camera", CAMERA_FRAME_ID);
    CAMERA_FRAME_ID.empty()? CAMERA_FRAME_ID = "camera" : CAMERA_FRAME_ID;
    
    ROS_INFO("frame_ids: world=%s body=%s camera=%s", WORLD_FRAME_ID.c_str(),
             BODY_FRAME_ID.c_str(), CAMERA_FRAME_ID.c_str());

    fsSettings.release();
}
