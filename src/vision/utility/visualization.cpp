/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "visualization.h"

// using namespace ros;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odometry, pub_latest_odometry;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path;
rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_point_cloud, pub_margin_cloud;
rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_key_poses;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_camera_pose;
rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_camera_pose_visual;
nav_msgs::msg::Path path;

rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_keyframe_pose;
rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_keyframe_point;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_extrinsic;

rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_image_track;

CameraPoseVisualization cameraposevisual(1, 0, 0, 1);
static double sum_of_path = 0;
static Eigen::Vector3d last_path(0.0, 0.0, 0.0);

size_t pub_counter = 0;

// 注册视觉模块常用的全部 ROS 发布器。
void registerPub(rclcpp::Node::SharedPtr n)
{
    pub_latest_odometry = n->create_publisher<nav_msgs::msg::Odometry>("imu_propagate", 10);
    pub_path = n->create_publisher<nav_msgs::msg::Path>("path", 10);
    pub_odometry = n->create_publisher<nav_msgs::msg::Odometry>("odometry", 10);
    pub_point_cloud = n->create_publisher<sensor_msgs::msg::PointCloud>("point_cloud", 10);
    pub_margin_cloud = n->create_publisher<sensor_msgs::msg::PointCloud>("margin_cloud", 10);
    pub_key_poses = n->create_publisher<visualization_msgs::msg::Marker>("key_poses", 10);
    pub_camera_pose = n->create_publisher<nav_msgs::msg::Odometry>("camera_pose", 10);
    pub_camera_pose_visual = n->create_publisher<visualization_msgs::msg::MarkerArray>("camera_pose_visual", 10);
    pub_keyframe_pose = n->create_publisher<nav_msgs::msg::Odometry>("keyframe_pose", 10);
    pub_keyframe_point = n->create_publisher<sensor_msgs::msg::PointCloud>("keyframe_point", 10);
    pub_extrinsic = n->create_publisher<nav_msgs::msg::Odometry>("extrinsic", 10);
    pub_image_track = n->create_publisher<sensor_msgs::msg::Image>("image_track", 1);

    cameraposevisual.setScale(0.1);
    cameraposevisual.setLineWidth(0.01);
}

// 发布高频传播得到的最新里程计结果。
void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, double t)
{
    if (!pub_latest_odometry)
        return;
    nav_msgs::msg::Odometry odometry;

    int sec_ts = (int)t;
    uint nsec_ts = (uint)((t - sec_ts) * 1e9);
    odometry.header.stamp.sec = sec_ts;
    odometry.header.stamp.nanosec = nsec_ts;

    odometry.header.frame_id = WORLD_FRAME_ID;
    odometry.pose.pose.position.x = P.x();
    odometry.pose.pose.position.y = P.y();
    odometry.pose.pose.position.z = P.z();
    odometry.pose.pose.orientation.x = Q.x();
    odometry.pose.pose.orientation.y = Q.y();
    odometry.pose.pose.orientation.z = Q.z();
    odometry.pose.pose.orientation.w = Q.w();
    odometry.twist.twist.linear.x = V.x();
    odometry.twist.twist.linear.y = V.y();
    odometry.twist.twist.linear.z = V.z();
    pub_latest_odometry->publish(odometry);
}

// 发布前端跟踪可视化图像。
void pubTrackImage(const cv::Mat &imgTrack, const double t)
{
    if (!pub_image_track)
        return;
    std_msgs::msg::Header header;
    header.frame_id = WORLD_FRAME_ID;

    int sec_ts = (int)t;
    uint nsec_ts = (uint)((t - sec_ts) * 1e9);
    header.stamp.sec = sec_ts;
    header.stamp.nanosec = nsec_ts;

    // sensor_msgs::msg::ImagePtr 
    sensor_msgs::msg::Image::SharedPtr imgTrackMsg = cv_bridge::CvImage(header, "bgr8", imgTrack).toImageMsg();
    pub_image_track->publish(*imgTrackMsg);
}


void printStatistics(const Estimator &estimator, double t)
{
    if (estimator.solver_flag != Estimator::NON_LINEAR)
        return;
    //printf("position: %f, %f, %f\r", estimator.states_[WINDOW_SIZE].pos_end.x(), estimator.states_[WINDOW_SIZE].pos_end.y(), estimator.states_[WINDOW_SIZE].pos_end.z());
    // ROS_DEBUG_STREAM("position: " << estimator.states_[WINDOW_SIZE].pos_end.transpose());
    // ROS_DEBUG_STREAM("orientation: " << estimator.states_[WINDOW_SIZE].vel_end.transpose());
    if (ESTIMATE_EXTRINSIC)
    {
        cv::FileStorage fs(EX_CALIB_RESULT_PATH, cv::FileStorage::WRITE);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            //ROS_DEBUG("calibration result for camera %d", i);
            // ROS_DEBUG_STREAM("extirnsic tic: " << estimator.tic[i].transpose());
            // ROS_DEBUG_STREAM("extrinsic ric: " << Utility::R2ypr(estimator.ric[i]).transpose());

            Eigen::Matrix4d eigen_T = Eigen::Matrix4d::Identity();
            eigen_T.block<3, 3>(0, 0) = estimator.ric[i];
            eigen_T.block<3, 1>(0, 3) = estimator.tic[i];
            cv::Mat cv_T;
            cv::eigen2cv(eigen_T, cv_T);
            if(i == 0)
                fs << "body_T_cam0" << cv_T ;
            else
                fs << "body_T_cam1" << cv_T ;
        }
        fs.release();
    }

    static double sum_of_time = 0;
    static int sum_of_calculation = 0;
    sum_of_time += t;
    sum_of_calculation++;
    ROS_DEBUG("vo solver costs: %f ms", t);
    ROS_DEBUG("average of time %f ms", sum_of_time / sum_of_calculation);

    sum_of_path += (estimator.states_[WINDOW_SIZE].pos_end - last_path).norm();
    last_path = estimator.states_[WINDOW_SIZE].pos_end;
    ROS_DEBUG("sum of path %f", sum_of_path);
    if (ESTIMATE_TD)
        ROS_INFO("td %f", estimator.td);
}

// 发布当前优化后的主里程计与轨迹。
void pubOdometry(const Estimator &estimator, const std_msgs::msg::Header &header)
{
    if (!pub_odometry || !pub_path)
        return;
    if (estimator.solver_flag == Estimator::NON_LINEAR)
    {
        nav_msgs::msg::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = WORLD_FRAME_ID;
        odometry.child_frame_id = BODY_FRAME_ID;
        Eigen::Quaterniond tmp_Q;
        tmp_Q = Eigen::Quaterniond(estimator.states_[WINDOW_SIZE].rot_end);
        odometry.pose.pose.position.x = estimator.states_[WINDOW_SIZE].pos_end.x();
        odometry.pose.pose.position.y = estimator.states_[WINDOW_SIZE].pos_end.y();
        odometry.pose.pose.position.z = estimator.states_[WINDOW_SIZE].pos_end.z();
        odometry.pose.pose.orientation.x = tmp_Q.x();
        odometry.pose.pose.orientation.y = tmp_Q.y();
        odometry.pose.pose.orientation.z = tmp_Q.z();
        odometry.pose.pose.orientation.w = tmp_Q.w();
        odometry.twist.twist.linear.x = estimator.states_[WINDOW_SIZE].vel_end.x();
        odometry.twist.twist.linear.y = estimator.states_[WINDOW_SIZE].vel_end.y();
        odometry.twist.twist.linear.z = estimator.states_[WINDOW_SIZE].vel_end.z();
        pub_odometry->publish(odometry);

        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header = header;
        pose_stamped.header.frame_id = WORLD_FRAME_ID;
        pose_stamped.pose = odometry.pose.pose;
        path.header = header;
        path.header.frame_id = WORLD_FRAME_ID;
        path.poses.push_back(pose_stamped);
        pub_path->publish(path);

        // write result to file
        ofstream foutC(VINS_RESULT_PATH, ios::app);
        foutC.setf(ios::fixed, ios::floatfield);
        foutC.precision(0);
        foutC << header.stamp.sec + header.stamp.nanosec * (1e-9) << ",";
        foutC.precision(5);
        foutC << estimator.states_[WINDOW_SIZE].pos_end.x() << ","
              << estimator.states_[WINDOW_SIZE].pos_end.y() << ","
              << estimator.states_[WINDOW_SIZE].pos_end.z() << ","
              << tmp_Q.w() << ","
              << tmp_Q.x() << ","
              << tmp_Q.y() << ","
              << tmp_Q.z() << ","
              << estimator.states_[WINDOW_SIZE].vel_end.x() << ","
              << estimator.states_[WINDOW_SIZE].vel_end.y() << ","
              << estimator.states_[WINDOW_SIZE].vel_end.z() << "," << endl;
        foutC.close();
        Eigen::Vector3d tmp_T = estimator.states_[WINDOW_SIZE].pos_end;
        printf("time: %f, t: %f %f %f q: %f %f %f %f \n", header.stamp.sec + header.stamp.nanosec * (1e-9),
                                                          tmp_T.x(), tmp_T.y(), tmp_T.z(),
                                                          tmp_Q.w(), tmp_Q.x(), tmp_Q.y(), tmp_Q.z());
    }
}

// 发布历史关键帧位置。
void pubKeyPoses(const Estimator &estimator, const std_msgs::msg::Header &header)
{
    if (!pub_key_poses)
        return;
    if (estimator.key_poses.size() == 0)
        return;
    visualization_msgs::msg::Marker key_poses;
    key_poses.header = header;
    key_poses.header.frame_id = WORLD_FRAME_ID;
    key_poses.ns = "key_poses";
    key_poses.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    key_poses.action = visualization_msgs::msg::Marker::ADD;
    key_poses.pose.orientation.w = 1.0;
    key_poses.lifetime = rclcpp::Duration(0, 0);

    //static int key_poses_id = 0;
    key_poses.id = 0; //key_poses_id++;
    key_poses.scale.x = 0.05;
    key_poses.scale.y = 0.05;
    key_poses.scale.z = 0.05;
    key_poses.color.r = 1.0;
    key_poses.color.a = 1.0;

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        geometry_msgs::msg::Point pose_marker;
        Eigen::Vector3d correct_pose;
        correct_pose = estimator.key_poses[i];
        pose_marker.x = correct_pose.x();
        pose_marker.y = correct_pose.y();
        pose_marker.z = correct_pose.z();
        key_poses.points.push_back(pose_marker);
    }
    pub_key_poses->publish(key_poses);
}

// 发布当前相机位姿与相机金字塔可视化。
void pubCameraPose(const Estimator &estimator, const std_msgs::msg::Header &header)
{
    if (!pub_camera_pose || !pub_camera_pose_visual)
        return;
    int idx2 = WINDOW_SIZE - 1;

    if (estimator.solver_flag == Estimator::NON_LINEAR)
    {
        int i = idx2;
        Eigen::Vector3d P = estimator.states_[i].pos_end + estimator.states_[i].rot_end * estimator.tic[0];
        Eigen::Quaterniond R = Eigen::Quaterniond(estimator.states_[i].rot_end * estimator.ric[0]);

        nav_msgs::msg::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = WORLD_FRAME_ID;
        odometry.pose.pose.position.x = P.x();
        odometry.pose.pose.position.y = P.y();
        odometry.pose.pose.position.z = P.z();
        odometry.pose.pose.orientation.x = R.x();
        odometry.pose.pose.orientation.y = R.y();
        odometry.pose.pose.orientation.z = R.z();
        odometry.pose.pose.orientation.w = R.w();

        pub_camera_pose->publish(odometry);

        cameraposevisual.reset();
        cameraposevisual.add_pose(P, R);
        if(STEREO)
        {
            Eigen::Vector3d P = estimator.states_[i].pos_end + estimator.states_[i].rot_end * estimator.tic[1];
            Eigen::Quaterniond R = Eigen::Quaterniond(estimator.states_[i].rot_end * estimator.ric[1]);
            cameraposevisual.add_pose(P, R);
        }
        cameraposevisual.publish_by(pub_camera_pose_visual, odometry.header);
    }
}


// 发布当前滑窗恢复的稀疏点云以及边缘化点云。
void pubPointCloud(const Estimator &estimator, const std_msgs::msg::Header &header)
{
    if (!pub_point_cloud || !pub_margin_cloud)
        return;
    sensor_msgs::msg::PointCloud point_cloud, loop_point_cloud;
    point_cloud.header = header;
    loop_point_cloud.header = header;


    for (auto &it_per_id : estimator.f_manager.feature)
    {
        int used_num;
        used_num = it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        if (it_per_id.start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id.solve_flag != 1)
            continue;
        int imu_i = it_per_id.start_frame;
        Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
        Eigen::Vector3d w_pts_i = estimator.states_[imu_i].rot_end * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.states_[imu_i].pos_end;

        geometry_msgs::msg::Point32 p;
        p.x = w_pts_i(0);
        p.y = w_pts_i(1);
        p.z = w_pts_i(2);
        point_cloud.points.push_back(p);
    }
    pub_point_cloud->publish(point_cloud);


    // pub margined potin
    sensor_msgs::msg::PointCloud margin_cloud;
    margin_cloud.header = header;

    for (auto &it_per_id : estimator.f_manager.feature)
    { 
        int used_num;
        used_num = it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        //if (it_per_id->start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id->solve_flag != 1)
        //        continue;

        if (it_per_id.start_frame == 0 && it_per_id.feature_per_frame.size() <= 2 
            && it_per_id.solve_flag == 1 )
        {
            int imu_i = it_per_id.start_frame;
            Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
            Eigen::Vector3d w_pts_i = estimator.states_[imu_i].rot_end * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.states_[imu_i].pos_end;

            geometry_msgs::msg::Point32 p;
            p.x = w_pts_i(0);
            p.y = w_pts_i(1);
            p.z = w_pts_i(2);
            margin_cloud.points.push_back(p);
        }
    }
    pub_margin_cloud->publish(margin_cloud);
}



// 发布世界系到 body/camera 的 TF。
void pubTF(const Estimator &estimator, const std_msgs::msg::Header &header)
{
    return; // tmp.


    cout << "tf 1" << endl;
    if( estimator.solver_flag != Estimator::NON_LINEAR)
        return;

    std::shared_ptr<tf2_ros::TransformBroadcaster> br;
    geometry_msgs::msg::TransformStamped transform, transform_cam;

    tf2::Quaternion q;
    // body frame
    Eigen::Vector3d correct_t;
    Eigen::Quaterniond correct_q;
    
    cout << "tf 2" << endl;
    correct_t = estimator.states_[WINDOW_SIZE].pos_end;
    correct_q = estimator.states_[WINDOW_SIZE].rot_end;

    cout << "tf 3" << endl;

    
    cout << header.stamp.sec + header.stamp.nanosec * (1e-9) << endl;
    cout << correct_t << endl;
    cout << correct_q.w() << " " << correct_q.x() << " " << correct_q.y() << " " << correct_q.z() << endl;


    // transform.header.stamp = header.stamp;
    transform.header.frame_id = WORLD_FRAME_ID;
    transform.child_frame_id = BODY_FRAME_ID;

    transform.transform.translation.x = correct_t(0);
    transform.transform.translation.y = correct_t(1);
    transform.transform.translation.z = correct_t(2);

    cout << "tf 4" << endl;


    q.setW(correct_q.w());
    q.setX(correct_q.x());
    q.setY(correct_q.y());
    q.setZ(correct_q.z());
    transform.transform.rotation.x = q.x();
    transform.transform.rotation.y = q.y();
    transform.transform.rotation.z = q.z();
    transform.transform.rotation.w = q.w();

    cout << "tf 5" << endl;

    br->sendTransform(transform);


    cout << "tf 6" << endl;



    // camera frame
    transform_cam.header.stamp = header.stamp;
    transform_cam.header.frame_id = BODY_FRAME_ID;
    transform_cam.child_frame_id = CAMERA_FRAME_ID;


    transform_cam.transform.translation.x = estimator.tic[0].x();
    transform_cam.transform.translation.y = estimator.tic[0].y();
    transform_cam.transform.translation.z = estimator.tic[0].z();

    q.setW(Eigen::Quaterniond(estimator.ric[0]).w());
    q.setX(Eigen::Quaterniond(estimator.ric[0]).x());
    q.setY(Eigen::Quaterniond(estimator.ric[0]).y());
    q.setZ(Eigen::Quaterniond(estimator.ric[0]).z());

    transform_cam.transform.rotation.x = q.x();
    transform_cam.transform.rotation.y = q.y();
    transform_cam.transform.rotation.z = q.z();
    transform_cam.transform.rotation.w = q.w();

    // br->sendTransform(transform_cam);

    cout << "tf 7" << endl;

    
    nav_msgs::msg::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = WORLD_FRAME_ID;
    odometry.pose.pose.position.x = estimator.tic[0].x();
    odometry.pose.pose.position.y = estimator.tic[0].y();
    odometry.pose.pose.position.z = estimator.tic[0].z();
    Eigen::Quaterniond tmp_q{estimator.ric[0]};
    odometry.pose.pose.orientation.x = tmp_q.x();
    odometry.pose.pose.orientation.y = tmp_q.y();
    odometry.pose.pose.orientation.z = tmp_q.z();
    odometry.pose.pose.orientation.w = tmp_q.w();

    cout << "tf 8" << endl;
    pub_extrinsic->publish(odometry);
    cout << "tf 9" << endl;

}


// void pubTF(const Estimator &estimator, const std_msgs::msg::Header &header)
// {
//     if( estimator.solver_flag != Estimator::NON_LINEAR)
//         return;
//     std::shared_ptr<tf2_ros::TransformBroadcaster> br;
//     tf2::Transform transform;
//     tf2::Quaternion q;
//     // body frame
//     Eigen::Vector3d correct_t;
//     Eigen::Quaterniond correct_q;
//     correct_t = estimator.states_[WINDOW_SIZE].pos_end;
//     correct_q = estimator.states_[WINDOW_SIZE].rot_end;

//     transform.setOrigin(tf2::Vector3(correct_t(0),
//                                     correct_t(1),
//                                     correct_t(2)));
//     q.setW(correct_q.w());
//     q.setX(correct_q.x());
//     q.setY(correct_q.y());
//     q.setZ(correct_q.z());
//     transform.setRotation(q);
//     // br->sendTransform(tf2::StampedTransform(transform, header.stamp, "world", "body"));
//     br->sendTransform(tf2::StampedTransform(transform, header.stamp, "world", "body"));

//     // camera frame
//     transform.setOrigin(tf2::Vector3(estimator.tic[0].x(),
//                                     estimator.tic[0].y(),
//                                     estimator.tic[0].z()));
//     q.setW(Eigen::Quaterniond(estimator.ric[0]).w());
//     q.setX(Eigen::Quaterniond(estimator.ric[0]).x());
//     q.setY(Eigen::Quaterniond(estimator.ric[0]).y());
//     q.setZ(Eigen::Quaterniond(estimator.ric[0]).z());
//     transform.setRotation(q);
//     // br->sendTransform(tf2::StampedTransform(transform, header.stamp, "body", "camera"));
//     br->sendTransform(tf2::StampedTransform(transform, header.stamp, "body", "camera"));

    
//     nav_msgs::msg::Odometry odometry;
//     odometry.header = header;
//     odometry.header.frame_id = WORLD_FRAME_ID;
//     odometry.pose.pose.position.x = estimator.tic[0].x();
//     odometry.pose.pose.position.y = estimator.tic[0].y();
//     odometry.pose.pose.position.z = estimator.tic[0].z();
//     Eigen::Quaterniond tmp_q{estimator.ric[0]};
//     odometry.pose.pose.orientation.x = tmp_q.x();
//     odometry.pose.pose.orientation.y = tmp_q.y();
//     odometry.pose.pose.orientation.z = tmp_q.z();
//     odometry.pose.pose.orientation.w = tmp_q.w();
//     pub_extrinsic->publish(odometry);

// }

void pubKeyframe(const Estimator &estimator)
{
    if (!pub_keyframe_pose || !pub_keyframe_point)
        return;
    // pub camera pose, 2D-3D points of keyframe
    if (estimator.solver_flag == Estimator::NON_LINEAR && estimator.marginalization_flag == 0)
    {
        int i = WINDOW_SIZE - 2;
        //Eigen::Vector3d P = estimator.states_[i].pos_end + estimator.states_[i].rot_end * estimator.tic[0];
        Eigen::Vector3d P = estimator.states_[i].pos_end;
        Eigen::Quaterniond R = Eigen::Quaterniond(estimator.states_[i].rot_end);

        nav_msgs::msg::Odometry odometry;

        int sec_ts = (int)estimator.Headers[WINDOW_SIZE - 2];
        uint nsec_ts = (uint)((estimator.Headers[WINDOW_SIZE - 2] - sec_ts) * 1e9);
        odometry.header.stamp.sec = sec_ts;
        odometry.header.stamp.nanosec = nsec_ts;

        odometry.header.frame_id = WORLD_FRAME_ID;
        odometry.pose.pose.position.x = P.x();
        odometry.pose.pose.position.y = P.y();
        odometry.pose.pose.position.z = P.z();
        odometry.pose.pose.orientation.x = R.x();
        odometry.pose.pose.orientation.y = R.y();
        odometry.pose.pose.orientation.z = R.z();
        odometry.pose.pose.orientation.w = R.w();
        //printf("time: %f t: %f %f %f r: %f %f %f %f\n", odometry.header.stamp.sec, P.x(), P.y(), P.z(), R.w(), R.x(), R.y(), R.z());

        pub_keyframe_pose->publish(odometry);


        sensor_msgs::msg::PointCloud point_cloud;

        sec_ts = (int)estimator.Headers[WINDOW_SIZE - 2];
        nsec_ts = (uint)((estimator.Headers[WINDOW_SIZE - 2] - sec_ts) * 1e9);
        point_cloud.header.stamp.sec = sec_ts;
        point_cloud.header.stamp.nanosec = nsec_ts;

        point_cloud.header.frame_id = WORLD_FRAME_ID;
        for (auto &it_per_id : estimator.f_manager.feature)
        {
            int frame_size = it_per_id.feature_per_frame.size();
            if(it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.start_frame + frame_size - 1 >= WINDOW_SIZE - 2 && it_per_id.solve_flag == 1)
            {

                int imu_i = it_per_id.start_frame;
                Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
                Eigen::Vector3d w_pts_i = estimator.states_[imu_i].rot_end * (estimator.ric[0] * pts_i + estimator.tic[0])
                                      + estimator.states_[imu_i].pos_end;
                geometry_msgs::msg::Point32 p;
                p.x = w_pts_i(0);
                p.y = w_pts_i(1);
                p.z = w_pts_i(2);
                point_cloud.points.push_back(p);

                int imu_j = WINDOW_SIZE - 2 - it_per_id.start_frame;
                sensor_msgs::msg::ChannelFloat32 p_2d;
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].point.x());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].point.y());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].uv.x());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].uv.y());
                p_2d.values.push_back(it_per_id.feature_id);
                point_cloud.channels.push_back(p_2d);
            }

        }
        pub_keyframe_point->publish(point_cloud);
    }
}
