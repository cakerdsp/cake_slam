/*
 * output_helper.h
 *
 *  Created on: Jan 20, 2013
 *      Author: cforster
 *  Update on: Feb 01, 2025
 *      Author: StrangeFly
 */

#ifndef VIKIT_OUTPUT_HELPER_H_
#define VIKIT_OUTPUT_HELPER_H_

#include <string>
#include <ros/ros.h>
#include <Eigen/Core>
#include <vikit/sophus_compat.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/TransformStamped.h>
#include <visualization_msgs/Marker.h>

namespace vk {
namespace output_helper {

using namespace std;
using namespace Eigen;

void
publishTfTransform      (const fast_livo::SE3d& T, const ros::Time& stamp,
                         const string& frame_id, const string& child_frame_id,
                         tf2_ros::TransformBroadcaster& br);

void
publishPointMarker      (const ros::Publisher pub,
                         const Vector3d& pos,
                         const string& ns,
                         const ros::Time& timestamp,
                         int id,
                         int action,
                         double marker_scale,
                         const Vector3d& color,
                         ros::Duration lifetime = ros::Duration(0,0));

void
publishLineMarker       (const ros::Publisher pub,
                         const Vector3d& start,
                         const Vector3d& end,
                         const string& ns,
                         const ros::Time& timestamp,
                         int id,
                         int action,
                         double marker_scale,
                         const Vector3d& color,
                         ros::Duration lifetime = ros::Duration(0,0));

void
publishArrowMarker      (const ros::Publisher pub,
                         const Vector3d& pos,
                         const Vector3d& dir,
                         double scale,
                         const string& ns,
                         const ros::Time& timestamp,
                         int id,
                         int action,
                         double marker_scale,
                         const Vector3d& color);

void
publishHexacopterMarker (const ros::Publisher pub,
                         const string& frame_id,
                         const string& ns,
                         const ros::Time& timestamp,
                         int id,
                         int action,
                         double marker_scale,
                         const Vector3d& color);

void
publishCameraMarker(const ros::Publisher pub,
                    const string& frame_id,
                    const string& ns,
                    const ros::Time& timestamp,
                    int id,
                    double marker_scale,
                    const Vector3d& color);
void
publishFrameMarker      (const ros::Publisher pub,
                        const Matrix3d& rot,
                        const Vector3d& pos,
                        const string& ns,
                        const ros::Time& timestamp,
                        int id,
                        int action,
                        double marker_scale,
                        ros::Duration lifetime = ros::Duration(0,0));


} // namespace output_helper
} // namespace vk


#endif /* VIKIT_OUTPUT_HELPER_H_ */
