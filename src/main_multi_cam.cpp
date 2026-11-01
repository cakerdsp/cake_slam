#include "LIVMapper_multi_cam.h"
#include <exception>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "laserMapping");
  try
  {
    ros::NodeHandle nh("~");
    image_transport::ImageTransport it_(nh);
    LIVMapper mapper(nh, "laserMapping");
    mapper.initializeSubscribersAndPublishers(nh, it_);
    mapper.run(nh);
    ros::shutdown();
  }
  catch (const std::exception &error)
  {
    ROS_FATAL_STREAM("[cake_slam] " << error.what());
    ros::shutdown();
    return 1;
  }
  return 0;
}
