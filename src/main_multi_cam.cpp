#include "LIVMapper_multi_cam.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "laserMapping");
  ros::NodeHandle nh("~");
  image_transport::ImageTransport it_(nh);
  LIVMapper mapper(nh, "laserMapping");
  mapper.initializeSubscribersAndPublishers(nh, it_);
  mapper.run(nh);
  ros::shutdown();
  return 0;
}
