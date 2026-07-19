#ifndef FAST_LIVO_ROS1_PARAM_H
#define FAST_LIVO_ROS1_PARAM_H

#include <algorithm>
#include <string>
#include <ros/ros.h>

inline std::string ros1ParamName(std::string name)
{
  std::replace(name.begin(), name.end(), '.', '/');
  return name;
}

template <typename T>
T declareOrGetRosParam(ros::NodeHandle &nh, const std::string &name, const T &default_value)
{
  T value;
  const std::string ros_name = ros1ParamName(name);
  if (nh.getParam(ros_name, value)) return value;
  nh.setParam(ros_name, default_value);
  return default_value;
}

template <typename T>
void getRosParam(ros::NodeHandle &nh, const std::string &name, T &value)
{
  nh.param(ros1ParamName(name), value, value);
}

#endif
