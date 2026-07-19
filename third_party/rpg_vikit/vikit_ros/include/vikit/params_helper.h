#ifndef ROS_PARAMS_HELPER_H_
#define ROS_PARAMS_HELPER_H_

#include <algorithm>
#include <string>
#include <type_traits>
#include <ros/ros.h>

namespace vk {

inline std::string rosParamName(std::string name)
{
  std::replace(name.begin(), name.end(), '.', '/');
  return name;
}

inline bool hasParam(const ros::NodeHandle &nh, const std::string &name)
{
  return nh.hasParam(rosParamName(name));
}

template<typename T>
T getParam(const ros::NodeHandle &nh, const std::string &name, const T &default_value)
{
  T value;
  if (nh.getParam(rosParamName(name), value)) return value;
  return default_value;
}

template<typename T>
typename std::enable_if<!std::is_same<T, std::string>::value, T>::type
getParam(const ros::NodeHandle &nh, const std::string &name)
{
  T value;
  if (!nh.getParam(rosParamName(name), value))
  {
    ROS_ERROR("Cannot find required parameter: %s", rosParamName(name).c_str());
    return T();
  }
  return value;
}

template<typename T>
typename std::enable_if<std::is_same<T, std::string>::value, T>::type
getParam(const ros::NodeHandle &nh, const std::string &name)
{
  std::string value;
  if (!nh.getParam(rosParamName(name), value))
  {
    ROS_ERROR("Cannot find required parameter: %s", rosParamName(name).c_str());
    return std::string();
  }
  return value;
}

} // namespace vk

#endif
