#ifndef FAST_LIVO_ROS1_RCLCPP_SHIM_HPP
#define FAST_LIVO_ROS1_RCLCPP_SHIM_HPP

#include <ros/ros.h>
#include <ros/console.h>
#include <XmlRpcValue.h>
#include <chrono>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace rclcpp
{

struct Logger
{
  const char *get_name() const { return ""; }
};
inline Logger get_logger(const std::string &) { return Logger(); }

class Time : public ros::Time
{
public:
  Time() : ros::Time() {}
  Time(uint32_t sec, uint32_t nsec) : ros::Time(sec, nsec) {}
  explicit Time(const ros::Time &stamp) : ros::Time(stamp) {}
  double seconds() const { return toSec(); }
};

class Duration : public ros::Duration
{
public:
  Duration() : ros::Duration() {}
  explicit Duration(double seconds) : ros::Duration(seconds) {}
  static Duration from_seconds(double seconds) { return Duration(seconds); }
};

using Rate = ros::Rate;

inline void init(int argc, char **argv) { ros::init(argc, argv, "laserMapping"); }
inline void shutdown() { ros::shutdown(); }
inline bool ok() { return ros::ok(); }

class Node;
inline void spin_some(const std::shared_ptr<Node> &) { ros::spinOnce(); }

class NodeOptions
{
public:
  NodeOptions &allow_undeclared_parameters(bool) { return *this; }
  NodeOptions &automatically_declare_parameters_from_overrides(bool) { return *this; }
};

template <typename T>
class Publisher
{
public:
  using SharedPtr = std::shared_ptr<Publisher<T>>;

  Publisher() = default;
  explicit Publisher(const ros::Publisher &publisher) : publisher_(publisher) {}

  void publish(const T &msg) const { publisher_.publish(msg); }

private:
  ros::Publisher publisher_;
};

class SubscriptionBase
{
public:
  using SharedPtr = std::shared_ptr<SubscriptionBase>;
  virtual ~SubscriptionBase() = default;
};

template <typename T>
class Subscription : public SubscriptionBase
{
public:
  using SharedPtr = std::shared_ptr<Subscription<T>>;
  explicit Subscription(const ros::Subscriber &subscriber) : subscriber_(subscriber) {}

private:
  ros::Subscriber subscriber_;
};

class TimerBase
{
public:
  using SharedPtr = std::shared_ptr<TimerBase>;
  explicit TimerBase(const ros::Timer &timer) : timer_(timer) {}

private:
  ros::Timer timer_;
};

class Clock
{
public:
  using SharedPtr = std::shared_ptr<Clock>;
  Time now() const { return Time(ros::Time::now()); }
};

struct KeepLast
{
  explicit KeepLast(size_t depth) : depth(depth) {}
  size_t depth;
};

class QoS
{
public:
  explicit QoS(const KeepLast &keep_last) : depth_(keep_last.depth) {}
  QoS &best_effort() { return *this; }
  QoS &durability_volatile() { return *this; }
  size_t depth() const { return depth_; }

private:
  size_t depth_;
};

class Parameter
{
public:
  Parameter(const ros::NodeHandle &nh, const std::string &key) : nh_(nh), key_(key) {}

  template <typename T>
  T get_value() const
  {
    T value{};
    getRosParam(nh_, key_, value);
    return value;
  }

private:
  template <typename T>
  static bool getRosParam(const ros::NodeHandle &nh, const std::string &key, T &value)
  {
    return nh.getParam(key, value);
  }

  static bool getRosParam(const ros::NodeHandle &nh, const std::string &key, std::vector<int64_t> &value)
  {
    XmlRpc::XmlRpcValue xml;
    if (!nh.getParam(key, xml) || xml.getType() != XmlRpc::XmlRpcValue::TypeArray) return false;
    value.clear();
    for (int i = 0; i < xml.size(); ++i)
      value.push_back(static_cast<int>(xml[i]));
    return true;
  }

  ros::NodeHandle nh_;
  std::string key_;
};

class Node : public std::enable_shared_from_this<Node>
{
public:
  using SharedPtr = std::shared_ptr<Node>;

  explicit Node(const std::string &, const NodeOptions & = NodeOptions()) : nh_("~") {}

  Logger get_logger() const { return Logger(); }
  Time now() const { return Time(ros::Time::now()); }
  Clock::SharedPtr get_clock() const { return std::make_shared<Clock>(); }
  ros::NodeHandle &ros_node_handle() { return nh_; }
  const ros::NodeHandle &ros_node_handle() const { return nh_; }

  template <typename T>
  T declare_parameter(const std::string &name, const T &default_value)
  {
    const std::string key = toRosParamName(name);
    T value = default_value;
    getParam(key, value);
    if (!nh_.hasParam(key)) setDefaultParam(key, value);
    return value;
  }

  template <typename T>
  bool get_parameter(const std::string &name, T &value) const
  {
    return getParam(toRosParamName(name), value);
  }

  Parameter get_parameter(const std::string &name) const { return Parameter(nh_, toRosParamName(name)); }

  template <typename T, typename CallbackT>
  typename Subscription<T>::SharedPtr create_subscription(const std::string &topic, const QoS &qos, CallbackT callback)
  {
    return std::make_shared<Subscription<T>>(nh_.subscribe<T>(topic, static_cast<uint32_t>(qos.depth()), callback));
  }

  template <typename T>
  typename Publisher<T>::SharedPtr create_publisher(const std::string &topic, size_t depth)
  {
    return std::make_shared<Publisher<T>>(nh_.advertise<T>(topic, static_cast<uint32_t>(depth)));
  }

  template <typename Rep, typename Period, typename CallbackT>
  TimerBase::SharedPtr create_wall_timer(const std::chrono::duration<Rep, Period> &period, CallbackT callback)
  {
    const double seconds = std::chrono::duration<double>(period).count();
    return std::make_shared<TimerBase>(nh_.createTimer(ros::Duration(seconds), [callback](const ros::TimerEvent &) { callback(); }));
  }

private:
  static std::string toRosParamName(std::string name)
  {
    for (char &ch : name)
      if (ch == '.') ch = '/';
    return name;
  }

  template <typename T>
  bool getParam(const std::string &key, T &value) const
  {
    return nh_.getParam(key, value);
  }

  bool getParam(const std::string &key, std::vector<int64_t> &value) const
  {
    XmlRpc::XmlRpcValue xml;
    if (!nh_.getParam(key, xml) || xml.getType() != XmlRpc::XmlRpcValue::TypeArray) return false;
    value.clear();
    for (int i = 0; i < xml.size(); ++i)
      value.push_back(static_cast<int>(xml[i]));
    return true;
  }

  template <typename T>
  void setDefaultParam(const std::string &key, const T &value)
  {
    nh_.setParam(key, value);
  }

  void setDefaultParam(const std::string &key, const std::vector<int64_t> &value)
  {
    std::vector<int> converted;
    converted.reserve(value.size());
    for (const int64_t item : value) converted.push_back(static_cast<int>(item));
    nh_.setParam(key, converted);
  }

  ros::NodeHandle nh_;
};

} // namespace rclcpp

namespace builtin_interfaces
{
namespace msg
{
using Time = ::ros::Time;
} // namespace msg
} // namespace builtin_interfaces

#define RCLCPP_INFO(logger, ...) ROS_INFO(__VA_ARGS__)
#define RCLCPP_WARN(logger, ...) ROS_WARN(__VA_ARGS__)
#define RCLCPP_ERROR(logger, ...) ROS_ERROR(__VA_ARGS__)
#define RCLCPP_INFO_ONCE(logger, ...) ROS_INFO_ONCE(__VA_ARGS__)
#define RCLCPP_WARN_ONCE(logger, ...) ROS_WARN_ONCE(__VA_ARGS__)
#define RCLCPP_ERROR_ONCE(logger, ...) ROS_ERROR_ONCE(__VA_ARGS__)
#define RCLCPP_ERROR_STREAM(logger, ...) ROS_ERROR_STREAM(__VA_ARGS__)

#endif
