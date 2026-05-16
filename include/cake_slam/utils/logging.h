// 模块功能：终端 stdout 日志工具。
// 避免重要运行信息走 ROS INFO 通道，保证 ros2 launch 时也能直接在屏幕观察。

#ifndef CAKE_SLAM_UTILS_LOGGING_H
#define CAKE_SLAM_UTILS_LOGGING_H

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "cake_slam/utils/color.h"

namespace cake_slam {

inline void ConsolePrintf(const char *fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  std::vprintf(fmt, args);
  va_end(args);
  std::printf("\n");
  std::fflush(stdout);
}

inline void PrintTimeTable(
    const char *title,
    const char *row_color,
    const std::vector<std::pair<std::string, double>> &rows,
    double current_total,
    double average_total)
{
  std::printf(BOLDBLUE "+-------------------------------------------------------------+" RESET "\n");
  std::printf(BOLDBLUE "| %-59s |" RESET "\n", title);
  std::printf(BOLDBLUE "+-------------------------------------------------------------+" RESET "\n");
  std::printf(BOLDBLUE "| %-29s | %-27s |" RESET "\n", "Algorithm Stage", "Time (secs)");
  std::printf(BOLDBLUE "+-------------------------------------------------------------+" RESET "\n");
  for (const auto &row : rows) {
    std::printf("%s| %-29s | %-27.6f |" RESET "\n",
                row_color, row.first.c_str(), row.second);
  }
  std::printf(BOLDBLUE "+-------------------------------------------------------------+" RESET "\n");
  std::printf("%s| %-29s | %-27.6f |" RESET "\n", row_color, "Current Total Time", current_total);
  std::printf("%s| %-29s | %-27.6f |" RESET "\n", row_color, "Average Total Time", average_total);
  std::printf(BOLDBLUE "+-------------------------------------------------------------+" RESET "\n");
  std::fflush(stdout);
}

inline void PrintKeyValueTable(
    const char *title,
    const char *row_color,
    const std::vector<std::pair<std::string, std::string>> &rows)
{
  std::printf(BOLDBLUE "+-------------------------------------------------------------+" RESET "\n");
  std::printf(BOLDBLUE "| %-59s |" RESET "\n", title);
  std::printf(BOLDBLUE "+-------------------------------------------------------------+" RESET "\n");
  std::printf(BOLDBLUE "| %-29s | %-27s |" RESET "\n", "Item", "Value");
  std::printf(BOLDBLUE "+-------------------------------------------------------------+" RESET "\n");
  for (const auto &row : rows) {
    std::printf("%s| %-29s | %-27s |" RESET "\n",
                row_color, row.first.c_str(), row.second.c_str());
  }
  std::printf(BOLDBLUE "+-------------------------------------------------------------+" RESET "\n");
  std::fflush(stdout);
}

} // namespace cake_slam

#define CAKE_INFO(fmt, ...) ::cake_slam::ConsolePrintf(fmt, ##__VA_ARGS__)
#define CAKE_INFO_THROTTLE_MS(interval_ms, fmt, ...)                                      \
  do {                                                                                    \
    static auto cake_last_print_time = std::chrono::steady_clock::time_point::min();      \
    const auto cake_now = std::chrono::steady_clock::now();                               \
    if (cake_last_print_time == std::chrono::steady_clock::time_point::min() ||           \
        std::chrono::duration_cast<std::chrono::milliseconds>(cake_now - cake_last_print_time).count() >= \
            static_cast<long long>(interval_ms)) {                                        \
      cake_last_print_time = cake_now;                                                    \
      ::cake_slam::ConsolePrintf(fmt, ##__VA_ARGS__);                                     \
    }                                                                                     \
  } while (0)

#endif // CAKE_SLAM_UTILS_LOGGING_H
