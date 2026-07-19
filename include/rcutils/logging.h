#pragma once

#define RCUTILS_LOG_SEVERITY_WARN 30

inline int rcutils_logging_set_logger_level(const char *, int)
{
  return 0;
}
