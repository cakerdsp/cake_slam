#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace visual_patch_budget
{
enum class Mode { Legacy, Sequential, Global };
enum class Score { ShiTomasi, Isotropic, LidarDirectional };

inline Mode parseMode(const std::string &name)
{
  if (name == "legacy") return Mode::Legacy;
  if (name == "sequential") return Mode::Sequential;
  if (name == "global") return Mode::Global;
  throw std::invalid_argument("vio.patch_budget_mode must be legacy, sequential or global");
}
inline Score parseScore(const std::string &name)
{
  if (name == "shi_tomasi") return Score::ShiTomasi;
  if (name == "isotropic") return Score::Isotropic;
  if (name == "lidar_directional") return Score::LidarDirectional;
  throw std::invalid_argument("vio.patch_score_mode must be shi_tomasi, isotropic or lidar_directional");
}

struct RuntimeSettings
{
  Mode mode = Mode::Legacy;
  Score score = Score::ShiTomasi;
  int points = 0;
  bool log_enabled = false;
  bool geometry_enabled = false;
};

// The master switch also isolates experimental settings from the baseline:
// disabled mode does not even parse score/mode names or apply its point limit.
inline RuntimeSettings resolveSettings(bool enabled, const std::string &mode, const std::string &score,
                                        int original_points, int module_points, bool log_enabled)
{
  RuntimeSettings out;
  out.points = original_points;
  if (!enabled) return out;
  out.mode = parseMode(mode);
  out.score = parseScore(score);
  out.points = module_points;
  out.log_enabled = log_enabled;
  out.geometry_enabled = out.score == Score::LidarDirectional;
  return out;
}

struct CameraScores
{
  int camera_id = -1; // Physical camera ID, independent of traversal order.
  std::vector<double> scores; // One score per camera/map-point group, descending.
};

// Exact maximum of the STATIC unit-cost primary-score sum, with lower/upper
// camera quotas. Execution charges both primary and fallback reference attempts
// against each quota, so fallback may consume slots planned for later primaries.
// With conditional fallback this allocation is a proxy, not an optimum of the
// realized accepted-point score, posterior information gain, or runtime.
inline std::vector<int> allocate(const std::vector<CameraScores> &cameras,
                                 int budget, int minimum, int maximum)
{
  if (budget < 0 || minimum < 0 || maximum < minimum)
    throw std::invalid_argument("invalid visual patch budget bounds");
  std::vector<int> quotas(cameras.size(), 0), caps(cameras.size(), 0);
  std::int64_t reserved = 0;
  for (size_t c = 0; c < cameras.size(); ++c)
  {
    for (size_t previous = 0; previous < c; ++previous)
      if (cameras[previous].camera_id == cameras[c].camera_id)
        throw std::invalid_argument("duplicate camera ID in visual patch budget");
    const auto &scores = cameras[c].scores;
    for (size_t i = 0; i < scores.size(); ++i)
      if (!std::isfinite(scores[i]) || scores[i] < 0 || (i && scores[i] > scores[i - 1]))
        throw std::invalid_argument("visual patch scores must be finite, nonnegative and descending");
    caps[c] = static_cast<int>(std::min(scores.size(), static_cast<size_t>(maximum)));
    quotas[c] = std::min(minimum, caps[c]);
    reserved += quotas[c];
  }
  if (reserved > budget)
    throw std::invalid_argument("visual patch budget cannot satisfy camera minimum quotas");
  for (int remaining = budget - static_cast<int>(reserved); remaining > 0; --remaining)
  {
    int best = -1;
    for (size_t c = 0; c < cameras.size(); ++c)
    {
      if (quotas[c] == caps[c]) continue;
      if (best < 0 || cameras[c].scores[quotas[c]] > cameras[best].scores[quotas[best]] ||
          (cameras[c].scores[quotas[c]] == cameras[best].scores[quotas[best]] &&
           cameras[c].camera_id < cameras[best].camera_id)) best = static_cast<int>(c);
    }
    if (best < 0) break;
    ++quotas[best];
  }
  return quotas;
}
} // namespace visual_patch_budget
