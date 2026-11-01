#include "visual_patch_budget.h"

#include <chrono>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>

using visual_patch_budget::CameraScores;
using visual_patch_budget::allocate;

static void require(bool condition, const char *message)
{
  if (!condition) throw std::runtime_error(message);
}

static double exhaustive(const std::vector<CameraScores> &cameras, int budget, int lower, int upper)
{
  double best = -1;
  std::function<void(size_t, int, double)> visit = [&](size_t c, int used, double score) {
    if (c == cameras.size()) { best = std::max(best, score); return; }
    const int cap = std::min<int>(upper, cameras[c].scores.size());
    const int floor = std::min(lower, cap);
    double prefix = 0;
    for (int q = 0; q <= cap; ++q)
    {
      if (q >= floor && used + q <= budget) visit(c + 1, used + q, score + prefix);
      if (q < cap) prefix += cameras[c].scores[q];
    }
  };
  visit(0, 0, 0);
  return best;
}

int main()
{
  // A single false must restore baseline settings, even if experimental
  // options remain populated, malformed or incompatible with that path.
  const auto disabled = visual_patch_budget::resolveSettings(false, "invalid-mode", "invalid-score", 500, -7, true);
  require(disabled.mode == visual_patch_budget::Mode::Legacy && disabled.score == visual_patch_budget::Score::ShiTomasi,
          "disabled master switch parsed experimental modes");
  require(disabled.points == 500 && !disabled.log_enabled && !disabled.geometry_enabled,
          "disabled master switch leaked module budget/logging/geometry into baseline");
  const auto custom_baseline = visual_patch_budget::resolveSettings(false, "global", "lidar_directional", 321, 400, true);
  require(custom_baseline.points == 321, "disabled switch must preserve the original configured limit, not hardcode 500");
  const auto enabled = visual_patch_budget::resolveSettings(true, "global", "lidar_directional", 500, 400, true);
  require(enabled.mode == visual_patch_budget::Mode::Global && enabled.score == visual_patch_budget::Score::LidarDirectional &&
          enabled.points == 400 && enabled.log_enabled && enabled.geometry_enabled, "enabled settings not applied");
  const auto isotropic = visual_patch_budget::resolveSettings(true, "global", "isotropic", 500, 400, false);
  require(!isotropic.geometry_enabled, "isotropic ablation must not collect LiDAR geometry");
  std::mt19937 random(12345);
  for (int trial = 0; trial < 2000; ++trial)
  {
    const int count = 1 + random() % 4, lower = random() % 3, upper = lower + random() % 5;
    const int budget = random() % 16;
    std::vector<CameraScores> cameras(count);
    for (int c = 0; c < count; ++c)
    {
      cameras[c].camera_id = 10 + c;
      cameras[c].scores.resize(random() % 7);
      for (auto &score : cameras[c].scores) score = random() % 10; // Ties and zeros are intentional.
      std::sort(cameras[c].scores.begin(), cameras[c].scores.end(), std::greater<double>());
    }
    const double optimum = exhaustive(cameras, budget, lower, upper);
    if (optimum < 0)
    {
      bool rejected = false;
      try { allocate(cameras, budget, lower, upper); } catch (const std::invalid_argument &) { rejected = true; }
      require(rejected, "infeasible minimum quotas must be rejected");
      continue;
    }
    const auto quotas = allocate(cameras, budget, lower, upper);
    double score = 0;
    std::map<int, int> expected;
    for (int c = 0; c < count; ++c)
    {
      const int cap = std::min<int>(upper, cameras[c].scores.size());
      require(quotas[c] >= std::min(lower, cap) && quotas[c] <= cap, "camera bounds violated");
      score += std::accumulate(cameras[c].scores.begin(), cameras[c].scores.begin() + quotas[c], 0.0);
      expected[cameras[c].camera_id] = quotas[c];
    }
    require(score == optimum, "allocator differs from exhaustive optimum");
    require(std::accumulate(quotas.begin(), quotas.end(), 0) <= budget, "global budget exceeded");
    std::shuffle(cameras.begin(), cameras.end(), random);
    const auto permuted = allocate(cameras, budget, lower, upper);
    for (int c = 0; c < count; ++c)
      require(permuted[c] == expected[cameras[c].camera_id], "camera order changed quota, including ties");
  }
  require(allocate({}, 400, 50, 250).empty(), "empty camera set");
  std::vector<CameraScores> full(3);
  for (int c = 0; c < 3; ++c) { full[c].camera_id = c; full[c].scores.assign(250, 1.0); }
  require(allocate(full, 400, 50, 250) == std::vector<int>({250, 100, 50}), "stable equal-score tie handling");
  for (int invalid_case = 0; invalid_case < 3; ++invalid_case)
  {
    auto bad = full;
    if (invalid_case == 0) bad[0].scores[3] = std::numeric_limits<double>::quiet_NaN();
    if (invalid_case == 1) bad[1].scores[3] = 5.0;
    if (invalid_case == 2) bad[1].camera_id = bad[0].camera_id;
    bool rejected = false;
    try { allocate(bad, 400, 50, 250); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "malformed allocator input accepted");
  }
  const auto start = std::chrono::steady_clock::now();
  int checksum = 0;
  for (int repeat = 0; repeat < 10000; ++repeat) checksum += allocate(full, 400, 50, 250)[0];
  const double micros = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / 10000;
  std::cout << "2000 exhaustive/permutation cases passed; allocation-only mean_us=" << micros
            << " checksum=" << checksum << '\n';
}
