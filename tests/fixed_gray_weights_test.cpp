#include "fixed_gray_weights.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cgi = color_gray_init;

namespace
{

void require(bool condition, const std::string &message)
{
  if (!condition) throw std::runtime_error(message);
}

double channel(std::uint32_t &state)
{
  state = state * 1664525u + 1013904223u;
  return 30.0 + static_cast<double>((state >> 8) % 111u);
}

std::vector<cgi::ColorSample> makeSamples(const Eigen::Vector3d &weights,
                                         double gain = 1.0)
{
  std::vector<cgi::ColorSample> samples;
  std::uint32_t random_state = 1234567u;
  for (int region = 0; region < 12; ++region)
  {
    for (int index = 0; index < 24; ++index)
    {
      cgi::ColorSample sample;
      // Separate statements avoid dependence on function-argument evaluation
      // order in deterministic synthetic test data.
      sample.rgb[0] = channel(random_state);
      sample.rgb[1] = channel(random_state);
      sample.rgb[2] = channel(random_state);
      sample.target = gain * weights.dot(sample.rgb);
      sample.region = region;
      samples.push_back(sample);
    }
  }
  return samples;
}

void testNoiselessRecoveryAndGainElimination()
{
  const Eigen::Vector3d expected(0.16, 0.31, 0.53);
  for (double gain : {0.5, 1.0, 1.6})
  {
    const cgi::WeightFit fit = cgi::fitWeights(makeSamples(expected, gain));
    require(fit.accepted, "noiseless valid model must be accepted: " + fit.reason);
    require(fit.reason == "validated_channel_weights", "test must exercise learned weights");
    require((fit.weights - expected).norm() < 1.0e-9,
            "overall brightness must not contaminate normalized channel weights");
    require(fit.validation_rmse < 1.0e-8, "noiseless model must fit held-out regions");
    require(std::abs(fit.weights.sum() - 1.0) < 1.0e-12, "weights must sum to one");
  }
}

void testDefaultWeightsRemainUnchangedWhenSupported()
{
  const cgi::WeightFit fit = cgi::fitWeights(makeSamples(cgi::defaultWeights(), 1.4));
  require(fit.accepted, "default weights supported by colorful independent regions must pass");
  require(fit.reason == "default_weights_supported", "unneeded adjustment must not be installed");
  require((fit.weights - cgi::defaultWeights()).norm() == 0.0,
          "retain the exact default coefficients");
  require(fit.baseline_rmse < 1.0e-8,
          "default baseline must be allowed its training-fitted brightness scalar");
}

void testExactSmallCorrectionIsNotSuppressedByEngineeringMargin()
{
  const Eigen::Vector3d expected =
      cgi::defaultWeights() + Eigen::Vector3d(0.002, -0.001, -0.001);
  const cgi::WeightFit fit = cgi::fitWeights(makeSamples(expected));
  require(fit.accepted && fit.reason == "validated_channel_weights",
          "an exactly verified small change must not acquire a selection bias");
  require((fit.weights - expected).norm() < 1.0e-9,
          "noiseless recovery must also hold below the noisy-fit improvement margin");
}

void testNegativeSolutionIsRejectedRatherThanClipped()
{
  const Eigen::Vector3d invalid(-0.2, 0.7, 0.5);
  const cgi::WeightFit fit = cgi::fitWeights(makeSamples(invalid));
  require(!fit.accepted, "negative coefficients must not be silently clipped into acceptance");
  require(fit.reason == "negative_or_invalid_weights", "negative-weight rejection must be explicit");
  require((fit.weights - cgi::defaultWeights()).norm() == 0.0,
          "rejected fit must not leak proposed weights");
}

void testUninformativeAndSingleRegionSamplesAreRejected()
{
  auto neutral = makeSamples(Eigen::Vector3d(0.2, 0.5, 0.3));
  for (std::size_t index = 0; index < neutral.size(); ++index)
  {
    const double value = 30.0 + static_cast<double>(index % 111);
    neutral[index].rgb = Eigen::Vector3d::Constant(value);
    neutral[index].target = value;
  }
  require(!cgi::fitWeights(neutral).accepted, "many neutral samples do not establish color rank");

  auto one_region = makeSamples(Eigen::Vector3d(0.2, 0.5, 0.3));
  for (auto &sample : one_region) sample.region = 7;
  const cgi::WeightFit region_fit = cgi::fitWeights(one_region);
  require(!region_fit.accepted && region_fit.reason == "insufficient_independent_regions",
          "one densely sampled patch must not masquerade as independent validation");

  auto small = makeSamples(Eigen::Vector3d(0.2, 0.5, 0.3));
  small.resize(20);
  require(!cgi::fitWeights(small).accepted, "too few samples must be rejected");

  auto weak_validation = makeSamples(Eigen::Vector3d(0.2, 0.5, 0.3));
  for (auto &sample : weak_validation)
  {
    if (sample.region % 3 != 2) continue;
    sample.rgb = Eigen::Vector3d::Constant(sample.rgb[0]);
    sample.target = sample.rgb[0];
  }
  require(!cgi::fitWeights(weak_validation).accepted,
          "colorful training alone is insufficient when held-out regions lack color rank");
}

void testRobustTrainingAndIndependentHoldoutRejection()
{
  const Eigen::Vector3d expected(0.16, 0.31, 0.53);
  auto corrupted_training = makeSamples(expected, 1.2);
  for (std::size_t index = 0; index < corrupted_training.size(); ++index)
  {
    auto &sample = corrupted_training[index];
    if (sample.region % 3 != 2 && index % 7 == 0) sample.target += 60.0;
  }
  const cgi::WeightFit robust = cgi::fitWeights(corrupted_training);
  require(robust.accepted, "sparse training outliers must not dominate: " + robust.reason);
  require((robust.weights - expected).norm() < 0.02,
          "robust estimate must stay near the true relative weights");

  auto inconsistent_holdout = makeSamples(expected);
  for (auto &sample : inconsistent_holdout)
    if (sample.region % 3 == 2) sample.target = 255.0 - sample.target;
  const cgi::WeightFit rejected = cgi::fitWeights(inconsistent_holdout);
  require(!rejected.accepted, "training fit cannot override incompatible unseen regions");
  require(rejected.reason == "held_out_mapping_mismatch",
          "held-out mapping mismatch must be distinguished from training success");
}

void testNonFiniteAndRangeInputsAreRejected()
{
  auto invalid = makeSamples(Eigen::Vector3d(0.2, 0.5, 0.3));
  invalid[0].rgb[1] = std::numeric_limits<double>::quiet_NaN();
  require(!cgi::fitWeights(invalid).accepted, "non-finite RGB must be rejected");
  invalid = makeSamples(Eigen::Vector3d(0.2, 0.5, 0.3));
  invalid[0].target = std::numeric_limits<double>::infinity();
  require(!cgi::fitWeights(invalid).accepted, "non-finite target must be rejected");
  invalid = makeSamples(Eigen::Vector3d(0.2, 0.5, 0.3));
  invalid[0].rgb[0] = 256.0;
  require(!cgi::fitWeights(invalid).accepted, "wrong intensity range must be explicit");
}

} // namespace

int main()
{
  try
  {
    testNoiselessRecoveryAndGainElimination();
    testDefaultWeightsRemainUnchangedWhenSupported();
    testExactSmallCorrectionIsNotSuppressedByEngineeringMargin();
    testNegativeSolutionIsRejectedRatherThanClipped();
    testUninformativeAndSingleRegionSamplesAreRejected();
    testRobustTrainingAndIndependentHoldoutRejection();
    testNonFiniteAndRangeInputsAreRejected();
    std::cout << "fixed_gray_weights_test passed\n";
  }
  catch (const std::exception &error)
  {
    std::cerr << "fixed_gray_weights_test failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
