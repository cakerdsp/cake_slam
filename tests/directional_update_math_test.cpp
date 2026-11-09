#include "directional_update.h"
#include "estimator_covariance.h"

#include <Eigen/SVD>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace
{
template <typename Left, typename Right>
bool near(const Eigen::MatrixBase<Left> &lhs, const Eigen::MatrixBase<Right> &rhs,
          double tolerance = 1.0e-9)
{
  return lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols() &&
         (lhs - rhs).cwiseAbs().maxCoeff() <= tolerance;
}

void testWeakStrongAndNullDirections()
{
  const Eigen::Matrix3d prior = Eigen::Vector3d(4.0, 1.0, 0.25).asDiagonal();
  const Eigen::Matrix3d information = Eigen::Vector3d(0.01, 2.0, 0.0).asDiagonal();
  const Eigen::Vector3d vector(0.2, -0.3, 0.0);
  directional_update::Result enhanced;
  assert(directional_update::filterInformation(prior, information, vector, 1.0, 0.5, enhanced));
  const double lambda = 0.04;
  const double fraction = 1.0 - (lambda / (1.0 + lambda)) / 0.5;
  const double w = 1.0 + fraction * fraction;
  const Eigen::Matrix3d expected_information = Eigen::Vector3d(0.01 * w, 2.0, 0.0).asDiagonal();
  const Eigen::Matrix3d expected_noise = Eigen::Vector3d(0.01 * w * w, 2.0, 0.0).asDiagonal();
  assert(near(enhanced.eigenvalues, Eigen::Vector3d(0.0, lambda, 2.0)));
  assert(near(enhanced.information, expected_information));
  assert(near(enhanced.noise_information, expected_noise));
  assert(near(enhanced.information_vector, Eigen::Vector3d(0.2 * w, -0.3, 0.0)));
  assert(enhanced.active_rank == 2 && enhanced.enhanced_rank == 1);
  const auto update = estimator_covariance::update(prior, enhanced.information,
      enhanced.information_vector, Eigen::Vector3d::Zero(), {0, 1, 2}, enhanced.noise_information);
  assert(near(update.correction, Eigen::Vector3d(4.0 * 0.2 * w / (1.0 + w * lambda), -0.1, 0.0)));
  const Eigen::Matrix3d expected_covariance = Eigen::Vector3d(
      4.0 * (1.0 + w * w * lambda) / std::pow(1.0 + w * lambda, 2), 1.0 / 3.0, 0.25).asDiagonal();
  assert(near(update.covariance, expected_covariance));
  Eigen::MatrixXd posterior;
  std::string error;
  assert(directional_update::posteriorCovariance(prior, enhanced.information,
      enhanced.noise_information, posterior, error));
  assert(near(posterior, expected_covariance));

  // Exactly at the cutoff, the original update and noise are retained.
  const Eigen::Matrix2d boundary_information = Eigen::Vector2d(1.0, 0.0).asDiagonal();
  assert(directional_update::filterInformation(Eigen::Matrix2d::Identity(), boundary_information,
      Eigen::Vector2d(1.0, 0.0), 1.0, 0.5, enhanced));
  assert(near(enhanced.information, boundary_information));
  assert(near(enhanced.noise_information, boundary_information));
  assert(enhanced.enhanced_rank == 0);
}

void testAgainstMeasurementSpaceSvd()
{
  using Eigen::MatrixXd;
  using Eigen::VectorXd;
  // Fixed-size and dynamic implementations, rank-deficient H, correlated
  // observation noise, prior displacement and frozen state rows.
  for (int n : {6, 18, 19, 20, 21, 28})
  {
    const int m = n - 3;
    const MatrixXd a = MatrixXd::Random(n, n), z = MatrixXd::Random(m, m);
    const MatrixXd prior = a * a.transpose() + 0.5 * MatrixXd::Identity(n, n);
    const MatrixXd noise = z * z.transpose() + 0.2 * MatrixXd::Identity(m, m);
    const MatrixXd h = 0.15 * MatrixXd::Random(m, n);
    const VectorXd residual = VectorXd::Random(m), delta = VectorXd::Random(n);
    const MatrixXd information = h.transpose() * noise.llt().solve(h);
    const VectorXd vector = h.transpose() * noise.llt().solve(residual);
    const MatrixXd l = prior.llt().matrixL(), r = noise.llt().matrixL();
    const MatrixXd whitening = r.triangularView<Eigen::Lower>().solve(MatrixXd::Identity(m, m));
    const MatrixXd whitened_h = whitening * h * l;
    const Eigen::JacobiSVD<MatrixXd> svd(whitened_h, Eigen::ComputeThinU | Eigen::ComputeThinV);

    for (double boost : {0.0, 1.0})
    {
      directional_update::Result enhanced;
      assert(directional_update::filterInformation(prior, information, vector, boost, 0.5, enhanced));
      VectorXd gain_singular_values(m), score_singular_values(m);
      for (int i = 0; i < m; ++i)
      {
        const double s = svd.singularValues()[i], lambda = s * s;
        const double fraction = std::max(0.0, 1.0 - (lambda / (1.0 + lambda)) / 0.5);
        const double w = 1.0 + boost * fraction * fraction;
        gain_singular_values[i] = w * s / (1.0 + w * lambda);
        score_singular_values[i] = w * s;
      }
      const MatrixXd dense_gain = l * svd.matrixV() * gain_singular_values.asDiagonal() *
                                  svd.matrixU().transpose() * whitening;
      const MatrixXd score_rhs = svd.matrixV() * score_singular_values.asDiagonal() *
                                 svd.matrixU().transpose() * whitening;
      const MatrixXd score = l.transpose().triangularView<Eigen::Upper>().solve(score_rhs);
      assert(near(enhanced.information, score * h));
      assert(near(enhanced.information_vector, score * residual));
      assert(near(enhanced.noise_information, score * noise * score.transpose()));
      for (bool freeze : {false, true})
      {
        std::vector<int> active;
        MatrixXd gain = dense_gain;
        VectorXd selected_delta = delta;
        for (int i = 0; i < n; ++i)
        {
          if (!freeze || i % 3 != 1) active.push_back(i);
          else { gain.row(i).setZero(); selected_delta[i] = 0.0; }
        }
        const auto update = estimator_covariance::update(prior, enhanced.information,
            enhanced.information_vector, delta, active, enhanced.noise_information);
        const MatrixXd identity = MatrixXd::Identity(n, n), transition = identity - gain * h;
        const MatrixXd covariance = transition * prior * transition.transpose() + gain * noise * gain.transpose();
        assert(near(update.gain_times_jacobian, gain * h));
        assert(near(update.correction, selected_delta + gain * (residual - h * delta)));
        assert(near(update.covariance, covariance));
        if (freeze)
          for (int i = 1; i < n; i += 3)
            for (int j = 1; j < n; j += 3)
              assert(std::fabs(update.covariance(i, j) - prior(i, j)) < 1.0e-9);
        if (boost == 0.0)
        {
          const auto original = estimator_covariance::update(prior, information, vector, delta, active);
          assert(near(update.correction, original.correction));
          assert(near(update.covariance, original.covariance));
        }
      }
    }
  }
}

void testSubspaceEmbeddingAndIndependentFactors()
{
  using Eigen::MatrixXd;
  using Eigen::VectorXd;
  constexpr int n = 7, q = 3, m = 4;
  const MatrixXd a = MatrixXd::Random(n, n);
  const MatrixXd prior = a * a.transpose() + MatrixXd::Identity(n, n);
  MatrixXd embed = MatrixXd::Zero(n, q);
  embed(0, 0) = embed(2, 1) = embed(5, 2) = 1.0;
  const MatrixXd local_h = 0.15 * MatrixXd::Random(m, q);
  const MatrixXd h = local_h * embed.transpose();
  const MatrixXd noise = 0.4 * MatrixXd::Identity(m, m);
  const VectorXd residual = VectorXd::Random(m), delta = VectorXd::Random(n);
  directional_update::Result enhanced;
  assert(directional_update::filterInformation(embed.transpose() * prior * embed,
      local_h.transpose() * noise.inverse() * local_h,
      local_h.transpose() * noise.inverse() * residual, 1.0, 0.5, enhanced));
  const MatrixXd local_transform = enhanced.prior_sqrt_lower.transpose().triangularView<Eigen::Upper>().solve(
      enhanced.normalized_eigenvectors * enhanced.weights.asDiagonal() *
      enhanced.normalized_eigenvectors.transpose() * enhanced.prior_sqrt_lower.transpose());
  const MatrixXd score = embed * local_transform * local_h.transpose() * noise.inverse();
  const MatrixXd h_extra = 0.1 * MatrixXd::Random(2, n);
  const MatrixXd noise_extra = 0.7 * MatrixXd::Identity(2, 2);
  const VectorXd residual_extra = VectorXd::Random(2);
  const MatrixXd info_extra = h_extra.transpose() * noise_extra.inverse() * h_extra;
  const MatrixXd info = embed * enhanced.information * embed.transpose() + info_extra;
  // Match VIO's J1 -> J2 replacement after adding independent factors.
  const MatrixXd score_noise = info + embed *
      (enhanced.noise_information - enhanced.information) * embed.transpose();
  const VectorXd vector = embed * enhanced.information_vector +
                          h_extra.transpose() * noise_extra.inverse() * residual_extra;
  const MatrixXd solve = (prior.inverse() + info).inverse();
  MatrixXd gain = solve * score;
  MatrixXd gain_extra = solve * h_extra.transpose() * noise_extra.inverse();
  gain.row(1).setZero(); gain_extra.row(1).setZero();
  VectorXd selected_delta = delta; selected_delta[1] = 0.0;
  const auto update = estimator_covariance::update(prior, info, vector, delta,
      {0, 2, 3, 4, 5, 6}, score_noise);
  const MatrixXd transition = MatrixXd::Identity(n, n) - gain * h - gain_extra * h_extra;
  assert(near(update.correction, selected_delta + gain * (residual - h * delta) +
      gain_extra * (residual_extra - h_extra * delta)));
  assert(near(update.covariance, transition * prior * transition.transpose() +
      gain * noise * gain.transpose() + gain_extra * noise_extra * gain_extra.transpose()));
}
} // namespace

int main()
{
  testWeakStrongAndNullDirections();
  testAgainstMeasurementSpaceSvd();
  testSubspaceEmbeddingAndIndependentFactors();
  std::cout << "directional_update_math_test passed\n";
  return 0;
}
