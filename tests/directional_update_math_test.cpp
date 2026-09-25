#include "directional_update.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

namespace
{

bool near(const Eigen::MatrixXd &lhs, const Eigen::MatrixXd &rhs, double tolerance = 1.0e-10)
{
  return lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols() &&
         (lhs - rhs).cwiseAbs().maxCoeff() <= tolerance;
}

bool near(const Eigen::VectorXd &lhs, const Eigen::VectorXd &rhs, double tolerance = 1.0e-10)
{
  return lhs.size() == rhs.size() && (lhs - rhs).cwiseAbs().maxCoeff() <= tolerance;
}

} // namespace

int main()
{
  {
    const Eigen::Matrix3d prior = (Eigen::Vector3d(4.0, 1.0, 0.25)).asDiagonal();
    const Eigen::Matrix3d information = (Eigen::Vector3d(0.25, 1.0, 0.0)).asDiagonal();
    const Eigen::Vector3d information_vector(2.0, -3.0, 7.0);
    directional_update::Result filtered;
    assert(directional_update::filterInformation(
        prior, information, information_vector, 0.05, 0.50, filtered));
    assert(near(filtered.eigenvalues, Eigen::Vector3d(0.0, 1.0, 1.0)));
    assert(near(filtered.variance_reductions, Eigen::Vector3d(0.0, 0.5, 0.5)));
    assert(near(filtered.information, information));
    assert(near(filtered.information_vector, Eigen::Vector3d(2.0, -3.0, 0.0)));
    assert(filtered.active_rank == 2);
    assert(filtered.full_rank == 2);

    Eigen::MatrixXd posterior;
    std::string error;
    assert(directional_update::posteriorCovariance(prior, filtered.information, posterior, error));
    assert(near(posterior, (Eigen::Vector3d(2.0, 0.5, 0.25)).asDiagonal()));
  }

  {
    constexpr double reduction = 0.275;
    const double lambda = reduction / (1.0 - reduction);
    const Eigen::Matrix2d prior = Eigen::Matrix2d::Identity();
    const Eigen::Matrix2d information = (Eigen::Vector2d(lambda, 0.0)).asDiagonal();
    const Eigen::Vector2d information_vector(4.0, 9.0);
    directional_update::Result filtered;
    assert(directional_update::filterInformation(
        prior, information, information_vector, 0.05, 0.50, filtered));
    assert(std::fabs(filtered.weights[1] - 0.5) < 1.0e-12);
    assert(std::fabs(filtered.information(0, 0) - 0.5 * lambda) < 1.0e-12);
    assert(std::fabs(filtered.information_vector[0] - 2.0) < 1.0e-12);
    assert(std::fabs(filtered.information_vector[1]) < 1.0e-12);
  }

  {
    Eigen::Matrix3d prior;
    prior << 2.0, 0.3, -0.1,
             0.3, 1.2,  0.2,
            -0.1, 0.2,  0.8;
    const Eigen::Vector3d j1(1.0, 2.0, -0.5);
    const Eigen::Vector3d j2(-0.4, 0.7, 1.5);
    const Eigen::Matrix3d component1 = j1 * j1.transpose();
    const Eigen::Matrix3d component2 = j2 * j2.transpose();
    const Eigen::Matrix3d information = component1 + component2;
    directional_update::Result filtered;
    assert(directional_update::filterInformation(
        prior, information, Eigen::Vector3d::Zero(), 0.05, 0.50, filtered));
    Eigen::MatrixXd filtered_component1;
    Eigen::MatrixXd filtered_component2;
    std::string error;
    assert(directional_update::filterInformationComponent(
        component1, filtered, filtered_component1, error));
    assert(directional_update::filterInformationComponent(
        component2, filtered, filtered_component2, error));
    assert(near(filtered_component1 + filtered_component2, filtered.information, 1.0e-9));
  }

  // Compare every fixed-size specialization and the dynamic fallback with
  // the original normalized-coordinate likelihood and component transforms.
  for (int n : {6, 18, 19, 20, 21, 28})
  {
    const Eigen::MatrixXd a = Eigen::MatrixXd::Random(n, n);
    const Eigen::MatrixXd prior = a * a.transpose() + 0.5 * Eigen::MatrixXd::Identity(n, n);
    const Eigen::MatrixXd h1 = 0.12 * Eigen::MatrixXd::Random(n, n);
    const Eigen::MatrixXd h2 = 0.12 * Eigen::MatrixXd::Random(n, n);
    const Eigen::MatrixXd c1 = h1.transpose() * h1, c2 = h2.transpose() * h2;
    const Eigen::MatrixXd information = c1 + c2;
    const Eigen::VectorXd vector = Eigen::VectorXd::Random(n);
    directional_update::Result filtered;
    assert(directional_update::filterInformation(prior, information, vector, 0.05, 0.50, filtered, true));
    const Eigen::MatrixXd l = prior.llt().matrixL();
    const Eigen::MatrixXd normalized = l.transpose() * information * l;
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(
        directional_update::symmetrize(normalized));
    const Eigen::MatrixXd u = solver.eigenvectors();
    Eigen::VectorXd weights(n), eigenvalues = solver.eigenvalues().cwiseMax(0.0);
    for (int i = 0; i < n; ++i)
    {
      const double rho = eigenvalues[i] / (1.0 + eigenvalues[i]);
      weights[i] = std::clamp((rho - 0.05) / 0.45, 0.0, 1.0);
    }
    const Eigen::MatrixXd normalized_h = u * (weights.array() * eigenvalues.array()).matrix().asDiagonal() * u.transpose();
    const Eigen::MatrixXd left = l.transpose().triangularView<Eigen::Upper>().solve(normalized_h);
    const Eigen::MatrixXd expected_h = l.transpose().triangularView<Eigen::Upper>().solve(left.transpose()).transpose();
    const Eigen::VectorXd normalized_g = u * weights.asDiagonal() * u.transpose() * l.transpose() * vector;
    const Eigen::VectorXd expected_g = l.transpose().triangularView<Eigen::Upper>().solve(normalized_g);
    assert(near(filtered.information, expected_h, 1.0e-9));
    assert(near(filtered.information_vector, expected_g, 1.0e-9));
    assert(near(filtered.weights, weights, 1.0e-9));
    Eigen::MatrixXd filtered_c1, filtered_c2, uncached_c1;
    std::string error;
    assert(directional_update::filterInformationComponent(c1, filtered, filtered_c1, error));
    assert(directional_update::filterInformationComponent(c2, filtered, filtered_c2, error));
    assert(near(Eigen::MatrixXd(filtered_c1 + filtered_c2), filtered.information, 1.0e-9));
    const Eigen::MatrixXd gate = u * weights.cwiseSqrt().asDiagonal() * u.transpose();
    const Eigen::MatrixXd normalized_c1 = gate * l.transpose() * c1 * l * gate;
    const Eigen::MatrixXd left_c1 = l.transpose().triangularView<Eigen::Upper>().solve(normalized_c1);
    const Eigen::MatrixXd expected_c1 = l.transpose().triangularView<Eigen::Upper>().solve(left_c1.transpose()).transpose();
    assert(near(filtered_c1, expected_c1, 1.0e-9));
    filtered.component_transform.resize(0, 0);
    assert(directional_update::filterInformationComponent(c1, filtered, uncached_c1, error));
    assert(near(filtered_c1, uncached_c1, 1.0e-9));
  }

  std::cout << "directional_update_math_test passed\n";
  return 0;
}
