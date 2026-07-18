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

  std::cout << "directional_update_math_test passed\n";
  return 0;
}
