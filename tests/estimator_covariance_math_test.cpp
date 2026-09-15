#include "estimator_covariance.h"
#include <iostream>
#include <numeric>

namespace ec = estimator_covariance;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Matrix3d;
using Eigen::Vector3d;

void require(bool condition, const char *message)
{
  if (!condition) throw std::runtime_error(message);
}

void near(const MatrixXd &actual, const MatrixXd &expected, double tol, const char *message)
{
  require(actual.rows() == expected.rows() && actual.cols() == expected.cols(), message);
  require((actual - expected).norm() <= tol * std::max(1.0, expected.norm()), message);
}

Matrix3d expSO3(const Vector3d &v)
{
  if (v.norm() < 1.0e-15) return Matrix3d::Identity();
  return Eigen::AngleAxisd(v.norm(), v.normalized()).toRotationMatrix();
}

Vector3d logSO3(const Matrix3d &r)
{
  const Eigen::AngleAxisd a(r);
  return a.angle() * a.axis();
}

void testGeometry()
{
  const Vector3d d(0.4, -0.2, 0.3), p_i(1.2, -0.8, 3.0);
  const Matrix3d r = expSO3(d), r_il = expSO3(Vector3d(-0.3, 0.5, 0.1));
  constexpr double eps = 1.0e-6;
  Matrix3d reset, prior_log;
  ec::PointJacobian point_j;
  for (int k = 0; k < 3; ++k)
  {
    const Vector3d e = eps * Vector3d::Unit(k);
    reset.col(k) = (logSO3(expSO3(-d) * expSO3(d + e)) -
                    logSO3(expSO3(-d) * expSO3(d - e))) / (2.0 * eps);
    prior_log.col(k) = (logSO3(r * expSO3(e)) - logSO3(r * expSO3(-e))) / (2.0 * eps);
    point_j.col(k) = (r * expSO3(e) * p_i - r * expSO3(-e) * p_i) / (2.0 * eps);
    point_j.col(k + 3) = Vector3d::Unit(k);
  }
  near(ec::rightJacobian(d), reset, 1.0e-8, "right reset finite difference");
  near(ec::rightJacobian(d) * prior_log, Matrix3d::Identity(), 1.0e-8, "prior tangent transport");
  near(ec::pointJacobian(r, p_i), point_j, 1.0e-8, "world point finite difference");
  const MatrixXd a = MatrixXd::Random(6, 6);
  const ec::Matrix6 p = a * a.transpose() + ec::Matrix6::Identity();
  const Matrix3d beam = Vector3d(0.01, 0.03, 0.05).asDiagonal();
  const Matrix3d q = ec::pointCovariance(r, r_il, p_i, beam, p);
  const Matrix3d world_rotation = expSO3(Vector3d(0.5, 0.2, -0.4));
  ec::Matrix6 change_world = ec::Matrix6::Identity();
  change_world.bottomRightCorner<3, 3>() = world_rotation;
  near(ec::pointCovariance(world_rotation * r, r_il, p_i, beam,
                          change_world * p * change_world.transpose()),
       world_rotation * q * world_rotation.transpose(), 1.0e-10, "point covariance world equivariance");

  // Reference residual -g^T R_ref (point - position), tested in mixed coordinates.
  const Matrix3d r_ref_i = r_il;
  const Matrix3d r_rw = r_ref_i * r.transpose();
  const Eigen::RowVector3d photo(2.1, -1.4, 0.8);
  const Vector3d position(0.7, -0.2, 0.5), point = r * p_i + position;
  Eigen::Matrix<double, 1, 9> numeric;
  for (int k = 0; k < 9; ++k)
  {
    auto residual = [&](double sign) {
      Vector3d dr = Vector3d::Zero(), dp = dr, dl = dr;
      if (k < 3) dr[k] = sign * eps;
      else if (k < 6) dp[k - 3] = sign * eps;
      else dl[k - 6] = sign * eps;
      return -(photo * r_ref_i * (r * expSO3(dr)).transpose() * (point + dl - position - dp))(0, 0);
    };
    numeric[k] = (residual(1.0) - residual(-1.0)) / (2.0 * eps);
  }
  near(ec::referencePhotometricJacobian(photo, r_rw, r, p_i), numeric, 1.0e-8,
       "reference residual mixed-coordinate finite difference");

  const Vector3d velocity(0.5, -0.2, 1.0), rate(0.3, 0.4, -0.2);
  const double dt = 0.04;
  const Matrix3d frame = r * expSO3(rate * dt);
  Eigen::Matrix<double, 6, 10> time_numeric;
  for (int k = 0; k < 10; ++k)
  {
    auto pose_error = [&](double sign) {
      Eigen::Matrix<double, 10, 1> e = Eigen::Matrix<double, 10, 1>::Zero();
      e[k] = sign * eps;
      Eigen::Matrix<double, 6, 1> error;
      error.head<3>() = logSO3(frame.transpose() * r * expSO3(e.head<3>()) * expSO3(rate * (dt + e[9])));
      error.tail<3>() = e.segment<3>(3) + (velocity + e.segment<3>(6)) * (dt + e[9]) - velocity * dt;
      return error;
    };
    time_numeric.col(k) = (pose_error(1.0) - pose_error(-1.0)) / (2.0 * eps);
  }
  near(ec::timeShiftPoseJacobian(expSO3(rate * dt).transpose(), velocity, rate, dt),
       time_numeric, 1.0e-8, "reference pose time-shift finite difference");
}

void testFullAndFrozenUpdate()
{
  for (int trial = 0; trial < 20; ++trial)
  {
    const int n = 13, m = 9;
    const MatrixXd a = MatrixXd::Random(n, n);
    const MatrixXd p = a * a.transpose() + MatrixXd::Identity(n, n);
    const MatrixXd h = MatrixXd::Random(m, n);
    const MatrixXd r = 0.3 * MatrixXd::Identity(m, m);
    const VectorXd residual = VectorXd::Random(m), prior_delta = VectorXd::Random(n);
    const MatrixXd info = h.transpose() * r.inverse() * h;
    const VectorXd vector = h.transpose() * r.inverse() * residual;
    const MatrixXd k = p * h.transpose() * (h * p * h.transpose() + r).inverse();
    for (bool freeze : {false, true})
    {
      std::vector<int> active;
      MatrixXd selected_k = k;
      VectorXd selected_delta = prior_delta;
      for (int i = 0; i < n; ++i)
      {
        if (!freeze || i % 3 != 1) active.push_back(i);
        else { selected_k.row(i).setZero(); selected_delta[i] = 0.0; }
      }
      const auto result = ec::update(p, info, vector, prior_delta, active);
      const MatrixXd identity = MatrixXd::Identity(n, n), j = identity - selected_k * h;
      near(result.correction, selected_delta + selected_k * (residual - h * prior_delta),
           1.0e-10, "correction against dense Kalman update");
      near(result.covariance, j * p * j.transpose() + selected_k * r * selected_k.transpose(),
           1.0e-10, "full Joseph covariance against dense update");
      require(Eigen::LLT<MatrixXd>(result.covariance).info() == Eigen::Success, "posterior SPD");
    }
  }
  Eigen::Matrix2d p; p << 1, 0.9, 0.9, 1;
  Eigen::Matrix2d info = Eigen::Matrix2d::Zero(); info(0, 0) = 100;
  const auto result = ec::update(p, info, Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero(), {0});
  require(std::abs(result.covariance(0, 1) - 0.9 / 101.0) < 1.0e-12, "stale cross-block regression");
  require(std::abs(result.covariance(1, 1) - 1.0) < 1.0e-12, "frozen marginal remains unchanged");
  p(0, 0) = -1.0;
  bool rejected = false;
  try { ec::update(p, info, Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero(), {0}); }
  catch (const std::runtime_error &) { rejected = true; }
  require(rejected, "indefinite full prior must not be silently repaired");
}

void testSharedResiduals()
{
  for (int trial = 0; trial < 20; ++trial)
  {
    const VectorXd d = VectorXd::Random(45).cwiseAbs().array() + 0.1;
    const MatrixXd u = MatrixXd::Random(45, 9), b = MatrixXd::Random(45, 15);
    const MatrixXd covariance = d.asDiagonal().toDenseMatrix() + u * u.transpose();
    near(ec::solveIndependentPlusShared(d, u, b), covariance.ldlt().solve(b), 1.0e-10,
         "Woodbury solve against dense residual covariance");
    std::vector<MatrixXd> terms;
    MatrixXd sum = MatrixXd::Zero(6, 8);
    for (int i = 0; i < 4; ++i)
    {
      const MatrixXd a = MatrixXd::Random(6, 8);
      sum += a; terms.push_back(a * a.transpose());
    }
    const MatrixXd slack = ec::unknownCorrelationBound(terms, 6) - sum * sum.transpose();
    require(Eigen::SelfAdjointEigenSolver<MatrixXd>(slack).eigenvalues().minCoeff() > -1.0e-9,
            "Cauchy bound covers arbitrary shared latent variables");
  }
  const int count = 100;
  const VectorXd d = VectorXd::Constant(count, 0.03 * 0.03);
  const MatrixXd u = MatrixXd::Constant(count, 1, 0.01);
  const MatrixXd ones = MatrixXd::Ones(count, 1);
  const double information = (ones.transpose() * ec::solveIndependentPlusShared(d, u, ones))(0, 0);
  require(std::abs(1.0 / information - (0.01 * 0.01 + 0.03 * 0.03 / count)) < 1.0e-12,
          "shared plane translation must not disappear as independent point count grows");
}

int main()
{
  try
  {
    testGeometry();
    testFullAndFrozenUpdate();
    testSharedResiduals();
    std::cout << "All estimator covariance math tests passed\n";
  }
  catch (const std::exception &error)
  {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
