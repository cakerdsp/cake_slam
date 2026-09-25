#ifndef CAKE_ESTIMATOR_COVARIANCE_H_
#define CAKE_ESTIMATOR_COVARIANCE_H_

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

// Error coordinates: right SO(3) increments and additive world-frame position.
// This header is independent of ROS so the covariance algebra can be tested.
namespace estimator_covariance
{
using Matrix6 = Eigen::Matrix<double, 6, 6>;
using PointJacobian = Eigen::Matrix<double, 3, 6>;

template <typename Derived>
inline typename Derived::PlainObject symmetric(const Eigen::MatrixBase<Derived> &a)
{
  // Evaluate products once before using both the matrix and its transpose.
  const typename Derived::PlainObject evaluated = a;
  return (0.5 * (evaluated + evaluated.transpose())).eval();
}

inline Eigen::Matrix3d skew(const Eigen::Vector3d &v)
{
  Eigen::Matrix3d a;
  a << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return a;
}

inline Eigen::Matrix3d rightJacobian(const Eigen::Vector3d &v)
{
  const double angle2 = v.squaredNorm();
  const Eigen::Matrix3d a = skew(v);
  double c1, c2;
  if (angle2 < 1.0e-8)
  {
    c1 = 0.5 - angle2 / 24.0 + angle2 * angle2 / 720.0;
    c2 = 1.0 / 6.0 - angle2 / 120.0 + angle2 * angle2 / 5040.0;
  }
  else
  {
    const double angle = std::sqrt(angle2);
    c1 = (1.0 - std::cos(angle)) / angle2;
    c2 = (angle - std::sin(angle)) / (angle2 * angle);
  }
  return Eigen::Matrix3d::Identity() - c1 * a + c2 * a * a;
}

// With d = current boxminus prior, Jr(d) transports the prior covariance
// into the current tangent space. With d = injected correction, the same
// Jacobian resets a local posterior into the corrected state's tangent space.
template <int Dimension>
inline Eigen::MatrixXd transportImpl(const Eigen::MatrixXd &p, const Eigen::VectorXd &d,
                                     const std::vector<int> &rotation_indices)
{
  using Matrix = Eigen::Matrix<double, Dimension, Dimension>;
  const Matrix covariance = p;
  Matrix j = Matrix::Identity(p.rows(), p.cols());
  for (int index : rotation_indices)
  {
    if (index < 0 || index + 3 > p.rows()) throw std::invalid_argument("invalid rotation covariance index");
    j.template block<3, 3>(index, index) = rightJacobian(d.segment<3>(index));
  }
  return symmetric(j * covariance * j.transpose());
}

inline Eigen::MatrixXd transport(const Eigen::MatrixXd &p, const Eigen::VectorXd &d,
                                 const std::vector<int> &rotation_indices)
{
  if (p.rows() != p.cols() || d.size() != p.rows() || !p.allFinite() || !d.allFinite())
    throw std::invalid_argument("covariance transport dimension mismatch");
  switch (p.rows())
  {
    case 19: return transportImpl<19>(p, d, rotation_indices);
    case 20: return transportImpl<20>(p, d, rotation_indices);
    case 21: return transportImpl<21>(p, d, rotation_indices);
    default: return transportImpl<Eigen::Dynamic>(p, d, rotation_indices);
  }
}

// Compact one/two/three-camera states use fixed-size arithmetic. The dynamic
// interface is retained for online-calibration layouts and larger camera rigs.
template <int Dimension>
inline Eigen::MatrixXd propagateImpl(const Eigen::MatrixXd &p, const Eigen::MatrixXd &f,
                                     const Eigen::MatrixXd &q)
{
  using Matrix = Eigen::Matrix<double, Dimension, Dimension>;
  const Matrix covariance = p;
  const Matrix transition = f;
  const Matrix noise = q;
  return symmetric(transition * covariance * transition.transpose() + noise);
}

inline Eigen::MatrixXd propagate(const Eigen::MatrixXd &p, const Eigen::MatrixXd &f,
                                 const Eigen::MatrixXd &q)
{
  switch (p.rows())
  {
    case 19: return propagateImpl<19>(p, f, q);
    case 20: return propagateImpl<20>(p, f, q);
    case 21: return propagateImpl<21>(p, f, q);
    default: return symmetric(f * p * f.transpose() + q);
  }
}

inline bool isPositiveDefinite(const Eigen::MatrixXd &p)
{
  switch (p.rows())
  {
    case 19: return Eigen::LLT<Eigen::Matrix<double, 19, 19>>(p).info() == Eigen::Success;
    case 20: return Eigen::LLT<Eigen::Matrix<double, 20, 20>>(p).info() == Eigen::Success;
    case 21: return Eigen::LLT<Eigen::Matrix<double, 21, 21>>(p).info() == Eigen::Success;
    default: return Eigen::LLT<Eigen::MatrixXd>(p).info() == Eigen::Success;
  }
}

template <int Dimension>
inline void normalEquationsFixed(const Eigen::MatrixXd &weighted_j, const Eigen::VectorXd &weighted_r,
                                  int rows, Eigen::MatrixXd &hessian, Eigen::VectorXd &gradient)
{
  // Map the existing workspace without allocating or changing its column stride.
  using Jacobian = Eigen::Matrix<double, Eigen::Dynamic, Dimension>;
  const Eigen::Map<const Jacobian> j(weighted_j.data(), weighted_j.rows(), Dimension);
  const Eigen::Matrix<double, Dimension, Dimension> h = j.topRows(rows).transpose() * j.topRows(rows);
  const Eigen::Matrix<double, Dimension, 1> g = j.topRows(rows).transpose() * weighted_r.head(rows);
  hessian = h;
  gradient = g;
}

inline void normalEquations(const Eigen::MatrixXd &weighted_j, const Eigen::VectorXd &weighted_r,
                            int rows, Eigen::MatrixXd &hessian, Eigen::VectorXd &gradient)
{
  switch (weighted_j.cols())
  {
    case 19: normalEquationsFixed<19>(weighted_j, weighted_r, rows, hessian, gradient); break;
    case 20: normalEquationsFixed<20>(weighted_j, weighted_r, rows, hessian, gradient); break;
    case 21: normalEquationsFixed<21>(weighted_j, weighted_r, rows, hessian, gradient); break;
    default:
      hessian.noalias() = weighted_j.topRows(rows).transpose() * weighted_j.topRows(rows);
      gradient.noalias() = weighted_j.topRows(rows).transpose() * weighted_r.head(rows);
      break;
  }
}

inline PointJacobian pointJacobian(const Eigen::Matrix3d &r_wi, const Eigen::Vector3d &p_i)
{
  PointJacobian j;
  j.leftCols<3>() = -r_wi * skew(p_i);
  j.rightCols<3>().setIdentity();
  return j;
}

// Jacobian of -I_ref with respect to [reference dtheta_I, reference dp_W,
// landmark dp_W]. photo is dI_ref/dp_ref, already including exposure or
// normalization where applicable. This uses mixed coordinates, not an SE3 adjoint.
inline Eigen::Matrix<double, 1, 9> referencePhotometricJacobian(
    const Eigen::RowVector3d &photo, const Eigen::Matrix3d &r_rw,
    const Eigen::Matrix3d &r_wi, const Eigen::Vector3d &p_i)
{
  Eigen::Matrix<double, 1, 9> j;
  j.segment<3>(0) = -photo * r_rw * r_wi * skew(p_i);
  j.segment<3>(3) = photo * r_rw;
  j.segment<3>(6) = -photo * r_rw;
  return j;
}

// [dtheta at adjusted frame, dp at adjusted frame] versus
// [dtheta at estimator time, dp, dv, d(time offset)]. Angular rate is frozen,
// matching the short time-shift model used by the visual frontend.
inline Eigen::Matrix<double, 6, 10> timeShiftPoseJacobian(
    const Eigen::Matrix3d &frame_from_state, const Eigen::Vector3d &velocity,
    const Eigen::Vector3d &angular_rate, double dt)
{
  Eigen::Matrix<double, 6, 10> j = Eigen::Matrix<double, 6, 10>::Zero();
  j.topLeftCorner<3, 3>() = frame_from_state;
  j.block<3, 3>(3, 3).setIdentity();
  j.block<3, 3>(3, 6) = dt * Eigen::Matrix3d::Identity();
  j.block<3, 1>(0, 9) = angular_rate;
  j.block<3, 1>(3, 9) = velocity;
  return j;
}

inline Eigen::Matrix3d pointCovariance(const Eigen::Matrix3d &r_wi,
                                      const Eigen::Matrix3d &r_il,
                                      const Eigen::Vector3d &p_i,
                                      const Eigen::Matrix3d &beam_cov,
                                      const Matrix6 &pose_cov)
{
  const Eigen::Matrix3d r_wl = r_wi * r_il;
  const PointJacobian j = pointJacobian(r_wi, p_i);
  return symmetric(r_wl * beam_cov * r_wl.transpose() + j * pose_cov * j.transpose());
}

struct Update
{
  Eigen::VectorXd correction;
  Eigen::MatrixXd gain_times_jacobian;
  Eigen::MatrixXd covariance;
};

// H' R^-1 H and H' R^-1 (z-h) are supplied in full state coordinates.
// Zero gain rows freeze means, while Joseph's formula still updates both
// cross-covariance blocks. Restoring old rows/columns is not a Schmidt update.
template <int Dimension>
inline Update updateImpl(const Eigen::MatrixXd &prior, const Eigen::MatrixXd &information,
                         const Eigen::VectorXd &information_vector,
                         const Eigen::VectorXd &prior_delta, const std::vector<int> &active)
{
  using Matrix = Eigen::Matrix<double, Dimension, Dimension>;
  using Vector = Eigen::Matrix<double, Dimension, 1>;
  const int n = static_cast<int>(prior.rows());
  const Matrix p = symmetric(Matrix(prior));
  const Matrix info = symmetric(Matrix(information));
  const Matrix identity = Matrix::Identity(n, n);
  const Eigen::LLT<Matrix> prior_llt(p);
  if (prior_llt.info() != Eigen::Success) throw std::runtime_error("full prior covariance is not positive definite");
  const Eigen::LLT<Matrix> posterior_llt(symmetric(prior_llt.solve(identity) + info));
  if (posterior_llt.info() != Eigen::Success) throw std::runtime_error("posterior information is not positive definite");
  const Matrix b = posterior_llt.solve(identity);
  Matrix selected_b = Matrix::Zero(n, n);
  Vector selected_prior = Vector::Zero(n);
  for (int index : active)
  {
    if (index < 0 || index >= n) throw std::invalid_argument("invalid active covariance index");
    selected_b.row(index) = b.row(index);
    selected_prior[index] = prior_delta[index];
  }
  const Matrix gain_times_jacobian = selected_b * info;
  const Vector rhs = information_vector;
  const Vector delta = prior_delta;
  const Vector correction = selected_b * rhs + selected_prior - gain_times_jacobian * delta;
  const Matrix a = identity - gain_times_jacobian;
  const Matrix covariance = symmetric(a * p * a.transpose() + selected_b * info * selected_b.transpose());
  if (!correction.allFinite() || !covariance.allFinite() ||
      Eigen::LLT<Matrix>(covariance).info() != Eigen::Success)
    throw std::runtime_error("invalid full posterior covariance");
  Update result;
  result.gain_times_jacobian = gain_times_jacobian;
  result.correction = correction;
  result.covariance = covariance;
  return result;
}

inline Update update(const Eigen::MatrixXd &prior, const Eigen::MatrixXd &information,
                     const Eigen::VectorXd &information_vector,
                     const Eigen::VectorXd &prior_delta, const std::vector<int> &active)
{
  const int n = static_cast<int>(prior.rows());
  if (prior.cols() != n || information.rows() != n || information.cols() != n ||
      information_vector.size() != n || prior_delta.size() != n ||
      !prior.allFinite() || !information.allFinite() ||
      !information_vector.allFinite() || !prior_delta.allFinite())
    throw std::runtime_error("invalid covariance update input");
  switch (n)
  {
    case 19: return updateImpl<19>(prior, information, information_vector, prior_delta, active);
    case 20: return updateImpl<20>(prior, information, information_vector, prior_delta, active);
    case 21: return updateImpl<21>(prior, information, information_vector, prior_delta, active);
    default: return updateImpl<Eigen::Dynamic>(prior, information, information_vector, prior_delta, active);
  }
}

// Covariance upper bound for a sum of vectors whose cross-correlations are
// unknown. Cauchy--Schwarz gives sum(C_i / w_i), sum(w_i)=1. The trace-optimal
// weights are proportional to sqrt(trace(C_i)); this is not a fitted noise scale.
inline Eigen::MatrixXd unknownCorrelationBound(const std::vector<Eigen::MatrixXd> &terms, int dimension)
{
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(dimension, dimension);
  double sum_root_trace = 0.0;
  for (const auto &term : terms)
  {
    if (term.rows() != dimension || term.cols() != dimension || !term.allFinite())
      throw std::runtime_error("invalid shared covariance term");
    sum_root_trace += std::sqrt(std::max(0.0, term.trace()));
  }
  for (const auto &term : terms)
  {
    const double root_trace = std::sqrt(std::max(0.0, term.trace()));
    if (root_trace > 0.0) result.noalias() += (sum_root_trace / root_trace) * term;
  }
  return symmetric(result);
}

inline Eigen::MatrixXd positiveSemidefiniteRoot(const Eigen::MatrixXd &covariance)
{
  if (covariance.rows() == 0 || covariance.rows() != covariance.cols() || !covariance.allFinite())
    throw std::runtime_error("invalid nuisance covariance dimensions or values");
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(symmetric(covariance));
  if (es.info() != Eigen::Success || !es.eigenvalues().allFinite() ||
      es.eigenvalues().minCoeff() < -1.0e-10 * std::max(1.0e-15, es.eigenvalues().cwiseAbs().maxCoeff()))
    throw std::runtime_error("invalid nuisance covariance");
  return es.eigenvectors() * es.eigenvalues().cwiseMax(0.0).cwiseSqrt().asDiagonal();
}

// (diag(d) + U U')^-1 B without a dense residual-sized factorization.
inline Eigen::MatrixXd solveIndependentPlusShared(const Eigen::VectorXd &d,
                                                  const Eigen::MatrixXd &u,
                                                  const Eigen::MatrixXd &b)
{
  if (d.size() != u.rows() || b.rows() != d.size() || !d.allFinite() ||
      (d.array() <= 0.0).any() || !u.allFinite() || !b.allFinite())
    throw std::runtime_error("invalid correlated residual covariance");
  const Eigen::MatrixXd di_b = d.cwiseInverse().asDiagonal() * b;
  const Eigen::MatrixXd di_u = d.cwiseInverse().asDiagonal() * u;
  const Eigen::LLT<Eigen::MatrixXd> llt(Eigen::MatrixXd::Identity(u.cols(), u.cols()) + u.transpose() * di_u);
  if (llt.info() != Eigen::Success) throw std::runtime_error("correlated residual factorization failed");
  return di_b - di_u * llt.solve(u.transpose() * di_b);
}
} // namespace estimator_covariance
#endif
