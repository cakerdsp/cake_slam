#ifndef DIRECTIONAL_UPDATE_H_
#define DIRECTIONAL_UPDATE_H_

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <string>

namespace directional_update
{

struct Result
{
  bool valid = false;
  std::string error;
  Eigen::MatrixXd information;
  Eigen::VectorXd information_vector;
  Eigen::VectorXd eigenvalues;
  Eigen::VectorXd variance_reductions;
  Eigen::VectorXd weights;
  Eigen::MatrixXd prior_sqrt_lower;
  Eigen::MatrixXd normalized_eigenvectors;
  Eigen::MatrixXd component_transform;
  int active_rank = 0;
  int full_rank = 0;
};

inline Eigen::MatrixXd symmetrize(const Eigen::MatrixXd &matrix)
{
  return (0.5 * (matrix + matrix.transpose())).eval();
}

namespace detail
{
template <int Dimension>
inline bool filterInformationImpl(const Eigen::MatrixXd &prior_covariance,
                                  const Eigen::MatrixXd &measurement_information,
                                  const Eigen::VectorXd &measurement_information_vector,
                                  double drop_variance_reduction,
                                  double full_variance_reduction,
                                  Result &result, bool prepare_components)
{
  using Matrix = Eigen::Matrix<double, Dimension, Dimension>;
  using Vector = Eigen::Matrix<double, Dimension, 1>;
  const Eigen::Index dimension = prior_covariance.rows();
  const Matrix prior = 0.5 * (prior_covariance + prior_covariance.transpose());
  const Eigen::LLT<Matrix> prior_llt(prior);
  if (prior_llt.info() != Eigen::Success)
  {
    result.error = "prior covariance is not positive definite";
    return false;
  }
  const Matrix information = 0.5 * (measurement_information + measurement_information.transpose());
  const Matrix lower = prior_llt.matrixL();
  Matrix normalized_information = lower.transpose() * information * lower;
  normalized_information = (0.5 * (normalized_information + normalized_information.transpose())).eval();

  const Eigen::SelfAdjointEigenSolver<Matrix> solver(normalized_information);
  if (solver.info() != Eigen::Success)
  {
    result.error = "normalized information eigendecomposition failed";
    return false;
  }

  result.eigenvalues = solver.eigenvalues();
  const double spectral_scale = std::max(1.0, result.eigenvalues.cwiseAbs().maxCoeff());
  const double negative_tolerance = 1.0e-10 * spectral_scale;
  if (result.eigenvalues.minCoeff() < -negative_tolerance)
  {
    result.error = "measurement information is not positive semidefinite";
    return false;
  }
  result.eigenvalues = result.eigenvalues.cwiseMax(0.0);
  result.variance_reductions.resize(dimension);
  result.weights.resize(dimension);
  for (Eigen::Index i = 0; i < dimension; ++i)
  {
    const double lambda = result.eigenvalues[i];
    const double reduction = lambda > 1.0e12 ? 1.0 : lambda / (1.0 + lambda);
    result.variance_reductions[i] = reduction;
    double weight = 0.0;
    if (reduction >= full_variance_reduction)
      weight = 1.0;
    else if (reduction > drop_variance_reduction)
      weight = (reduction - drop_variance_reduction) /
               (full_variance_reduction - drop_variance_reduction);
    result.weights[i] = std::clamp(weight, 0.0, 1.0);
    if (result.weights[i] > 0.0) ++result.active_rank;
    if (result.weights[i] >= 1.0) ++result.full_rank;
  }

  const Matrix eigenvectors = solver.eigenvectors();
  result.prior_sqrt_lower = lower;
  result.normalized_eigenvectors = eigenvectors;
  // B = L^-T U. Reuse this basis for both likelihood terms rather than
  // reconstructing in normalized coordinates and solving each term again.
  const Matrix basis = lower.transpose().template triangularView<Eigen::Upper>().solve(eigenvectors);
  const Vector weighted_eigenvalues = result.weights.array() * result.eigenvalues.array();
  const Matrix weighted_basis = basis * weighted_eigenvalues.asDiagonal();
  result.information.noalias() = weighted_basis * basis.transpose();
  result.information = symmetrize(result.information);
  const Vector normalized_vector = lower.transpose() * measurement_information_vector;
  Vector modal_vector = eigenvectors.transpose() * normalized_vector;
  modal_vector.array() *= result.weights.array();
  result.information_vector.noalias() = basis * modal_vector;

  if (prepare_components)
  {
    // T = L^-T U sqrt(W) U^T L^T applies the identical gate to any additive
    // information component. Build once; all usage categories share it.
    const Matrix weighted_component_basis = basis * result.weights.cwiseSqrt().asDiagonal();
    const Matrix component_gate = weighted_component_basis * eigenvectors.transpose();
    result.component_transform.noalias() = component_gate * lower.transpose();
  }
  if (!result.information.array().isFinite().all() || !result.information_vector.array().isFinite().all())
  {
    result.error = "filtered information reconstruction produced non-finite values";
    return false;
  }
  result.valid = true;
  return true;
}

} // namespace detail

inline bool filterInformation(const Eigen::MatrixXd &prior_covariance,
                              const Eigen::MatrixXd &measurement_information,
                              const Eigen::VectorXd &measurement_information_vector,
                              double drop_variance_reduction,
                              double full_variance_reduction,
                              Result &result,
                              bool prepare_components = false)
{
  result = Result();
  const Eigen::Index dimension = prior_covariance.rows();
  if (dimension <= 0 || prior_covariance.cols() != dimension ||
      measurement_information.rows() != dimension || measurement_information.cols() != dimension ||
      measurement_information_vector.size() != dimension)
  {
    result.error = "inconsistent matrix dimensions";
    return false;
  }
  if (!prior_covariance.array().isFinite().all() ||
      !measurement_information.array().isFinite().all() ||
      !measurement_information_vector.array().isFinite().all())
  {
    result.error = "non-finite covariance or measurement information";
    return false;
  }
  if (!std::isfinite(drop_variance_reduction) || !std::isfinite(full_variance_reduction) ||
      drop_variance_reduction < 0.0 || full_variance_reduction > 1.0 ||
      full_variance_reduction <= drop_variance_reduction)
  {
    result.error = "variance-reduction thresholds must satisfy 0 <= drop < full <= 1";
    return false;
  }

  // Common rigs use fixed-size decompositions; online calibration and larger
  // rigs retain the same dynamic fallback and the complete joint covariance.
  switch (dimension)
  {
    case 18: return detail::filterInformationImpl<18>(prior_covariance, measurement_information,
        measurement_information_vector, drop_variance_reduction, full_variance_reduction, result, prepare_components);
    case 19: return detail::filterInformationImpl<19>(prior_covariance, measurement_information,
        measurement_information_vector, drop_variance_reduction, full_variance_reduction, result, prepare_components);
    case 20: return detail::filterInformationImpl<20>(prior_covariance, measurement_information,
        measurement_information_vector, drop_variance_reduction, full_variance_reduction, result, prepare_components);
    case 21: return detail::filterInformationImpl<21>(prior_covariance, measurement_information,
        measurement_information_vector, drop_variance_reduction, full_variance_reduction, result, prepare_components);
    default: return detail::filterInformationImpl<Eigen::Dynamic>(prior_covariance, measurement_information,
        measurement_information_vector, drop_variance_reduction, full_variance_reduction, result, prepare_components);
  }
}

// Applies the same symmetric directional gate to one additive information
// component.  Components filtered this way remain positive semidefinite and
// sum exactly to the filtered total information matrix.
inline bool filterInformationComponent(const Eigen::MatrixXd &component_information,
                                       const Result &filter,
                                       Eigen::MatrixXd &filtered_component,
                                       std::string &error)
{
  const Eigen::Index dimension = component_information.rows();
  if (!filter.valid || dimension <= 0 || component_information.cols() != dimension ||
      filter.prior_sqrt_lower.rows() != dimension || filter.prior_sqrt_lower.cols() != dimension ||
      filter.normalized_eigenvectors.rows() != dimension ||
      filter.normalized_eigenvectors.cols() != dimension || filter.weights.size() != dimension)
  {
    error = "invalid directional filter or component dimensions";
    return false;
  }
  if (!component_information.array().isFinite().all())
  {
    error = "non-finite information component";
    return false;
  }

  Eigen::MatrixXd uncached_transform;
  const Eigen::MatrixXd *transform = &filter.component_transform;
  if (transform->rows() != dimension || transform->cols() != dimension)
  {
    const Eigen::MatrixXd basis = filter.prior_sqrt_lower.transpose()
        .triangularView<Eigen::Upper>().solve(filter.normalized_eigenvectors);
    uncached_transform = basis * filter.weights.cwiseSqrt().asDiagonal() *
        filter.normalized_eigenvectors.transpose() * filter.prior_sqrt_lower.transpose();
    transform = &uncached_transform;
  }
  const Eigen::MatrixXd left = (*transform) * symmetrize(component_information);
  filtered_component.noalias() = left * transform->transpose();
  filtered_component = symmetrize(filtered_component);
  if (!filtered_component.array().isFinite().all())
  {
    error = "filtered information component contains non-finite values";
    return false;
  }
  return true;
}

inline bool posteriorCovariance(const Eigen::MatrixXd &prior_covariance,
                                const Eigen::MatrixXd &total_measurement_information,
                                Eigen::MatrixXd &posterior_covariance,
                                std::string &error)
{
  const Eigen::Index dimension = prior_covariance.rows();
  if (dimension <= 0 || prior_covariance.cols() != dimension ||
      total_measurement_information.rows() != dimension ||
      total_measurement_information.cols() != dimension)
  {
    error = "inconsistent posterior covariance dimensions";
    return false;
  }
  if (!prior_covariance.array().isFinite().all() ||
      !total_measurement_information.array().isFinite().all())
  {
    error = "non-finite posterior covariance input";
    return false;
  }

  const Eigen::MatrixXd prior = symmetrize(prior_covariance);
  const Eigen::LLT<Eigen::MatrixXd> prior_llt(prior);
  if (prior_llt.info() != Eigen::Success)
  {
    error = "prior covariance is not positive definite";
    return false;
  }
  const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(dimension, dimension);
  Eigen::MatrixXd posterior_information =
      prior_llt.solve(identity) + symmetrize(total_measurement_information);
  posterior_information = symmetrize(posterior_information);
  const Eigen::LLT<Eigen::MatrixXd> posterior_llt(posterior_information);
  if (posterior_llt.info() != Eigen::Success)
  {
    error = "posterior information is not positive definite";
    return false;
  }
  posterior_covariance = symmetrize(posterior_llt.solve(identity));
  if (!posterior_covariance.array().isFinite().all())
  {
    error = "posterior covariance contains non-finite values";
    return false;
  }
  return true;
}

} // namespace directional_update

#endif // DIRECTIONAL_UPDATE_H_
