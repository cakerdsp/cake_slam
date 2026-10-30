#include "split_state_math.h"
#include <chrono>
#include <iostream>
#include <numeric>
#include <limits>

namespace ec = estimator_covariance;
using Eigen::MatrixXd;
using Eigen::VectorXd;

void require(bool ok, const char *message)
{
  if (!ok) throw std::runtime_error(message);
}

double relativeError(const MatrixXd &a, const MatrixXd &b)
{
  return (a - b).norm() / std::max(1.0, b.norm());
}

// Original production calculation, including its fixed-size dispatch. This
// stays in the test so the comparison cannot accidentally follow the new path.
template <int N>
ec::Update joseph(const MatrixXd &prior, const MatrixXd &information,
                  const VectorXd &rhs, const VectorXd &delta, const std::vector<int> &active)
{
  using Matrix = Eigen::Matrix<double, N, N>;
  using Vector = Eigen::Matrix<double, N, 1>;
  const int n = prior.rows();
  const Matrix p = ec::symmetric(Matrix(prior));
  const Matrix info = ec::symmetric(Matrix(information));
  const Matrix identity = Matrix::Identity(n, n);
  const Eigen::LLT<Matrix> prior_llt(p);
  const Eigen::LLT<Matrix> posterior_llt(ec::symmetric(prior_llt.solve(identity) + info));
  const Matrix b = posterior_llt.solve(identity);
  Matrix selected_b = Matrix::Zero(n, n);
  Vector selected_prior = Vector::Zero(n);
  for (int i : active) { selected_b.row(i) = b.row(i); selected_prior[i] = delta[i]; }
  const Matrix gain = selected_b * info;
  const Vector correction = selected_b * Vector(rhs) + selected_prior - gain * Vector(delta);
  const Matrix a = identity - gain;
  const Matrix covariance = ec::symmetric(a * p * a.transpose() + selected_b * info * selected_b.transpose());
  require(correction.allFinite() && covariance.allFinite() &&
              Eigen::LLT<Matrix>(covariance).info() == Eigen::Success, "invalid Joseph reference");
  ec::Update result;
  result.correction = correction;
  result.gain_times_jacobian = gain;
  result.covariance = covariance;
  return result;
}

ec::Update reference(const MatrixXd &p, const MatrixXd &h, const VectorXd &g,
                     const VectorXd &d, const std::vector<int> &active)
{
  switch (p.rows())
  {
    case 19: return joseph<19>(p, h, g, d, active);
    case 20: return joseph<20>(p, h, g, d, active);
    case 21: return joseph<21>(p, h, g, d, active);
    default: return joseph<Eigen::Dynamic>(p, h, g, d, active);
  }
}

void testUpdates()
{
  ec::UpdateWorkspace workspace;
  ec::Update result;
  int cases = 0;
  double worst_covariance = 0.0, worst_measurement = 0.0, worst_gain = 0.0;
  double worst_old_gain_truth = 0.0, worst_new_gain_truth = 0.0;
  for (int n : {2, 13, 18, 19, 20, 21, 22, 39, 61})
    for (double condition : {1.0, 1.0e4, 1.0e8})
      for (double scale : {1.0e-5, 1.0, 1.0e5})
        for (int rank : {0, 1, 6, n})
        {
          const MatrixXd random = MatrixXd::Random(n, n);
          const MatrixXd q = random.householderQr().householderQ();
          VectorXd eigenvalues(n);
          for (int i = 0; i < n; ++i)
            eigenvalues[i] = scale * std::pow(condition, -static_cast<double>(i) / (n - 1));
          const MatrixXd p = ec::symmetric(q * eigenvalues.asDiagonal() * q.transpose());
          const int m = std::max(1, rank);
          MatrixXd h = MatrixXd::Random(m, n);
          if (rank == 0) h.setZero();
          const VectorXd residual = VectorXd::Random(m);
          const MatrixXd information = h.transpose() * h;
          const VectorXd rhs = h.transpose() * residual;
          const VectorXd delta = VectorXd::Random(n) * 0.01;
          for (int mode = 0; mode < 4; ++mode)
          {
            std::vector<int> active;
            for (int i = n - 1; i >= 0; --i)
              if (mode == 0 || (mode == 1 && i % 3 != 1) || (mode == 3 && i < 18))
                active.push_back(i);
            if (!active.empty()) active.push_back(active.front()); // duplicate/order invariance
            ec::updateInto(p, information, rhs, delta, active, workspace, result);
            const auto old = reference(p, information, rhs, delta, active);
            const double covariance_error = relativeError(result.covariance, old.covariance);
            worst_covariance = std::max(worst_covariance, covariance_error);
            require(covariance_error < 2.0e-7, "Joseph covariance mismatch");
            // Large P with low-rank information also makes P^-1 + information
            // ill-conditioned, even if P itself has condition number one.
            // Weyl gives kappa(posterior_information) <= condition + scale*||info||.
            // Use a bounded roundoff budget for these stress cases; separately
            // check BOTH implementations against measurement-space Kalman below.
            const double roundoff_budget = std::min(2.0e-7, std::max(2.0e-10,
                8.0 * n * std::numeric_limits<double>::epsilon() * (condition + scale * information.norm())));
            require(relativeError(result.correction, old.correction) < roundoff_budget, "correction changed");
            const double gain_error = relativeError(result.gain_times_jacobian, old.gain_times_jacobian);
            worst_gain = std::max(worst_gain, gain_error);
            if (gain_error >= roundoff_budget)
              std::cerr << "gain mismatch n=" << n << " condition=" << condition << " scale=" << scale
                        << " rank=" << rank << " mode=" << mode << " error="
                        << gain_error << " roundoff_budget=" << roundoff_budget
                        << " correction_error=" << relativeError(result.correction, old.correction) << '\n';
            require(gain_error < roundoff_budget, "gain changed");
            require(!result.used_joseph_fallback, "unexpected fallback for SPD test input");

            // Independent measurement-space Kalman calculation (R=I).
            const MatrixXd innovation = h * p * h.transpose() + MatrixXd::Identity(m, m);
            const MatrixXd gain = innovation.llt().solve(h * p).transpose();
            MatrixXd selected_gain = MatrixXd::Zero(n, m);
            VectorXd selected_delta = VectorXd::Zero(n);
            std::vector<bool> selected(n, false);
            for (int i : active)
            { selected_gain.row(i) = gain.row(i); selected_delta[i] = delta[i]; selected[i] = true; }
            const MatrixXd a = MatrixXd::Identity(n, n) - selected_gain * h;
            const MatrixXd expected_gain = selected_gain * h;
            worst_old_gain_truth = std::max(worst_old_gain_truth, relativeError(old.gain_times_jacobian, expected_gain));
            worst_new_gain_truth = std::max(worst_new_gain_truth, relativeError(result.gain_times_jacobian, expected_gain));
            require(relativeError(old.gain_times_jacobian, expected_gain) < 2.0e-7 &&
                        relativeError(result.gain_times_jacobian, expected_gain) < 2.0e-7,
                    "measurement-space gain mismatch");
            const MatrixXd expected = ec::symmetric(a * p * a.transpose() + selected_gain * selected_gain.transpose());
            const double measurement_error = relativeError(result.covariance, expected);
            worst_measurement = std::max(worst_measurement, measurement_error);
            require(measurement_error < 2.0e-7, "measurement-space covariance mismatch");
            require(relativeError(result.correction, selected_gain * residual + selected_delta - selected_gain * h * delta)
                        < 2.0e-7, "measurement-space correction mismatch");
            for (int r = 0; r < n; ++r) if (!selected[r])
            {
              require(result.correction[r] == 0.0, "frozen mean changed");
              for (int c = 0; c < n; ++c) if (!selected[c])
                require(result.covariance(r, c) == p(r, c), "frozen marginal changed");
            }
            const VectorXd reset = VectorXd::Random(n) * 0.001;
            const auto transported = ec::transport(result.covariance, reset, n >= 6 ? std::vector<int>{0, 3} : std::vector<int>{});
            const auto old_transported = ec::transport(old.covariance, reset, n >= 6 ? std::vector<int>{0, 3} : std::vector<int>{});
            require(relativeError(transported, old_transported) < 2.0e-7, "SO3 reset mismatch");
            ++cases;
          }
        }
  std::cout << "update_cases=" << cases << " max_scaled_Joseph_error=" << worst_covariance
            << " max_scaled_measurement_error=" << worst_measurement << " max_scaled_gain_difference=" << worst_gain
            << " old_gain_vs_measurement=" << worst_old_gain_truth << " new_gain_vs_measurement=" << worst_new_gain_truth << '\n';
}

void testFailureAndReuse()
{
  ec::UpdateWorkspace workspace;
  ec::Update result;
  for (int n : {19, 20, 21, 39, 61, 21})
  {
    MatrixXd p = MatrixXd::Identity(n, n);
    MatrixXd info = p * 2.0;
    VectorXd rhs = VectorXd::Ones(n), delta = VectorXd::Zero(n);
    std::vector<int> active(n);
    std::iota(active.begin(), active.end(), 0);
    ec::updateInto(p, info, rhs, delta, active, workspace, result);
    const ec::Update accepted = result;
    // No Eigen allocation is allowed after layout warmup, including LLTs and outputs.
    Eigen::internal::set_is_malloc_allowed(false);
    for (int repeat = 0; repeat < 20; ++repeat)
      ec::updateInto(p, info, rhs, delta, active, workspace, result);
    Eigen::internal::set_is_malloc_allowed(true);
    for (int invalid = 0; invalid < 6; ++invalid)
    {
      MatrixXd bad_p = p, bad_info = info;
      VectorXd bad_rhs = rhs;
      std::vector<int> bad_active = active;
      if (invalid == 0) bad_p(0, 0) = -1.0;
      if (invalid == 1) bad_info = -2.0 * p;
      if (invalid == 2) bad_rhs[0] = std::numeric_limits<double>::quiet_NaN();
      if (invalid == 3) bad_active.push_back(n);
      if (invalid == 4) bad_info.resize(n - 1, n - 1);
      // Posterior information is SPD but the partially frozen covariance is
      // indefinite. Exercise both the candidate check and Joseph rejection.
      if (invalid == 5)
      {
        bad_info.setZero();
        bad_info.topLeftCorner<2, 2>() << 9.0, -3.0, -3.0, 0.0;
        bad_active = {0};
      }
      bool rejected = false;
      try { ec::updateInto(bad_p, bad_info, bad_rhs, delta, bad_active, workspace, result); }
      catch (const std::exception &) { rejected = true; }
      require(rejected, "invalid trial accepted");
      require(relativeError(result.covariance, accepted.covariance) == 0.0 &&
                  relativeError(result.correction, accepted.correction) == 0.0 &&
                  relativeError(result.gain_times_jacobian, accepted.gain_times_jacobian) == 0.0 &&
                  result.used_joseph_fallback == accepted.used_joseph_fallback,
              "rejected trial overwrote accepted output");
    }
    // A failed trial must not poison reusable scratch either.
    ec::updateInto(p, info, rhs, delta, active, workspace, result);
    require(relativeError(result.covariance, accepted.covariance) == 0.0, "scratch not recoverable");
  }
}

void testPatchStorage()
{
  split_state_math::PatchWorkspace patch;
  split_state_math::PairWorkspace pair;
  const std::vector<int> indices{18, 20};
  patch.configure(64, 21, indices, true);
  pair.reset(64, 21, true, true);
  const MatrixXd source = MatrixXd::Random(64, 21), target = MatrixXd::Random(64, 21);
  const VectorXd residuals = VectorXd::Random(64);
  const MatrixXd jacobians = 0.7 * (source - target);
  const MatrixXd expected_h = jacobians.transpose() * jacobians;
  const VectorXd expected_g = jacobians.transpose() * residuals;
  for (int repeat = 0; repeat < 4; ++repeat)
  {
    Eigen::internal::set_is_malloc_allowed(false);
    patch.configure(64, 21, indices, true);
    patch.pose_affine_delta.setZero();
    pair.reset(64, 21, true, true);
    for (int i = 0; i < 64; ++i)
    {
      pair.source_jacobian = source.row(i).transpose();
      pair.target_jacobian = target.row(i).transpose();
      pair.jacobian = 0.7 * (pair.source_jacobian - pair.target_jacobian);
      pair.hessian.noalias() += pair.jacobian * pair.jacobian.transpose();
      pair.gradient.noalias() += pair.jacobian * residuals[i];
    }
    Eigen::internal::set_is_malloc_allowed(true);
    require(relativeError(pair.hessian, expected_h) < 1.0e-14 && relativeError(pair.gradient, expected_g) < 1.0e-14,
            "reused pair statistics differ from dense reference");
    require(patch.pose_affine_delta.isZero(0), "previous patch nuisance leaked");
    patch.pose_affine_delta.setConstant(13.0);
    pair.hessian.setConstant(17.0); // simulate a rejected incomplete pair
    pair.gradient.setConstant(19.0);
  }
  pair.reset(7, 39, true, true);
  require(pair.hessian.rows() == 39 && pair.hessian.isZero(0) && pair.gradient.isZero(0), "layout change failed");
  patch.configure(7, 39, {18, 24, 25}, true);
  require(patch.pose_affine_delta.rows() == 7 && patch.camera.cols() == 3, "patch layout change failed");
}

void benchmark()
{
  volatile double checksum = 0.0;
  for (int n : {21, 39})
  {
    const MatrixXd random = MatrixXd::Random(n, n);
    const MatrixXd p = random * random.transpose() + MatrixXd::Identity(n, n);
    const MatrixXd h = random.transpose() * random;
    const VectorXd rhs = VectorXd::Random(n), delta = VectorXd::Random(n);
    std::vector<int> active(n);
    std::iota(active.begin(), active.end(), 0);
    ec::UpdateWorkspace workspace;
    ec::Update result;
    ec::updateInto(p, h, rhs, delta, active, workspace, result);
    std::vector<double> old_times, new_times;
    constexpr int repeats = 2000;
    for (int trial = 0; trial < 7; ++trial)
      for (int pass = 0; pass < 2; ++pass)
      {
        const bool old = ((trial + pass) % 2 == 0);
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; ++i)
        {
          if (old) result = reference(p, h, rhs, delta, active);
          else ec::updateInto(p, h, rhs, delta, active, workspace, result);
          checksum += result.covariance(0, 0);
        }
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / repeats;
        (old ? old_times : new_times).push_back(us);
      }
    std::sort(old_times.begin(), old_times.end());
    std::sort(new_times.begin(), new_times.end());
    std::cout << "microbenchmark_n=" << n << " old_us=" << old_times[3] << " new_us=" << new_times[3]
              << " speedup=" << old_times[3] / new_times[3] << '\n';
  }
  require(checksum != 0.0, "benchmark optimized away");
}

int main(int argc, char **argv)
{
  try
  {
    std::srand(73);
    testUpdates();
    testFailureAndReuse();
    testPatchStorage();
    if (argc > 1 && std::string(argv[1]) == "--benchmark") benchmark();
    std::cout << "Update/workspace regression and allocation tests passed\n";
  }
  catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
