#pragma once

// Conditional observation scoring only. This header never writes the estimator
// covariance: its posterior is a temporary design covariance for selection.
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace photometric_selection {
using Matrix = Eigen::MatrixXd;
using SourceKey = std::tuple<int, uint64_t, uint64_t, uint64_t>;

struct Candidate {
  Matrix h;                         // Full state columns; private noise is I.
  std::map<SourceKey, Matrix> sources;  // B sqrt(Sigma), same columns per key.
  uint64_t point = 0;
  int camera = 0;
  int slot = 0;
  int cell = -1;                    // At most one patch per current image cell.
  int cost = 0;                      // Actual pixels, before score projection.
};

struct Options {
  int patch_budget = 150;
  int pixel_budget = 9600;
  bool shared_errors = true;
};

struct Result {
  std::vector<int> indices;
  std::vector<double> gains;
  Matrix design_covariance;
  int pixels = 0;
  double pose_logdet_gain = 0.0;
};

inline Matrix symmetric(const Matrix &m) { return 0.5 * (m + m.transpose()); }

inline double poseLogdet(const Matrix &p) {
  if (p.rows() < 6 || !p.allFinite()) throw std::runtime_error("invalid selection covariance");
  Eigen::LLT<Matrix> llt(symmetric(p.topLeftCorner(6, 6)));
  if (llt.info() != Eigen::Success) throw std::runtime_error("non-positive selection pose covariance");
  const Matrix l = llt.matrixL();
  return 2.0 * l.diagonal().array().log().sum();
}

inline Matrix crossCovariance(const Candidate &a, const Candidate &b) {
  Matrix r = Matrix::Zero(a.h.rows(), b.h.rows());
  auto i = a.sources.begin(), j = b.sources.begin();
  while (i != a.sources.end() && j != b.sources.end()) {
    if (i->first < j->first) { ++i; continue; }
    if (j->first < i->first) { ++j; continue; }
    if (i->second.cols() != j->second.cols()) throw std::runtime_error("inconsistent shared source dimensions");
    r.noalias() += i->second * j->second.transpose();
    ++i; ++j;
  }
  return r;
}

// Keep the full state-sensitive row subspace. For independent noise this is
// lossless. With shared noise this deliberately defines a projected scoring
// model: discarded pure-noise rows could otherwise identify shared nuisance.
inline bool project(Candidate &c) {
  if (!c.h.allFinite() || c.h.rows() == 0 || c.cost <= 0) return false;
  Eigen::ColPivHouseholderQR<Matrix> qr(c.h);
  qr.setThreshold(1.e-10);
  const int rank = qr.rank();
  if (rank == 0) return false;
  const Matrix q = qr.householderQ() * Matrix::Identity(c.h.rows(), rank);
  c.h = (q.transpose() * c.h).eval();
  for (auto it = c.sources.begin(); it != c.sources.end();) {
    if (it->second.rows() != q.rows() || !it->second.allFinite()) return false;
    it->second = (q.transpose() * it->second).eval();
    if (it->second.squaredNorm() == 0.0) it = c.sources.erase(it);
    else ++it;
  }
  return true;
}

inline Matrix posterior(const Matrix &p, const Matrix &a) {
  const Matrix v = p * a.transpose();
  Eigen::LLT<Matrix> llt(symmetric(Matrix::Identity(a.rows(), a.rows()) + a * v));
  if (llt.info() != Eigen::Success) throw std::runtime_error("invalid selection innovation");
  return symmetric(p - v * llt.solve(v.transpose()));
}

inline double poseLogdetAfter(const Matrix &p, const Matrix &a) {
  const Matrix v = p * a.transpose();
  Eigen::LLT<Matrix> llt(symmetric(Matrix::Identity(a.rows(), a.rows()) + a * v));
  if (llt.info() != Eigen::Success) throw std::runtime_error("invalid selection innovation");
  const Matrix pose_cross = v.topRows(6);
  return poseLogdet(symmetric(p.topLeftCorner(6, 6) - pose_cross * llt.solve(pose_cross.transpose())));
}

inline Result select(const std::vector<Candidate> &candidates, const Matrix &prior, const Options &options) {
  if (prior.rows() != prior.cols() || options.patch_budget < 0 || options.pixel_budget < 0)
    throw std::runtime_error("invalid photometric selection input");
  Eigen::LLT<Matrix> prior_llt(symmetric(prior));
  if (prior_llt.info() != Eigen::Success) throw std::runtime_error("non-positive full selection prior");
  Result result;
  result.design_covariance = symmetric(prior);
  double logdet = poseLogdet(prior);
  const double initial_logdet = logdet;
  const int n = static_cast<int>(candidates.size());
  std::vector<int> parent(n);
  std::iota(parent.begin(), parent.end(), 0);
  auto root = [&](int v) { while (parent[v] != v) { parent[v] = parent[parent[v]]; v = parent[v]; } return v; };
  std::map<SourceKey, int> owners;
  for (int i = 0; i < n; ++i) {
    const auto &c = candidates[i];
    if (c.h.cols() != prior.rows() || c.h.rows() == 0 || c.cost <= 0 || !c.h.allFinite())
      throw std::runtime_error("invalid selection candidate");
    for (const auto &source : c.sources) {
      if (source.second.rows() != c.h.rows() || !source.second.allFinite())
        throw std::runtime_error("invalid selection source");
      if (!options.shared_errors) continue;
      const auto inserted = owners.emplace(source.first, i);
      if (!inserted.second) parent[root(i)] = root(inserted.first->second);
    }
  }
  struct Component { Matrix l, whitened_h; std::vector<int> selected; int version = 0; };
  struct Cache {
    Matrix a, lower_cross, conditional_l, conditional_h, conditional_r;
    int version = -1;
  };
  std::vector<Component> components(n);
  std::vector<Cache> cache(n);
  std::vector<Matrix> diagonal(n);
  for (int i = 0; i < n; ++i) {
    parent[i] = root(i);
    diagonal[i] = Matrix::Identity(candidates[i].h.rows(), candidates[i].h.rows()) +
                  crossCovariance(candidates[i], candidates[i]);
  }
  std::set<std::pair<uint64_t, int>> used_current_observations;
  std::set<std::pair<int, int>> used_cells;
  for (int step = 0; step < options.patch_budget; ++step) {
    int best = -1;
    double best_score = 0.0, best_gain = 0.0;
    for (int i = 0; i < n; ++i) {
      const auto &c = candidates[i];
      if (result.pixels + c.cost > options.pixel_budget ||
          used_current_observations.count({c.point, c.camera}) != 0 ||
          (c.cell >= 0 && used_cells.count({c.camera, c.cell}) != 0)) continue;
      auto &component = components[parent[i]];
      auto &item = cache[i];
      if (item.version != component.version) {
        if (item.version < 0) {
          item.conditional_h = c.h;
          item.conditional_r = diagonal[i];
          item.lower_cross.resize(0, c.h.rows());
          item.version = 0;
        }
        // Append only newly selected innovations. Re-solving the growing
        // component covariance for every candidate would be quadratic in all
        // already selected rows at every greedy step.
        for (int step_index = item.version; step_index < component.version; ++step_index) {
          const int j = component.selected[step_index];
          const int old_rows = item.lower_cross.rows(), added = candidates[j].h.rows();
          Matrix cross = crossCovariance(candidates[j], c);
          if (old_rows > 0)
            cross.noalias() -= component.l.block(old_rows, 0, added, old_rows) * item.lower_cross;
          const Matrix innovation_cross = component.l.block(old_rows, old_rows, added, added)
              .triangularView<Eigen::Lower>().solve(cross);
          item.conditional_r.noalias() -= innovation_cross.transpose() * innovation_cross;
          item.conditional_h.noalias() -= innovation_cross.transpose() * component.whitened_h.middleRows(old_rows, added);
          item.lower_cross.conservativeResize(old_rows + added, c.h.rows());
          item.lower_cross.bottomRows(added) = innovation_cross;
        }
        Eigen::LLT<Matrix> llt(symmetric(item.conditional_r));
        if (llt.info() != Eigen::Success) throw std::runtime_error("invalid conditional selection covariance");
        item.conditional_l = llt.matrixL();
        item.a = llt.matrixL().solve(item.conditional_h);
        item.version = component.version;
      }
      const double gain = 0.5 * (logdet - poseLogdetAfter(result.design_covariance, item.a));
      const double score = gain / c.cost;
      // Stable input-order tie breaking; no random or eigen-direction gate.
      if (score > best_score) { best = i; best_score = score; best_gain = gain; }
    }
    if (best < 0) break;
    const auto &c = candidates[best];
    result.indices.push_back(best);
    result.gains.push_back(best_gain);
    result.pixels += c.cost;
    used_current_observations.emplace(c.point, c.camera);
    if (c.cell >= 0) used_cells.emplace(c.camera, c.cell);
    result.design_covariance = posterior(result.design_covariance, cache[best].a);
    logdet = poseLogdet(result.design_covariance);
    auto &component = components[parent[best]];
    auto &item = cache[best];
    const int old_rows = component.whitened_h.rows(), added = c.h.rows();
    Matrix next_l = Matrix::Zero(old_rows + added, old_rows + added);
    if (old_rows > 0) {
      next_l.topLeftCorner(old_rows, old_rows) = component.l;
      next_l.bottomLeftCorner(added, old_rows) = item.lower_cross.transpose();
    }
    next_l.bottomRightCorner(added, added) = item.conditional_l;
    component.l.swap(next_l);
    component.whitened_h.conservativeResize(old_rows + added, c.h.cols());
    component.whitened_h.bottomRows(added) = item.a;
    component.selected.push_back(best);
    ++component.version;
  }
  result.pose_logdet_gain = 0.5 * (initial_logdet - logdet);
  return result;
}
}  // namespace photometric_selection
