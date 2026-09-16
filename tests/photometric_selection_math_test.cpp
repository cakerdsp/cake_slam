#include "photometric_selection.h"
#include <iostream>
#include <limits>

using namespace photometric_selection;
static void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
static void close(const Matrix &a, const Matrix &b, double tol = 1.e-9) {
  require(a.rows() == b.rows() && a.cols() == b.cols() &&
          (a - b).norm() <= tol * std::max(1.0, b.norm()), "matrix comparison failed");
}

// Independent dense oracle: assemble the complete joint covariance, then
// invert its information system. No incremental Schur/Cholesky cache is used.
static Matrix densePosterior(const std::vector<Candidate> &c, const std::vector<int> &indices,
                             const Matrix &p, bool shared) {
  int rows = 0;
  for (int index : indices) rows += c[index].h.rows();
  if (rows == 0) return p;
  Matrix h(rows, p.rows()), r = Matrix::Identity(rows, rows);
  int a = 0;
  for (int i : indices) {
    h.middleRows(a, c[i].h.rows()) = c[i].h;
    int b = 0;
    for (int j : indices) {
      if (i == j || shared)
        r.block(a, b, c[i].h.rows(), c[j].h.rows()) += crossCovariance(c[i], c[j]);
      b += c[j].h.rows();
    }
    a += c[i].h.rows();
  }
  return (p.inverse() + h.transpose() * r.ldlt().solve(h)).inverse();
}

static Candidate scalar(int direction, double h, double noise, uint64_t source, uint64_t point,
                        int camera = 0, int dim = 7) {
  Candidate c;
  c.h = Matrix::Zero(1, dim); c.h(0, direction) = h;
  if (noise != 0.) c.sources[{0, source, 0, 0}] = Matrix::Constant(1, 1, noise);
  c.point = point; c.camera = camera; c.cost = 64;
  return c;
}

int main() {
  const Matrix prior = Matrix::Identity(7, 7);
  {
    // Correlated duplicates saturate; choose a genuinely complementary patch.
    std::vector<Candidate> c{scalar(0, 10., 10., 1, 1), scalar(0, 10., 10., 1, 2), scalar(1, .6, 0., 2, 3)};
    const auto joint = select(c, prior, {2, 128, true});
    const auto independent = select(c, prior, {2, 128, false});
    require(joint.indices == std::vector<int>({0, 2}), "shared duplicate did not saturate");
    require(independent.indices == std::vector<int>({0, 1}), "independent ablation changed marginals");
    close(joint.design_covariance, densePosterior(c, joint.indices, prior, true));
    close(independent.design_covariance, densePosterior(c, independent.indices, prior, false));
  }
  {
    // Negative nuisance correlation can increase conditional value: recompute
    // affected candidates instead of using a submodular lazy-greedy assumption.
    std::vector<Candidate> c{scalar(0, 2., 5., 1, 1), scalar(0, 2., -5., 1, 2)};
    const auto r = select(c, prior, {2, 128, true});
    require(r.gains[1] > r.gains[0], "conditional gain incorrectly forced to decrease");
    close(r.design_covariance, densePosterior(c, r.indices, prior, true));
  }
  {
    // Same current pixels through alternate references cannot be selected twice.
    std::vector<Candidate> c{scalar(0, 2., 0., 0, 42), scalar(1, 1., 0., 0, 42), scalar(2, 1., 0., 0, 42, 1)};
    auto r = select(c, prior, {3, 192, true});
    require(r.indices.size() == 2 && r.pixels == 128, "reference exclusivity/camera key broken");
    require(select(c, prior, {3, 63, true}).indices.empty(), "raw pixel cost bypassed");
    require(select(c, prior, {0, 192, true}).indices.empty(), "zero patch budget ignored");
    c[1].point = 7; c[0].cell = c[1].cell = c[2].cell = 3;
    require(select(c, prior, {3, 192, true}).indices.size() == 2, "per-camera coverage cell constraint broken");
  }
  {
    // Exposure can explain a large image response. Scoring only H_pose would
    // choose candidate 0; the full-state marginal correctly chooses candidate 1.
    Matrix p = prior; p(6, 6) = 100.;
    Candidate contaminated = scalar(0, 10., 0., 0, 1);
    contaminated.h(0, 6) = 10.;
    std::vector<Candidate> c{contaminated, scalar(0, 1., 0., 0, 2)};
    require(select(c, p, {1, 64, true}).indices[0] == 1, "pose-only scoring leaked nuisance information");
  }
  {
    // Compression retains independent information, while preserving original cost.
    Candidate c; c.h = Matrix::Random(13, 7); c.h.rightCols(4).setZero(); c.cost = 64; c.point = 1;
    const Matrix information = c.h.transpose() * c.h;
    require(project(c), "projection rejected a valid Jacobian");
    close(c.h.transpose() * c.h, information);
    require(c.h.rows() == 3 && c.cost == 64, "projected rank substituted for pixel cost");
  }
  {
    // Multiple connected components, partially shared keys, vector nuisance,
    // state cross covariance, and conditional cache updates against a dense oracle.
    Matrix seed = Matrix::Random(9, 9);
    const Matrix p = seed * seed.transpose() + .5 * Matrix::Identity(9, 9);
    std::vector<Candidate> c;
    for (int i = 0; i < 8; ++i) {
      Candidate a; a.h = Matrix::Random(3, 9); a.point = i; a.camera = i % 2; a.cost = 64;
      a.sources[{0, static_cast<uint64_t>(i / 3), 0, 0}] = Matrix::Random(3, 2);
      if (i % 2 == 0) a.sources[{1, 100, 0, 0}] = Matrix::Random(3, 1);
      c.push_back(a);
    }
    for (bool shared : {false, true}) {
      const auto result = select(c, p, {8, 512, shared});
      std::vector<int> prefix;
      Matrix previous = p;
      for (size_t step = 0; step < result.indices.size(); ++step) {
        int best = -1; double best_gain = -1.;
        for (int i = 0; i < static_cast<int>(c.size()); ++i) {
          if (std::find(prefix.begin(), prefix.end(), i) != prefix.end()) continue;
          auto trial = prefix; trial.push_back(i);
          const double gain = .5 * (poseLogdet(previous) - poseLogdet(densePosterior(c, trial, p, shared)));
          if (gain > best_gain) { best = i; best_gain = gain; }
        }
        require(best == result.indices[step], "incremental greedy differs from dense oracle");
        require(std::abs(best_gain - result.gains[step]) < 1.e-9, "incorrect conditional gain");
        prefix.push_back(best); previous = densePosterior(c, prefix, p, shared);
      }
      close(result.design_covariance, previous);
    }
  }
  {
    Matrix bad = prior; bad(0, 0) = -1.;
    bool rejected = false;
    try { select({}, bad, {}); } catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "invalid prior silently repaired");
  }
  {
    // Zero-column marginalization must retain uncertainty and cross covariance
    // of the full state, not condition inactive IMU/calibration states away.
    const int dim = 32;
    Matrix seed = Matrix::Random(dim, dim);
    const Matrix p = seed * seed.transpose() + Matrix::Identity(dim, dim);
    std::vector<Candidate> c;
    for (int i = 0; i < 7; ++i) {
      Candidate a;
      a.h = Matrix::Zero(3, dim);
      a.h.leftCols(6) = Matrix::Random(3, 6);
      a.h.col(9) = Matrix::Random(3, 1);
      a.h.col(17) = Matrix::Random(3, 1);
      a.sources[{0, static_cast<uint64_t>(i / 3), 0, 0}] = Matrix::Random(3, 2);
      a.sources[{1, 0, 0, static_cast<uint64_t>(i % 3)}] = Matrix::Random(3, 1);
      a.point = i; a.cost = 64; c.push_back(a);
    }
    for (bool shared : {false, true}) {
      const auto result = select(c, p, {5, 320, shared});
      require(result.active_state_dimension == 8, "unused columns were not marginalized");
      std::vector<int> prefix;
      Matrix previous = p;
      for (size_t step = 0; step < result.indices.size(); ++step) {
        int best = -1; double best_gain = -1.;
        for (int i = 0; i < static_cast<int>(c.size()); ++i) {
          if (std::find(prefix.begin(), prefix.end(), i) != prefix.end()) continue;
          auto trial = prefix; trial.push_back(i);
          const double gain = .5 * (poseLogdet(previous) - poseLogdet(densePosterior(c, trial, p, shared)));
          if (gain > best_gain) { best = i; best_gain = gain; }
        }
        require(best == result.indices[step], "compressed/cached greedy differs from full dense oracle");
        require(std::abs(best_gain - result.gains[step]) < 1.e-9, "compressed gain mismatch");
        prefix.push_back(best); previous = densePosterior(c, prefix, p, shared);
      }
      close(result.design_covariance, previous);
    }
  }
  {
    // The >=96-candidate branch is parallel in OpenMP builds. Check ordering,
    // shared noise and inactive-state recovery against a serial dense oracle.
    std::vector<Candidate> c;
    for (int i = 0; i < 100; ++i)
      c.push_back(scalar(i % 6, 0.5 + .013 * i, .3, i / 4, i, i % 2));
    const auto result = select(c, prior, {6, 384, true});
    std::vector<int> prefix;
    Matrix previous = prior;
    for (int selected : result.indices) {
      int best = -1; double best_gain = -1.;
      for (int i = 0; i < static_cast<int>(c.size()); ++i) {
        if (std::find(prefix.begin(), prefix.end(), i) != prefix.end()) continue;
        auto trial = prefix; trial.push_back(i);
        const double gain = .5 * (poseLogdet(previous) - poseLogdet(densePosterior(c, trial, prior, true)));
        if (gain > best_gain) { best = i; best_gain = gain; }
      }
      require(best == selected, "parallel ordered reduction differs from serial oracle");
      prefix.push_back(selected); previous = densePosterior(c, prefix, prior, true);
    }
    close(result.design_covariance, previous);
  }
  std::cout << "photometric_selection_math_test passed\n";
}
