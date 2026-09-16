// Standalone selector benchmark; excludes image retrieval/linearization and ROS.
// Synthetic timings are not an end-to-end real-time claim.
#include "photometric_selection.h"
#include <chrono>
#include <iostream>

int main()
{
  using namespace photometric_selection;
  constexpr int count = 600, dimension = 32, rank = 7;
  const Matrix prior = Matrix::Identity(dimension, dimension);
  std::vector<Candidate> candidates;
  candidates.reserve(count);
  for (int i = 0; i < count; ++i) {
    Candidate c;
    c.h = Matrix::Zero(rank, dimension);
    c.h.leftCols(6) = Matrix::Random(rank, 6);
    c.camera = (i / 3) % 2;
    c.h.col(6 + c.camera) = Matrix::Random(rank, 1);
    c.point = i / 3; c.slot = i; c.cost = 64;
    c.sources[{0, static_cast<uint64_t>(i / 6), 0, 0}] = .1 * Matrix::Random(rank, 3);
    for (int pixel = 0; pixel < 200; ++pixel)
      c.sources[{1, static_cast<uint64_t>(c.camera), static_cast<uint64_t>(i / 3),
                 static_cast<uint64_t>(pixel)}] = .01 * Matrix::Random(rank, 1);
    candidates.push_back(std::move(c));
  }
  for (bool shared : {false, true}) {
    for (int run = 0; run < 3; ++run) {
      const auto start = std::chrono::steady_clock::now();
      const Result result = select(candidates, prior, {150, 9600, shared});
      const double ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start).count();
      std::cout << "shared=" << shared << " run=" << run << " candidates=" << count
                << " selected=" << result.indices.size() << " score_evals=" << result.score_evaluations
                << " state_dim=" << result.active_state_dimension
                << " shared_blocks=" << result.shared_covariance_blocks << " choose_ms=" << ms << '\n';
    }
  }
}
