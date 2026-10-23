// Random number helpers mirroring the parts of `np.random` used in the
// exercises. Note: the C++ streams differ from NumPy's, so simulated noise (and
// therefore plots) will not be bit-identical to the Python notebooks.
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace pora {

class Rng {
 public:
  explicit Rng(unsigned long long seed = 0) : engine_(seed) {}

  void seed(unsigned long long s) { engine_.seed(s); }

  // Standard normal sample, like np.random.randn().
  double normal() { return std::normal_distribution<double>(0.0, 1.0)(engine_); }

  // Normal sample with mean `loc` and standard deviation `scale`, like np.random.normal(loc, scale).
  double normal(double loc, double scale) {
    if (scale == 0.0) return loc;
    return std::normal_distribution<double>(loc, scale)(engine_);
  }

  // Matrix of standard normal samples, like np.random.normal(size=(rows, cols)).
  Eigen::MatrixXd normal(int rows, int cols) {
    Eigen::MatrixXd out(rows, cols);
    for (int j = 0; j < cols; ++j)
      for (int i = 0; i < rows; ++i) out(i, j) = normal();
    return out;
  }

  // Vector of standard normal samples, like np.random.randn(n).
  Eigen::VectorXd randn(int n) { return normal(n, 1).col(0); }

  // Uniform sample in [low, high), like np.random.uniform(low, high).
  double uniform(double low = 0.0, double high = 1.0) {
    return std::uniform_real_distribution<double>(low, high)(engine_);
  }

  // Uniform sample in [0, 1), like np.random.rand().
  double rand() { return uniform(0.0, 1.0); }

  // Uniform integer in [low, high), like rng.integers(low, high).
  int integers(int low, int high) { return std::uniform_int_distribution<int>(low, high - 1)(engine_); }

  // Index sampled from the discrete distribution `p`, like np.random.choice(len(p), p=p).
  int choice(const std::vector<double>& p) {
    return std::discrete_distribution<int>(p.begin(), p.end())(engine_);
  }
  int choice(const Eigen::VectorXd& p) {
    return choice(std::vector<double>(p.data(), p.data() + p.size()));
  }

  // `size` indices sampled with replacement from the distribution `p`,
  // like np.random.choice(len(p), size=size, p=p).
  std::vector<int> choice(const Eigen::VectorXd& p, int size) {
    std::discrete_distribution<int> dist(p.data(), p.data() + p.size());
    std::vector<int> out(size);
    for (auto& v : out) v = dist(engine_);
    return out;
  }

  // Random permutation of 0..n-1, like rng.permutation(n).
  std::vector<int> permutation(int n) {
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), engine_);
    return idx;
  }

  // Sample from N(mean, cov), like np.random.multivariate_normal(mean, cov).
  // Works with singular (positive semi-definite) covariances.
  Eigen::VectorXd multivariate_normal(const Eigen::VectorXd& mean, const Eigen::MatrixXd& cov) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(cov);
    Eigen::VectorXd sqrt_eig = es.eigenvalues().cwiseMax(0.0).cwiseSqrt();
    Eigen::MatrixXd L = es.eigenvectors() * sqrt_eig.asDiagonal();
    return mean + L * randn(static_cast<int>(mean.size()));
  }

  std::mt19937_64& engine() { return engine_; }

 private:
  std::mt19937_64 engine_;
};

// Seed of the global generator: the environment variable PORA_SEED if set, else 0.
inline unsigned long long default_seed() {
  const char* s = std::getenv("PORA_SEED");
  return s ? std::stoull(s) : 0ULL;
}

// Global generator, the counterpart of NumPy's global random state. Exercises
// that do not call pora::seed() explicitly (like the unseeded notebooks) give
// different random outcomes when run with different PORA_SEED values.
inline Rng& rng() {
  static Rng global(default_seed());
  return global;
}

// Seed the global generator, like np.random.seed(s).
inline void seed(unsigned long long s) { rng().seed(s); }

}  // namespace pora
