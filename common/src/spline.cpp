#include "pora/spline.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pora {

namespace {

struct Fit {
  Eigen::VectorXd g, gamma;
  double rss;
};

// Penalised fit  min ||y - g||^2 + lambda * ∫ g''^2  (Green & Silverman, ch. 2).
Fit fit(const Eigen::VectorXd& x, const Eigen::VectorXd& y, double lambda) {
  const int n = static_cast<int>(x.size());
  Fit f;
  if (n < 3) {
    f.g = y;
    f.gamma = Eigen::VectorXd::Zero(n);
    f.rss = 0;
    return f;
  }
  Eigen::VectorXd h = x.tail(n - 1) - x.head(n - 1);
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(n, n - 2), R = Eigen::MatrixXd::Zero(n - 2, n - 2);
  for (int j = 1; j < n - 1; ++j) {
    Q(j - 1, j - 1) = 1.0 / h[j - 1];
    Q(j, j - 1) = -1.0 / h[j - 1] - 1.0 / h[j];
    Q(j + 1, j - 1) = 1.0 / h[j];
    R(j - 1, j - 1) = (h[j - 1] + h[j]) / 3.0;
    if (j < n - 2) R(j - 1, j) = R(j, j - 1) = h[j] / 6.0;
  }
  Eigen::VectorXd gamma_inner = (R + lambda * Q.transpose() * Q).ldlt().solve(Q.transpose() * y);
  f.g = y - lambda * Q * gamma_inner;
  f.gamma = Eigen::VectorXd::Zero(n);
  f.gamma.segment(1, n - 2) = gamma_inner;
  f.rss = (y - f.g).squaredNorm();
  return f;
}

}  // namespace

SmoothingSpline::SmoothingSpline(const Eigen::VectorXd& x, const Eigen::VectorXd& y, double s) : x_(x) {
  if (x.size() != y.size() || x.size() < 2) throw std::invalid_argument("SmoothingSpline: need >= 2 points");
  for (int i = 1; i < x.size(); ++i)
    if (!(x[i] > x[i - 1])) throw std::invalid_argument("SmoothingSpline: x must be strictly increasing");

  Fit best = fit(x, y, 0.0);
  if (s > 0 && x.size() >= 3) {
    // RSS grows monotonically with lambda: bisect on log(lambda) for RSS == s.
    double lo = -12, hi = 12;
    Fit f_hi = fit(x, y, std::pow(10.0, hi));
    if (f_hi.rss <= s) {
      best = f_hi;
    } else {
      for (int it = 0; it < 80; ++it) {
        double mid = 0.5 * (lo + hi);
        Fit f = fit(x, y, std::pow(10.0, mid));
        if (f.rss > s) hi = mid;
        else { lo = mid; best = f; }
      }
    }
  }
  g_ = best.g;
  gamma_ = best.gamma;
}

double SmoothingSpline::operator()(double t) const {
  const int n = static_cast<int>(x_.size());
  int i = static_cast<int>(std::upper_bound(x_.data(), x_.data() + n, t) - x_.data()) - 1;
  i = std::min(std::max(i, 0), n - 2);
  double h = x_[i + 1] - x_[i];
  if (t < x_[0] || t > x_[n - 1]) {
    // linear extrapolation (natural spline)
    int e = t < x_[0] ? 0 : n - 1;
    double slope = t < x_[0] ? (g_[1] - g_[0]) / h - h / 6.0 * gamma_[1]
                             : (g_[n - 1] - g_[n - 2]) / h + h / 6.0 * gamma_[n - 2];
    return g_[e] + slope * (t - x_[e]);
  }
  double a = t - x_[i], b = x_[i + 1] - t;
  return (a * g_[i + 1] + b * g_[i]) / h -
         a * b / 6.0 * ((1.0 + a / h) * gamma_[i + 1] + (1.0 + b / h) * gamma_[i]);
}

Eigen::VectorXd SmoothingSpline::operator()(const Eigen::VectorXd& t) const {
  Eigen::VectorXd out(t.size());
  for (int i = 0; i < t.size(); ++i) out[i] = (*this)(t[i]);
  return out;
}

}  // namespace pora
