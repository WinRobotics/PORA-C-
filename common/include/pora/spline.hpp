// Cubic smoothing spline, the C++ stand-in for
// `scipy.interpolate.make_splrep(x, y, s=s)` + `scipy.interpolate.splev`.
//
// It finds the natural cubic spline g that minimises  ∫ g''(t)^2 dt  subject to
// sum_i (y_i - g(x_i))^2 <= s  (Reinsch's formulation, which is also the
// smoothing condition used by FITPACK/SciPy). s = 0 gives the interpolating spline.
#pragma once

#include <Eigen/Dense>

namespace pora {

class SmoothingSpline {
 public:
  SmoothingSpline() = default;
  // x must be strictly increasing.
  SmoothingSpline(const Eigen::VectorXd& x, const Eigen::VectorXd& y, double s);

  double operator()(double t) const;
  Eigen::VectorXd operator()(const Eigen::VectorXd& t) const;

  const Eigen::VectorXd& knots() const { return x_; }
  const Eigen::VectorXd& values() const { return g_; }

 private:
  Eigen::VectorXd x_, g_, gamma_;  // knots, fitted values, second derivatives
};

// Same as SmoothingSpline(x, y, s); named after the SciPy function it replaces.
inline SmoothingSpline make_splrep(const Eigen::VectorXd& x, const Eigen::VectorXd& y, double s) {
  return SmoothingSpline(x, y, s);
}

// Evaluate a spline at the points t, like scipy.interpolate.splev(t, spline, der=0).
inline Eigen::VectorXd splev(const Eigen::VectorXd& t, const SmoothingSpline& spline) { return spline(t); }

}  // namespace pora
