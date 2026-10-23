// Numerical helpers that stand in for the NumPy / SciPy / JAX functions used by
// the exercises (np.linspace, np.interp, scipy.integrate.odeint, jax.jacobian,
// scipy.linalg.solve_discrete_are, ...).
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

namespace pora {

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

constexpr double kPi = 3.14159265358979323846;
constexpr double kInf = std::numeric_limits<double>::infinity();

inline double deg2rad(double deg) { return deg * kPi / 180.0; }
inline double rad2deg(double rad) { return rad * 180.0 / kPi; }

// np.linspace(start, stop, num)
inline Vec linspace(double start, double stop, int num) {
  if (num == 1) return Vec::Constant(1, start);
  return Vec::LinSpaced(num, start, stop);
}

// np.arange(start, stop, step)
inline Vec arange(double start, double stop, double step) {
  int n = static_cast<int>(std::ceil((stop - start) / step));
  n = std::max(n, 0);
  Vec out(n);
  for (int i = 0; i < n; ++i) out[i] = start + i * step;
  return out;
}

// np.cumsum(x)
inline Vec cumsum(const Vec& x) {
  Vec out(x.size());
  double s = 0.0;
  for (int i = 0; i < x.size(); ++i) out[i] = (s += x[i]);
  return out;
}

// np.interp(x, xp, fp) for increasing xp (values outside are clamped).
inline double interp(double x, const Vec& xp, const Vec& fp) {
  const int n = static_cast<int>(xp.size());
  if (x <= xp[0]) return fp[0];
  if (x >= xp[n - 1]) return fp[n - 1];
  int hi = static_cast<int>(std::upper_bound(xp.data(), xp.data() + n, x) - xp.data());
  int lo = hi - 1;
  double w = (x - xp[lo]) / (xp[hi] - xp[lo]);
  return (1.0 - w) * fp[lo] + w * fp[hi];
}

// np.clip for scalars
inline double clip(double x, double lo, double hi) { return std::min(std::max(x, lo), hi); }

// Jacobian of f at x by central finite differences. This is the C++ stand-in
// for `jax.jacobian(f)(x)`: J(i, j) = d f_i / d x_j.
inline Mat jacobian(const std::function<Vec(const Vec&)>& f, const Vec& x, double eps = 1e-6) {
  Vec fx = f(x);
  Mat J(fx.size(), x.size());
  for (int j = 0; j < x.size(); ++j) {
    double h = eps * std::max(1.0, std::abs(x[j]));
    Vec xp = x, xm = x;
    xp[j] += h;
    xm[j] -= h;
    J.col(j) = (f(xp) - f(xm)) / (2.0 * h);
  }
  return J;
}

// Rank of a matrix, like np.linalg.matrix_rank.
inline int matrix_rank(const Mat& M) {
  if (M.size() == 0) return 0;
  Eigen::JacobiSVD<Mat> svd(M);
  const Vec& s = svd.singularValues();
  double tol = s.size() ? s[0] * std::max(M.rows(), M.cols()) * std::numeric_limits<double>::epsilon() : 0.0;
  int rank = 0;
  for (int i = 0; i < s.size(); ++i) rank += (s[i] > tol);
  return rank;
}

// scipy.linalg.block_diag(A, B)
inline Mat block_diag(const Mat& A, const Mat& B) {
  Mat out = Mat::Zero(A.rows() + B.rows(), A.cols() + B.cols());
  out.topLeftCorner(A.rows(), A.cols()) = A;
  out.bottomRightCorner(B.rows(), B.cols()) = B;
  return out;
}

// Solution P of the discrete-time algebraic Riccati equation
//   P = A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A + Q,
// like scipy.linalg.solve_discrete_are(A, B, Q, R). Uses the structure
// preserving doubling algorithm.
inline Mat solve_discrete_are(const Mat& A, const Mat& B, const Mat& Q, const Mat& R,
                              double tol = 1e-12, int max_iters = 200) {
  const int n = static_cast<int>(A.rows());
  const Mat I = Mat::Identity(n, n);
  Mat Ak = A;
  Mat Gk = B * R.ldlt().solve(B.transpose());
  Mat Hk = Q;
  for (int k = 0; k < max_iters; ++k) {
    Eigen::PartialPivLU<Mat> W(I + Gk * Hk);
    Mat WinvA = W.solve(Ak);
    Mat WinvG = W.solve(Gk);
    Mat A_next = Ak * WinvA;
    Mat G_next = Gk + Ak * WinvG * Ak.transpose();
    Mat H_next = Hk + Ak.transpose() * Hk * WinvA;
    double change = (H_next - Hk).norm() / std::max(1.0, H_next.norm());
    Ak = A_next;
    Gk = 0.5 * (G_next + G_next.transpose());
    Hk = 0.5 * (H_next + H_next.transpose());
    if (change < tol) return Hk;
  }
  throw std::runtime_error("solve_discrete_are did not converge");
}

// Integrate the ODE dx/dt = f(x, t) from x(t[0]) = x0 and return the solution
// at every time in `t` (one row per time), like scipy.integrate.odeint(f, x0, t).
// Uses an adaptive Dormand-Prince 5(4) Runge-Kutta scheme.
inline Mat odeint(const std::function<Vec(const Vec&, double)>& f, const Vec& x0, const Vec& t,
                  double rtol = 1.49012e-8, double atol = 1.49012e-8) {
  static const double c2 = 1.0 / 5, c3 = 3.0 / 10, c4 = 4.0 / 5, c5 = 8.0 / 9;
  static const double a21 = 1.0 / 5;
  static const double a31 = 3.0 / 40, a32 = 9.0 / 40;
  static const double a41 = 44.0 / 45, a42 = -56.0 / 15, a43 = 32.0 / 9;
  static const double a51 = 19372.0 / 6561, a52 = -25360.0 / 2187, a53 = 64448.0 / 6561, a54 = -212.0 / 729;
  static const double a61 = 9017.0 / 3168, a62 = -355.0 / 33, a63 = 46732.0 / 5247, a64 = 49.0 / 176,
                      a65 = -5103.0 / 18656;
  static const double b1 = 35.0 / 384, b3 = 500.0 / 1113, b4 = 125.0 / 192, b5 = -2187.0 / 6784, b6 = 11.0 / 84;
  static const double e1 = 71.0 / 57600, e3 = -71.0 / 16695, e4 = 71.0 / 1920, e5 = -17253.0 / 339200,
                      e6 = 22.0 / 525, e7 = -1.0 / 40;

  const int n = static_cast<int>(x0.size());
  Mat out(t.size(), n);
  if (t.size() == 0) return out;
  out.row(0) = x0.transpose();
  Vec x = x0;
  double h = 0.0;
  for (int k = 1; k < t.size(); ++k) {
    double tc = t[k - 1];
    const double tf = t[k];
    const double span = tf - tc;
    if (span == 0.0) {
      out.row(k) = x.transpose();
      continue;
    }
    const double dir = span > 0 ? 1.0 : -1.0;
    if (h == 0.0 || std::abs(h) > std::abs(span)) h = span;
    h = dir * std::abs(h);
    int steps = 0;
    Vec k1 = f(x, tc);
    while (dir * (tf - tc) > 1e-14 * std::max(1.0, std::abs(tf))) {
      if (++steps > 1000000) throw std::runtime_error("odeint: too many steps");
      if (dir * (tc + h - tf) > 0) h = tf - tc;
      Vec k2 = f(x + h * (a21 * k1), tc + c2 * h);
      Vec k3 = f(x + h * (a31 * k1 + a32 * k2), tc + c3 * h);
      Vec k4 = f(x + h * (a41 * k1 + a42 * k2 + a43 * k3), tc + c4 * h);
      Vec k5 = f(x + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4), tc + c5 * h);
      Vec k6 = f(x + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5), tc + h);
      Vec x_new = x + h * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6);
      Vec k7 = f(x_new, tc + h);
      Vec err = h * (e1 * k1 + e3 * k3 + e4 * k4 + e5 * k5 + e6 * k6 + e7 * k7);
      Vec scale = (atol + rtol * x.cwiseAbs().cwiseMax(x_new.cwiseAbs()).array()).matrix();
      double err_norm = std::sqrt((err.cwiseQuotient(scale)).squaredNorm() / std::max(n, 1));
      if (!std::isfinite(err_norm)) err_norm = 1e10;
      if (err_norm <= 1.0) {
        tc += h;
        x = x_new;
        k1 = k7;
        double factor = err_norm == 0.0 ? 5.0 : std::min(5.0, std::max(0.2, 0.9 * std::pow(err_norm, -0.2)));
        h *= factor;
      } else {
        h *= std::max(0.2, 0.9 * std::pow(err_norm, -0.2));
      }
      if (std::abs(h) < 1e-14 * std::max(1.0, std::abs(tc))) throw std::runtime_error("odeint: step size too small");
    }
    out.row(k) = x.transpose();
  }
  return out;
}

}  // namespace pora
