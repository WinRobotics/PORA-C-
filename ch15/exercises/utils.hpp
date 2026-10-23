// C++ port of ch15/exercises/utils.py
#pragma once

#include <pora/numerics.hpp>
#include <pora/plot.hpp>

#include <Eigen/Dense>

#include <cmath>
#include <string>

/** Compute root mean square error. */
inline double rmse(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
  return std::sqrt((y - x).array().square().mean());
}

/**
 * Determine whether the system x+ = Ax + Bu with measurement
 * y = Cx is observable.
 */
inline bool observable(const Eigen::MatrixXd& A, const Eigen::MatrixXd& C) {
  const int n = static_cast<int>(A.rows());
  const int o = static_cast<int>(C.rows());
  Eigen::MatrixXd O = Eigen::MatrixXd::Zero(n * o, n);
  Eigen::MatrixXd powerA = Eigen::MatrixXd::Identity(n, n);
  for (int i = 0; i < n; ++i) {
    O.middleRows(i * o, o) = C * powerA;
    powerA = powerA * A;
  }
  return pora::matrix_rank(O) == n;
}

/** Plot car state and estimate trajectories (saved as `filename`). */
inline void plot_car_estimation(const Eigen::MatrixXd& x, const Eigen::MatrixXd& mean, double T,
                                const std::string& filename) {
  const bool with_bias = x.rows() == 4;
  pora::plt::Figure fig(with_bias ? 4 : 3, 1, 1200, with_bias ? 800 : 600);
  Eigen::VectorXd t = T * Eigen::VectorXd::LinSpaced(x.cols(), 0, x.cols() - 1);
  const char* ylabels[] = {"Position [m]", "Velocity [m]", "Acceleration [m]", "GNSS Bias [m]"};
  for (int i = 0; i < x.rows(); ++i) {
    auto& ax = fig.ax(i);
    ax.plot(t, x.row(i), "-", "true");
    ax.plot(t, mean.row(i), "--", "est");
    ax.set_xlabel("Time [s]");
    ax.set_ylabel(ylabels[i]);
    ax.legend();
  }
  fig.save(filename);
}
