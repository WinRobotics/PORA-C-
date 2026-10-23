// =============================================================================
// Exercise 1: Kalman Filter for Landmark Localization
// (C++ port of ch12/exercises/kalman_filter.ipynb)
// =============================================================================
#include <pora/core.hpp>
#include <pora/numerics.hpp>
#include <pora/plot.hpp>
#include <pora/random.hpp>

#include <Eigen/Dense>

#include <cmath>
#include <utility>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// In this problem, we are going to localize a set of landmarks given knowledge about a robot's
// motion. First, we can define the robot's motion model (not to be confused with the dynamics
// model that we will use to formulate the Kalman filter later).

/**
 * Robot motion model.
 *
 * Args:
 *     x: robot state (pose), [px, py, θ]
 *     v: linear velocity command
 *     omega: angular velocity command
 *     dt: timestep to apply command input
 *
 * Returns:
 *     Next robot state (pose)
 */
VectorXd robot_motion_model(const VectorXd& x, double v, double omega, double dt) {
  VectorXd out(3);
  out << x[0] + dt * v * std::cos(x[2]), x[1] + dt * v * std::sin(x[2]), x[2] + dt * omega;
  return out;
}

/**
 * Open-loop robot linear and angular velocity commands.
 *
 * Args:
 *     t: current time
 *
 * Returns:
 *     Robot linear and angular velocity commands, shape (2,)
 */
VectorXd robot_commands(double t) {
  VectorXd u(2);
  u << 1, std::sin(t);
  return u;
}

// State transition and observation matrices of the Kalman filter (defined in main()).
MatrixXd A, C;

// Process and observation noise.
const MatrixXd Q = 0.0 * MatrixXd::Identity(8, 8);   // Landmarks are stationary, assume zero process noise.
const MatrixXd R = 0.25 * MatrixXd::Identity(8, 8);  // Measurements contain noise, assume observation noise.

/**
 * Given the prior mean and covariance for the set of landmarks, update
 * the belief using the given measurements.
 *
 * Args:
 *     prior_mean: prior mean of landmark positions, shape (8,)
 *     prior_cov: prior covariance matrix, shape (8,8)
 *     z: position measurement array, shape (8,)
 *
 * Returns:
 *     Updated mean pose
 *     Update covariance matrix
 */
std::pair<VectorXd, MatrixXd> kalman_filter_update(const VectorXd& prior_mean, const MatrixXd& prior_cov,
                                                   const VectorXd& z) {
  VectorXd mean;
  MatrixXd cov;
  // ##### YOUR CODE STARTS HERE #####
  // Implement Kalman filtering predict and update steps.
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
  return {mean, cov};
}

void plot_error_ellipse(pora::plt::Axes& ax, const VectorXd& mean, const MatrixXd& cov, double alpha = 1,
                        const std::string& label = "") {
  // Calculate the error ellipse parameters
  Eigen::SelfAdjointEigenSolver<MatrixXd> es(cov);
  VectorXd eigenvalues = es.eigenvalues().reverse();      // descending order
  MatrixXd eigenvectors = es.eigenvectors().rowwise().reverse();
  double angle = pora::rad2deg(std::atan2(eigenvectors(1, 0), eigenvectors(0, 0)));

  // Compute the radius of the ellipse to correspond to the desired confidence level
  double chi2_val = 2.4477;  // Corresponds to 95% conf. interval
  double width = 2 * chi2_val * std::sqrt(eigenvalues[0]);
  double height = 2 * chi2_val * std::sqrt(eigenvalues[1]);

  // Draw the ellipse
  ax.ellipse(mean[0], mean[1], width, height, angle, pora::plt::fmt("r-").lw(1).alpha(alpha).label(label));
}

int main() {
  return pora::run([] {
    pora::seed(123456);

    // Create array of timesteps
    const double dt = 0.2;
    const double tf = 15;
    VectorXd time = pora::arange(0, tf + dt, dt);
    const int T = static_cast<int>(time.size());

    // Ground-truth landmark positions that we will try to estimate.
    MatrixXd xm(4, 2);
    xm << 0, 0,
          2, 8,
          8, 2,
          10, 10;

    // Ground-truth robot states.
    MatrixXd xr = MatrixXd::Zero(3, T);
    xr.col(0) << 1, 1, 0;
    for (int i = 1; i < T; ++i) {
      VectorXd u = robot_commands(time[i]);
      xr.col(i) = robot_motion_model(xr.col(i - 1), u[0], u[1], dt);
    }

    // -------------------------------------------------------------------------
    // ### Exercise 1.1 and 1.2: Kalman Filter Implementation
    // First, assuming we can directly measure the landmark positions, define the matrices A and
    // C that we will use in our Kalman filter implementation (and implement
    // `kalman_filter_update` above). What is the corresponding state the Kalman filter will
    // estimate?
    // -------------------------------------------------------------------------
    // ##### YOUR CODE STARTS HERE #####
    // What are the state transition and observation matrices? (assign the globals A and C)
    throw pora::NotImplementedError("Need to implement code here.");
    // ###### YOUR CODE ENDS HERE ######

    // -------------------------------------------------------------------------
    // ### Exercise 1.3: Simulate and Analyze Results
    // Run the code provided below to simulate your Kalman filter with a noisy initial estimate.
    // -------------------------------------------------------------------------
    VectorXd xm_flat(8);
    xm_flat << xm(0, 0), xm(0, 1), xm(1, 0), xm(1, 1), xm(2, 0), xm(2, 1), xm(3, 0), xm(3, 1);

    // Ground-truth states (simulated).
    MatrixXd x = MatrixXd::Zero(8, T);
    x.col(0) = xm_flat;

    // Estimated states (Kalman Filter).
    MatrixXd mean_kf = MatrixXd::Zero(8, T);
    std::vector<MatrixXd> cov_kf(T, MatrixXd::Identity(8, 8));

    // Initial state and state covariance estimate.
    double init_mean_std = 0.5;
    mean_kf.col(0) = xm_flat + init_mean_std * pora::rng().randn(8);
    cov_kf[0] = init_mean_std * MatrixXd::Identity(8, 8);

    // Run simulation.
    for (int i = 1; i < T; ++i) {
      // Simulate true landmark dynamics
      x.col(i) = A * x.col(i - 1);

      // Simulate noisy received measurement
      VectorXd z_noise = pora::rng().multivariate_normal(VectorXd::Zero(8), R);
      VectorXd z = C * x.col(i) + z_noise;

      // Estimation.
      auto [mean_i, cov_i] = kalman_filter_update(mean_kf.col(i - 1), cov_kf[i - 1], z);
      mean_kf.col(i) = mean_i;
      cov_kf[i] = cov_i;
    }

    // ### Plot Mean Estimates of 2D Landmark Positions
    // Run the provided code below to plot the mean estimates to see how well the filter performed.
    {
      pora::plt::Figure fig(1, 2, 1400, 600);
      const char* coord[2] = {"X", "Y"};
      for (int c = 0; c < 2; ++c) {
        auto& ax = fig.ax(c);
        ax.set_title(std::string("Kalman Filter Estimates of Landmark ") + coord[c] + "-coordinates");
        for (int i = 0; i < 4; ++i) {
          ax.plot(time, x.row(2 * i + c), pora::plt::fmt("-").lw(2).c(pora::plt::colors::red).label(i == 0 ? "Ground Truth" : ""));
          ax.plot(time, mean_kf.row(2 * i + c), pora::plt::fmt(".").ms(3).c(pora::plt::colors::tab_blue).label(i == 0 ? "Estimate" : ""));
        }
        ax.set_xlabel("Time");
        ax.set_ylabel(std::string("Landmark ") + (c == 0 ? "x" : "y") + "-coordinates");
        ax.legend();
        ax.grid(true);
      }
      fig.save("landmark_estimates.png");
    }

    // ### Plot Error Ellipses of 2D Landmarks Along Trajectory
    // Run the provided code below to see a visualization of the uncertainty in the 2D plane.
    {
      pora::plt::Figure fig(1, 1, 1000, 800);
      auto& ax = fig.ax();
      ax.set_title("Estimated Landmark Positions and Error");
      ax.plot(xr.row(0), xr.row(1), pora::plt::fmt("-").lw(2).label("Robot Path"));
      for (int i = 0; i < 4; ++i)
        ax.plot(x.row(2 * i), x.row(2 * i + 1), pora::plt::fmt("b.").ms(12).label(i == 0 ? "Landmark" : ""));
      for (int i = 0; i < T; i += 10) {
        double alpha = 0.5 + 0.5 * i / T;
        for (int j = 0; j < 4; ++j) {
          std::string label = (i == 0 && j == 0) ? "Uncertainty" : "";
          plot_error_ellipse(ax, mean_kf.block(2 * j, i, 2, 1), cov_kf[i].block(2 * j, 2 * j, 2, 2), alpha, label);
        }
      }
      ax.set_xlabel("X position");
      ax.set_ylabel("Y position");
      ax.legend();
      ax.grid(true);
      fig.save("error_ellipses.png");
    }
  });
}
