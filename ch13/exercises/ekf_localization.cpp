// =============================================================================
// Exercise 1: Extended Kalman Filter for Robot Localization
// (C++ port of ch13/exercises/ekf_localization.ipynb)
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

// Global constants
const double dt = 0.1;                                          // discretization timestep
const MatrixXd Q = (dt * dt * 0.1) * MatrixXd::Identity(3, 3);  // process noise
const MatrixXd R = 0.25 * MatrixXd::Identity(8, 8);             // observation noise
const VectorXd m = (VectorXd(8) << 0, 0, 2, 8, 8, 2, 10, 10).finished();  // ground-truth landmark positions.

// -----------------------------------------------------------------------------
// ### Exercise 1.1: Dynamics and Measurement Equations
// Consider the discrete-time robot dynamics model:
//     x_{t+1} = x_{t} + V_{t} \cos(\theta_{t}) \Delta t,
//     y_{t+1} = y_{t} + V_{t} \sin(\theta_{t}) \Delta t
//     \theta_{t+1} = \theta_{t} + \omega_t \Delta t,
// where (x,y) is the robot position, θ is the heading, Δt is the discretization timestep, and
// (V,ω) are the speed and angular velocity commands.
//
// In this problem, we assume the global ground truth positions of the four landmarks are
// known. The landmarks are stationary objects in the environment, and their combined state
// vector is:
//     m = [m_{1,x}, m_{1,y}, m_{2,x}, m_{2,y}, m_{3,x}, m_{3,y}, m_{4,x}, m_{4,y}]^T.
// As the robot navigates through its environment, it receives noisy measurements of the
// positions of four landmarks in the environment relative to the robot's current pose.
// The measurement for landmark i is the relative position with the measurement model:
//     z_t^i = [cos(θ_t), sin(θ_t); -sin(θ_t), cos(θ_t)] ([m_{i,x}; m_{i,y}] - [x_t; y_t]).
// The full measurement vector of all landmarks is:
//     z_t = [z_t^1; z_t^2; z_t^3; z_t^4].
//
// Implement these models in the functions `robot_dynamics` and `robot_measurement`.
// -----------------------------------------------------------------------------

/**
 * Robot dynamics model.
 *
 * Args:
 *     x: robot state, [x, y, θ]
 *     u: robot control vector [v, omega] where v is the
 *        linear velocity and omega is the angular velocity
 *
 * Returns:
 *     Expected next robot state.
 */
VectorXd robot_dynamics(const VectorXd& x, const VectorXd& u) {
  // ##### YOUR CODE STARTS HERE #####
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
}

/**
 * Robot measurement model that measures the relative position of the landmarks.
 *
 * Args:
 *     x: robot state, [px, py, θ]
 *     m: landmark positions, [m_1,x, m_1,y, ..., m_4,y], size (8,)
 *
 * Returns:
 *     Expected measurement from the given robot pose.
 */
VectorXd robot_measurement(const VectorXd& x, const VectorXd& m) {
  const double px = x[0];
  const double py = x[1];
  const double th = x[2];
  // ##### YOUR CODE STARTS HERE #####
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
}

// -----------------------------------------------------------------------------
// ### Exercise 1.2 and 1.3: Model Jacobians
// Implement the functions `dynamics_jacobian` and `measurement_jacobian` to define the
// Jacobians of the dynamics and measurement models you implemented above.
// -----------------------------------------------------------------------------

/**
 * Computes the Jacobian of the dynamics model with respect
 * to the robot state and input commands.
 *
 * Args:
 *     x: robot state, [x, y, θ]
 *     u: robot control vector [v, omega] where v is the
 *        linear velocity and omega is the angular velocity
 *
 * Returns:
 *     Jacobian of the dynamics model, F, shape (3,3)
 */
MatrixXd dynamics_jacobian(const VectorXd& x, const VectorXd& u) {
  const double v = u[0], omega = u[1];
  MatrixXd F;
  // ##### YOUR CODE STARTS HERE #####
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
  return F;
}

/**
 * Computes the Jacobian of the measurement model with respect
 * to the robot state given the landmark positions.
 *
 * Args:
 *     x: robot state, [px, py, θ]
 *     m: landmark positions, [m_1,x, m_1,y, ..., m_4,y], size (8,)
 *
 * Returns:
 *     Jacobian of the measurement model, H, shape (8,3)
 */
MatrixXd measurement_jacobian(const VectorXd& x, const VectorXd& m) {
  const double px = x[0];
  const double py = x[1];
  const double th = x[2];
  MatrixXd H;
  // ##### YOUR CODE STARTS HERE #####
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
  return H;
}

// -----------------------------------------------------------------------------
// ### Exercise 1.4: Extended Kalman Filter Implementation
// Implement the function `ekf_localization_update` to implement the EKF update.
// -----------------------------------------------------------------------------

/**
 * Given the prior mean and covariance, update the belief using the
 * given control input and measurements.
 *
 * Args:
 *     prior_mean: prior mean robot pose, shape (3,)
 *     prior_cov: prior covariance matrix, shape (3,3)
 *     u: control input, shape (2,)
 *     z: measurement array, shape (8,)
 *
 * Returns:
 *     Updated mean pose
 *     Update covariance matrix
 */
std::pair<VectorXd, MatrixXd> ekf_localization_update(const VectorXd& prior_mean, const MatrixXd& prior_cov,
                                                      const VectorXd& u, const VectorXd& z) {
  VectorXd mean;
  MatrixXd cov;
  // ##### YOUR CODE STARTS HERE #####
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
  return {mean, cov};
}

void plot_error_ellipse(pora::plt::Axes& ax, const VectorXd& mean, const MatrixXd& cov) {
  // Calculate the error ellipse parameters
  Eigen::SelfAdjointEigenSolver<MatrixXd> es(cov);
  VectorXd eigenvalues = es.eigenvalues().reverse();  // descending order
  MatrixXd eigenvectors = es.eigenvectors().rowwise().reverse();
  double angle = pora::rad2deg(std::atan2(eigenvectors(1, 0), eigenvectors(0, 0)));

  // Compute the radius of the ellipse to correspond to the desired confidence level
  double chi2_val = 2.4477;  // Corresponds to 95% conf. interval
  double width = 2 * chi2_val * std::sqrt(eigenvalues[0]);
  double height = 2 * chi2_val * std::sqrt(eigenvalues[1]);

  // Draw the ellipse
  ax.ellipse(mean[0], mean[1], width, height, angle, pora::plt::fmt("r-").lw(2));
}

int main() {
  return pora::run([] {
    pora::seed(123456);

    // -------------------------------------------------------------------------
    // ### Exercise 1.5: Simulate EKF Localization
    // Run the provided code below to simulate the robot EKF localization given an open-loop
    // control sequence and noisy measurements and dynamics.
    // -------------------------------------------------------------------------

    // Time horizon to simulate
    const double tf = 15;
    VectorXd time = pora::arange(0, tf + dt, dt);
    const int T = static_cast<int>(time.size());

    // Ground-truth states (simulated)
    MatrixXd x = MatrixXd::Zero(3, T);
    x.col(0) << 1, 1, 0;

    // Estimated states
    MatrixXd mean_ekf = MatrixXd::Zero(3, T);
    std::vector<MatrixXd> cov_ekf(T, MatrixXd::Identity(3, 3));

    // Initial state and state covariance estimate
    double init_mu_std = 0.1;
    mean_ekf.col(0) = Eigen::Vector3d(1, 1, 0) +
                      pora::rng().multivariate_normal(VectorXd::Zero(3), init_mu_std * MatrixXd::Identity(3, 3));

    for (int i = 1; i < T; ++i) {
      // Simulation
      // True robot commands
      double v = 1;
      double omega = std::sin(time[i]);
      VectorXd u(2);
      u << v, omega;

      // True robot dynamics with noise
      VectorXd w_noise = pora::rng().multivariate_normal(VectorXd::Zero(3), Q);
      x.col(i) = robot_dynamics(x.col(i - 1), u) + w_noise;

      // True received measurement
      VectorXd v_noise = pora::rng().multivariate_normal(VectorXd::Zero(8), R);
      VectorXd z = robot_measurement(x.col(i), m) + v_noise;

      // Estimation
      auto [mean_i, cov_i] = ekf_localization_update(mean_ekf.col(i - 1), cov_ekf[i - 1], u, z);
      mean_ekf.col(i) = mean_i;
      cov_ekf[i] = cov_i;
    }

    // Finally, run the code below to generate some plots to visualize the results.
    {
      pora::plt::Figure fig(1, 3, 1800, 400);
      const char* titles[] = {"X Position", "Y Position", "Orientation"};
      const char* ylabels[] = {"x_r", "y_r", "theta_r"};
      for (int i = 0; i < 3; ++i) {
        auto& ax = fig.ax(i);
        ax.set_title(titles[i]);
        ax.plot(time, x.row(i), pora::plt::fmt("-").lw(2).label("True"));
        ax.plot(time, mean_ekf.row(i), pora::plt::fmt(".").ms(3).label("EKF"));
        ax.set_xlabel("Time");
        ax.set_ylabel(ylabels[i]);
        ax.legend();
        ax.grid(true);
      }
      fig.save("ekf_states.png");
    }

    // #### Plot Error Ellipses Along Trajectory
    {
      pora::plt::Figure fig(1, 1, 1000, 800);
      auto& ax = fig.ax();
      ax.set_title("Estimated Robot Trajectory and Uncertainty");
      ax.plot(x.row(0), x.row(1), pora::plt::fmt("-").lw(2).label("Ground truth robot"));
      ax.plot(mean_ekf.row(0), mean_ekf.row(1), pora::plt::fmt("-").lw(2).c(pora::plt::colors::orange).label("EKF estimate"));
      std::vector<double> mx = {m[0], m[2], m[4], m[6]}, my = {m[1], m[3], m[5], m[7]};
      ax.plot(mx, my, pora::plt::fmt("b.").ms(12).label("Landmark"));
      for (int i = 0; i < T; i += 10) plot_error_ellipse(ax, mean_ekf.block(0, i, 2, 1), cov_ekf[i].topLeftCorner(2, 2));
      ax.set_xlabel("X");
      ax.set_ylabel("Y");
      ax.legend();
      ax.grid(true);
      fig.save("ekf_trajectory.png");
    }
  });
}
