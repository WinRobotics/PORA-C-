// =============================================================================
// Exercise 1: Extended Kalman Filter for SLAM
// (C++ port of ch14/exercises/ekf_slam.ipynb)
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
const double dt = 0.1;  // discretization timestep
const MatrixXd Q = [] {
  MatrixXd q = (dt * dt * 0.1) * MatrixXd::Identity(11, 11);  // process noise
  q.bottomRightCorner(8, 8).setZero();
  return q;
}();
const MatrixXd R = 0.25 * MatrixXd::Identity(8, 8);  // observation noise
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
// unknown, along with the robot pose. However, we assume the landmarks are stationary objects
// in the environment, and their combined state vector is described by:
//     m = [m_{1,x}, m_{1,y}, m_{2,x}, m_{2,y}, m_{3,x}, m_{3,y}, m_{4,x}, m_{4,y}]^T.
// As the robot navigates through its environment, it receives noisy measurements of the
// positions of four landmarks in the environment relative to the robot's current pose.
// The measurement for landmark i is the relative position with the measurement model:
//     z_t^i = [cos(θ_t), sin(θ_t); -sin(θ_t), cos(θ_t)] ([m_{i,x}; m_{i,y}] - [x_t; y_t]).
// The full measurement vector of all landmarks is:
//     z_t = [z_t^1; z_t^2; z_t^3; z_t^4].
//
// Implement these models in the functions `robot_dynamics` and `robot_measurement`. Then,
// implement the function `state_dynamics` to define the dynamics of the SLAM state that
// consists of both the robot pose and the landmark positions,
// y = [x, y, θ, m_{1,x}, ..., m_{4,y}]^T.
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
 *     x: combined robot + landmarks state, [x, y, θ, m_1x, m_1y, ...], size (11,)
 *
 * Returns:
 *     Expected measurement from the given SLAM state.
 */
VectorXd robot_measurement(const VectorXd& x) {
  const double px = x[0];
  const double py = x[1];
  const double th = x[2];
  const VectorXd m = x.tail(8);
  // ##### YOUR CODE STARTS HERE #####
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
}

/**
 * Dynamics model for the full combined state of the SLAM problem,
 * which consists of the robot pose and the landmark positions.
 *
 * Args:
 *     y: combined robot + landmark state, [x, y, θ, m_1x, m_1y, ...], size (11,)
 *     u: robot control vector [v, omega] where v is the
 *        linear velocity and omega is the angular velocity
 *
 * Returns:
 *     Expected next SLAM state.
 */
VectorXd state_dynamics(const VectorXd& y, const VectorXd& u) {
  const VectorXd x_robot = y.head(3);
  const VectorXd m = y.tail(8);
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
 * to the SLAM state and robot input commands.
 *
 * Args:
 *     y: combined robot + landmark state, [x, y, θ, m_1x, m_1y, ...], size (11,)
 *     u: robot control vector [v, omega] where v is the
 *        linear velocity and omega is the angular velocity
 *
 * Returns:
 *     Jacobian of the dynamics model, F, shape (11,11)
 */
MatrixXd dynamics_jacobian(const VectorXd& y, const VectorXd& u) {
  const double v = u[0], omega = u[1];
  MatrixXd F;
  // ##### YOUR CODE STARTS HERE #####
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
  return F;
}

/**
 * Computes the Jacobian of the measurement model with respect
 * to the SLAM state.
 *
 * Args:
 *     y: combined robot + landmark state, [x, y, θ, m_1x, m_1y, ...], size (11,)
 *
 * Returns:
 *     Jacobian of the measurement model, H, shape (8,11)
 */
MatrixXd measurement_jacobian(const VectorXd& y) {
  const double px = y[0];
  const double py = y[1];
  const double th = y[2];
  const VectorXd m = y.tail(8);
  MatrixXd H;
  // ##### YOUR CODE STARTS HERE #####
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
  return H;
}

// -----------------------------------------------------------------------------
// ### Exercise 1.4: Extended Kalman Filter SLAM Implementation
// Implement the function `ekf_slam_update` to implement the EKF update.
// -----------------------------------------------------------------------------

/**
 * Given the prior mean and covariance, update the belief using the
 * given control input and measurements.
 *
 * Args:
 *     prior_mean: prior mean SLAM state, shape (11,)
 *     prior_cov: prior covariance matrix, shape (11,11)
 *     u: control input, shape (2,)
 *     z: measurement array, shape (8,)
 *
 * Returns:
 *     Updated mean SLAM state
 *     Update covariance matrix
 */
std::pair<VectorXd, MatrixXd> ekf_slam_update(const VectorXd& prior_mean, const MatrixXd& prior_cov,
                                              const VectorXd& u, const VectorXd& z) {
  VectorXd mean;
  MatrixXd cov;
  // ##### YOUR CODE STARTS HERE #####
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
  return {mean, cov};
}

void plot_error_ellipse(pora::plt::Axes& ax, const VectorXd& mean, const MatrixXd& cov, double alpha = 1,
                        const std::string& label = "") {
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
  ax.ellipse(mean[0], mean[1], width, height, angle, pora::plt::fmt("r-").lw(1).alpha(alpha).label(label));
}

int main() {
  return pora::run([] {
    pora::seed(123456);

    // -------------------------------------------------------------------------
    // ### Exercise 1.5: Simulate EKF SLAM
    // Run the provided code below to simulate the robot EKF SLAM performance given an open-loop
    // control sequence and noisy measurements and dynamics.
    // -------------------------------------------------------------------------

    // Time horizon to simulate
    const double tf = 15;
    VectorXd time = pora::arange(0, tf + dt, dt);
    const int T = static_cast<int>(time.size());

    // Ground-truth states (simulated)
    MatrixXd y = MatrixXd::Zero(11, T);
    y.block(0, 0, 3, 1) << 1, 1, 0;
    y.block(3, 0, 8, 1) = m;

    // Estimated states
    MatrixXd mean_ekf = MatrixXd::Zero(11, T);
    std::vector<MatrixXd> cov_ekf(T, MatrixXd::Identity(11, 11));

    // Initial state and state covariance estimate
    double init_mu_std = 0.1;
    mean_ekf.col(0) = y.col(0) + pora::rng().multivariate_normal(VectorXd::Zero(11), init_mu_std * MatrixXd::Identity(11, 11));

    for (int i = 1; i < T; ++i) {
      // Simulation
      // True robot commands
      double v = 1;
      double omega = std::sin(time[i]);
      VectorXd u(2);
      u << v, omega;

      // True robot dynamics with noise
      VectorXd w_noise = pora::rng().multivariate_normal(VectorXd::Zero(11), Q);
      y.col(i) = state_dynamics(y.col(i - 1), u) + w_noise;

      // True received measurement
      VectorXd v_noise = pora::rng().multivariate_normal(VectorXd::Zero(8), R);
      VectorXd z = robot_measurement(y.col(i)) + v_noise;

      // Estimation
      auto [mean_i, cov_i] = ekf_slam_update(mean_ekf.col(i - 1), cov_ekf[i - 1], u, z);
      mean_ekf.col(i) = mean_i;
      cov_ekf[i] = cov_i;
    }

    // Finally, run the code below to generate some plots to visualize the results.
    {
      pora::plt::Figure fig(1, 3, 1800, 400);
      const char* titles[] = {"Robot X Position", "Robot Y Position", "Robot Orientation"};
      const char* ylabels[] = {"x_r", "y_r", "theta_r"};
      for (int i = 0; i < 3; ++i) {
        auto& ax = fig.ax(i);
        ax.set_title(titles[i]);
        ax.plot(time, y.row(i), pora::plt::fmt("-").lw(2).label("True"));
        ax.plot(time, mean_ekf.row(i), pora::plt::fmt(".").ms(3).label("EKF"));
        ax.set_xlabel("Time");
        ax.set_ylabel(ylabels[i]);
        ax.legend();
        ax.grid(true);
      }
      fig.save("robot_states.png");
    }
    {
      pora::plt::Figure fig(1, 2, 1400, 600);
      const char* titles[] = {"Landmark X Position", "Landmark Y Position"};
      const char* ylabels[] = {"m_x", "m_y"};
      const pora::plt::Color colors[] = {pora::plt::colors::red, pora::plt::colors::tab_blue, pora::plt::colors::green,
                                         pora::plt::colors::black};
      for (int i = 0; i < 2; ++i) {
        auto& ax = fig.ax(i);
        ax.set_title(titles[i]);
        for (int j = 0; j < 4; ++j) {
          ax.plot(time, y.row(3 + i + 2 * j), pora::plt::fmt("-").lw(2).c(colors[j]).label("True"));
          ax.plot(time, mean_ekf.row(3 + i + 2 * j), pora::plt::fmt(".").ms(3).c(colors[j]).label("EKF"));
        }
        ax.set_xlabel("Time");
        ax.set_ylabel(ylabels[i]);
        ax.legend();
        ax.grid(true);
      }
      fig.save("landmark_states.png");
    }

    // #### Plot Error Ellipses Along Trajectory and Landmarks
    {
      pora::plt::Figure fig(1, 1, 1000, 800);
      auto& ax = fig.ax();
      ax.set_title("Estimated Robot Trajectory and Uncertainty");
      ax.plot(y.row(0), y.row(1), pora::plt::fmt("-").lw(2).label("Ground truth robot"));
      ax.plot(mean_ekf.row(0), mean_ekf.row(1), pora::plt::fmt("-").lw(2).c(pora::plt::colors::orange).label("EKF estimate"));
      for (int j = 0; j < 4; ++j) ax.plot(mean_ekf.row(3 + 2 * j), mean_ekf.row(4 + 2 * j), "g+");
      for (int i = 0; i < T; i += 10) {
        std::string label = i == 0 ? "Uncertainty" : "";
        double alpha = 0.5 + 0.5 * i / T;
        plot_error_ellipse(ax, mean_ekf.block(0, i, 2, 1), cov_ekf[i].topLeftCorner(2, 2), alpha, label);
        for (int j = 0; j < 4; ++j) {
          int start = 3 + 2 * j;
          plot_error_ellipse(ax, mean_ekf.block(start, i, 2, 1), cov_ekf[i].block(start, start, 2, 2), alpha);
        }
      }
      ax.set_xlabel("X");
      ax.set_ylabel("Y");
      ax.legend();
      ax.grid(true);
      fig.save("slam_trajectory.png");
    }
  });
}
