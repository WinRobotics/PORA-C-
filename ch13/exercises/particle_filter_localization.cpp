// =============================================================================
// Exercise 2: Particle Filter for Robot Localization
// (C++ port of ch13/exercises/particle_filter_localization.ipynb)
// =============================================================================
#include <pora/core.hpp>
#include <pora/numerics.hpp>
#include <pora/plot.hpp>
#include <pora/random.hpp>

#include <Eigen/Dense>

#include <cmath>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Global constants
const double dt = 0.1;                                          // discretization timestep
const MatrixXd Q = (dt * dt * 0.1) * MatrixXd::Identity(3, 3);  // process noise
const MatrixXd R = 0.25 * MatrixXd::Identity(8, 8);             // observation noise
const MatrixXd R_inv = R.inverse();                             // precompute meas. noise covariance inverse
const VectorXd m = (VectorXd(8) << 0, 0, 2, 8, 8, 2, 10, 10).finished();  // ground-truth landmark positions.

// -----------------------------------------------------------------------------
// ### Exercise 2.1: Dynamics and Measurement Equations
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
//
// NOTE: if you have already completed Exercise 1: Extended Kalman Filter for Robot
// Localization, you may reuse your solution for this part.
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
// ### Exercise 2.2: Particle Filter Implementation
// Implement the function `particle_filter_update` to implement the particle filter update.
// -----------------------------------------------------------------------------

/**
 * Helper function to compute (unnormalized) likelihood of a point x
 * for a Gaussian distribution with given mean, and *inverse* covariance cov_inv.
 *
 * Args:
 *     x: value to get pdf value at, shape (n,)
 *     mean: mean of the Gaussian, shape (n,)
 *     cov_inv: inverse covariance matrix, shape (n,n)
 *
 * Returns:
 *     Gaussian pdf evaluated at x.
 */
double gaussian_pdf(const VectorXd& x, const VectorXd& mean, const MatrixXd& cov_inv) {
  // note: don't compute the normalization factor because it doesn't matter.
  return std::exp(-0.5 * (x - mean).dot(cov_inv * (x - mean)));
}

/**
 * Given the prior belief particle set, update the belief using the
 * given control input and measurements.
 *
 * Args:
 *     particles: array representing particles, shape (3, num_particles)
 *     u: control input, shape (2,)
 *     z: measurement array, shape (8,)
 *
 * Returns:
 *     Updated particle set, shape (3, num_particles)
 */
MatrixXd particle_filter_update(MatrixXd particles, const VectorXd& u, const VectorXd& z) {
  // Sample particle noises
  const int n = static_cast<int>(particles.rows()), num_particles = static_cast<int>(particles.cols());
  MatrixXd W_particles = Q.llt().matrixL() * pora::rng().normal(n, num_particles);

  // ##### YOUR CODE STARTS HERE #####
  // Hint: Use functions 'robot_dynamics()', 'robot_measurement()' and 'gaussian_pdf()'.
  // Hint: Do not forget to add the process noise to each particle, i.e., 'W_particles'.
  // Hint: Use 'pora::rng().choice(weights, num_particles)' (the counterpart of
  //       np.random.choice()) to resample.
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
  return particles;
}

// Sample covariance of the columns of X (like np.cov(X)).
MatrixXd sample_cov(const MatrixXd& X) {
  MatrixXd centered = X.colwise() - X.rowwise().mean();
  return centered * centered.transpose() / static_cast<double>(X.cols() - 1);
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
    // ### Exercise 2.3: Simulate PF Localization
    // Run the provided code below to simulate the robot PF localization given an open-loop
    // control sequence and noisy measurements and dynamics.
    // -------------------------------------------------------------------------

    // Time horizon to simulate
    const double tf = 15;
    VectorXd time = pora::arange(0, tf + dt, dt);

    // Ground-truth states (simulated)
    MatrixXd x = MatrixXd::Zero(3, time.size());
    x.col(0) << 1, 1, 0;

    // Initial state and state covariance estimate
    double init_mu_std = 0.1;
    VectorXd init_mean = Eigen::Vector3d(1, 1, 0) +
                         pora::rng().multivariate_normal(VectorXd::Zero(3), init_mu_std * MatrixXd::Identity(3, 3));
    MatrixXd init_cov = MatrixXd::Identity(3, 3);

    // Containers for belief
    const int T = static_cast<int>(time.size());
    const int n = static_cast<int>(init_mean.size());
    MatrixXd mean_pf = MatrixXd::Zero(n, T);
    std::vector<MatrixXd> cov_pf(T, MatrixXd::Zero(n, n));

    // Containers for particles
    const int num_particles = 1000;

    // Sample particles (init_cov is symmetric positive definite: its matrix square root is
    // computed from the eigen-decomposition, like scipy.linalg.sqrtm)
    Eigen::SelfAdjointEigenSolver<MatrixXd> es(init_cov);
    MatrixXd sqrt_init_cov = es.operatorSqrt();
    MatrixXd particles = (sqrt_init_cov * pora::rng().normal(n, num_particles)).colwise() + init_mean;
    for (int i = 0; i < T - 1; ++i) {
      // Simulation
      // True robot commands
      double v = 1;
      double omega = std::sin(time[i]);
      VectorXd u(2);
      u << v, omega;

      // True robot dynamics with noise
      VectorXd w_noise = pora::rng().multivariate_normal(VectorXd::Zero(3), Q);
      x.col(i + 1) = robot_dynamics(x.col(i), u) + w_noise;

      // True received measurement
      VectorXd v_noise = pora::rng().multivariate_normal(VectorXd::Zero(8), R);
      VectorXd z = robot_measurement(x.col(i + 1), m) + v_noise;

      // Store the current belief's mean and covariance to 'mean_pf' and 'cov_pf'.
      mean_pf.col(i) = particles.rowwise().mean();
      cov_pf[i] = sample_cov(particles);

      // Estimation.
      particles = particle_filter_update(particles, u, z);
    }

    // Store final belief
    mean_pf.col(T - 1) = particles.rowwise().mean();
    cov_pf[T - 1] = sample_cov(particles);

    // Finally, run the code below to generate some plots to visualize the results.
    {
      pora::plt::Figure fig(1, 3, 1800, 400);
      const char* titles[] = {"X Position", "Y Position", "Orientation"};
      const char* ylabels[] = {"x_r", "y_r", "theta_r"};
      for (int i = 0; i < 3; ++i) {
        auto& ax = fig.ax(i);
        ax.set_title(titles[i]);
        ax.plot(time, x.row(i), pora::plt::fmt("-").lw(2).label("True"));
        ax.plot(time, mean_pf.row(i), pora::plt::fmt(".").ms(3).label("PF"));
        ax.set_xlabel("Time");
        ax.set_ylabel(ylabels[i]);
        ax.legend();
        ax.grid(true);
      }
      fig.save("pf_states.png");
    }

    // #### Plot Error Ellipses Along Trajectory
    {
      pora::plt::Figure fig(1, 1, 1000, 800);
      auto& ax = fig.ax();
      ax.set_title("Estimated Robot Trajectory and Uncertainty");
      ax.plot(x.row(0), x.row(1), pora::plt::fmt("-").lw(2).label("Ground truth robot"));
      ax.plot(mean_pf.row(0), mean_pf.row(1), pora::plt::fmt("-").lw(2).c(pora::plt::colors::orange).label("PF estimate"));
      std::vector<double> mx = {m[0], m[2], m[4], m[6]}, my = {m[1], m[3], m[5], m[7]};
      ax.plot(mx, my, pora::plt::fmt("b.").ms(12).label("Landmark"));
      for (int i = 0; i < T; i += 10) plot_error_ellipse(ax, mean_pf.block(0, i, 2, 1), cov_pf[i].topLeftCorner(2, 2));
      ax.set_xlabel("X");
      ax.set_ylabel("Y");
      ax.legend();
      ax.grid(true);
      fig.save("pf_trajectory.png");
    }
  });
}
