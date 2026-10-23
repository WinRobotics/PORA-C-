// =============================================================================
// Exercise 3: Double Integrator Motion Simulation
// (C++ port of ch01/exercises/double_integrator_dynamics.ipynb)
//
// The goal of this exercise is to build a C++ class that defines the dynamics of a
// double integrator.
//
// Unlike in the `nonholonomic_wheeled_robot_dynamics.cpp` exercise, here we will leverage
// *batching* to rollout several trajectories simultaneously (each column of an
// Eigen::MatrixXd is one rollout).
// =============================================================================
#include "dynamics.hpp"

#include <pora/core.hpp>
#include <pora/numerics.hpp>
#include <pora/plot.hpp>
#include <pora/random.hpp>

#include <Eigen/Dense>

#include <cmath>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// #### Double Integrator Dynamics
// The double integrator dynamics are:
//     \dot{x} = v_x, \quad \dot{y} = v_y, \quad \dot{v}_x = a_x, \quad \dot{v}_y = a_y.
// where (x,y) is the position, (v_x, v_y) is the velocity, and (a_x, a_y) is the
// acceleration control input.

// -----------------------------------------------------------------------------
// ### Exercise 3.1 and 3.2: Define Double Integrator Dynamics Class Using Batching
// Complete the implementations of the `step` and `rollout` functions. Note that in the
// `step` method you should leverage the variables `A` and `B` to define a single equation
// with batching to compute multiple trajectories in parallel.
// -----------------------------------------------------------------------------
class DoubleIntegratorDynamics : public Dynamics {
 public:
  DoubleIntegratorDynamics() : Dynamics() {}

  double xdd_max = 0.5;  // m/s^2
  double ydd_max = 0.5;  // m/s^2
  int n = 4;
  int m = 2;

  /**
   * Compute the next state from the current state and control.
   *
   * Args:
   *     x_batch: batch of current states, [x, y, vx, vy], of each rollout, shape (n, num_rollouts)
   *     u: control input, [ax, ay], shape (m,)
   *
   * Returns:
   *     Batch of next states, `dt` seconds in the future, shape (n, num_rollouts)
   */
  MatrixXd step(const MatrixXd& x_batch, const VectorXd& u) override {
    const int num_rollouts = static_cast<int>(x_batch.cols());

    // Define Gaussian Noise
    VectorXd var(4);
    if (noisy == false) {
      var << 0.0, 0.0, 0.0, 0.0;
    } else {
      var << 0.01, 0.01, 0.001, 0.001;
    }

    // Batch the noise
    MatrixXd w_batch(n, num_rollouts);
    for (int j = 0; j < num_rollouts; ++j)
      for (int i = 0; i < n; ++i) w_batch(i, j) = pora::rng().normal(0.0, var[i]);

    // State space dynamics
    MatrixXd A(4, 4);
    A << 1.0, 0.0, dt, 0.0,
         0.0, 1.0, 0.0, dt,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0;

    MatrixXd B(4, 2);
    B << 0.0, 0.0,
         0.0, 0.0,
         dt, 0.0,
         0.0, dt;

    MatrixXd x_batch_next;
    // ##### YOUR CODE STARTS HERE #####
    // Construct `x_batch_next` by applying discrete time dynamics vectorized equations.
    // Will require use of `A` and `B`
    // Hint: Eigen's `.colwise()` / `.replicate()` are the counterparts of NumPy broadcasting.
    throw pora::NotImplementedError("Need to implement code here.");
    // ###### YOUR CODE ENDS HERE ######

    // Add noise
    x_batch_next += w_batch;

    return x_batch_next;
  }

  /**
   * Rollout the trajectory from x0 using the given control sequence.
   *
   * Args:
   *     x0: initial state, [x, y, vx, vy]
   *     u_sequence: sequence of control inputs to apply, shape (m, num_steps)
   *     num_rollouts: number of rollouts to perform
   *
   * Returns:
   *     Vector of length num_steps + 1; rollouts[k] has shape (n, num_rollouts)
   *     (the Python version returns an array of shape (num_steps + 1, n, num_rollouts)).
   */
  std::vector<MatrixXd> rollout(const VectorXd& x0, const MatrixXd& u_sequence, int num_rollouts) override {
    const int num_steps = static_cast<int>(u_sequence.cols());

    std::vector<MatrixXd> rollouts(num_steps + 1, MatrixXd::Zero(n, num_rollouts));
    rollouts[0] = x0.replicate(1, num_rollouts);
    // ##### YOUR CODE STARTS HERE #####
    // Use ONLY one for-loop to loop `num_steps` to populate `rollouts`.
    // Use the `step` function above.
    throw pora::NotImplementedError("Need to implement code here.");
    // ###### YOUR CODE ENDS HERE ######

    return rollouts;
  }
};

int main() {
  return pora::run([] {
    pora::seed(0);

    // Run the script below to simulate the double integrator motion and plot the state and
    // control trajectories.

    // Constants
    const int num_rollouts = 10;
    const int num_steps = 100;

    // Define the dynamics class
    DoubleIntegratorDynamics dynamics;

    // Define the control inputs
    MatrixXd u_sequence = MatrixXd::Zero(dynamics.m, num_steps);
    u_sequence.row(0) = pora::linspace(0, 2 * pora::kPi, num_steps).array().sin().transpose();
    u_sequence.row(1) = pora::linspace(0, 2 * pora::kPi, num_steps).array().cos().transpose();

    // Define the initial state
    VectorXd x0 = VectorXd::Zero(4);

    // Rollouts dynamics
    std::vector<MatrixXd> rollouts = dynamics.rollout(x0, u_sequence, num_rollouts);

    // Plot control inputs
    pora::plt::Figure fig_ctrl(dynamics.m, 1, 700, 450);
    const char* ylabels_ctrl[] = {"xdd", "ydd"};
    double ylims_ctrl[2][2] = {{-1.1, 1.1}, {-1.1, 1.1}};
    for (int j = 0; j < dynamics.m; ++j) {
      fig_ctrl.ax(j).plot(pora::linspace(0, num_steps, num_steps), u_sequence.row(j));
      fig_ctrl.ax(j).set_ylabel(ylabels_ctrl[j]);
      fig_ctrl.ax(j).set_ylim(ylims_ctrl[j][0], ylims_ctrl[j][1]);
    }
    fig_ctrl.suptitle("Control");
    fig_ctrl.save("control.png");

    // Plots state trajectory rollouts
    pora::plt::Figure fig_state(dynamics.n, 1, 700, 800);
    const char* ylabels_state[] = {"x", "y", "xd", "yd"};
    double ylims_state[4][2] = {{-0.5, 0.5}, {-0.5, 0.5}, {-0.1, 0.5}, {-0.3, 0.3}};
    for (int i = 0; i < num_rollouts; ++i) {
      for (int j = 0; j < dynamics.n; ++j) {
        VectorXd traj(num_steps + 1);
        for (int k = 0; k <= num_steps; ++k) traj[k] = rollouts[k](j, i);
        fig_state.ax(j).plot(pora::linspace(0, num_steps + 1, num_steps + 1), traj, "c-");
        fig_state.ax(j).set_ylabel(ylabels_state[j]);
        fig_state.ax(j).set_ylim(ylims_state[j][0], ylims_state[j][1]);
      }
    }
    fig_state.suptitle("State");
    fig_state.save("state.png");
  });
}
