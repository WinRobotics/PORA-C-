// =============================================================================
// Exercise 2: Nonholonomic Wheeled Robot Motion Simulation
// (C++ port of ch01/exercises/nonholonomic_wheeled_robot_dynamics.ipynb)
//
// The goal of this exercise is to build a C++ class that defines the dynamics of a
// nonholonomic wheeled robot that we can model using a kinematic unicycle model.
//
// Unlike in the `simulation.cpp` exercise, here we will leverage C++ classes and
// inheritance which are general programming fundamentals that we will use throughout
// the later exercises.
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

// #### Dynamics Base Class
// We will first define a base `Dynamics` class that will be the core structure for future
// dynamics models, including the nonholonomic wheeled robot model. This base class is
// defined in `ch01/exercises/dynamics.hpp`.

// #### Nonholonomic Wheeled Robot Dynamics
// We model the robot dynamics using the kinematic unicycle model:
//     \dot{x} = v \cos(\theta), \quad \dot{y} = v \sin(\theta), \quad \dot{\theta} = \omega.
// where (x,y) is the position, θ is the heading angle with respect to the x-axis, and the
// control inputs are the speed v and the angular velocity ω.

// -----------------------------------------------------------------------------
// ### Exercise 2.1: Define Nonholonomic Wheeled Robot Dynamics Class
// Below we define the `RobotDynamics` class by inheriting from the `Dynamics` base class.
// Complete the implementations of the `step` and `rollout` functions.
// -----------------------------------------------------------------------------

/**
 * A class to describe the nonholonomic wheeled robot's dynamics.
 */
class RobotDynamics : public Dynamics {
 public:
  /** Initialize robot dynamics class. */
  RobotDynamics() : Dynamics() {}

  int n = 3;  // length of state
  int m = 2;  // length of control

  /**
   * Compute the next state from the current state and control.
   *
   * Args:
   *     x: current state, [x, y, θ] (column vector, shape (n, 1))
   *     u: control input, [v, ω]
   *
   * Returns:
   *     Next state, `dt` seconds in the future.
   */
  MatrixXd step(const MatrixXd& x, const VectorXd& u) override {
    // Define Gaussian Noise
    VectorXd var(3);
    if (noisy == false) {
      var << 0.0, 0.0, 0.0;
    } else {
      var << 0.01, 0.01, 0.001;
    }
    VectorXd w(3);
    for (int i = 0; i < 3; ++i) w[i] = pora::rng().normal(0.0, var[i]);

    VectorXd x_next(3);
    // ##### YOUR CODE STARTS HERE #####
    // Populate `x_next` by applying discrete time dynamics equations.
    // Use `dt` from the Dynamics base class
    throw pora::NotImplementedError("Need to implement code here.");
    // ###### YOUR CODE ENDS HERE ######

    // Add noise
    x_next += w;
    return x_next;
  }

  /**
   * Rollout the robot's trajectory from x0 using the given control sequence.
   *
   * Args:
   *     x0: initial robot state, [x0, y0, θ0]
   *     u_sequence: sequence of control inputs to apply, shape (m, num_steps)
   *     num_rollouts: number of rollouts to perform
   *
   * Returns:
   *     Vector of rollouts of length num_rollouts; rollouts[i] has shape (n, num_steps + 1)
   *     (the Python version returns an array of shape (num_rollouts, n, num_steps + 1)).
   */
  std::vector<MatrixXd> rollout(const VectorXd& x0, const MatrixXd& u_sequence, int num_rollouts) override {
    const int num_steps = static_cast<int>(u_sequence.cols());

    std::vector<MatrixXd> rollouts(num_rollouts, MatrixXd::Zero(n, num_steps + 1));
    for (auto& r : rollouts) r.col(0) = x0;
    // ##### YOUR CODE STARTS HERE #####
    // Use two for-loops to loop through `num_rollouts` and `num_steps` to populate `rollouts`.
    // Use the `step` function above.
    throw pora::NotImplementedError("Need to implement code here.");
    // ###### YOUR CODE ENDS HERE ######

    return rollouts;
  }
};

int main() {
  return pora::run([] {
    pora::seed(0);

    // Run the script below to simulate the robot motion and plot the state and control trajectories.

    // Constants
    const int num_rollouts = 10;
    const int num_steps = 100;

    // Define the dynamics class
    RobotDynamics dynamics;

    // Define the control inputs
    MatrixXd u_sequence = MatrixXd::Ones(dynamics.m, num_steps);
    u_sequence.row(1) = pora::linspace(0, 2 * pora::kPi, num_steps).array().sin().transpose();

    // Define inital state
    VectorXd x0 = VectorXd::Zero(3);

    // Rollouts dynamics
    std::vector<MatrixXd> rollouts = dynamics.rollout(x0, u_sequence, num_rollouts);

    // Plot control inputs
    pora::plt::Figure fig_ctrl(dynamics.m, 1, 700, 450);
    const char* ylabels_ctrl[] = {"v", "omega"};
    double ylims_ctrl[2][2] = {{0.0, 2.0}, {-1.1, 1.1}};
    for (int j = 0; j < dynamics.m; ++j) {
      fig_ctrl.ax(j).plot(pora::linspace(0, num_steps, num_steps), u_sequence.row(j));
      fig_ctrl.ax(j).set_ylabel(ylabels_ctrl[j]);
      fig_ctrl.ax(j).set_ylim(ylims_ctrl[j][0], ylims_ctrl[j][1]);
    }
    fig_ctrl.suptitle("Control");
    fig_ctrl.save("control.png");

    // Plot state trajectory rollouts
    pora::plt::Figure fig_state(dynamics.n, 1, 700, 650);
    const char* ylabels_state[] = {"x", "y", "theta"};
    double ylims_state[3][2] = {{-0.1, 1.1}, {-0.5, 0.5}, {-0.1, 0.5}};
    for (int i = 0; i < num_rollouts; ++i) {
      for (int j = 0; j < dynamics.n; ++j) {
        fig_state.ax(j).plot(pora::linspace(0, num_steps + 1, num_steps + 1), rollouts[i].row(j), "c-");
        fig_state.ax(j).set_ylabel(ylabels_state[j]);
        fig_state.ax(j).set_ylim(ylims_state[j][0], ylims_state[j][1]);
      }
    }
    fig_state.suptitle("RobotState");
    fig_state.save("state.png");
  });
}
