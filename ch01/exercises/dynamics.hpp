// C++ port of ch01/exercises/dynamics.py
#pragma once

#include <pora/core.hpp>

#include <Eigen/Dense>

#include <vector>

class Dynamics {
 public:
  Dynamics() = default;
  virtual ~Dynamics() = default;

  double dt = 0.01;    // timestep for integration
  bool noisy = true;   // whether the dynamics are noisy

  /**
   * Compute the next state at time t + Δt given the state at
   * time t and the control input.
   *
   * Args:
   *   x: state at time t
   *   u: control at time t
   *
   * Returns:
   *   The next state, at time t + Δt.
   */
  virtual Eigen::MatrixXd step(const Eigen::MatrixXd& x, const Eigen::VectorXd& u) {
    (void)x;
    (void)u;
    throw pora::NotImplementedError("Calling abstract function");
  }

  /**
   * Compute a number of state trajectories (rollouts) starting
   * from an initial state given a control sequence.
   *
   * Args:
   *   x0: initial state
   *   u_sequence: control sequence
   *   num_rollouts: number of rollouts to compute
   *
   * Returns:
   *   Set of rollouts starting from the initial and applying
   *   the given control sequence.
   */
  virtual std::vector<Eigen::MatrixXd> rollout(const Eigen::VectorXd& x0, const Eigen::MatrixXd& u_sequence,
                                               int num_rollouts) {
    (void)x0;
    (void)u_sequence;
    (void)num_rollouts;
    throw pora::NotImplementedError("Calling abstract function");
  }
};
