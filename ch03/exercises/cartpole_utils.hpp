// C++ port of ch03/exercises/cartpole_utils.py
#pragma once

#include <Eigen/Dense>

#include <string>
#include <utility>

struct CartPole {
  /**
   * Initialize cart-pole physical properties
   *
   * Args:
   *     mp: pendulum mass [kg]
   *     mc: cart mass [kg]
   *     L: pendulum length [m]
   *     g: acceleration due to gravity [m/s^2]
   */
  CartPole(double mp, double mc, double L, double g) : mp(mp), mc(mc), L(L), g(g) {}
  double mp, mc, L, g;
};

/**
 * Compute the cart-pole state derivative
 *
 * Args:
 *     s: The cartpole state: [x, theta, x_dot, theta_dot], shape (n,)
 *     u: The cartpole control: [F_x], shape (m,)
 *     cartpole: object storing cartpole params
 *
 * Returns:
 *     The state derivative, shape (n,)
 */
Eigen::VectorXd cartpole_dynamics(const Eigen::VectorXd& s, const Eigen::VectorXd& u, const CartPole& cartpole);

// Any controller with compute_control(k, s, closed_loop) can be simulated (the Python
// version uses duck typing for this).
class CartpoleController {
 public:
  virtual ~CartpoleController() = default;
  /** Control u for state s at time step k. */
  virtual Eigen::VectorXd compute_control(int k, const Eigen::VectorXd& s, bool closed_loop = true) const = 0;
};

/**
 * Simulate the cartpole
 *
 * Args:
 *     cartpole: Cartpole object storing params
 *     t: Evaluation times, shape (num_timesteps,)
 *     s0: Initial state, shape (n,)
 *     controller: Controller with function compute_control(k, s, closed_loop) where
 *                 k is the time step, s is the state at time k, which outputs the
 *                 control u for this state and time.
 *
 * Returns:
 *     Pair of:
 *         The state history, shape (num_timesteps, n)
 *         The control history, shape (num_timesteps-1, m)
 */
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> simulate_cartpole(const CartPole& cartpole, const Eigen::VectorXd& t,
                                                              const Eigen::VectorXd& s0,
                                                              const CartpoleController& controller,
                                                              bool closed_loop = true);

/**
 * Animate the cart-pole system from given position data and save it as a video
 * (`<name>.mp4` in the output directory).
 *
 * All arguments are assumed to be 1-D arrays, where `x` and `θ` are the
 * degrees of freedom of the cart-pole over time `t`.
 */
void animate_cartpole(const Eigen::VectorXd& t, const Eigen::VectorXd& x, const Eigen::VectorXd& theta,
                      const std::string& name);

/**
 * Helper function for cartpole visualization
 *
 * Args:
 *     s: State history, shape (num_timesteps, n)
 *     u: Control history, shape (num_timesteps, m)
 *     t: Times, shape (num_timesteps,)
 *     s_ref: Reference state s_bar, evaluated at each time t. Shape (num_timesteps, n)
 *     name: Filename prefix for saving figures
 */
void plot_state_and_control_history(const Eigen::MatrixXd& s, const Eigen::MatrixXd& u, const Eigen::VectorXd& t,
                                    const Eigen::MatrixXd& s_ref, const std::string& name);
