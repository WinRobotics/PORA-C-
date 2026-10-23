// =============================================================================
// Exercise 2: Cartpole LQR Control
// (C++ port of ch03/exercises/cartpole_lqr_control.ipynb)
//
// In this problem we consider controlling a cart-pole system, which consists of a "cart"
// that can travel linearly along a one-dimensional track, and has a pendulum attached to it.
// The objective is to control the cart position to stablize the pendulum about its
// "inverted" position.
// =============================================================================
#include "cartpole_utils.hpp"

#include <pora/core.hpp>
#include <pora/numerics.hpp>

#include <Eigen/Dense>

#include <cmath>
#include <stdexcept>
#include <utility>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// -----------------------------------------------------------------------------
// ### Exercise 2.1: Linearizing the Cartpole Dynamics
// Implement the function `compute_lti_matrices` to compute the linear time-invariant (LTI)
// system matrices A and B for the cart-pole system about the upright equilibrium state.
// -----------------------------------------------------------------------------

/**
 * Compute the linearized dynamics matrices A and B of the LTI system.
 *
 * Args:
 *     cartpole: CartPole object storing params
 *     dt: time step
 *
 * Returns:
 *     Pair of:
 *         The A (dynamics) matrix, shape (n, n)
 *         The B (controls) matrix, shape (n, m)
 */
std::pair<MatrixXd, MatrixXd> compute_lti_matrices(const CartPole& cartpole, double dt) {
  const double mp = cartpole.mp, mc = cartpole.mc, L = cartpole.L, g = cartpole.g;
  // ##### YOUR CODE STARTS HERE #####
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
}

// -----------------------------------------------------------------------------
// ### Exercise 2.2: Implement Discrete-time Riccati Equations
// Implement the function `discreteLQR::solve` to solve the LQR problem by implementing the
// Riccati recursion to approximate the optimal gain matrix K_∞ for the linearized and
// discrete-time cart-pole system.
// -----------------------------------------------------------------------------
class discreteLQR : public CartpoleController {
 public:
  /**
   * Initialize the discrete-time LQR problem with cost function matrices.
   *
   * Args:
   *     Q: State cost matrix, shape (n, n)
   *     R: Control cost matrix, shape (m, m)
   *     eps: termination threshold for Riccati recursion
   *     max_iters: maximum number of Riccati recursion iterations
   */
  discreteLQR(const MatrixXd& Q, const MatrixXd& R, double eps = 1e-4, int max_iters = 1000)
      : Q(Q), R(R), eps(eps), max_iters(max_iters) {
    if (max_iters <= 1) throw std::invalid_argument("Argument `max_iters` must be at least 1.");
    n = static_cast<int>(Q.rows());  // state dimension
    m = static_cast<int>(R.rows());  // control dimension

    // Initialize gain matrix
    K = MatrixXd::Zero(m, n);

    // Initialize the reference points
    u_ref = VectorXd::Zero(m);
    s_ref = MatrixXd::Zero(1, n);
  }

  /**
   * Compute the gain matrix K through Ricatti recursion. Stores the gain matrix
   * in K and returns it.
   *
   * Args:
   *     A: Dynamics matrix, shape (n, n)
   *     B: Controls matrix, shape (n, m)
   *     s_ref: Reference trajectory that A, B are linearized about, shape (num_timesteps, n)
   *            Note, every point of the reference trajectory must have the same A, B linearization.
   *     u_ref: Reference control that A, B are linearized about, shape (m,)
   *
   * Returns:
   *     Gain matrix K, shape (m, n)
   */
  MatrixXd solve(const MatrixXd& A, const MatrixXd& B, const MatrixXd& s_ref_, const VectorXd& u_ref_) {
    u_ref = u_ref_;
    s_ref = s_ref_;
    MatrixXd P_prev = MatrixXd::Zero(n, n);  // initialization
    MatrixXd K_new = MatrixXd::Zero(m, n);
    bool converged = false;
    for (int i = 0; i < max_iters; ++i) {
      // ##### YOUR CODE STARTS HERE #####
      // Implement Riccati recursion to compute gain matrix, K_new
      throw pora::NotImplementedError("Need to implement code here.");
      // ###### YOUR CODE ENDS HERE ######
    }
    if (!converged) throw std::runtime_error("Ricatti recursion did not converge!");
    K = K_new;
    return K;
  }

  /**
   * Compute the control value for the LQR controller. If `closed_loop` is false
   * just return the open loop u_ref value.
   *
   * Args:
   *     k: current time step
   *     s: current state
   *
   * Returns:
   *     Control value to apply
   */
  VectorXd compute_control(int k, const VectorXd& s, bool closed_loop = true) const override {
    return closed_loop ? VectorXd(K * (s - s_ref.row(k).transpose()) + u_ref) : u_ref;
  }

  MatrixXd Q, R;
  double eps;
  int max_iters;
  int n, m;
  MatrixXd K;
  VectorXd u_ref;
  MatrixXd s_ref;
};

// -----------------------------------------------------------------------------
// ### Exercise 2.4: Use LQR to Track Time-varying Reference Trajectory
// Implement the function `reference` below to generate a time-varying reference trajectory
// that keeps the pendulum upright while moving the cart's linear position according to
// x̄(t) = a sin(2πt / T). Then, run the code below to see the simulated results. Try playing
// with the state penalty matrix Q to see if you can improve the tracking error.
// -----------------------------------------------------------------------------

/**
 * Compute the reference state (s_bar) at time t.
 *
 * Args:
 *     t: Evaluation time
 *
 * Returns:
 *     Reference state, shape (n,)
 */
VectorXd reference(double t) {
  const double a = 10.0;  // Amplitude
  const double T = 10.0;  // Period

  // ##### YOUR CODE STARTS HERE #####
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
}

int main() {
  return pora::run([] {
    // -------------------------------------------------------------------------
    // ### Exercise 2.3: Simulate Stabilized Cartpole
    // Run the code below to simulate the stabilized cartpole with an LQR feedback controller
    // defined using the functions you implemented above.
    // -------------------------------------------------------------------------
    VectorXd t = pora::arange(0.0, 30.0, 0.1);
    {
      // Define constants for the cart-pole system
      double mp = 2.0;   // pendulum mass
      double mc = 10.0;  // cart mass
      double L = 1.0;    // pendulum length
      double g = 9.81;   // gravitational acceleration
      CartPole cartpole(mp, mc, L, g);

      // Compute discrete LTI model with 10Hz sampling rate
      double dt = 0.1;
      auto [A, B] = compute_lti_matrices(cartpole, dt);

      // Compute LQR gain matrix
      int n = 4;
      int m = 1;
      MatrixXd Q = MatrixXd::Identity(n, n);  // state cost matrix
      MatrixXd R = MatrixXd::Identity(m, m);  // control cost matrix
      discreteLQR lqr(Q, R);
      Eigen::RowVectorXd s_eq(4);
      s_eq << 0.0, pora::kPi, 0.0, 0.0;
      MatrixXd s_ref = s_eq.replicate(t.size(), 1);  // stablize at equilibrium
      VectorXd u_ref = VectorXd::Zero(1);
      lqr.solve(A, B, s_ref, u_ref);

      // Simulate and plot results
      VectorXd s0(4);
      s0 << 0.0, 3 * pora::kPi / 4, 0.0, 0.0;
      auto [s, u] = simulate_cartpole(cartpole, t, s0, lqr);
      plot_state_and_control_history(s, u, t, lqr.s_ref, "cartpole_balance");
      animate_cartpole(t, s.col(0), s.col(1), "cartpole_balance");
    }

    // Exercise 2.4: time-varying reference
    {
      // Define constants for the cart-pole system
      double mp = 2.0;   // pendulum mass
      double mc = 10.0;  // cart mass
      double L = 1.0;    // pendulum length
      double g = 9.81;   // gravitational acceleration
      CartPole cartpole(mp, mc, L, g);

      // Compute discrete LTI model with 10Hz sampling rate
      double dt = 0.1;
      auto [A, B] = compute_lti_matrices(cartpole, dt);

      // Compute LQR gain matrix
      int n = 4;
      int m = 1;
      MatrixXd Q = MatrixXd::Identity(n, n);  // state cost matrix
      MatrixXd R = MatrixXd::Identity(m, m);  // control cost matrix
      discreteLQR lqr(Q, R);
      MatrixXd s_ref(t.size(), n);
      for (int i = 0; i < t.size(); ++i) s_ref.row(i) = reference(t[i]).transpose();
      VectorXd u_ref = VectorXd::Zero(1);
      lqr.solve(A, B, s_ref, u_ref);

      // Simulate with time-varying reference and plot results
      VectorXd s0(4);
      s0 << 0.0, pora::kPi, 0.0, 0.0;
      auto [s, u] = simulate_cartpole(cartpole, t, s0, lqr);
      plot_state_and_control_history(s, u, t, lqr.s_ref, "cartpole_balance_tv");
      animate_cartpole(t, s.col(0), s.col(1), "cartpole_balance_tv");
    }
  });
}
