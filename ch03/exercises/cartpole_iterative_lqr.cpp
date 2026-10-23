// =============================================================================
// Exercise 3: Cartpole iLQR Control
// (C++ port of ch03/exercises/cartpole_iterative_lqr.ipynb)
//
// In this problem we consider controlling a cart-pole system, which consists of a "cart"
// that can travel linearly along a one-dimensional track, and has a pendulum attached to it.
// The objective is to implement a controller to solve the "swing up" problem to bring the
// pendulum from the downard hanging position to a stabilized upright position.
// =============================================================================
#include "cartpole_utils.hpp"

#include <pora/core.hpp>
#include <pora/numerics.hpp>

#include <Eigen/Dense>

#include <chrono>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

using DynamicsFn = std::function<VectorXd(const VectorXd&, const VectorXd&)>;

// -----------------------------------------------------------------------------
// ### Exercise 3.1: Linearizing the Cartpole Dynamics
// Implement the function `linearize` to linearize the function `f(s,u)` about the point
// `(s,u)` (the Python version uses JAX; here use `pora::jacobian`).
// -----------------------------------------------------------------------------

/**
 * Linearize the function f(s, u) around the point (s, u).
 *
 * Args:
 *     f: a nonlinear function, f(s, u)
 *     s: the state to linear about
 *     u: the control input to linearize about
 *
 * Returns:
 *     The pair (A, B) where:
 *         A: the Jacobian of f at (s, u), with respect to s.
 *         B: the Jacobian of f at (s, u), with respect to u.
 */
std::pair<MatrixXd, MatrixXd> linearize(const DynamicsFn& f, const VectorXd& s, const VectorXd& u) {
  MatrixXd A, B;
  // ##### YOUR CODE STARTS HERE #####
  // Use pora::jacobian (the counterpart of jax.jacobian) to compute the matrices A and B.
  // Hint: wrap f in lambdas of a single argument, e.g. [&](const VectorXd& ss) { return f(ss, u); }
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
  return {A, B};
}

// -----------------------------------------------------------------------------
// ### Exercises 3.3 and 3.4: Implement iLQR
// Implement the function `discreteIterativeLQR::solve` below to compute the iLQR controller
// gain matrices and offset terms. Note here we are referring to the gain matrix as Y and the
// offset term as y to avoid confusion with respect to the time step variable k. Concretely,
// the controller we are looking to define is of the form:
//     u_k = \bar u_k - y_k - Y_k(s_k - \bar s_k),
//
// Then, implement the function `discreteIterativeLQR::compute_control` to compute the control
// value based on either the open or closed-loop case.
// -----------------------------------------------------------------------------
class discreteIterativeLQR : public CartpoleController {
 public:
  /**
   * Initialize the discrete-time iLQR problem with cost function and horizon.
   *
   * Args:
   *     N: the time horizon of the LQR cost function
   *     Q: the state cost matrix, shape (n,n)
   *     R: the control cost matrix, shape (m,m)
   *     QN: the terminal state cost matrix, shape (n,n)
   *     eps: termination threshold for iLQR
   *     max_iters: maximum number of iLQR iterations.
   */
  discreteIterativeLQR(int N, const MatrixXd& Q, const MatrixXd& R, const MatrixXd& QN, double eps = 1e-3,
                       int max_iters = 1000)
      : N(N), Q(Q), R(R), QN(QN), eps(eps), max_iters(max_iters) {
    if (max_iters <= 1) throw std::invalid_argument("Argument `max_iters` must be at least 1.");
    n = static_cast<int>(Q.rows());  // state dimension
    m = static_cast<int>(R.rows());  // control dimension

    // Initialize gains `Y` and offsets `y` for the policy
    Y.assign(N, MatrixXd::Zero(m, n));
    y = MatrixXd::Zero(N, m);

    // Initialize the nominal trajectory `(s_bar, u_bar`),
    u_bar = MatrixXd::Zero(N, m);
    s_bar = MatrixXd::Zero(N + 1, n);
  }

  /**
   * Compute the iLQR set-point tracking solution for given discrete-time dynamics and start
   * goal states. Sets the s_bar, u_bar, Y, and y member variables.
   *
   * Args:
   *     f: a function describing the discrete-time dynamics, such that
   *        s[k+1] = f(s[k], u[k])
   *     s0: the initial state, shape (n,)
   *     s_goal: the goal state, shape (n,)
   *
   * Sets:
   *     s_bar: array of shape (N+1, n) where s_bar.row(k) is the nominal state at time step k
   *     u_bar: array of shape (N, m) where u_bar.row(k) is the nominal control at time step k
   *     Y: vector of N arrays of shape (m, n) where Y[k] is the matrix gain term of the iLQR
   *        control law at time step k
   *     y: array of shape (N, m) where y.row(k) is the offset term of the iLQR control law
   *        at time step k
   */
  void solve(const DynamicsFn& f, const VectorXd& s0, const VectorXd& s_goal) {
    // Initialize gains Y and offsets y for the policy
    std::vector<MatrixXd> Y(N, MatrixXd::Zero(m, n));
    MatrixXd y = MatrixXd::Zero(N, m);

    // Initialize the nominal trajectory (s_bar, u_bar), and the
    // deviations (ds, du)
    MatrixXd u_bar = MatrixXd::Zero(N, m);
    MatrixXd s_bar = MatrixXd::Zero(N + 1, n);
    s_bar.row(0) = s0.transpose();
    for (int k = 0; k < N; ++k) s_bar.row(k + 1) = f(s_bar.row(k).transpose(), u_bar.row(k).transpose()).transpose();
    MatrixXd ds = MatrixXd::Zero(N + 1, n);
    MatrixXd du = MatrixXd::Zero(N, m);

    // iLQR loop
    bool converged = false;
    for (int iter = 0; iter < max_iters; ++iter) {
      // Linearize the dynamics at each step k of (s_bar, u_bar)
      std::vector<MatrixXd> A(N), B(N);
      for (int k = 0; k < N; ++k)
        std::tie(A[k], B[k]) = linearize(f, s_bar.row(k).transpose(), u_bar.row(k).transpose());

      // ##### YOUR CODE STARTS HERE #####
      // Update Y, y, ds, du, s_bar, and u_bar
      // Backward pass
      throw pora::NotImplementedError("Need to implement code here.");
      // ###### YOUR CODE ENDS HERE ######

      if (du.cwiseAbs().maxCoeff() < eps) {
        converged = true;
        break;
      }
    }
    if (!converged) throw std::runtime_error("iLQR did not converge!");
    this->s_bar = s_bar;
    this->u_bar = u_bar;
    this->Y = Y;
    this->y = y;
  }

  /**
   * Compute the control value for the iLQR controller. If `closed_loop` is false
   * just return the open loop u_bar value.
   *
   * Args:
   *     k: current time step
   *     s: current state
   *     closed_loop: whether to run in closed-loop or not
   *
   * Returns:
   *     Control value to apply.
   */
  VectorXd compute_control(int k, const VectorXd& s, bool closed_loop = true) const override {
    VectorXd u;
    // ##### YOUR CODE STARTS HERE #####
    throw pora::NotImplementedError("Need to implement code here.");
    // ###### YOUR CODE ENDS HERE ######
    return u;
  }

  int N;
  MatrixXd Q, R, QN;
  double eps;
  int max_iters;
  int n, m;
  std::vector<MatrixXd> Y;
  MatrixXd y, u_bar, s_bar;
};

int main() {
  return pora::run([] {
    // Finally, run the code below to simulate the iLQR controller in both open-loop and
    // closed-loop.

    // Define constants for the cart-pole system
    double mp = 2.0;   // pendulum mass
    double mc = 10.0;  // cart mass
    double L = 1.0;    // pendulum length
    double g = 9.81;   // gravitational acceleration
    CartPole cartpole(mp, mc, L, g);

    // Define problem set up constants
    VectorXd s0(4), s_goal(4);
    s0 << 0.0, 0.0, 0.0, 0.0;            // initial state
    s_goal << 0.0, pora::kPi, 0.0, 0.0;  // goal state
    double T = 10.0;                     // simulation time
    double dt = 0.1;                     // sampling time

    // Define iLQR Problem
    int n = 4;  // state dimension
    int m = 1;  // control dimension
    VectorXd t = pora::arange(0.0, T, dt);
    int N = static_cast<int>(t.size()) - 1;
    MatrixXd Q = VectorXd((VectorXd(4) << 10.0, 10.0, 2.0, 2.0).finished()).asDiagonal();  // state cost matrix
    MatrixXd R = 1e-2 * MatrixXd::Identity(m, m);  // control cost matrix
    MatrixXd QN = 1e2 * MatrixXd::Identity(n, n);  // terminal state cost matrix
    discreteIterativeLQR ilqr(N, Q, R, QN);

    // Initialize continuous-time and discretized dynamics
    auto f = [&](const VectorXd& s, const VectorXd& u) { return cartpole_dynamics(s, u, cartpole); };
    DynamicsFn fd = [&](const VectorXd& s, const VectorXd& u) { return VectorXd(s + dt * f(s, u)); };

    // Compute the iLQR solution with the discretized dynamics
    std::printf("Computing iLQR solution ... ");
    std::fflush(stdout);
    auto start = std::chrono::steady_clock::now();
    ilqr.solve(fd, s0, s_goal);
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("done! (%.2f s)\n", elapsed);

    // Simulate on the true continuous-time system and plot results
    {
      bool closed_loop = false;  // open-loop
      auto [s, u] = simulate_cartpole(cartpole, t, s0, ilqr, closed_loop);
      plot_state_and_control_history(s, u, t, ilqr.s_bar, "cartpole_swingup_open_loop");
      animate_cartpole(t, s.col(0), s.col(1), "cartpole_swingup_open_loop");
    }

    // Simulate on the true continuous-time system and plot results
    {
      bool closed_loop = true;  // closed-loop
      auto [s, u] = simulate_cartpole(cartpole, t, s0, ilqr, closed_loop);
      plot_state_and_control_history(s, u, t, ilqr.s_bar, "cartpole_swingup_closed_loop");
      animate_cartpole(t, s.col(0), s.col(1), "cartpole_swingup_closed_loop");
    }
  });
}
