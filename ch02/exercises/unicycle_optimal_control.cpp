// =============================================================================
// Exercise 5: Unicycle Optimal Control
// (C++ port of ch02/exercises/unicycle_optimal_control.ipynb)
//
// Consider the kinematic model of a unicycle:
//     \dot{x} = v \cos(\theta), \quad \dot{y} = v \sin(\theta), \quad \dot{\theta} = \omega.
// where (x,y) is the position, θ is the heading angle with respect to the x-axis, and the
// control inputs are the speed v and the angular velocity ω. We define the state vector
// x = (x, y, θ).
//
// In this problem we will compute the optimal control to drive the unicycle vehicle from a
// starting configuration x(0) = (0, 0, π/2) to the target configuration x(T) = (5, 5, π/2).
// We seek to find a solution that minimizes time to the target configuration while also
// minimizing the control effort required. Specifically, we want to minimize:
//     J(x,u) = \int_0^T  \alpha + v(t)^2 + \omega(t)^2  dt,
// where α > 0 is a constant weighting factor and T is the free final time.
//
// NOTE: The Python version uses `jax.numpy` and JAX's automatic differentiation. In C++ we
// use Eigen for the linear algebra, `pora::odeint` for the ODE integration and
// `pora::jacobian` (central finite differences) in place of `jax.jacobian`.
// =============================================================================
#include <pora/core.hpp>
#include <pora/numerics.hpp>
#include <pora/plot.hpp>

#include <Eigen/Dense>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <tuple>
#include <utility>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// -----------------------------------------------------------------------------
// ### Exercise 5.2: Define the dynamics, Hamiltonian, and necessary optimality conditions
//
// First, implement the function `unicycle_dynamics`:
// -----------------------------------------------------------------------------

/**
 * Evaluate the continuous-time dynamics of a unicycle.
 *
 * Args:
 *     x: an array of shape (3,) containing the unicycle pose (x, y, θ).
 *     u: an array of shape (2,) containing the velocity controls (v, ω).
 *
 * Returns:
 *     An array of shape (3,) containing the time derivative of the state (dx/dt, dy/dt, dθ/dt)
 */
VectorXd unicycle_dynamics(const VectorXd& x, const VectorXd& u) {
  // ##### YOUR CODE STARTS HERE #####
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
}

// Next, derive the Hamiltonian for the optimal control problem and implement the function
// `hamiltonian` below to compute it:

/**
 * Evaluate the Hamiltonian.
 *
 * Args:
 *     x: an array of shape (3,) containing the unicycle pose (x, y, θ).
 *     p: an array of shape (3,) containing the co-state (px, py, pθ).
 *     u: an array of shape (2,) containing the velocity controls (v, ω).
 *
 * Returns:
 *     The value of the Hamiltonian.
 */
double hamiltonian(const VectorXd& x, const VectorXd& p, const VectorXd& u) {
  const double alpha = 0.25;
  // ##### YOUR CODE STARTS HERE #####
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
}

// From the Hamiltonian, we can now derive the optimal control function u*(t) and the set of
// ordinary differential equations and their boundary conditions that constitute the
// necessary optimality conditions. Implement the function u*(t) in `optimal_control` and the
// ordinary differential equations for the NOCs in `noc_ode`.

/**
 * Compute an optimal control as a function of the state and co-state.
 *
 * Args:
 *     x: an array of shape (3,) containing the unicycle pose (x, y, θ).
 *     p: an array of shape (3,) containing the co-state (px, py, pθ).
 *
 * Returns:
 *     An array of shape (2,) containing an optimal control u* = (v*, ω*).
 */
VectorXd optimal_control(const VectorXd& x, const VectorXd& p) {
  // ##### YOUR CODE STARTS HERE #####
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
}

/**
 * Evaluate the ODE that an optimal state and co-state must obey.
 *
 * Args:
 *     x_and_p: a pair of arrays (x, p), where x is an array of shape
 *              (3,) containing the unicycle pose (x, y, θ), and p is
 *              an array of shape (3,) containing the co-state (px, py, pθ).
 *     t: the current time (required for use with `odeint`, but can be ignored here).
 *
 * Returns:
 *     A pair of arrays (dx/dt, dp/dt) containing the time derivatives of the
 *     state and co-state, respectively.
 */
std::pair<VectorXd, VectorXd> noc_ode(const std::pair<VectorXd, VectorXd>& x_and_p, double t) {
  const VectorXd& x = x_and_p.first;
  const VectorXd& p = x_and_p.second;
  // ##### YOUR CODE STARTS HERE #####
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
}

// -----------------------------------------------------------------------------
// ### Exercise 5.3: Define the boundary condition residual function
// Now, use the `noc_trajectories` function defined below to complete the code for
// `boundary_residual`, where we want to compute the residual error of the boundary
// conditions for the NOCs. In other words, we want to express the boundary conditions in a
// standard form l(z(t_0), z(t_f)) = 0 where z represents a vector of the states and costates
// and have `boundary_residual` return l(z(t_0), z(t_f)).
// -----------------------------------------------------------------------------

/**
 * Integrate the optimal state and co-state ODE forward in time.
 *
 * Args:
 *     x0: an array of shape (3,) containing the initial state (x(0), y(0), θ(0)).
 *     p0: an array of shape (3,) containing the initial co-state (px(0), py(0), pθ(0)).
 *     T: the final time T.
 *     N: the number of nodes along the ODE solution at which to report the
 *        solution values.
 *
 * Returns:
 *     A tuple of arrays (ts, xs, us, ps) where:
 *         ts: an array of shape (N,) containing a sequence of times
 *              spanning [0, T].
 *         xs: an array of shape (N, 3) containing the states at ts.
 *         us: an array of shape (N, 2) containing the optimal control
 *              inputs at ts.
 *         ps: an array of shape (N, 3) containing the co-states at ts.
 */
std::tuple<VectorXd, MatrixXd, MatrixXd, MatrixXd> noc_trajectories(const VectorXd& x0, const VectorXd& p0, double T,
                                                                     int N = 20) {
  VectorXd ts = pora::linspace(0, T, N);
  // Stack (x, p) into a single vector z = [x; p] for the integrator.
  auto f = [](const VectorXd& z, double t) {
    auto [dx, dp] = noc_ode({z.head(3), z.tail(3)}, t);
    VectorXd dz(6);
    dz << dx, dp;
    return dz;
  };
  VectorXd z0(6);
  z0 << x0, p0;
  // Tight tolerances so that finite-difference Jacobians of the solution are accurate.
  MatrixXd zs = pora::odeint(f, z0, ts, 1e-11, 1e-11);
  MatrixXd xs = zs.leftCols(3), ps = zs.rightCols(3);
  MatrixXd us(N, 2);
  for (int i = 0; i < N; ++i) us.row(i) = optimal_control(xs.row(i).transpose(), ps.row(i).transpose()).transpose();
  return {ts, xs, us, ps};
}

/**
 * Compute the residual error of the boundary conditions for the NOCs.
 *
 * Args:
 *     p0: an array of shape (3,) containing the initial co-state (px(0), py(0), pθ(0))
 *          estimate.
 *     T:  the final time T estimate.
 *     x0: an array of shape (3,) containing the initial state (x(0), y(0), θ(0)).
 *     xT: an array of shape (3,) containing the final state x(T), y(T), θ(T)).
 *
 * Returns:
 *     The array of shape (4,) we want to drive to zero through appropriate
 *     selection of p0 and T.
 */
VectorXd boundary_residual(const VectorXd& p0, double T, const VectorXd& x0, const VectorXd& xT) {
  // ##### YOUR CODE STARTS HERE #####
  // Hint: Use `noc_trajectories` here to compute xs, ps
  // auto [ts, xs, us, ps] = noc_trajectories(x0, p0, T);
  // xT_sim = xs.row(N - 1), pT = ps.row(N - 1)
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
}

// Now that we have implemented `boundary_residual` our goal is to find a way to get the
// boundary conditions to be satisfied, or in other words have `boundary_residual` return a
// zero vector. To do this, we need a method to search over a set of initial costate values
// and the final time T to find a solution with zero residual. We will use a Newton-Raphson
// method to accomplish this. Implement the Newton-Raphson step in the function `newton_step`
// below.

/**
 * Implement a step of the Newton-Raphson method for `boundary_residual`.
 *
 * Args:
 *     p0: an array of shape (3,) containing the current initial co-state
 *          (px(0), py(0), pθ(0)) estimate.
 *     T: the current final time T estimate.
 *     x0: an array of shape (3,) containing the initial state (x(0), y(0), θ(0)).
 *     xT: an array of shape (3,) containing the final state (x(T), y(T), θ(T)).
 *
 * Returns:
 *     A pair containing the next estimate of p0 and T computed by the
 *     Newton-Raphson method.
 */
std::pair<VectorXd, double> newton_step(VectorXd p0, double T, const VectorXd& x0, const VectorXd& xT) {
  // ##### YOUR CODE STARTS HERE #####
  // Hint: Use `pora::jacobian` (the C++ counterpart of `jax.jacobian`) on a function of the
  //       stacked decision vector [p0; T], and Eigen's `.partialPivLu().solve()` (the
  //       counterpart of `jnp.linalg.solve`).
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
  return {p0, T};
}

// Now, run the code below to run iterations of the Newton-Raphson method to find the initial
// costate parameters and the final time that will satisfy the necessary optimality
// conditions, and thus will give us the optimal control for the unicycle problem.

/**
 * Single shooting for the unicycle using a Newton-Raphson root finding method.
 *
 * Args:
 *     p0: an array of shape (3,) containing an initial guess for the
 *         initial co-state (px(0), py(0), pθ(0)).
 *     T: an initial guess for the final time T.
 *     x0: an array of shape (3,) containing the initial state (x(0), y(0), θ(0)).
 *     xT: an array of shape (3,) containing the desired final state (x(T), y(T), θ(T)).
 *     max_iters: the maximum number of Newton-Raphson steps to take.
 *     tol: the convergence tolerance.
 *
 * Returns:
 *     A pair containing the optimized initial co-state p0 and final time T.
 */
std::pair<VectorXd, double> single_shooting(VectorXd p0, double T, const VectorXd& x0, const VectorXd& xT,
                                            int max_iters = 10, double tol = 1e-4) {
  bool converged = false;
  for (int k = 0; k < max_iters; ++k) {
    std::tie(p0, T) = newton_step(p0, T, x0, xT);
    VectorXd r = boundary_residual(p0, T, x0, xT);
    double error = r.cwiseAbs().maxCoeff();
    std::printf("Iteration %d (error = %g)\n", k, error);
    if (error < tol) {
      converged = true;
      break;
    }
  }
  if (!converged) throw std::runtime_error("Single shooting did not converge!");
  return {p0, T};
}

int main() {
  return pora::run([] {
    // NOTE: the Python notebook uses p0_init = (1, 1, -1). With the Hamiltonian convention
    // H = α + v² + ω² + pᵀf(x, u), that guess drives the unicycle backwards and plain
    // Newton-Raphson diverges (in JAX as well as here), so the sign is flipped below.
    VectorXd p0_init(3);
    p0_init << -1., -1., 1.;
    double T_init = 20.;

    VectorXd x0(3), xT(3);
    x0 << 0, 0, pora::kPi / 2;
    xT << 5, 5, pora::kPi / 2;
    auto [p0, T] = single_shooting(p0_init, T_init, x0, xT);
    auto [ts, xs, us, ps] = noc_trajectories(x0, p0, T);
    std::printf("Optimal final time T = %.4f\n", T);

    pora::plt::Figure fig(1, 2, 1200, 450);
    auto& ax0 = fig.ax(0);
    ax0.plot(xs.col(0), xs.col(1), pora::plt::fmt("k-").lw(2));
    VectorXd c = xs.col(2).array().cos(), s = xs.col(2).array().sin();
    ax0.quiver(xs.col(0), xs.col(1), c, s, 0.3);
    ax0.scatter(std::vector<double>{x0[0]}, std::vector<double>{x0[1]}, pora::plt::fmt("go").ms(15));
    ax0.scatter(std::vector<double>{xT[0]}, std::vector<double>{xT[1]}, pora::plt::fmt("ro").ms(15));
    ax0.grid(true);
    ax0.axis(-1, 6, -1, 6);
    ax0.set_xlabel("x [m]");
    ax0.set_ylabel("y [m]");
    ax0.set_title("Optimal trajectory");

    auto& ax1 = fig.ax(1);
    ax1.plot(ts, us.col(0), pora::plt::fmt("-").lw(2).label("v* [m/s]"));
    ax1.plot(ts, us.col(1), pora::plt::fmt("-").lw(2).label("omega* [rad/s]"));
    ax1.grid(true);
    ax1.set_xlabel("t [s]");
    ax1.legend();
    ax1.set_title("Optimal control sequence");
    fig.save("unicycle_optimal_control.png");
  });
}
