// =============================================================================
// Exercise 4: Linear MPC
// (C++ port of ch03/exercises/linear_mpc.ipynb)
//
// NOTE: The Python version formulates the MPC problem with CVXPY. In C++ we use the small
// QP solver `pora::QP` (an OSQP-style ADMM solver) which solves
//     minimize    0.5 z^T P z + q^T z
//     subject to  l <= A z <= u
// so the MPC problem has to be written in terms of one stacked decision vector z.
// =============================================================================
#include <pora/core.hpp>
#include <pora/numerics.hpp>
#include <pora/plot.hpp>
#include <pora/qp.hpp>

#include <Eigen/Dense>

#include <cmath>
#include <cstdio>
#include <string>
#include <tuple>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Constant problem variables
const int n = 2, m = 1;
const MatrixXd A = (MatrixXd(2, 2) << 1., 1., 0., 1.).finished();
const MatrixXd B = (MatrixXd(2, 1) << 0., 1.).finished();
const MatrixXd Q = MatrixXd::Identity(n, n);
const MatrixXd R = 10. * MatrixXd::Identity(m, m);

// -----------------------------------------------------------------------------
// ### Exercise 4.1: MPC Problem
// Complete the implementation of the MPC optimization problem in `solve_mpc`. Be sure to
// include the following:
// 1. quadratic running cost along with a quadratic terminal cost,
//    x_N^T Q_f x_N + \sum_{k=0}^{N-1} ( x_k^T Q x_k + u_k^T R u_k ),
// 2. dynamics constraints, x_{k+1} = Ax_k + Bu_k for k=0,...,N-1,
// 3. state constraints, x_k <= x_u and x_k >= x_l for k=0,...,N,
// 4. control constraints ||u_k||_∞ <= r_u for k=0,...,N-1,
// 5. initial condition constraint,
// 6. terminal state constraint, ||x_N||_∞ <= r_N.
// -----------------------------------------------------------------------------

/**
 * Solve the MPC problem starting at state x0.
 *
 * Args:
 *     x0: initial state
 *     A: system matrix, x_next = Ax + Bu, shape (n, n)
 *     B: control matrix, x_next = Ax + Bu, shape (n, m)
 *     Q_f: terminal cost matrix, shape (n, n)
 *     Q: running cost matrix, shape (n, n)
 *     R: control cost matrix, shape (m, m)
 *     N: MPC horizon
 *     x_upper: array describing upper limits on state x, shape (n,)
 *     x_lower: array describing lower limits on state x, shape (n,)
 *     r_u: limit on control u, |u|_inf <= r_u
 *     r_N: terminal state constraint magnitude |x|_inf <= r_N
 *
 * Returns:
 *     x: array of shape (N + 1, n) where x.row(k) is the solution state at time k
 *     u: array of shape (N, m) where u.row(k) is the solution control at time k
 *     status: status message from the optimizer
 */
std::tuple<MatrixXd, MatrixXd, std::string> solve_mpc(const VectorXd& x0, const MatrixXd& A, const MatrixXd& B,
                                                      const MatrixXd& Q_f, const MatrixXd& Q, const MatrixXd& R, int N,
                                                      const VectorXd& x_upper, const VectorXd& x_lower, double r_u,
                                                      double r_N) {
  const int n = static_cast<int>(Q.rows()), m = static_cast<int>(R.rows());

  // Decision vector z = [x_0; x_1; ...; x_N; u_0; ...; u_{N-1}]
  // (the counterpart of the CVXPY variables x_cvx (N+1, n) and u_cvx (N, m)).
  const int nz = (N + 1) * n + N * m;
  auto ix = [&](int k) { return k * n; };                // offset of x_k inside z
  auto iu = [&](int k) { return (N + 1) * n + k * m; };  // offset of u_k inside z
  pora::QP prob(nz);  // cost 0.5 z^T P z + q^T z, constraints l <= A z <= u

  // ##### YOUR CODE STARTS HERE #####
  // Construct the MPC problem: fill `prob.P` (and `prob.q`) with the quadratic cost and add the
  // constraints with `prob.add_equality(rows, rhs)` and `prob.add_constraints(rows, lower, upper)`,
  // where `rows` has `nz` columns.
  // Hint: x_k^T Q x_k = 0.5 z^T P z with P.block(ix(k), ix(k), n, n) = 2 Q (the counterpart of cvx.quad_form)
  // Hint: use pora::kInf for unbounded sides of a constraint (e.g. when r_N is infinite).
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######

  pora::QPResult result = prob.solve();
  MatrixXd x(N + 1, n), u(N, m);
  for (int k = 0; k <= N; ++k) x.row(k) = result.z.segment(ix(k), n).transpose();
  for (int k = 0; k < N; ++k) u.row(k) = result.z.segment(iu(k), m).transpose();
  std::string status = result.status;

  return {x, u, status};
}

void plot(const std::string& title, const std::vector<MatrixXd>& x_mpc, const std::vector<MatrixXd>& u_mpc,
          const VectorXd& x_upper, const VectorXd& x_lower, double r_u, const std::string& filename) {
  pora::plt::Figure fig(1, 2, 1200, 550);
  const int T = static_cast<int>(x_mpc.size());
  std::vector<double> x0s, x1s, us;
  for (int t = 0; t < T; ++t) {
    x0s.push_back(x_mpc[t](0, 0));
    x1s.push_back(x_mpc[t](0, 1));
    us.push_back(u_mpc[t](0, 0));
  }
  fig.ax(0).plot(x0s, x1s, pora::plt::fmt("-o").ms(5));
  for (int t = 0; t < T; ++t)
    fig.ax(0).plot(x_mpc[t].col(0), x_mpc[t].col(1), pora::plt::fmt("--*").c(pora::plt::colors::red).alpha(0.5).lw(0.5));
  fig.ax(1).plot(us, pora::plt::fmt("-o").ms(5));

  // Plot constraints
  fig.ax(0).plot(std::vector<double>{x_upper[0], x_upper[0]}, std::vector<double>{x_lower[1], x_upper[1]},
                 pora::plt::fmt("k--").lw(0.5));
  fig.ax(1).plot(std::vector<double>{0, double(T)}, std::vector<double>{r_u, r_u}, pora::plt::fmt("k--").lw(0.5));
  fig.ax(1).plot(std::vector<double>{0, double(T)}, std::vector<double>{-r_u, -r_u}, pora::plt::fmt("k--").lw(0.5));

  // Labels
  if (!title.empty()) {
    fig.ax(0).set_title(title);
    fig.ax(1).set_title(title);
  }
  fig.ax(0).set_xlabel("x_k,1");
  fig.ax(1).set_xlabel("k");
  fig.ax(0).set_ylabel("x_k,2");
  fig.ax(1).set_ylabel("u_k");
  fig.save(filename);
}

/**
 * Simulate the closed-loop system from the initial state using an MPC controller.
 *
 * Args:
 *     x0: initial state
 *     T: simulation time
 *     N: MPC horizon
 *     x_upper: array describing upper limits on state x, shape (n,)
 *     x_lower: array describing lower limits on state x, shape (n,)
 *     r_u: limit on control u
 *     Q_f: terminal cost matrix, shape (n, n)
 *     r_N: terminal state constraint magnitude |x|_inf <= r_N
 *     title: plot title (optional)
 */
void simulate(const VectorXd& x0, int T, int N, const VectorXd& x_upper, const VectorXd& x_lower, double r_u,
              const MatrixXd& Q_f, double r_N, const std::string& title, const std::string& filename) {
  VectorXd x = x0;
  std::vector<MatrixXd> x_mpc, u_mpc;
  double cost = 0.;
  bool success = true;
  for (int t = 0; t < T; ++t) {
    // Solve the finite horizon MPC problem at each time step
    auto [x_sol, u_sol, status] = solve_mpc(x, A, B, Q_f, Q, R, N, x_upper, x_lower, r_u, r_N);

    if (status == "infeasible") {
      cost += pora::kInf;
      if (t == 0) {
        success = false;
        std::printf("Failed to find initial feasible solution\n");
      }
      break;
    }
    x_mpc.push_back(x_sol);
    u_mpc.push_back(u_sol);

    // Extract the first control input from this step's MPC solution
    VectorXd u = u_sol.row(0).transpose();

    // Incrememt running cost
    cost += x.dot(Q * x) + u.dot(R * u);

    // Simulate the system forward one time step
    x = A * x + B * u;
  }
  std::printf("%s: Cost = %.2f\n", title.c_str(), cost);
  if (success) plot(title, x_mpc, u_mpc, x_upper, x_lower, r_u, filename);
}

int main() {
  return pora::run([] {
    // -------------------------------------------------------------------------
    // ### Exercise 4.2: Terminal Cost Only
    // Run the provided code below to simulate the closed-loop system under the MPC controller
    // with a terminal cost Q_f = Q and no terminal constraint. Take a look at the resulting
    // plots. Why is there a difference between the trajectories computed by the MPC controller
    // at each iteration and the actual closed-loop trajectory?
    // -------------------------------------------------------------------------
    int N = 3;   // mpc horizon
    int T = 20;  // simulation time
    VectorXd x_upper(2), x_lower(2), x0(2);
    x_upper << 1., 5.;
    x_lower << -6., -5.;
    double r_u = 1.0;         // control constraint
    double r_N = pora::kInf;  // no terminal constraint
    MatrixXd Q_f = Q;         // terminal cost matrix
    x0 << -5.5, 3.;

    // Simulate the system
    simulate(x0, T, N, x_upper, x_lower, r_u, Q_f, r_N, "Q_f = Q", "mpc_terminal_cost_Q.png");

    // -------------------------------------------------------------------------
    // ### Exercise 4.3: Terminal Cost from DARE
    // Run the provided code below that defines the terminal cost matrix Q_f by the solution to
    // the discrete-time algebraic Riccati equation (DARE) (recall the relationship to the
    // optimal discrete time LQR controller).
    // What do you notice about the difference between the MPC trajectory computed at each time
    // step and the closed-loop system behavior in this case?
    // Why is this different than when using Q_f = Q?
    // Notice that the total cost of the closed-loop trajectory is lower when using the DARE
    // terminal cost matrix, why is that?
    // -------------------------------------------------------------------------
    N = 3;   // mpc horizon
    T = 20;  // simulation time
    r_N = pora::kInf;  // no terminal constraint
    Q_f = pora::solve_discrete_are(A, B, Q, R);

    // Simulate the system
    simulate(x0, T, N, x_upper, x_lower, r_u, Q_f, r_N, "DARE", "mpc_terminal_cost_dare.png");

    // -------------------------------------------------------------------------
    // ### Exercise 4.4: Persistent Feasibility
    // Persistent feasibility is a challenge with MPC, where we want to guarantee that if we find
    // a feasible solution at one time step then we will be able to find a feasible solution at
    // future time steps.
    // One simple solution to this problem is to apply a terminal constraint x_N = 0, which
    // forces the optimizer to find a trajectory to the origin by the end of the finite horizon,
    // thereby eliminating the issue with being short-sighted.
    // However, this will cause the controller to be sub-optimal and can require a longer
    // horizon to find a feasible solution.
    // Run the provided code below and play around with the controller's horizon, how does the
    // closed-loop trajectory's cost change as the horizon changes?
    // -------------------------------------------------------------------------
    N = 3;     // mpc horizon
    T = 20;    // simulation time
    r_N = 0.0; // add terminal constraint
    Q_f = pora::solve_discrete_are(A, B, Q, R);

    // Simulate the system
    simulate(x0, T, N, x_upper, x_lower, r_u, Q_f, r_N, "Terminal constraint", "mpc_terminal_constraint.png");
  });
}
