// Small dense convex QP solver (ADMM, following the OSQP algorithm). It is the
// C++ stand-in for CVXPY in the linear MPC exercise:
//
//     minimize    0.5 z^T P z + q^T z
//     subject to  l <= A z <= u
//
// Equality constraints use l == u, one-sided constraints use +/-infinity.
#pragma once

#include <Eigen/Dense>

#include <limits>
#include <string>

namespace pora {

struct QPSettings {
  double rho = 0.1;
  double sigma = 1e-6;
  double alpha = 1.6;
  double eps_abs = 1e-6;
  double eps_rel = 1e-6;
  double eps_prim_inf = 1e-7;
  double eps_dual_inf = 1e-7;
  int max_iter = 100000;
};

struct QPResult {
  Eigen::VectorXd z;   // primal solution
  Eigen::VectorXd y;   // dual solution
  std::string status;  // "optimal", "infeasible", "unbounded" or "max_iter"
  double cost = std::numeric_limits<double>::quiet_NaN();
  int iterations = 0;
};

class QP {
 public:
  // Problem with `num_vars` decision variables, zero cost and no constraints.
  explicit QP(int num_vars);

  int num_vars() const { return n_; }
  int num_constraints() const { return static_cast<int>(A_.rows()); }

  Eigen::MatrixXd P;  // quadratic cost (n x n, symmetric PSD)
  Eigen::VectorXd q;  // linear cost (n)

  // Append the constraints lower <= rows * z <= upper.
  void add_constraints(const Eigen::MatrixXd& rows, const Eigen::VectorXd& lower, const Eigen::VectorXd& upper);
  // Append the equality constraints rows * z == rhs.
  void add_equality(const Eigen::MatrixXd& rows, const Eigen::VectorXd& rhs) { add_constraints(rows, rhs, rhs); }

  QPResult solve(const QPSettings& settings = QPSettings()) const;

 private:
  int n_;
  Eigen::MatrixXd A_;
  Eigen::VectorXd l_, u_;
};

}  // namespace pora
