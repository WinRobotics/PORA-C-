#include "pora/qp.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pora {

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();

double inf_norm(const Eigen::VectorXd& v) { return v.size() ? v.cwiseAbs().maxCoeff() : 0.0; }
}  // namespace

QP::QP(int num_vars) : P(Eigen::MatrixXd::Zero(num_vars, num_vars)), q(Eigen::VectorXd::Zero(num_vars)), n_(num_vars) {
  A_.resize(0, num_vars);
}

void QP::add_constraints(const Eigen::MatrixXd& rows, const Eigen::VectorXd& lower, const Eigen::VectorXd& upper) {
  if (rows.cols() != n_ || rows.rows() != lower.size() || rows.rows() != upper.size())
    throw std::invalid_argument("QP::add_constraints: inconsistent dimensions");
  const Eigen::Index m0 = A_.rows(), m = rows.rows();
  A_.conservativeResize(m0 + m, n_);
  l_.conservativeResize(m0 + m);
  u_.conservativeResize(m0 + m);
  A_.bottomRows(m) = rows;
  l_.tail(m) = lower;
  u_.tail(m) = upper;
}

QPResult QP::solve(const QPSettings& s) const {
  const int n = n_;
  const int m = static_cast<int>(A_.rows());
  const Eigen::MatrixXd& A = A_;
  Eigen::VectorXd l = l_.cwiseMax(-1e20), u = u_.cwiseMin(1e20);

  // Per-constraint step sizes: stiffer for equalities, tiny for free rows.
  auto make_rho = [&](double rho) {
    Eigen::VectorXd r(m);
    for (int i = 0; i < m; ++i) {
      if (l[i] <= -1e19 && u[i] >= 1e19) r[i] = 1e-6;
      else if (std::abs(u[i] - l[i]) < 1e-12) r[i] = 1e3 * rho;
      else r[i] = rho;
    }
    return r;
  };
  double rho = s.rho;
  Eigen::VectorXd rho_vec = make_rho(rho);
  auto factor = [&](const Eigen::VectorXd& rv) {
    Eigen::MatrixXd K = P + s.sigma * Eigen::MatrixXd::Identity(n, n) + A.transpose() * rv.asDiagonal() * A;
    return Eigen::LDLT<Eigen::MatrixXd>(K);
  };
  Eigen::LDLT<Eigen::MatrixXd> kkt = factor(rho_vec);

  Eigen::VectorXd x = Eigen::VectorXd::Zero(n), z = Eigen::VectorXd::Zero(m), y = Eigen::VectorXd::Zero(m);
  QPResult res;
  res.status = "max_iter";
  for (int k = 1; k <= s.max_iter; ++k) {
    Eigen::VectorXd rhs = s.sigma * x - q + A.transpose() * (rho_vec.cwiseProduct(z) - y);
    Eigen::VectorXd xt = kkt.solve(rhs);
    Eigen::VectorXd zt = A * xt;
    Eigen::VectorXd x_new = s.alpha * xt + (1 - s.alpha) * x;
    Eigen::VectorXd z_relax = s.alpha * zt + (1 - s.alpha) * z;
    Eigen::VectorXd z_new = (z_relax + y.cwiseQuotient(rho_vec)).cwiseMax(l).cwiseMin(u);
    Eigen::VectorXd y_new = y + rho_vec.cwiseProduct(z_relax - z_new);
    Eigen::VectorXd dx = x_new - x, dy = y_new - y;
    x = x_new;
    z = z_new;
    y = y_new;
    res.iterations = k;

    // Convergence check
    Eigen::VectorXd Ax = A * x, Px = P * x, Aty = A.transpose() * y;
    double r_prim = inf_norm(Ax - z), r_dual = inf_norm(Px + q + Aty);
    double eps_prim = s.eps_abs + s.eps_rel * std::max(inf_norm(Ax), inf_norm(z));
    double eps_dual = s.eps_abs + s.eps_rel * std::max({inf_norm(Px), inf_norm(Aty), inf_norm(q)});
    if (r_prim <= eps_prim && r_dual <= eps_dual) {
      res.status = "optimal";
      break;
    }

    // Primal infeasibility certificate: A^T dy ~ 0 and u^T max(dy,0) + l^T min(dy,0) < 0
    double ndy = inf_norm(dy);
    if (ndy > 1e-12) {
      double support = 0.0;
      bool finite = true;
      for (int i = 0; i < m; ++i) {
        if (dy[i] > 0) {
          if (u[i] >= 1e19) { finite = false; break; }
          support += u[i] * dy[i];
        } else if (dy[i] < 0) {
          if (l[i] <= -1e19) { finite = false; break; }
          support += l[i] * dy[i];
        }
      }
      if (finite && inf_norm(A.transpose() * dy) <= s.eps_prim_inf * ndy && support <= -s.eps_prim_inf * ndy) {
        res.status = "infeasible";
        break;
      }
    }

    // Dual infeasibility (unbounded problem) certificate
    double ndx = inf_norm(dx);
    if (ndx > 1e-12 && inf_norm(P * dx) <= s.eps_dual_inf * ndx && q.dot(dx) <= -s.eps_dual_inf * ndx) {
      Eigen::VectorXd Adx = A * dx;
      bool ok = true;
      for (int i = 0; i < m && ok; ++i) {
        double tol = s.eps_dual_inf * ndx;
        if (u[i] < 1e19 && Adx[i] > tol) ok = false;
        if (l[i] > -1e19 && Adx[i] < -tol) ok = false;
      }
      if (ok) {
        res.status = "unbounded";
        break;
      }
    }

    // Adaptive step size (as in OSQP)
    if (k % 50 == 0) {
      double num = r_prim / std::max(std::max(inf_norm(Ax), inf_norm(z)), 1e-10);
      double den = r_dual / std::max(std::max({inf_norm(Px), inf_norm(Aty), inf_norm(q)}), 1e-10);
      double rho_new = std::min(std::max(rho * std::sqrt(num / std::max(den, 1e-10)), 1e-6), 1e6);
      if (rho_new > 5 * rho || rho_new < rho / 5) {
        rho = rho_new;
        rho_vec = make_rho(rho);
        kkt = factor(rho_vec);
      }
    }
  }
  res.z = x;
  res.y = y;
  if (res.status == "optimal" || res.status == "max_iter") res.cost = 0.5 * x.dot(P * x) + q.dot(x);
  else res.cost = res.status == "infeasible" ? kInf : -kInf;
  return res;
}

}  // namespace pora
