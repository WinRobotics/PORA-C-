// C++ port of ch03/exercises/unicycle_utils.py
#include "unicycle_utils.hpp"

#include <pora/numerics.hpp>
#include <pora/plot.hpp>
#include <pora/random.hpp>

#include <iostream>
#include <stdexcept>

using Eigen::MatrixXd;
using Eigen::VectorXd;

VectorXd unicycle_dynamics(const VectorXd& x, double /*t*/, const Eigen::Vector2d& u, const Eigen::Vector2d& noise) {
  const double v = x[2] + noise[0];
  const double om = u[1] + noise[1];
  VectorXd dxdt(4);
  dxdt << v * std::cos(x[3]), v * std::sin(x[3]), u[0], om;
  return dxdt;
}

std::pair<MatrixXd, MatrixXd> simulate_unicycle(double x_0, double y_0, double v_0, double th_0, const VectorXd& times,
                                                UnicycleController* controller, const MatrixXd* open_loop_control,
                                                double noise_scale) {
  bool feedback = false;
  if (controller) {
    feedback = true;
  } else if (open_loop_control == nullptr) {
    std::cout << "Either provide a controller or a sequence of open loop actions" << std::endl;
    throw std::invalid_argument("simulate_unicycle: no controller");
  }

  VectorXd x(4);
  x << x_0, y_0, v_0, th_0;
  const int N = static_cast<int>(times.size());
  MatrixXd traj = MatrixXd::Zero(N, 4);
  MatrixXd noise = noise_scale * pora::rng().normal(N, 2);  // control noise
  MatrixXd ctrl = MatrixXd::Zero(N, 2);
  for (int i = 0; i < N - 1; ++i) {
    const double t = times[i];
    traj.row(i) = x.transpose();

    // Compute control
    double a = 0, om = 0;
    if (feedback) {
      Eigen::Vector2d u = controller->compute_control(State(x[0], x[1], x[2], x[3]), t);
      a = u[0];
      om = u[1];
    } else {
      a = (*open_loop_control)(i, 0);
      om = (*open_loop_control)(i, 1);
    }

    ctrl(i, 0) = a;
    ctrl(i, 1) = om;

    // Apply control and simulate forward
    Eigen::Vector2d ui = ctrl.row(i).transpose(), ni = noise.row(i).transpose();
    VectorXd span(2);
    span << t, times[i + 1];
    MatrixXd d_state = pora::odeint([&](const VectorXd& xx, double tt) { return unicycle_dynamics(xx, tt, ui, ni); },
                                    x, span);
    x = d_state.row(1).transpose();
  }

  // Log final state
  traj.row(N - 1) = x.transpose();

  return {traj, ctrl};
}

void plot_unicycle_traj(const VectorXd& t, const std::vector<MatrixXd>& trajs, const std::vector<MatrixXd>& ctrls,
                        const std::vector<std::string>& labels, const std::string& filename,
                        const std::vector<std::string>& linestyles, const std::vector<double>& alphas) {
  using pora::plt::fmt;
  namespace colors = pora::plt::colors;
  pora::plt::Figure fig(3, 2, 1100, 650);
  auto style = [&](size_t i, const pora::plt::Color& c) {
    return fmt(linestyles.empty() ? "-" : linestyles[i]).c(c).alpha(alphas.empty() ? 1.0 : alphas[i]).label(labels[i]);
  };

  auto& ax_xy = fig.span(0, 0, 3, 1);
  for (size_t i = 0; i < trajs.size(); ++i) ax_xy.plot(trajs[i].col(0), trajs[i].col(1), style(i, colors::black));
  ax_xy.set_xlabel("x");
  ax_xy.set_ylabel("y");
  ax_xy.legend();
  ax_xy.grid(true);

  auto& ax_th = fig.ax(0, 1);
  for (size_t i = 0; i < trajs.size(); ++i) ax_th.plot(t, trajs[i].col(3), style(i, colors::green));
  ax_th.set_ylabel("th");
  ax_th.legend();
  ax_th.grid(true);
  ax_th.set_xlabel("t");

  auto& ax_v = fig.ax(1, 1);
  for (size_t i = 0; i < trajs.size(); ++i) ax_v.plot(t, trajs[i].col(2), style(i, colors::blue));
  ax_v.grid(true);
  ax_v.set_xlabel("t");
  ax_v.set_ylabel("v");
  ax_v.legend();

  auto& ax_om = fig.ax(2, 1);
  const int n = static_cast<int>(t.size()) - 1;
  for (size_t i = 0; i < ctrls.size(); ++i) ax_om.plot(t.head(n), ctrls[i].col(1).head(n), style(i, colors::magenta));
  ax_om.grid(true);
  ax_om.set_xlabel("t");
  ax_om.set_ylabel("om");
  ax_om.legend();
  fig.save(filename);
}
