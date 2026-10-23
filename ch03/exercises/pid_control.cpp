// =============================================================================
// Exercise 1: PD Control of Inverted Pendulum
// (C++ port of ch03/exercises/pid_control.ipynb)
// =============================================================================
#include <pora/core.hpp>
#include <pora/numerics.hpp>
#include <pora/plot.hpp>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// -----------------------------------------------------------------------------
// ### Inverted Pendulum Dynamics
// The dynamics for a controlled inverted pendulum is defined by the second order equation:
//     \ddot{\theta} = \frac{g}{l}\sin\theta + \frac{1}{ml^2}u,
// where θ is the angle of the pendulum with respect to the positive vertical axis, g is the
// acceleration due to gravity, l is the length of the pendulum, m is the pendulum mass, and
// u is the control input (a torque about the axis of rotation).
//
// The linearized approximation to these dynamics are simply:
//     \ddot{\theta} = \frac{g}{l}\theta + \frac{1}{ml^2}u.
//
// A simple PD controller for this system could take the form:
//     u = k_p\theta + k_d \dot{\theta},
// which would give linear closed-loop dynamics:
//     [\dot{\theta}; \ddot{\theta}] = [0, 1; g/l + k_p/(ml^2), k_d/(ml^2)] [\theta; \dot{\theta}]
// -----------------------------------------------------------------------------

// ### Exercise 1.1: Implement the PD controller and the closed-loop inverted pendulum dynamics

/**
 * Evaluate closed-loop inverted pendulum PD controller.
 *
 * Args:
 *     x: state vector, [θ, dθ_dt]
 *     kp: proportional gain
 *     kd: derivative gain
 *
 * Returns:
 *     u: control input for inverted pendulum
 */
double inverted_pendulum_controller(const VectorXd& x, double kp, double kd) {
  const double theta = x[0], dtheta = x[1];

  double u;
  // Implement the PD controller below:
  // ##### YOUR CODE STARTS HERE #####
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
  return u;
}

/**
 * Evaluate closed-loop inverted pendulum dynamics.
 *
 * Args:
 *     x: state vector, [θ, dθ_dt]
 *     t: time [s]
 *     kp: proportional gain
 *     kd: derivative gain
 *     g: acceleration due to gravity [m/s^2]
 *     m: mass [kg]
 *     l: pendulum length [m]
 *
 * Returns:
 *     State vector derivative, dx/dt = [dθ_dt, d2θ_dt2]
 */
VectorXd closed_loop_inverted_pendulum(const VectorXd& x, double t, double kp, double kd, double g = 9.81,
                                       double m = 1., double l = 1.) {
  const double theta = x[0], dtheta = x[1];

  // Compute control input
  double u = inverted_pendulum_controller(x, kp, kd);

  VectorXd dx_dt(2);
  // Apply the control to the dynamics
  // ##### YOUR CODE STARTS HERE #####
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
  return dx_dt;
}

int main() {
  return pora::run([] {
    // -------------------------------------------------------------------------
    // ### Exercise 1.2: Linearize the closed-loop dynamics, tune gains, and verify eigenvalues
    // -------------------------------------------------------------------------

    // Linearize around the stationary upright position with zero control
    // (i.e. the pendulum is perfectly balanced)
    VectorXd x_eq = VectorXd::Zero(2);

    // Tune these gains to ensure the linear system is stable
    double kp = -10.;
    double kd = -0.5;

    // Compute the Jacobian, f_jac, of closed_loop_inverted_pendulum at x_eq
    // (f_jac(x, t) must return the 2x2 matrix d(dx/dt)/dx)
    std::function<MatrixXd(const VectorXd&, double)> f_jac;
    // ##### YOUR CODE STARTS HERE #####
    // Hint: use pora::jacobian (the C++ counterpart of jax.jacobian), e.g. inside a lambda.
    throw pora::NotImplementedError("Need to implement code here.");
    // ###### YOUR CODE ENDS HERE ######
    MatrixXd A = f_jac(x_eq, 0);  // Evaluate Jacobian at equilibrium point
    std::cout << A << std::endl;

    // Compute the eigenvalues of the Jacobian, and verify they are stable (negative real parts)
    Eigen::EigenSolver<MatrixXd> es(A);
    std::cout << es.eigenvalues().transpose() << std::endl;

    // -------------------------------------------------------------------------
    // ### Exercise 1.3: Simulate nonlinear closed-loop dynamics
    // Run the code below to simulate the nonlinear closed-loop dynamics using the tuned PD
    // controller.
    // -------------------------------------------------------------------------
    double theta0 = pora::deg2rad(80.);
    VectorXd x0(2);
    x0 << theta0, 0.;
    double T = 30.;

    VectorXd t = pora::linspace(0, T, 1000);
    MatrixXd x = pora::odeint(
        [&](const VectorXd& xx, double tt) { return closed_loop_inverted_pendulum(xx, tt, kp, kd); }, x0, t);
    VectorXd theta = x.col(0), dtheta = x.col(1);

    pora::plt::Figure fig(1, 1, 750, 300);
    VectorXd theta_deg = theta * 180.0 / pora::kPi;
    fig.ax().plot(t, theta_deg);
    char title[64];
    std::snprintf(title, sizeof(title), "kappa_p = %g, kappa_d = %g", kp, kd);
    fig.ax().set_title(title);
    fig.ax().set_xlabel("t");
    fig.ax().set_ylabel("theta(t) [deg]");
    fig.save("inverted_pendulum_pd.png");
  });
}
