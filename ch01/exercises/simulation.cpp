// =============================================================================
// Exercise 1: Numerical Integration Methods
// (C++ port of ch01/exercises/simulation.ipynb)
// =============================================================================
#include <pora/core.hpp>
#include <pora/numerics.hpp>
#include <pora/plot.hpp>

#include <Eigen/Dense>

#include <cmath>
#include <string>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using pora::odeint;

// -----------------------------------------------------------------------------
// ### Exercise 1.1: Simulate Dampled Pendulum Dynamics
//
// The dynamics for a simple (damped) pendulum are defined by the second order equation:
//     \ddot{\theta} = -\frac{g}{l}\sin \theta - \gamma \dot{\theta},
// where θ is the angle of the pendulum with respect to the negative vertical axis, g is
// the acceleration due to gravity, l is the length of the pendulum, and γ is a damping
// coefficient.
// This second-order differential equation is equivalent to the system of first-order
// differential equations:
//     \dot{x}_0 = x_1, \quad \dot{x}_1 = -\frac{g}{l}\sin x_0 - \gamma x_1,
// where x_0 = θ and x_1 = dθ/dt.
//
// Implement the damped pendulum dynamics in the function below:
// -----------------------------------------------------------------------------

/**
 * Evaluate pendulum dynamics.
 *
 * Args:
 *     x: state vector, [θ, dθ_dt]
 *     t: time [s]
 *     g: acceleration due to gravity [m/s^2]
 *     l: pendulum length [m]
 *     γ: damping coefficient [1/s]
 *
 * Returns:
 *     dx_dt : state vector derivative, [dθ_dt, d2θ_dt2]
 */
VectorXd dampled_pendulum(const VectorXd& x, double t, double g = 9.81, double l = 1., double gamma = 1.) {
  VectorXd dx(2);
  // ##### YOUR CODE STARTS HERE #####
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
  return dx;
}

// -----------------------------------------------------------------------------
// ### Exercise 2: Simulate Bicycle Dynamics
//
// A simple motion model for a bicycle/simple car is:
//     \dot{x} = v\cos\theta, \quad \dot{y} = v\sin\theta, \quad \dot{\theta} = \frac{v}{L}\tan\phi,
// where (x,y) is the position, v is the speed control, θ is the heading angle, L is the
// wheelbase length, and ϕ is the steering angle.
// This is a controlled system with the control inputs being u = [v, ϕ]^T.
// Here we can assume simple simple open-loop control with:
//     v = 1, \quad \phi = \sin(t).
//
// Implement the `open_loop_bicycle` function to model the bicycle/simple car dynamics
// with the specified open-loop controller.
// -----------------------------------------------------------------------------

/**
 * Evaluate bicycle dynamics with a simple open-loop control law.
 *
 * Args:
 *     x: state vector, [x, y, θ]
 *     t: time [s]
 *     L: wheelbase length [m]
 *
 * Returns:
 *     dx_dt: state vector derivative, [dx_dt, dy_dt, dθ_dt]
 */
VectorXd open_loop_bicycle(const VectorXd& x, double t, double L = 1.) {
  // Open-loop controller definition
  double v = 1;  // constant speed
  double phi = std::sin(t);

  VectorXd dx_dt(3);
  // ##### YOUR CODE STARTS HERE #####
  // ### Implement bicycle dynamics
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
  return dx_dt;
}

int main() {
  return pora::run([] {
    // -------------------------------------------------------------------------
    // We can simulate general robot motion models by solving the differential
    // equations via numerical integration, for example using the Euler, midpoint,
    // or Runge-Kutta methods. In Python, the `SciPy` package offers a method for
    // numerical integration called `odeint`; here `pora::odeint(f, x0, t)` plays
    // that role (adaptive Runge-Kutta, f has the signature f(x, t), and it returns
    // one row per time in t). Implement the call to `odeint` and run the code
    // below to simulate the damped pendulum.
    // -------------------------------------------------------------------------
    {
      // Define IVP and model constants
      double theta0 = pora::deg2rad(30.);  // initial angle
      VectorXd x0(2);
      x0 << theta0, 0.;
      double T = 10.;    // simulation time
      double l = 1;      // pendulum length
      double gamma = 0.5;  // damping coefficient

      // Use odeint to simulated the dynamics
      VectorXd t = pora::linspace(0, T, 100);
      MatrixXd x_odeint;  // shape (len(t), 2)
      // ##### YOUR CODE STARTS HERE #####
      // Compute x_odeint using pora::odeint (the C++ counterpart of SciPy's odeint)
      // Hint: wrap dampled_pendulum in a lambda [&](const VectorXd& x, double t) {...}
      //       to pass the extra arguments l and gamma.
      throw pora::NotImplementedError("Need to implement code here.");
      // ###### YOUR CODE ENDS HERE ######
      VectorXd theta_odeint = x_odeint.col(0), dtheta_odeint = x_odeint.col(1);

      // Plot results
      pora::plt::Figure fig(1, 1, 700, 300);
      VectorXd theta_deg = theta_odeint * 180.0 / pora::kPi;
      fig.ax().plot(t, theta_deg);
      char title[64];
      std::snprintf(title, sizeof(title), "l = %g, gamma = %g", l, gamma);
      fig.ax().set_title(title);
      fig.ax().set_xlabel("t");
      fig.ax().set_ylabel("theta(t) [deg]");
      fig.save("damped_pendulum.png");
    }

    // -------------------------------------------------------------------------
    // Implement the call to `odeint` for the bicycle dynamics and run the script
    // below to simulate the dynamics.
    // -------------------------------------------------------------------------
    {
      double dt = 0.01;
      double T = 10.;  // simulation time
      VectorXd t = pora::linspace(0, T, static_cast<int>(T / dt));
      VectorXd x0 = VectorXd::Zero(3);

      MatrixXd x_bike;  // shape (len(t), 3)
      // ##### YOUR CODE STARTS HERE #####
      // Compute x_bike using pora::odeint
      throw pora::NotImplementedError("Need to implement code here.");
      // ###### YOUR CODE ENDS HERE ######
      VectorXd pos_x_bike = x_bike.col(0), pos_y_bike = x_bike.col(1), theta_bike = x_bike.col(2);

      // Plot results
      pora::plt::Figure fig(1, 1, 600, 600);
      fig.ax().plot(pos_x_bike, pos_y_bike);
      fig.ax().set_xlabel("x(t)");
      fig.ax().set_ylabel("y(t)");
      fig.save("bicycle.png");
    }
  });
}
