// =============================================================================
// Exercise 5: Extended Unicycle Trajectory Tracking Control
// (C++ port of ch03/exercises/unicycle_trajectory_tracking.ipynb)
//
// Consider the extended kinematic model of a unicycle:
//     \dot{x} = v \cos(\theta), \quad \dot{y} = v \sin(\theta), \quad \dot{v}=a, \quad \dot{\theta} = \omega.
// where (x,y) is the position, θ is the heading angle with respect to the x-axis, v is the
// speed, and the control inputs are the acceleration a and the angular velocity ω. We define
// the state vector x = (x, y, v, θ).
//
// In this exercise we will implement a differential flatness-based trajectory tracking
// controller for this robot.
// =============================================================================
#include "unicycle_utils.hpp"

#include <pora/core.hpp>
#include <pora/numerics.hpp>

#include <Eigen/Dense>

#include <cmath>
#include <utility>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// -----------------------------------------------------------------------------
// ### Exercise 5.1: Differentially Flat Trajectory Open-loop Control
// Implement the `compute_traj_coeffs`, `compute_traj`, and `compute_controls` functions below
// to compute a differentially flat trajectory for the extended kinematic unicycle model above.
// -----------------------------------------------------------------------------

/**
 * Compute the flat trajectory coefficients to connect the initial to the final state.
 *
 * Args:
 *     s_0: the initial state
 *     s_f: the final state
 *     t_f: final time
 *
 * Returns:
 *     Coefficients on the basis functions, array of shape (8,)
 */
VectorXd compute_traj_coeffs(const State& s_0, const State& s_f, double t_f) {
  VectorXd coeffs(8);
  // ##### YOUR CODE STARTS HERE #####
  // Hint: Use Eigen's `.colPivHouseholderQr().solve()` (the counterpart of np.linalg.solve).
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
  return coeffs;
}

/**
 * Compute the trajectory from the basis function coefficients. Each point in time
 * of the output trajectory is the vector [x, y, v, th, dx, dy, ddx, ddy].
 *
 * Args:
 *     s_0: the initial state
 *     s_f: the final state
 *     t_f: final_time
 *     N: number of points
 *
 * Returns:
 *     t: evenly spaced time points from 0 to t_f, shape (N,)
 *     traj: N points along the trajectory from t=0 to t=t_f, evenly spaced in time, shape (N, 8)
 */
std::pair<VectorXd, MatrixXd> compute_traj(const State& s_0, const State& s_f, double t_f, int N) {
  VectorXd coeffs = compute_traj_coeffs(s_0, s_f, t_f);
  VectorXd t = pora::linspace(0, t_f, N);  // generate evenly spaced points from 0 to t_f
  MatrixXd traj = MatrixXd::Zero(N, 8);

  // ##### YOUR CODE STARTS HERE #####
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######

  return {t, traj};
}

/**
 * Convert the differentially flat trajectory into the controls [a, ω].
 *
 * Args:
 *     traj: differentially flat trajectory, shape (N, 8)
 *
 * Returns:
 *     Controls [a, ω] for each point of the trajectory, shape (N, 2)
 */
MatrixXd compute_controls(const MatrixXd& traj) {
  const int N = static_cast<int>(traj.rows());
  MatrixXd ctrl = MatrixXd::Zero(N, 2);
  // ##### YOUR CODE STARTS HERE #####
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
  return ctrl;
}

// -----------------------------------------------------------------------------
// ### Exercise 5.2: Closed-loop Tracking Control
// In the class `TrajectoryTracker` implement the function `compute_control` to compute the
// controls for the differentially flat trajectory tracking controller from Example 3.2.2 in
// the text.
// -----------------------------------------------------------------------------

/**
 * Trajectory tracking controller using differential flatness
 */
class TrajectoryTracker : public UnicycleController {
 public:
  /**
   * Initialize tracker variables.
   *
   * Args:
   *     kpx: x proportional gain
   *     kpy: y proportional gain
   *     kdx: x derivative gain
   *     kdy: y derivative gain
   *     om_max: max angular velocity
   */
  TrajectoryTracker(double kpx, double kpy, double kdx, double kdy, double om_max = 1)
      : kpx(kpx), kpy(kpy), kdx(kdx), kdy(kdy), om_max(om_max) {
    traj_times = VectorXd::Zero(1);
    traj = MatrixXd::Zero(1, 8);
  }

  /**
   * Loads in a new trajectory to follow, and resets the time. Assumes the
   * traj is of shape (N, 8) where at each time is the vector
   * [x, y, v, th, dx, dy, ddx, ddy].
   *
   * Args:
   *     times: array of times
   *     traj: array defining the shape at the times, shape (N, 8)
   */
  void load_traj(const VectorXd& times, const MatrixXd& traj_) {
    traj_times = times;
    traj = traj_;
  }

  /**
   * Get the flat output state (x_d, xd_d, xdd_d, y_d, yd_d, ydd_d) for the given
   * time, t, using the tracker's traj.
   *
   * Args:
   *     t: Current time
   *
   * Returns:
   *     Array (x_d, xd_d, xdd_d, y_d, yd_d, ydd_d) defining the desired state
   *     and derivatives at time t according to traj.
   */
  VectorXd get_desired_state(double t) const {
    VectorXd d(6);
    d << pora::interp(t, traj_times, traj.col(0)),  // x_d
         pora::interp(t, traj_times, traj.col(4)),  // xd_d
         pora::interp(t, traj_times, traj.col(6)),  // xdd_d
         pora::interp(t, traj_times, traj.col(1)),  // y_d
         pora::interp(t, traj_times, traj.col(5)),  // yd_d
         pora::interp(t, traj_times, traj.col(7));  // ydd_d
    return d;
  }

  /**
   * Compute the control (a, ω) using a PD controller to track the traj reference.
   *
   * Args:
   *     s: current state
   *     t: current time
   *
   * Returns:
   *     Control array (a, ω).
   */
  Eigen::Vector2d compute_control(const State& s, double t) override {
    VectorXd d = get_desired_state(t);
    const double x_d = d[0], xd_d = d[1], xdd_d = d[2], y_d = d[3], yd_d = d[4], ydd_d = d[5];

    const double V_THRES = 0.0001;
    double a, om;
    // ##### YOUR CODE STARTS HERE #####
    // Hint: Use V_THRES to avoid a potential singularity if v < V_THRES
    // Avoid singularity
    throw pora::NotImplementedError("Need to implement code here.");
    // ###### YOUR CODE ENDS HERE ######

    // Apply control limits
    double omega = pora::clip(om, -om_max, om_max);
    return {a, omega};
  }

  double kpx, kpy, kdx, kdy, om_max;
  VectorXd traj_times;
  MatrixXd traj;
};

int main() {
  return pora::run([] {
    // Now, run the code below for the provided initial and final states to see what the open
    // loop differentially flat trajectory would look like for the extended kinematic unicycle.

    // Define constants
    double t_f = 25;  // final time (sec)
    int N = 501;      // number of timesteps
    State s_0(0, 0, 0.5, -pora::kPi / 2);
    State s_f(5, 5, 0.5, -pora::kPi / 2);

    // Compute trajectory
    auto [t, traj] = compute_traj(s_0, s_f, t_f, N);
    MatrixXd ctrl = compute_controls(traj);

    // Plot results
    plot_unicycle_traj(t, {traj}, {ctrl}, {"nominal"}, "nominal.png");

    // Next, we will simulate how well we can actually execute these trajectories in the
    // presense of noise. We will make use of the `simulate_unicycle` function to do the
    // simulation of the same control sequence computed above.
    double noise_scale = std::sqrt(0.2);
    auto [traj_ol, ctrl_ol] = simulate_unicycle(s_0.x, s_0.y, s_0.v, s_0.th, t, nullptr, &ctrl, noise_scale);

    // Plot results
    plot_unicycle_traj(t, {traj, traj_ol}, {ctrl}, {"nominal", "ol"}, "open_loop.png", {":", "-"}, {0.7, 1});

    // Run the code below to see how the simulation differs for the unicycle with the same
    // amount of noise as before, but this time applying the closed-loop tracking controller.
    TrajectoryTracker controller(2, 2, 2, 2, 1);
    controller.load_traj(t, traj);
    auto [traj_cl, ctrl_cl] = simulate_unicycle(s_0.x, s_0.y, s_0.v, s_0.th, t, &controller, nullptr, noise_scale);

    // Plot results
    plot_unicycle_traj(t, {traj, traj_cl}, {ctrl, ctrl_cl}, {"nominal", "closed-loop"}, "closed_loop.png",
                       {":", "-"}, {0.7, 1.0});
  });
}
