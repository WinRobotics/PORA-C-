// C++ port of ch03/exercises/unicycle_utils.py
#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

/**
 * Define the state of the extended kinematic unicycle
 */
struct State {
  State(double x, double y, double v, double th) : x(x), y(y), v(v), th(th) {}
  double x, y, v, th;

  double xd() const { return v * std::cos(th); }
  double yd() const { return v * std::sin(th); }
};

// Interface of a feedback controller usable by simulate_unicycle (duck-typed in Python).
class UnicycleController {
 public:
  virtual ~UnicycleController() = default;
  /** Compute the control (a, ω) for the current state and time. */
  virtual Eigen::Vector2d compute_control(const State& s, double t) = 0;
};

/**
 * Differential equation describing the kinematic unicycle with state x = [x, y, v, th] and
 * control u = [a, om], with assumed noise on the control inputs.
 */
Eigen::VectorXd unicycle_dynamics(const Eigen::VectorXd& x, double t, const Eigen::Vector2d& u,
                                  const Eigen::Vector2d& noise);

/**
 * Simulate the unicycle model given a closed-loop controller OR a sequence of control inputs
 * to apply open-loop.
 *
 * inputs: x_0, y_0, v_0, th_0 (floats) initial state
 *         times (length N) sequence of times at which to apply control
 *         controller: controller object to use to compute feedback control (or nullptr)
 *         open_loop_control: (shape: N-1, 2) list of control inputs to apply (used if no controller)
 *         noise_scale: (float) standard deviation of control noise
 *
 *         if controller is provided, simulates feedback control by calling
 *             controller->compute_control(State(x, y, v, th), t) at each time step
 *         otherwise, if the array open_loop_control is specified, they are applied open loop
 *
 *         (one of controller or open_loop_control must be specified)
 *
 * outputs: traj (shape (N, 4)) sequence of [x,y,v,th] state vectors
 *          ctrl (shape (N, 2)) sequence of [a, om] control vectors
 */
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> simulate_unicycle(double x_0, double y_0, double v_0, double th_0,
                                                              const Eigen::VectorXd& times,
                                                              UnicycleController* controller = nullptr,
                                                              const Eigen::MatrixXd* open_loop_control = nullptr,
                                                              double noise_scale = 0.);

/**
 * Plot one or more trajectories (x-y path, heading, speed) and control sequences (ω), and
 * save the figure as `filename` in the output directory.
 */
void plot_unicycle_traj(const Eigen::VectorXd& t, const std::vector<Eigen::MatrixXd>& trajs,
                        const std::vector<Eigen::MatrixXd>& ctrls, const std::vector<std::string>& labels,
                        const std::string& filename, const std::vector<std::string>& linestyles = {},
                        const std::vector<double>& alphas = {});
