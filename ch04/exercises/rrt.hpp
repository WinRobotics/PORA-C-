// C++ port of ch04/exercises/rrt.py (class declarations)
#pragma once

#include "utils.hpp"

#include <pora/plot.hpp>

#include <Eigen/Dense>

#include <string>
#include <vector>

/** Represents a motion planning problem to be solved using the RRT algorithm. */
class RRT {
 public:
  RRT(const Workspace2D& workspace, const Eigen::Vector2d& x_init, const Eigen::Vector2d& x_goal);
  virtual ~RRT() = default;

  /** Plot the workspace, start, and goal points. */
  void plot_problem(pora::plt::Axes& ax) const;

  /**
   * Subject to the robot dynamics, returns whether a point robot moving
   * along the shortest path from x1 to x2 would collide with any obstacles
   * (implemented as a "black box")
   *
   * Args:
   *     x1: start state of motion
   *     x2: end state of motion
   *
   * Returns:
   *     Whether the motion from x1 to x2 is collision free.
   */
  virtual bool is_free_motion(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2) const = 0;

  /**
   * Given a query state x, returns the index of V such that the steering
   * distance (subject to robot dynamics) from V[i] to x is minimized.
   *
   * Args:
   *     x: query state
   *
   * Returns:
   *     Integer index of nearest point in V to x
   */
  virtual int nearest_neighbor(const Eigen::Vector2d& x) const = 0;

  /**
   * Given a query state x, returns the indices of V such that the distance
   * from the query state to the states is less than r
   *
   * Args:
   *     x: query state
   *     r: query radius
   *
   * Returns:
   *     Indices (into V) of the nodes within radius r of the query state.
   *     (The Python version returns the Node objects themselves.)
   */
  virtual std::vector<int> neighbors(const Eigen::Vector2d& x, double r) const = 0;

  /**
   * Steers from x1 towards x2 along the shortest path (subject to robot
   * dynamics). Returns x2 if the length of this shortest path is less than
   * eps, otherwise returns the point at distance eps along the path from
   * x1 to x2.
   *
   * Args:
   *     x1: start state
   *     x2: target state
   *     eps: maximum steering distance
   *
   * Returns:
   *     State resulting from bounded steering
   */
  virtual Eigen::Vector2d steer_towards(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2, double eps) const = 0;

  /**
   * Constructs an RRT rooted at x_init with the aim of producing a
   * dynamically-feasible and obstacle-free trajectory from x_init
   * to x_goal.
   *
   * Updates V and path if successful. The result is plotted into `figure_name`.
   *
   * Args:
   *     eps: maximum steering distance
   *     max_iters: maximum number of RRT iterations (early termination
   *                is possible when a feasible solution is found)
   *     goal_bias: probability during each iteration of setting
   *                x_rand = x_goal (instead of uniformly randly sampling
   *                from the state space)
   *
   * Returns:
   *     Whether the solution was found
   */
  bool solve(double eps, int max_iters = 1000, double goal_bias = 0.05, bool shortcut = false,
             const std::string& figure_name = "rrt.png");

  /**
   * Iteratively removes nodes from solution path to find a shorter path
   * which is still collision-free.
   *
   * Modifies path.
   */
  void shortcut_path();

  /**
   * Compute a path from a start point to a goal region, avoiding obstacles,
   * using the RRT* algorithm.
   *
   * Args:
   *     eps: maximum steering distance
   *     max_iters: maximum number of RRT iterations (early termination
   *                is possible when a feasible solution is found)
   *     goal_bias: probability during each iteration of setting
   *                x_rand = x_goal (instead of uniformly randly sampling
   *                from the state space)
   *
   * Returns:
   *     Cost-to-arrive at the goal node from the starting node, or inf if
   *     no solution was found.
   */
  double solve_optimal(double eps, int max_iters = 1000, double goal_bias = 0.05,
                       const std::string& figure_name = "rrt_star.png");

  virtual void plot_tree(pora::plt::Axes& ax, const pora::plt::Style& style) const = 0;
  virtual void plot_path(pora::plt::Axes& ax, const pora::plt::Style& style) const = 0;

  const Workspace2D& workspace;
  Eigen::Vector2d x_init;  // initial state
  Eigen::Vector2d x_goal;  // goal state
  std::vector<Node> V;     // list of Nodes in the tree
  std::vector<Node> path;  // the final path as a list of Nodes
};

/**
 * Represents a geometric planning problem, where the steering solution
 * between two points is a straight line (Euclidean metric).
 */
class GeometricRRT : public RRT {
 public:
  using RRT::RRT;

  int nearest_neighbor(const Eigen::Vector2d& x) const override;
  std::vector<int> neighbors(const Eigen::Vector2d& x, double r) const override;
  Eigen::Vector2d steer_towards(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2, double eps) const override;
  bool is_free_motion(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2) const override;
  void plot_tree(pora::plt::Axes& ax, const pora::plt::Style& style) const override;
  void plot_path(pora::plt::Axes& ax, const pora::plt::Style& style) const override;
};
