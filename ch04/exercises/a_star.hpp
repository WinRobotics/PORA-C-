// C++ port of ch04/exercises/a_star.py (class declaration)
#pragma once

#include "utils.hpp"

#include <pora/plot.hpp>

#include <map>
#include <set>
#include <utility>
#include <vector>

// A state on the discrete grid, the counterpart of the Python `tuple[float]`.
using GridState = std::pair<double, double>;

/**
 * Class to define a motion planning problem to be solved using A*
 */
class AStar {
 public:
  /**
   * Initialize AStar problem.
   *
   * Args:
   *     workspace: 2D workspace to consider
   *     x_init: initial position
   *     x_goal: goal position
   *     resolution: state space resoltion
   */
  AStar(const Workspace2D& workspace, const GridState& x_init, const GridState& x_goal, double resolution = 1.);

  /**
   * Returns the closest point on a discrete state grid.
   *
   * Args:
   *     x: position
   *
   * Returns:
   *     Position that represents the closest point to x on the discrete state grid.
   */
  GridState snap_to_grid(const GridState& x) const;

  /**
   * Gets the state in the open set that has the lowest estimated cost from
   * start to goal that passes through the state.
   *
   * Returns:
   *     The state found in the open set that has the lowest lowest estimated
   *     cost from start to goal that passes through the state.
   */
  GridState find_best_est_cost_through() const;

  /**
   * Use the came_from map to reconstruct a path from the initial location to
   * the goal location.
   *
   * Returns:
   *     A list of states that go from start to goal.
   */
  std::vector<GridState> reconstruct_path() const;

  /** Plots the path found in path and the obstacles. */
  void plot_path(pora::plt::Axes& ax, bool show_init_label = true) const;

  /** Plot the tree. */
  void plot_tree(pora::plt::Axes& ax, double point_size = 4) const;

  /**
   * Checks if a give state x is free, meaning it is inside the bounds of
   * the map and is not inside any obstacle.
   */
  bool is_free(const GridState& x) const;

  /** Computes the Euclidean distance between two states. */
  double distance(const GridState& x1, const GridState& x2) const;

  /**
   * Gets the FREE neighbor states of a given state x. Assumes a motion model
   * where we can move up, down, left, right, or along the diagonals by an
   * amount equal to res.
   */
  std::vector<GridState> get_neighbors(const GridState& x) const;

  /**
   * Solves the planning problem using the A* search algorithm. It places
   * the solution as a list of states that go from x_init to x_goal inside
   * the variable path.
   *
   * Returns:
   *     True if a solution from x_init to x_goal was found.
   */
  bool solve();

  const Workspace2D& workspace;  // workspace grid (a Workspace2D object)
  double res;                    // resolution of the discretization of state space (cell/m)
  GridState x_offset;
  GridState x_init;  // initial state
  GridState x_goal;  // goal state

  std::set<GridState> closed_set;  // the set containing the states that have been visited
  std::set<GridState> open_set;    // the set containing the states that are condidate for future expension

  std::map<GridState, double> est_cost_through;  // estimated cost from start to goal passing through state (f score)
  std::map<GridState, double> cost_to_arrive;    // cost-to-arrive at state from start (g score)
  std::map<GridState, GridState> came_from;      // each state's parent to reconstruct the path

  std::vector<GridState> path;  // the final path as a list of states
};
