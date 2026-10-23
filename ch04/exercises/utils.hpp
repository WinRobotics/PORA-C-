// C++ port of ch04/exercises/utils.py
// (shapely boxes are replaced by the small axis-aligned `Box` type below)
#pragma once

#include <pora/plot.hpp>

#include <Eigen/Dense>

#include <limits>
#include <vector>

// Axis-aligned rectangular obstacle, the counterpart of `shapely.box(minx, miny, maxx, maxy)`.
struct Box {
  double minx, miny, maxx, maxy;

  double area() const { return (maxx - minx) * (maxy - miny); }
  // Euclidean distance from a point to the box (0 if the point is inside), like Point(x).distance(box).
  double distance(const Eigen::Vector2d& p) const;
  // Whether the segment from a to b touches the box, like LineString([a, b]).intersects(box).
  bool intersects_segment(const Eigen::Vector2d& a, const Eigen::Vector2d& b) const;
};

inline Box box(double minx, double miny, double maxx, double maxy) { return Box{minx, miny, maxx, maxy}; }

/**
 * A 2D state space grid with a set of rectangular obstacles. The grid is
 * fully deterministic.
 */
class Workspace2D {
 public:
  Workspace2D(double width, double height);

  void set_obstacles(const std::vector<Box>& obstacles);

  /**
   * Generate planning problem within the 2D workspace by defining
   * a set of obstacles at random.
   */
  void generate_random_box_obstacles(int num_obs, double min_obs_size, double max_obs_size);

  double width() const { return upper[0] - lower[0]; }
  double height() const { return upper[1] - lower[1]; }

  /** Checks if the point is inside the valid statespace */
  bool in_statespace(const Eigen::Vector2d& x) const;

  /** Compute a random sample point within the statespace */
  Eigen::Vector2d random_sample() const;

  /**
   * Verifies that point is not inside any obstacles by some margin
   * that is scaled by the max workspace dimension.
   */
  bool is_free(const Eigen::Vector2d& x, double margin = 0.01) const;

  /** Checks if line from `x1` to `x2` is collision free. */
  bool is_free_line(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2) const;

  /** Compute a random sample point within the statespace that is free */
  Eigen::Vector2d free_random_sample() const;

  /** Plots the space and its obstacles into the given axes. */
  void plot(pora::plt::Axes& ax) const;

  Eigen::Vector2d lower;      // state space lower bound
  Eigen::Vector2d upper;      // state space upper bound
  std::vector<Box> obstacles; // list of boxes
};

struct Node {
  Node(const Eigen::Vector2d& point, int parent = -1, double cost = std::numeric_limits<double>::infinity(),
       int idx = -1)
      : point(point), parent(parent), cost(cost), idx(idx) {}
  Eigen::Vector2d point;
  int parent;  // index of the parent node in the tree, -1 for none (Python: None)
  double cost;
  int idx;
};
