// =============================================================================
// Exercise 1: A* Motion Planning
// (C++ port of ch04/exercises/a_star.ipynb; the A* class lives in a_star.hpp / a_star.cpp)
// =============================================================================
#include "a_star.hpp"
#include "utils.hpp"

#include <pora/core.hpp>
#include <pora/plot.hpp>
#include <pora/random.hpp>
#include <pora/spline.hpp>

#include <Eigen/Dense>

#include <iostream>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// -----------------------------------------------------------------------------
// #### Exercise 1.2: Smooth A* Path
// The A* path is built on a discrete grid and is therefore not a smooth path. We can smooth
// the path using splines, which may be more practically useful for robot path planning.
//
// In the function `compute_smooth_plan`, implement the function to compute smoothing spline
// objects (`pora::SmoothingSpline`, the counterpart of `scipy.interpolate.BSpline`) from the
// A* path.
// -----------------------------------------------------------------------------

/**
 * Compute a smoothed path from the given A* path.
 *
 * Args:
 *     path: A* solution path
 *     spline_alpha: smoothing factor, larger means more smoothing
 *
 * Returns:
 *     Smoothed path array, shape (N, 2) where N is the number of path points.
 */
MatrixXd compute_smooth_plan(const std::vector<GridState>& path_, double spline_alpha = 0.05) {
  // Ensure path is an Eigen array, shape (N, 2)
  MatrixXd path(path_.size(), 2);
  for (size_t i = 0; i < path_.size(); ++i) path.row(i) << path_[i].first, path_[i].second;

  // Compute and set the following variables:
  //   1. s:
  //      Compute an array of arclengths, s, for each planned waypoint based on
  //      the Euclidean distance between points.
  //
  //   2. path_x_spline, path_y_spline (pora::SmoothingSpline):
  //      Fit cubic splines to the x and y coordinates of the path separately
  //      with respect to the computed time stamp array.
  VectorXd s;
  pora::SmoothingSpline path_x_spline, path_y_spline;

  // ##### YOUR CODE STARTS HERE #####
  // Hint: Use pora::make_splrep (the counterpart of scipy.interpolate.make_splrep) and use `spline_alpha`
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######

  // Compute smooth path from the interpolation
  VectorXd x_smooth = pora::splev(s, path_x_spline);
  VectorXd y_smooth = pora::splev(s, path_y_spline);
  MatrixXd out(s.size(), 2);
  out << x_smooth, y_smooth;
  return out;
}

int main() {
  return pora::run([] {
    // For the A* motion planning problem we will start with a simple 2D environment with a set
    // of obstacles defined below.
    Workspace2D workspace(10, 10);
    std::vector<Box> obstacles = {box(2, 2, 4, 3), box(6, 7, 8, 8), box(6, 3, 8, 5), box(2, 5, 4, 7)};
    workspace.set_obstacles(obstacles);
    {
      pora::plt::Figure fig(1, 1, 600, 600);
      workspace.plot(fig.ax());
      fig.save("workspace.png");
    }

    // -------------------------------------------------------------------------
    // ### Exercise 1.1: Implement A* Algorithm and Solve a Simple 2D Motion Planning Problem
    // First, in the file `a_star.cpp` implement the functions:
    // 1. `is_free`: a function to check if a given state is free (in the map and not in
    //    collision with an obstacle).
    // 2. `distance`: the Euclidean distance between two states
    // 3. `get_neighbors`: get the free neighbors from a given state
    // 4. `solve`: solve the motion planning problem using A*
    //
    // Run the code below to solve the simple 2D problem defined above.
    // -------------------------------------------------------------------------
    {
      GridState x_init = {1, 9};
      GridState x_goal = {9, 1};
      AStar astar(workspace, x_init, x_goal);
      if (!astar.solve()) {
        std::cout << "No path found" << std::endl;
      } else {
        pora::plt::Figure fig(1, 1, 600, 640);
        astar.plot_path(fig.ax());
        astar.plot_tree(fig.ax());
        fig.save("a_star_simple.png");
      }
    }

    // -------------------------------------------------------------------------
    // ### Exercise 1.1 (continued): Motion Planning in a Random Cluttered Environment
    // Now we will run the code below to see our A* implementation run in a randomly generated
    // cluttered environment.
    // -------------------------------------------------------------------------

    // Specify dimensions of the workspace, and some specifications for obstacle generation
    int num_obs = 25;
    double min_size = .5;
    double max_size = 3;
    workspace.generate_random_box_obstacles(num_obs, min_size, max_size);
    Eigen::Vector2d xi = workspace.free_random_sample(), xg = workspace.free_random_sample();
    GridState x_init = {xi.x(), xi.y()}, x_goal = {xg.x(), xg.y()};

    // Solve the problem using A*
    AStar astar(workspace, x_init, x_goal, 0.1);
    if (!astar.solve()) {
      std::cout << "No path found! (This is normal, try re-running with another PORA_SEED.)" << std::endl;
      return;
    }
    {
      pora::plt::Figure fig(1, 1, 800, 840);
      astar.plot_path(fig.ax());
      astar.plot_tree(fig.ax(), 2);
      fig.save("a_star_cluttered.png");
    }

    // Compute smooth plan from A* path
    MatrixXd smooth_plan = compute_smooth_plan(astar.path);

    // Plot A* path and the smooth path
    pora::plt::Figure fig(1, 1, 800, 840);
    astar.plot_path(fig.ax());
    fig.ax().plot(smooth_plan.col(0), smooth_plan.col(1), pora::plt::fmt("r-").label("Smoothed Path"));
    fig.ax().legend("lower right");
    fig.ax().set_title("Path Smoothing");
    fig.save("a_star_smoothed.png");
  });
}
