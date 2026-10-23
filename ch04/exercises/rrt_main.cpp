// =============================================================================
// Exercise 2: RRT Sampling-Based Motion Planning
// (C++ port of ch04/exercises/rrt.ipynb; the RRT classes live in rrt.hpp / rrt.cpp)
// =============================================================================
#include "rrt.hpp"
#include "utils.hpp"

#include <pora/core.hpp>
#include <pora/plot.hpp>

#include <cstdio>
#include <vector>

int main() {
  return pora::run([] {
    // First, we will define a simple 2D workspace with some obstacles.
    Workspace2D workspace(10, 10);
    std::vector<Box> obstacles = {box(2, 2, 4, 3), box(6, 7, 8, 8), box(6, 3, 8, 5), box(2, 5, 4, 7)};
    workspace.set_obstacles(obstacles);
    {
      pora::plt::Figure fig(1, 1, 600, 600);
      workspace.plot(fig.ax());
      fig.save("workspace.png");
    }

    // -------------------------------------------------------------------------
    // ### Exercise 2.1: Geometric RRT Planning
    // We are going to implement a geometric version of the RRT algorithm where we connect nodes
    // with simple straight line paths.
    //
    // In the file `rrt.cpp`, implement the following methods:
    // 1. `RRT::solve` to run the RRT algorithm
    // 2. `GeometricRRT::nearest_neighbor` to compute the nearest neighbor in the current tree
    //    from a given point
    // 3. `GeometricRRT::steer_towards` to compute a new state from a target state following a
    //    straight line path
    //
    // Then, run the code below to test your implementation on our workspace.
    // -------------------------------------------------------------------------
    Eigen::Vector2d x_init(1, 9);
    Eigen::Vector2d x_goal(9, 1);
    GeometricRRT grrt(workspace, x_init, x_goal);
    int max_iters = 500;
    bool success = grrt.solve(1.0, max_iters, 0.05, false, "rrt.png");

    // -------------------------------------------------------------------------
    // ### Exercise 2.2: Adding Shortcutting
    // Now we will add a simple shortcutting technique to bypass nodes of the RRT path that
    // aren't strictly needed (i.e. if a node can be dropped and the path between the parent and
    // child nodes is collision free). Implement the function `RRT::shortcut_path` and then run
    // the code below.
    // -------------------------------------------------------------------------
    success = grrt.solve(1.0, max_iters, 0.05, true, "rrt_shortcut.png");
    (void)success;

    // -------------------------------------------------------------------------
    // ### Exercise 2.3: Compare to RRT*
    // Run the code below to run the geometric problem with the RRT* optimal path planning
    // algorithm. Explore the code in `RRT::solve_optimal` to see how the algorithm differs from
    // the regular RRT.
    // -------------------------------------------------------------------------
    double cost = grrt.solve_optimal(1.0, max_iters, 0.05, "rrt_star.png");
    std::printf("Path cost: %.3f\n", cost);
  });
}
