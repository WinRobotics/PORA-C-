// C++ port of ch04/exercises/rrt.py
#include "rrt.hpp"

#include <pora/core.hpp>
#include <pora/numerics.hpp>
#include <pora/random.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <utility>

using Eigen::Vector2d;
namespace plt = pora::plt;

RRT::RRT(const Workspace2D& workspace, const Vector2d& x_init, const Vector2d& x_goal)
    : workspace(workspace), x_init(x_init), x_goal(x_goal) {}

void RRT::plot_problem(plt::Axes& ax) const {
  workspace.plot(ax);
  ax.scatter(std::vector<double>{x_init[0], x_goal[0]}, std::vector<double>{x_init[1], x_goal[1]},
             plt::fmt("o").c(plt::colors::green).ms(6));
  ax.text(x_init[0] + .2, x_init[1], "x_init", plt::colors::black, 0.6);
  ax.text(x_goal[0] + .2, x_goal[1], "x_goal", plt::colors::black, 0.6);
}

bool RRT::solve(double eps, int max_iters, double goal_bias, bool shortcut, const std::string& figure_name) {
  V = {Node(x_init)};  // reset the list of nodes
  path.clear();

  bool success = false;

  // ##### YOUR CODE STARTS HERE #####
  // ## Intermediate Outputs
  // You must update and/or populate:
  //    - V: the represention of the planning tree
  //    - success: whether or not you've found a solution within max_iters RRT iterations
  //    - path: if success is true, then must contain list of Nodes
  //          [x_init, ..., x_goal] such that the global trajectory made by linking steering
  //          trajectories connecting the states in order is obstacle-free.
  //
  // ## Hints:
  //   - use the helper functions nearest_neighbor, steer_towards, and is_free_motion
  //   - the order in which you pass in arguments to steer_towards and is_free_motion is important
  //   - use pora::rng().rand() and workspace.random_sample() for sampling
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######

  // Plot the results
  plt::Figure fig(1, 1, 650, 700);
  plt::Axes& ax = fig.ax();
  plot_problem(ax);
  plot_tree(ax, plt::fmt("-").c(plt::colors::blue).lw(.5).alpha(0.5).label("RRT tree"));
  if (success) {
    if (shortcut) {
      plot_path(ax, plt::fmt("-").c(plt::colors::purple).lw(2).label("Original solution path"));
      shortcut_path();
      plot_path(ax, plt::fmt("-").c(plt::colors::green).lw(2).label("Shortcut solution path"));
    } else {
      plot_path(ax, plt::fmt("-").c(plt::colors::green).lw(2).label("Solution path"));
    }
    ax.legend("lower right");
    std::vector<double> xs, ys;
    for (const auto& node : V) {
      xs.push_back(node.point[0]);
      ys.push_back(node.point[1]);
    }
    ax.scatter(xs, ys, plt::fmt("o").ms(4));
  } else {
    std::cout << "Solution not found!" << std::endl;
  }
  fig.save(figure_name);

  return success;
}

void RRT::shortcut_path() {
  // ##### YOUR CODE STARTS HERE #####
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
}

double RRT::solve_optimal(double eps, int max_iters, double goal_bias, const std::string& figure_name) {
  path.clear();
  Node start_node(x_init, -1, 0, 0);

  V = {start_node};

  // Compute connection radius
  double mu = workspace.width() * workspace.height();
  for (const auto& obstacle : workspace.obstacles) mu -= obstacle.area();
  auto r = [&](int N) { return std::min(1.1 * std::sqrt(3 * mu / pora::kPi * std::log(N) / N), eps); };

  // Search
  int goal_idx = -1;
  for (int k = 0; k < max_iters; ++k) {
    // Sample a new point randomly, select the goal occasionally
    Vector2d x;
    if (pora::rng().rand() < goal_bias && goal_idx < 0) {
      x = x_goal;
    } else {
      x = workspace.random_sample();
    }

    // Get nearest neighbor in current tree to sampled point, try to steer
    // towards it to define new sample point
    int nn_idx = nearest_neighbor(x);
    Vector2d x_near = V[nn_idx].point;
    Vector2d x_new = steer_towards(x_near, x, eps);

    // Check if the new sample point is in the workspace and there is a free motion to
    // connect to the tree
    if (workspace.in_statespace(x_new) && is_free_motion(x_near, x_new)) {
      // Create new tree node, default assign parent to the nearest neighbor
      const int N = static_cast<int>(V.size());
      double cost_to_arrive = V[nn_idx].cost + (x_new - x_near).norm();
      Node new_node(x_new, nn_idx, cost_to_arrive, N);

      // Compute neighbors of the new node, within radius r
      std::vector<int> near_nodes = neighbors(x_new, r(N));

      // Compute cost to near nodes, which is defined as the distance for this example
      std::vector<double> cost_to_near_nodes;
      for (int i : near_nodes) cost_to_near_nodes.push_back((V[i].point - x_new).norm());

      // Sort the cost-to-arrive from lowest to highest for each nearby node and then
      // figure out the best parent node that is collision free
      std::vector<std::pair<double, int>> sorted_costs;
      for (size_t i = 0; i < near_nodes.size(); ++i)
        sorted_costs.emplace_back(cost_to_near_nodes[i] + V[near_nodes[i]].cost, static_cast<int>(i));
      std::stable_sort(sorted_costs.begin(), sorted_costs.end(),
                       [](const auto& a, const auto& b) { return a.first < b.first; });
      for (const auto& [cost, i] : sorted_costs) {
        if (cost < new_node.cost && is_free_motion(x_new, V[near_nodes[i]].point)) {
          new_node.parent = V[near_nodes[i]].idx;
          new_node.cost = cost;
          break;
        }
      }

      // Add new node to the tree
      V.push_back(new_node);
      if (new_node.point == x_goal) goal_idx = N;

      // For each nearby node to the new node, check if going through the new node to get to the
      // nearby node would have lower cost-to-arrive
      std::vector<int> nodes_to_rewire;
      for (size_t i = 0; i < near_nodes.size(); ++i)
        if (cost_to_near_nodes[i] + new_node.cost < V[near_nodes[i]].cost) nodes_to_rewire.push_back(near_nodes[i]);

      // Rewire nearby nodes to new node as parent
      for (int near_idx : nodes_to_rewire) {
        Node& near_node = V[near_idx];
        if (!is_free_motion(x_new, near_node.point)) continue;
        double cost = (x_new - near_node.point).norm();
        double new_cost_to_arrive = new_node.cost + cost;
        double cost_change = new_cost_to_arrive - near_node.cost;

        // Update the nearby node parent and cost
        near_node.parent = new_node.idx;
        near_node.cost = new_cost_to_arrive;

        std::vector<int> descendants_to_process = {near_node.idx};
        while (!descendants_to_process.empty()) {
          int current_node_idx = descendants_to_process.back();
          descendants_to_process.pop_back();
          for (auto& node : V) {
            if (node.parent == current_node_idx) {
              descendants_to_process.push_back(node.idx);
              node.cost += cost_change;
            }
          }
        }
      }
    }
  }

  plt::Figure fig(1, 1, 650, 700);
  plt::Axes& ax = fig.ax();
  plot_problem(ax);
  plot_tree(ax, plt::fmt("-").c(plt::colors::blue).lw(.5).alpha(0.5).label("RRT tree"));
  if (goal_idx < 0) {
    std::cout << "Solution not found!" << std::endl;
    fig.save(figure_name);
    return pora::kInf;
  }

  // Construct path
  path.clear();
  int current_idx = goal_idx;
  while (current_idx >= 0) {
    path.push_back(V[current_idx]);
    current_idx = V[current_idx].parent;
  }
  std::reverse(path.begin(), path.end());

  // Plot path
  plot_path(ax, plt::fmt("-").c(plt::colors::green).lw(2).label("Solution path"));
  ax.legend("lower right");
  std::vector<double> xs, ys;
  for (const auto& node : V) {
    xs.push_back(node.point[0]);
    ys.push_back(node.point[1]);
  }
  ax.scatter(xs, ys, plt::fmt("o").ms(4));
  fig.save(figure_name);
  return V[goal_idx].cost;
}

int GeometricRRT::nearest_neighbor(const Vector2d& x) const {
  // ##### YOUR CODE STARTS HERE #####
  // Consult function specification in parent (RRT) class.
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
}

std::vector<int> GeometricRRT::neighbors(const Vector2d& x, double r) const {
  // Computes neighbors with Euclidean distance to x that is < r.
  std::vector<int> out;
  for (size_t i = 0; i < V.size(); ++i)
    if ((V[i].point - x).norm() < r) out.push_back(static_cast<int>(i));
  return out;
}

Vector2d GeometricRRT::steer_towards(const Vector2d& x1, const Vector2d& x2, double eps) const {
  // Consult function specification in parent (RRT) class.
  // ##### YOUR CODE STARTS HERE #####
  // Hint: This should take 1-4 lines.
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
}

bool GeometricRRT::is_free_motion(const Vector2d& x1, const Vector2d& x2) const {
  // Consult function specification in parent (RRT) class.
  for (const auto& obstacle : workspace.obstacles)
    if (obstacle.intersects_segment(x1, x2)) return false;
  return true;
}

void GeometricRRT::plot_tree(plt::Axes& ax, const plt::Style& style) const {
  // Plot the searched tree (one line series, edges separated by NaNs).
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> xs, ys;
  for (const auto& node : V) {
    if (node.parent >= 0) {
      const Node& parent_node = V[node.parent];
      xs.insert(xs.end(), {node.point[0], parent_node.point[0], nan});
      ys.insert(ys.end(), {node.point[1], parent_node.point[1], nan});
    }
  }
  plt::Style s = style;
  s.lw(1).alpha(0.2).label("");  // like the Python version: style kwargs are ignored here
  ax.plot(xs, ys, s);
}

void GeometricRRT::plot_path(plt::Axes& ax, const plt::Style& style) const {
  // Plot the solution path.
  if (path.size() > 1) {
    std::vector<double> xs, ys;
    for (const auto& node : path) {
      xs.push_back(node.point[0]);
      ys.push_back(node.point[1]);
    }
    ax.plot(xs, ys, style);
  }
}
