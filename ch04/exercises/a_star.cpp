// C++ port of ch04/exercises/a_star.py
#include "a_star.hpp"

#include <pora/core.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

AStar::AStar(const Workspace2D& workspace, const GridState& x_init_, const GridState& x_goal_, double resolution)
    : workspace(workspace), res(resolution), x_offset(x_init_) {
  x_init = snap_to_grid(x_init_);  // initial state
  x_goal = snap_to_grid(x_goal_);  // goal state

  open_set.insert(x_init);
  cost_to_arrive[x_init] = 0;
  est_cost_through[x_init] = distance(x_init, x_goal);
}

GridState AStar::snap_to_grid(const GridState& x) const {
  double x_snap = res * std::round((x.first - x_offset.first) / res) + x_offset.first;
  double y_snap = res * std::round((x.second - x_offset.second) / res) + x_offset.second;
  return {x_snap, y_snap};
}

GridState AStar::find_best_est_cost_through() const {
  return *std::min_element(open_set.begin(), open_set.end(), [&](const GridState& a, const GridState& b) {
    return est_cost_through.at(a) < est_cost_through.at(b);
  });
}

std::vector<GridState> AStar::reconstruct_path() const {
  std::vector<GridState> p = {x_goal};
  GridState current = p.back();
  while (current != x_init) {
    p.push_back(came_from.at(current));
    current = p.back();
  }
  std::reverse(p.begin(), p.end());
  return p;
}

void AStar::plot_path(pora::plt::Axes& ax, bool show_init_label) const {
  if (path.empty()) return;

  workspace.plot(ax);

  std::vector<double> xs, ys;
  for (const auto& s : path) {
    xs.push_back(s.first);
    ys.push_back(s.second);
  }
  ax.plot(xs, ys, pora::plt::fmt("-").c(pora::plt::colors::green).lw(2).label("A* solution path"));
  ax.scatter(std::vector<double>{x_init.first, x_goal.first}, std::vector<double>{x_init.second, x_goal.second},
             pora::plt::fmt("o").c(pora::plt::colors::green).ms(6));
  if (show_init_label) ax.text(x_init.first + .2, x_init.second + .2, "x_init", pora::plt::colors::black, 0.6);
  ax.text(x_goal.first + .2, x_goal.second + .2, "x_goal", pora::plt::colors::black, 0.6);
  ax.legend("lower right");

  ax.axis(0, workspace.width(), 0, workspace.height());
}

void AStar::plot_tree(pora::plt::Axes& ax, double point_size) const {
  // All tree edges are drawn as one line series separated by NaNs.
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> ex, ey, px, py;
  auto add = [&](const std::set<GridState>& set) {
    for (const auto& x : set) {
      if (x == x_init) continue;
      const GridState& parent = came_from.at(x);
      ex.insert(ex.end(), {x.first, parent.first, nan});
      ey.insert(ey.end(), {x.second, parent.second, nan});
      if (x != x_goal) {
        px.push_back(x.first);
        py.push_back(x.second);
      }
    }
  };
  add(open_set);
  add(closed_set);
  ax.plot(ex, ey, pora::plt::fmt("-").c(pora::plt::colors::blue).lw(1).alpha(0.2));
  ax.scatter(px, py, pora::plt::fmt("o").c(pora::plt::colors::blue).ms(point_size).alpha(0.2));
}

bool AStar::is_free(const GridState& x) const {
  // ##### YOUR CODE STARTS HERE #####
  // Hint: workspace is a Workspace2D object, take a look at its methods
  // for what might be useful here (convert x to Eigen::Vector2d(x.first, x.second))
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
}

double AStar::distance(const GridState& x1, const GridState& x2) const {
  // ##### YOUR CODE STARTS HERE #####
  // HINT: This should take one line.
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
}

std::vector<GridState> AStar::get_neighbors(const GridState& x) const {
  std::vector<GridState> neighbors;

  // ##### YOUR CODE STARTS HERE #####
  // HINTS: Use is_free to check whether a given state is indeed free.
  //        Use snap_to_grid (see above) to ensure that the neighbors
  //        you compute are actually on the discrete grid, i.e., if you were
  //        to compute neighbors by adding/subtracting res from x,
  //        numerical errors could creep in over the course of many additions
  //        and cause grid point equality checks to fail. To remedy this, you
  //        should make sure that every neighbor is snapped to the grid as it
  //        is computed.
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
  return neighbors;
}

bool AStar::solve() {
  // ##### YOUR CODE STARTS HERE #####
  // HINTS:  We're representing the open and closed sets using std::set. This allows
  //         easily adding and removing items using .insert(item) and .erase(item)
  //         respectively, as well as checking for set membership efficiently using
  //         the syntax "set.count(item)". The maps (std::map) work the same way:
  //         cost_to_arrive.count(x) checks whether x has a value yet.
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
}
