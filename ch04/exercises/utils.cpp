// C++ port of ch04/exercises/utils.py
#include "utils.hpp"

#include <pora/random.hpp>

#include <algorithm>
#include <cmath>

double Box::distance(const Eigen::Vector2d& p) const {
  double dx = std::max({minx - p.x(), 0.0, p.x() - maxx});
  double dy = std::max({miny - p.y(), 0.0, p.y() - maxy});
  return std::hypot(dx, dy);
}

bool Box::intersects_segment(const Eigen::Vector2d& a, const Eigen::Vector2d& b) const {
  // Liang-Barsky clipping of the segment against the (closed) box
  double t0 = 0.0, t1 = 1.0;
  const Eigen::Vector2d d = b - a;
  const double p[4] = {-d.x(), d.x(), -d.y(), d.y()};
  const double q[4] = {a.x() - minx, maxx - a.x(), a.y() - miny, maxy - a.y()};
  for (int i = 0; i < 4; ++i) {
    if (p[i] == 0.0) {
      if (q[i] < 0.0) return false;
    } else {
      double r = q[i] / p[i];
      if (p[i] < 0.0) t0 = std::max(t0, r);
      else t1 = std::min(t1, r);
      if (t0 > t1) return false;
    }
  }
  return true;
}

Workspace2D::Workspace2D(double width, double height) : lower(0, 0), upper(width, height) {}

void Workspace2D::set_obstacles(const std::vector<Box>& obs) { obstacles = obs; }

void Workspace2D::generate_random_box_obstacles(int num_obs, double min_obs_size, double max_obs_size) {
  const double x_margin = width() * 0.1;
  const double y_margin = height() * 0.1;
  obstacles.clear();
  std::vector<double> cx(num_obs), cy(num_obs);
  for (auto& c : cx) c = pora::rng().uniform(-x_margin, width() + x_margin);
  for (auto& c : cy) c = pora::rng().uniform(-y_margin, height() + y_margin);
  for (int i = 0; i < num_obs; ++i) {
    double sx = pora::rng().uniform(min_obs_size, max_obs_size);
    double sy = pora::rng().uniform(min_obs_size, max_obs_size);
    obstacles.push_back(box(cx[i], cy[i], cx[i] + sx, cy[i] + sy));
  }
}

bool Workspace2D::in_statespace(const Eigen::Vector2d& x) const {
  return (x.array() <= upper.array()).all() && (x.array() >= lower.array()).all();
}

Eigen::Vector2d Workspace2D::random_sample() const {
  Eigen::Vector2d r(pora::rng().rand(), pora::rng().rand());
  return lower + r.cwiseProduct(upper - lower);
}

bool Workspace2D::is_free(const Eigen::Vector2d& x, double margin) const {
  const double scale = std::max(width(), height());
  for (const auto& obs : obstacles)
    if (obs.distance(x) < scale * margin) return false;
  return true;
}

bool Workspace2D::is_free_line(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2) const {
  for (const auto& obstacle : obstacles)
    if (obstacle.intersects_segment(x1, x2)) return false;
  return true;
}

Eigen::Vector2d Workspace2D::free_random_sample() const {
  while (true) {
    Eigen::Vector2d x = random_sample();
    if (is_free(x)) return x;
  }
}

void Workspace2D::plot(pora::plt::Axes& ax) const {
  for (const auto& obs : obstacles)
    ax.rectangle(obs.minx, obs.miny, obs.maxx, obs.maxy, pora::plt::fmt("-").c(pora::plt::colors::tab_blue).fill());
  ax.set_xlim(0, width());
  ax.set_ylim(0, height());
  ax.axis_equal();
}
