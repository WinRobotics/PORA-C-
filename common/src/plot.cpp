#include "pora/plot.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>

namespace pora::plt {

namespace {

constexpr int kShift = 4;  // sub-pixel precision for OpenCV drawing
constexpr double kScale = 1 << kShift;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInfinity = std::numeric_limits<double>::infinity();
const int kFont = cv::FONT_HERSHEY_SIMPLEX;

cv::Point sp(double x, double y) {  // sub-pixel point
  return {static_cast<int>(std::lround(x * kScale)), static_cast<int>(std::lround(y * kScale))};
}

std::vector<double> nice_ticks(double lo, double hi, int target = 6) {
  if (lo > hi) std::swap(lo, hi);
  double range = hi - lo;
  if (!(range > 0) || !std::isfinite(range)) return {lo};
  double raw = range / target;
  double mag = std::pow(10.0, std::floor(std::log10(raw)));
  double norm = raw / mag;
  double step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  std::vector<double> ticks;
  for (double t = std::ceil(lo / step - 1e-9) * step; t <= hi + 1e-9 * step; t += step)
    ticks.push_back(std::abs(t) < 1e-12 * step ? 0.0 : t);
  return ticks;
}

std::string tick_label(double v, double step) {
  int decimals = 0;
  if (step > 0 && step < 1) decimals = static_cast<int>(std::ceil(-std::log10(step) - 1e-9));
  decimals = std::min(decimals, 6);
  char buf[64];
  if (std::abs(v) >= 1e5 || (std::abs(v) > 0 && std::abs(v) < 1e-4 && decimals > 5))
    std::snprintf(buf, sizeof(buf), "%.2g", v);
  else
    std::snprintf(buf, sizeof(buf), "%.*f", decimals, v);
  return buf;
}

void put_text(cv::Mat& img, const std::string& s, cv::Point org, double scale, const Color& color,
              int thickness = 1) {
  cv::putText(img, s, org, kFont, scale, color, thickness, cv::LINE_AA);
}

cv::Size text_size(const std::string& s, double scale, int thickness = 1) {
  int base = 0;
  return cv::getTextSize(s, kFont, scale, thickness, &base);
}

void put_text_centered(cv::Mat& img, const std::string& s, cv::Point center, double scale, const Color& color,
                       int thickness = 1) {
  cv::Size ts = text_size(s, scale, thickness);
  put_text(img, s, {center.x - ts.width / 2, center.y + ts.height / 2}, scale, color, thickness);
}

void put_text_vertical(cv::Mat& img, const std::string& s, cv::Point center, double scale, const Color& color) {
  cv::Size ts = text_size(s, scale);
  cv::Mat tmp(ts.height + 8, ts.width + 4, CV_8UC3, cv::Scalar(255, 255, 255));
  put_text(tmp, s, {2, ts.height + 2}, scale, color);
  cv::Mat rot;
  cv::rotate(tmp, rot, cv::ROTATE_90_COUNTERCLOCKWISE);
  int x0 = center.x - rot.cols / 2, y0 = center.y - rot.rows / 2;
  cv::Rect dst(x0, y0, rot.cols, rot.rows);
  cv::Rect clipped = dst & cv::Rect(0, 0, img.cols, img.rows);
  if (clipped.area() <= 0) return;
  cv::Mat src = rot(cv::Rect(clipped.x - x0, clipped.y - y0, clipped.width, clipped.height));
  cv::Mat roi = img(clipped);
  cv::min(roi, src, roi);  // dark text over the background
}

// Dash pattern (on/off lengths in pixels) for a matplotlib line style.
std::vector<double> dash_pattern(const std::string& line, double lw) {
  double w = std::max(lw, 1.0);
  if (line == "--") return {3.7 * w, 1.6 * w};
  if (line == ":") return {1.0 * w, 1.65 * w};
  if (line == "-.") return {6.4 * w, 1.6 * w, 1.0 * w, 1.6 * w};
  return {};
}

void draw_polyline(cv::Mat& img, const std::vector<cv::Point2d>& pts, const Color& color, double lw,
                   const std::string& line) {
  if (pts.size() < 2 || line.empty()) return;
  int thick = std::max(1, static_cast<int>(std::lround(0.75 * lw)));
  std::vector<double> pattern = dash_pattern(line, lw);
  if (pattern.empty()) {
    std::vector<cv::Point> p;
    p.reserve(pts.size());
    for (const auto& q : pts) p.push_back(sp(q.x, q.y));
    cv::polylines(img, p, false, color, thick, cv::LINE_AA, kShift);
    return;
  }
  size_t pi = 0;
  double remaining = pattern[0];
  bool on = true;
  for (size_t i = 0; i + 1 < pts.size(); ++i) {
    cv::Point2d a = pts[i], b = pts[i + 1];
    double len = cv::norm(b - a);
    double pos = 0;
    while (pos < len) {
      double step = std::min(remaining, len - pos);
      cv::Point2d s = a + (b - a) * (pos / len), e = a + (b - a) * ((pos + step) / len);
      if (on) cv::line(img, sp(s.x, s.y), sp(e.x, e.y), color, thick, cv::LINE_AA, kShift);
      pos += step;
      remaining -= step;
      if (remaining <= 1e-9) {
        pi = (pi + 1) % pattern.size();
        remaining = pattern[pi];
        on = !on;
      }
    }
  }
}

void draw_marker(cv::Mat& img, cv::Point2d p, char marker, double size, const Color& color, bool filled) {
  double r = size / 2.0;
  int thick = 1;
  switch (marker) {
    case '.':
      cv::circle(img, sp(p.x, p.y), static_cast<int>(std::max(1.0, r * 0.6) * kScale), color, cv::FILLED,
                 cv::LINE_AA, kShift);
      break;
    case 'o':
      cv::circle(img, sp(p.x, p.y), static_cast<int>(r * kScale), color, filled ? cv::FILLED : thick, cv::LINE_AA,
                 kShift);
      break;
    case '+':
      cv::line(img, sp(p.x - r, p.y), sp(p.x + r, p.y), color, thick, cv::LINE_AA, kShift);
      cv::line(img, sp(p.x, p.y - r), sp(p.x, p.y + r), color, thick, cv::LINE_AA, kShift);
      break;
    case 'x':
      cv::line(img, sp(p.x - r, p.y - r), sp(p.x + r, p.y + r), color, thick, cv::LINE_AA, kShift);
      cv::line(img, sp(p.x - r, p.y + r), sp(p.x + r, p.y - r), color, thick, cv::LINE_AA, kShift);
      break;
    case '*': {
      std::vector<cv::Point> star;
      for (int k = 0; k < 10; ++k) {
        double ang = -1.5707963267948966 + k * 3.14159265358979 / 5;
        double rr = (k % 2 == 0) ? r * 1.2 : r * 0.5;
        star.push_back(sp(p.x + rr * std::cos(ang), p.y + rr * std::sin(ang)));
      }
      cv::fillPoly(img, std::vector<std::vector<cv::Point>>{star}, color, cv::LINE_AA, kShift);
      break;
    }
    case '^': {
      std::vector<cv::Point> tri = {sp(p.x, p.y - r), sp(p.x - r, p.y + r), sp(p.x + r, p.y + r)};
      cv::fillPoly(img, std::vector<std::vector<cv::Point>>{tri}, color, cv::LINE_AA, kShift);
      break;
    }
    case 's':
      cv::rectangle(img, sp(p.x - r, p.y - r), sp(p.x + r, p.y + r), color, filled ? cv::FILLED : thick,
                    cv::LINE_AA, kShift);
      break;
    default:
      break;
  }
}

}  // namespace


Color cycle_color(int i) {
  static const Color cycle[] = {colors::tab_blue, colors::tab_orange, colors::tab_green, colors::tab_red,
                                colors::tab_purple, colors::tab_brown, colors::tab_pink, colors::tab_gray,
                                colors::tab_olive, colors::tab_cyan};
  return cycle[((i % 10) + 10) % 10];
}

Style fmt(const std::string& format) {
  Style s;
  bool has_line = false, has_marker = false;
  for (size_t i = 0; i < format.size(); ++i) {
    char ch = format[i];
    if (ch == '-' && i + 1 < format.size() && (format[i + 1] == '-' || format[i + 1] == '.')) {
      s.line = format.substr(i, 2);
      has_line = true;
      ++i;
      continue;
    }
    switch (ch) {
      case '-': s.line = "-"; has_line = true; break;
      case ':': s.line = ":"; has_line = true; break;
      case 'b': s.c(colors::tab_blue); break;
      case 'g': s.c(colors::green); break;
      case 'r': s.c(colors::red); break;
      case 'c': s.c(colors::cyan); break;
      case 'm': s.c(colors::magenta); break;
      case 'y': s.c(colors::yellow); break;
      case 'k': s.c(colors::black); break;
      case 'w': s.c(colors::white); break;
      case '.': case 'o': case '+': case 'x': case '*': case '^': case 's':
        s.marker = ch;
        has_marker = true;
        break;
      default: break;
    }
  }
  if (has_marker && !has_line) s.line = "";
  if (s.marker == 'o') s.filled = true;
  return s;
}

Color colormap(ColorMap map, double v) {
  if (!std::isfinite(v)) v = 0;
  v = std::min(1.0, std::max(0.0, v));
  if (map == ColorMap::Gray) return Color(255 * v, 255 * v, 255 * v);
  if (map == ColorMap::Blues) {
    // white (247,251,255) -> dark blue (8,48,107) in RGB
    return Color(255 + (107 - 255) * v, 251 + (48 - 251) * v, 247 + (8 - 247) * v);
  }
  // OpenCV colour maps, evaluated once into 256-entry lookup tables
  auto make_lut = [](int cm) {
    cv::Mat in(1, 256, CV_8UC1), out;
    for (int i = 0; i < 256; ++i) in.at<uchar>(0, i) = static_cast<uchar>(i);
    cv::applyColorMap(in, out, cm);
    return out;
  };
  static const cv::Mat viridis = make_lut(cv::COLORMAP_VIRIDIS), inferno = make_lut(cv::COLORMAP_INFERNO),
                       jet = make_lut(cv::COLORMAP_JET);
  const cv::Mat& lut = map == ColorMap::Viridis ? viridis : map == ColorMap::Inferno ? inferno : jet;
  cv::Vec3b c = lut.at<cv::Vec3b>(0, static_cast<int>(std::lround(v * 255)));
  return Color(c[0], c[1], c[2]);
}

// ----------------------------------------------------------------------------
// Axes
// ----------------------------------------------------------------------------

Style Axes::next_style(Style s) {
  if (!s.has_color) {
    s.color = cycle_color(color_index_++);
    s.has_color = true;
  }
  return s;
}

void Axes::plot_impl(const std::vector<double>& x, const std::vector<double>& y, Style style,
                     const std::vector<Color>& point_colors) {
  Item it;
  it.kind = Kind::Line;
  it.x = x;
  it.y = y;
  if (it.x.size() != it.y.size()) throw std::invalid_argument("plot: x and y must have the same length");
  it.point_colors = point_colors;
  it.style = next_style(style);
  items_.push_back(std::move(it));
}

void Axes::axhline(double y, const Style& style) {
  Item it;
  it.kind = Kind::HLine;
  it.a = y;
  it.style = next_style(style);
  items_.push_back(it);
}

void Axes::axvline(double x, const Style& style) {
  Item it;
  it.kind = Kind::VLine;
  it.a = x;
  it.style = next_style(style);
  items_.push_back(it);
}

void Axes::ellipse(double cx, double cy, double width, double height, double angle_deg, const Style& style) {
  Item it;
  it.kind = Kind::Ellipse;
  it.a = cx; it.b = cy; it.c = width; it.d = height; it.e = angle_deg;
  it.style = next_style(style);
  items_.push_back(it);
}

void Axes::circle(double cx, double cy, double radius, const Style& style) {
  Item it;
  it.kind = Kind::Circle;
  it.a = cx; it.b = cy; it.c = radius;
  it.style = next_style(style);
  items_.push_back(it);
}

void Axes::rectangle(double x0, double y0, double x1, double y1, const Style& style) {
  polygon({x0, x1, x1, x0}, {y0, y0, y1, y1}, style);
}

void Axes::polygon(const std::vector<double>& xs, const std::vector<double>& ys, const Style& style) {
  Item it;
  it.kind = Kind::Polygon;
  it.x = xs;
  it.y = ys;
  it.style = next_style(style);
  items_.push_back(it);
}

void Axes::quiver_impl(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& u,
                       const std::vector<double>& v, double scale, const Style& style) {
  Item it;
  it.kind = Kind::Quiver;
  it.x = x; it.y = y; it.u = u; it.v = v; it.a = scale;
  it.style = next_style(style);
  items_.push_back(it);
}

void Axes::text(double x, double y, const std::string& s, const Color& color, double font_scale) {
  Item it;
  it.kind = Kind::Text;
  it.a = x; it.b = y; it.c = font_scale;
  it.text = s;
  it.style = fmt("-").c(color);
  items_.push_back(it);
}

void Axes::imshow(const Eigen::MatrixXd& data, bool origin_lower, ColorMap map, const std::string& colorbar_label,
                  bool annotate) {
  Item it;
  it.kind = Kind::Heatmap;
  it.data = data;
  it.flag = annotate;
  it.map = map;
  items_.push_back(it);
  has_colorbar_ = !colorbar_label.empty();
  colorbar_label_ = colorbar_label;
  cbar_min_ = data.minCoeff();
  cbar_max_ = data.maxCoeff();
  cbar_map_ = map;
  if (!has_xlim_) set_xlim(-0.5, data.cols() - 0.5);
  if (!has_ylim_) {
    if (origin_lower) set_ylim(-0.5, data.rows() - 0.5);
    else set_ylim(data.rows() - 0.5, -0.5);
  }
  equal_ = true;
}

void Axes::image(const cv::Mat& img, double alpha) {
  Item it;
  it.kind = Kind::Image;
  cv::Mat img8;
  if (img.depth() != CV_8U) {
    cv::Mat tmp;
    cv::normalize(img, tmp, 0, 255, cv::NORM_MINMAX, CV_64F);
    tmp.convertTo(img8, CV_8U);
  } else {
    img8 = img;
  }
  if (img8.channels() == 1) cv::cvtColor(img8, it.img, cv::COLOR_GRAY2BGR);
  else it.img = img8.clone();
  it.a = alpha;
  items_.push_back(it);
  if (!has_xlim_) set_xlim(-0.5, img.cols - 0.5);
  if (!has_ylim_) set_ylim(img.rows - 0.5, -0.5);
  equal_ = true;
  image_axes_ = true;
}

void Axes::scatter3d(const Eigen::MatrixXd& points, double azim, double elev, ColorMap map, double point_size) {
  const double az = azim * 3.14159265358979 / 180.0, el = elev * 3.14159265358979 / 180.0;
  Eigen::Vector3d ex(-std::sin(az), std::cos(az), 0.0);
  Eigen::Vector3d ey(-std::sin(el) * std::cos(az), -std::sin(el) * std::sin(az), std::cos(el));
  Eigen::Vector3d eye(std::cos(el) * std::cos(az), std::cos(el) * std::sin(az), std::sin(el));
  const int n = static_cast<int>(points.rows());
  double zmax = points.col(2).maxCoeff();
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  Eigen::VectorXd depth = points * eye;
  std::sort(order.begin(), order.end(), [&](int a, int b) { return depth[a] < depth[b]; });
  Item it;
  it.kind = Kind::Points3D;
  it.style = fmt("o").ms(point_size);
  for (int i : order) {
    Eigen::Vector3d p = points.row(i).transpose();
    it.x.push_back(p.dot(ex));
    it.y.push_back(p.dot(ey));
    it.point_colors.push_back(colormap(map, zmax != 0 ? p.z() / zmax : 0.0));
  }
  it.u = {ex.x(), ex.y(), ex.z(), ey.x(), ey.y(), ey.z()};  // view axes for the triad
  items_.push_back(it);
  equal_ = true;
  axis_off_ = true;
}

void Axes::render(cv::Mat& canvas, const cv::Rect& area) const {
  // ---- data limits -------------------------------------------------------
  double xmin = kInfinity, xmax = -kInfinity, ymin = kInfinity, ymax = -kInfinity;
  auto acc = [&](double x, double y) {
    if (std::isfinite(x)) { xmin = std::min(xmin, x); xmax = std::max(xmax, x); }
    if (std::isfinite(y)) { ymin = std::min(ymin, y); ymax = std::max(ymax, y); }
  };
  for (const auto& it : items_) {
    switch (it.kind) {
      case Kind::Line: case Kind::Polygon: case Kind::Points3D:
        for (size_t i = 0; i < it.x.size(); ++i) acc(it.x[i], it.y[i]);
        break;
      case Kind::Quiver:
        for (size_t i = 0; i < it.x.size(); ++i) {
          acc(it.x[i], it.y[i]);
          acc(it.x[i] + it.a * it.u[i], it.y[i] + it.a * it.v[i]);
        }
        break;
      case Kind::HLine: acc(kNaN, it.a); break;
      case Kind::VLine: acc(it.a, kNaN); break;
      case Kind::Ellipse: {
        double r = 0.5 * std::max(it.c, it.d);
        acc(it.a - r, it.b - r);
        acc(it.a + r, it.b + r);
        break;
      }
      case Kind::Circle: acc(it.a - it.c, it.b - it.c); acc(it.a + it.c, it.b + it.c); break;
      case Kind::Text: acc(it.a, it.b); break;
      default: break;
    }
  }
  if (!(xmin <= xmax)) { xmin = 0; xmax = 1; }
  if (!(ymin <= ymax)) { ymin = 0; ymax = 1; }
  auto pad = [](double& lo, double& hi) {
    double r = hi - lo;
    if (r <= 0) {
      double d = std::abs(lo) > 0 ? 0.05 * std::abs(lo) : 0.5;
      lo -= d; hi += d;
    } else {
      lo -= 0.05 * r; hi += 0.05 * r;
    }
  };
  pad(xmin, xmax);
  pad(ymin, ymax);
  double x0 = has_xlim_ ? xlim_.first : xmin, x1 = has_xlim_ ? xlim_.second : xmax;
  double y0 = has_ylim_ ? ylim_.first : ymin, y1 = has_ylim_ ? ylim_.second : ymax;

  // ---- layout ------------------------------------------------------------
  const double fs = 0.42;  // tick font scale
  int left = axis_off_ ? 8 : 64, right = has_colorbar_ ? 90 : 16;
  int top = title_.empty() ? 10 : 30, bottom = axis_off_ ? 8 : (xlabel_.empty() ? 28 : 46);
  if (hide_yticks_) left = 20;
  if (!ylabel_.empty() && !axis_off_) left += 6;
  cv::Rect pr(area.x + left, area.y + top, std::max(10, area.width - left - right),
              std::max(10, area.height - top - bottom));

  if (equal_) {
    double sx = pr.width / std::abs(x1 - x0), sy = pr.height / std::abs(y1 - y0);
    if (image_axes_ || (has_xlim_ && has_ylim_) || (!items_.empty() && items_[0].kind == Kind::Heatmap)) {
      // shrink the box to keep the aspect ratio
      double s = std::min(sx, sy);
      int w = static_cast<int>(s * std::abs(x1 - x0)), h = static_cast<int>(s * std::abs(y1 - y0));
      pr = cv::Rect(pr.x + (pr.width - w) / 2, pr.y + (pr.height - h) / 2, std::max(w, 2), std::max(h, 2));
    } else if (sx > sy) {
      double half = 0.5 * pr.width / sy, c = 0.5 * (x0 + x1), sgn = x1 >= x0 ? 1 : -1;
      x0 = c - sgn * half; x1 = c + sgn * half;
    } else {
      double half = 0.5 * pr.height / sx, c = 0.5 * (y0 + y1), sgn = y1 >= y0 ? 1 : -1;
      y0 = c - sgn * half; y1 = c + sgn * half;
    }
  }

  auto px = [&](double x) { return (x - x0) / (x1 - x0) * pr.width; };   // relative to pr
  auto py = [&](double y) { return (y1 - y) / (y1 - y0) * pr.height; };  // relative to pr

  // ---- background, grid, ticks --------------------------------------------
  cv::Mat plot = canvas(pr);
  plot.setTo(colors::white);
  std::vector<double> xt = nice_ticks(x0, x1), yt = nice_ticks(y0, y1);
  double xstep = xt.size() > 1 ? xt[1] - xt[0] : 1, ystep = yt.size() > 1 ? yt[1] - yt[0] : 1;
  if (grid_ && !axis_off_) {
    for (double t : xt) cv::line(plot, sp(px(t), 0), sp(px(t), pr.height), Color(220, 220, 220), 1, cv::LINE_AA, kShift);
    for (double t : yt) cv::line(plot, sp(0, py(t)), sp(pr.width, py(t)), Color(220, 220, 220), 1, cv::LINE_AA, kShift);
  }

  // ---- items --------------------------------------------------------------
  for (const auto& it : items_) {
    const double alpha = it.kind == Kind::Image ? it.a : it.style.alpha_;
    cv::Mat layer = alpha < 1.0 ? plot.clone() : plot;
    const Color& col = it.style.color;
    switch (it.kind) {
      case Kind::Line: case Kind::Points3D: {
        std::vector<cv::Point2d> seg;
        auto flush = [&]() { draw_polyline(layer, seg, col, it.style.linewidth, it.style.line); seg.clear(); };
        for (size_t i = 0; i < it.x.size(); ++i) {
          if (!std::isfinite(it.x[i]) || !std::isfinite(it.y[i])) { flush(); continue; }
          seg.emplace_back(px(it.x[i]), py(it.y[i]));
        }
        flush();
        if (it.style.marker) {
          for (size_t i = 0; i < it.x.size(); ++i) {
            if (!std::isfinite(it.x[i]) || !std::isfinite(it.y[i])) continue;
            const Color& mc = it.point_colors.empty() ? col : it.point_colors[i];
            draw_marker(layer, {px(it.x[i]), py(it.y[i])}, it.style.marker, it.style.markersize, mc, it.style.filled);
          }
        }
        break;
      }
      case Kind::HLine:
        draw_polyline(layer, {{0, py(it.a)}, {double(pr.width), py(it.a)}}, col, it.style.linewidth, it.style.line);
        break;
      case Kind::VLine:
        draw_polyline(layer, {{px(it.a), 0}, {px(it.a), double(pr.height)}}, col, it.style.linewidth, it.style.line);
        break;
      case Kind::Ellipse: {
        std::vector<cv::Point2d> pts;
        double ang = it.e * 3.14159265358979 / 180.0;
        for (int k = 0; k <= 72; ++k) {
          double t = 2 * 3.14159265358979 * k / 72;
          double ex = 0.5 * it.c * std::cos(t), ey = 0.5 * it.d * std::sin(t);
          pts.emplace_back(px(it.a + ex * std::cos(ang) - ey * std::sin(ang)),
                           py(it.b + ex * std::sin(ang) + ey * std::cos(ang)));
        }
        draw_polyline(layer, pts, col, it.style.linewidth, it.style.line.empty() ? "-" : it.style.line);
        break;
      }
      case Kind::Circle: {
        double rpx = std::abs(px(it.a + it.c) - px(it.a));
        cv::circle(layer, sp(px(it.a), py(it.b)), static_cast<int>(rpx * kScale), col,
                   it.style.filled ? cv::FILLED : std::max(1, (int)std::lround(0.75 * it.style.linewidth)), cv::LINE_AA, kShift);
        break;
      }
      case Kind::Polygon: {
        std::vector<cv::Point> pts;
        for (size_t i = 0; i < it.x.size(); ++i) pts.push_back(sp(px(it.x[i]), py(it.y[i])));
        if (it.style.filled) cv::fillPoly(layer, std::vector<std::vector<cv::Point>>{pts}, col, cv::LINE_AA, kShift);
        else cv::polylines(layer, pts, true, col, std::max(1, (int)std::lround(0.75 * it.style.linewidth)), cv::LINE_AA, kShift);
        break;
      }
      case Kind::Quiver:
        for (size_t i = 0; i < it.x.size(); ++i) {
          cv::Point2d a(px(it.x[i]), py(it.y[i])), b(px(it.x[i] + it.a * it.u[i]), py(it.y[i] + it.a * it.v[i]));
          cv::arrowedLine(layer, sp(a.x, a.y), sp(b.x, b.y), col, 1, cv::LINE_AA, kShift, 0.3);
        }
        break;
      case Kind::Text:
        put_text(layer, it.text, {static_cast<int>(px(it.a)), static_cast<int>(py(it.b))}, it.c, col);
        break;
      case Kind::Heatmap: {
        double lo = it.data.minCoeff(), hi = it.data.maxCoeff();
        for (int r = 0; r < it.data.rows(); ++r)
          for (int c = 0; c < it.data.cols(); ++c) {
            double v = hi > lo ? (it.data(r, c) - lo) / (hi - lo) : 0.5;
            Color cc = colormap(it.map, v);
            cv::Point2d a(px(c - 0.5), py(r - 0.5)), b(px(c + 0.5), py(r + 0.5));
            cv::rectangle(layer, sp(std::min(a.x, b.x), std::min(a.y, b.y)), sp(std::max(a.x, b.x), std::max(a.y, b.y)),
                          cc, cv::FILLED, cv::LINE_8, kShift);
            if (it.flag) {
              char buf[32];
              std::snprintf(buf, sizeof(buf), "%g", it.data(r, c));
              Color tc = v > 0.5 ? colors::white : colors::black;
              put_text_centered(layer, buf, {(int)px(c), (int)py(r)}, 0.38, tc);
            }
          }
        break;
      }
      case Kind::Image: {
        double ax0 = px(-0.5), ax1 = px(it.img.cols - 0.5), ay0 = py(-0.5), ay1 = py(it.img.rows - 0.5);
        int w = std::max(1, (int)std::lround(std::abs(ax1 - ax0))), h = std::max(1, (int)std::lround(std::abs(ay1 - ay0)));
        cv::Mat scaled;
        cv::resize(it.img, scaled, {w, h}, 0, 0, cv::INTER_AREA);
        int ox = (int)std::lround(std::min(ax0, ax1)), oy = (int)std::lround(std::min(ay0, ay1));
        cv::Rect dst(ox, oy, w, h), clip = dst & cv::Rect(0, 0, layer.cols, layer.rows);
        if (clip.area() > 0) scaled(cv::Rect(clip.x - ox, clip.y - oy, clip.width, clip.height)).copyTo(layer(clip));
        break;
      }
    }
    if (alpha < 1.0) cv::addWeighted(layer, alpha, plot, 1.0 - alpha, 0.0, plot);
    if (it.kind == Kind::Points3D && it.u.size() == 6) {
      // small X/Y/Z triad in the lower-left corner
      cv::Point2d o(28, pr.height - 28);
      const char* names[3] = {"X", "Y", "Z"};
      for (int k = 0; k < 3; ++k) {
        cv::Point2d d(it.u[k] * 20, -it.u[3 + k] * 20);
        cv::arrowedLine(plot, sp(o.x, o.y), sp(o.x + d.x, o.y + d.y), colors::black, 1, cv::LINE_AA, kShift, 0.25);
        put_text(plot, names[k], {(int)(o.x + d.x * 1.3) - 4, (int)(o.y + d.y * 1.3) + 4}, 0.35, colors::black);
      }
    }
  }

  // ---- frame, ticks, labels -----------------------------------------------
  if (!axis_off_) {
    cv::rectangle(canvas, pr, colors::black, 1);
    for (size_t i = 0; i < xt.size(); ++i) {
      int x = pr.x + (int)std::lround(px(xt[i]));
      cv::line(canvas, {x, pr.y + pr.height}, {x, pr.y + pr.height + 4}, colors::black, 1);
      std::string s = xticklabels_.empty() ? tick_label(xt[i], xstep) : "";
      put_text_centered(canvas, s, {x, pr.y + pr.height + 14}, fs, colors::black);
    }
    if (!xticklabels_.empty()) {
      for (size_t i = 0; i < xticklabels_.size(); ++i) {
        int x = pr.x + (int)std::lround(px(double(i)));
        put_text_centered(canvas, xticklabels_[i], {x, pr.y + pr.height + 14}, 0.36, colors::black);
      }
    }
    if (!hide_yticks_) {
      for (size_t i = 0; i < yt.size() && yticklabels_.empty(); ++i) {
        int y = pr.y + (int)std::lround(py(yt[i]));
        cv::line(canvas, {pr.x - 4, y}, {pr.x, y}, colors::black, 1);
        std::string s = tick_label(yt[i], ystep);
        cv::Size ts = text_size(s, fs);
        put_text(canvas, s, {pr.x - 7 - ts.width, y + ts.height / 2}, fs, colors::black);
      }
      for (size_t i = 0; i < yticklabels_.size(); ++i) {
        int y = pr.y + (int)std::lround(py(double(i)));
        cv::Size ts = text_size(yticklabels_[i], 0.36);
        put_text(canvas, yticklabels_[i], {pr.x - 6 - ts.width, y + ts.height / 2}, 0.36, colors::black);
      }
    }
    if (!xlabel_.empty()) put_text_centered(canvas, xlabel_, {pr.x + pr.width / 2, pr.y + pr.height + 34}, 0.5, colors::black);
    if (!ylabel_.empty()) put_text_vertical(canvas, ylabel_, {area.x + 12, pr.y + pr.height / 2}, 0.5, colors::black);
  } else if (!xlabel_.empty()) {
    put_text_centered(canvas, xlabel_, {pr.x + pr.width / 2, pr.y + pr.height - 10}, 0.45, colors::black);
  }
  if (!title_.empty()) put_text_centered(canvas, title_, {pr.x + pr.width / 2, area.y + 14}, 0.52, colors::black);

  // ---- colour bar -----------------------------------------------------------
  if (has_colorbar_) {
    cv::Rect cb(pr.x + pr.width + 12, pr.y, 16, pr.height);
    for (int r = 0; r < cb.height; ++r) {
      double v = 1.0 - double(r) / std::max(1, cb.height - 1);
      cv::line(canvas, {cb.x, cb.y + r}, {cb.x + cb.width, cb.y + r}, colormap(cbar_map_, v), 1);
    }
    cv::rectangle(canvas, cb, colors::black, 1);
    std::vector<double> ct = nice_ticks(cbar_min_, cbar_max_, 5);
    double cstep = ct.size() > 1 ? ct[1] - ct[0] : 1;
    for (double t : ct) {
      if (cbar_max_ <= cbar_min_) break;
      int y = cb.y + (int)std::lround((cbar_max_ - t) / (cbar_max_ - cbar_min_) * (cb.height - 1));
      cv::line(canvas, {cb.x + cb.width, y}, {cb.x + cb.width + 3, y}, colors::black, 1);
      put_text(canvas, tick_label(t, cstep), {cb.x + cb.width + 5, y + 4}, 0.36, colors::black);
    }
    put_text_vertical(canvas, colorbar_label_, {cb.x + cb.width + 58, cb.y + cb.height / 2}, 0.45, colors::black);
  }

  // ---- legend ---------------------------------------------------------------
  if (legend_) {
    std::vector<const Item*> labeled;
    for (const auto& it : items_)
      if (!it.style.label_.empty()) labeled.push_back(&it);
    if (!labeled.empty()) {
      int w = 0, rowh = 18;
      for (auto* it : labeled) w = std::max(w, text_size(it->style.label_, 0.42).width);
      int bw = w + 50, bh = rowh * (int)labeled.size() + 8;
      bool upper = legend_loc_.find("lower") == std::string::npos;
      bool rightside = legend_loc_.find("left") == std::string::npos;
      int bx = rightside ? pr.x + pr.width - bw - 6 : pr.x + 6;
      int by = upper ? pr.y + 6 : pr.y + pr.height - bh - 6;
      cv::Rect box(bx, by, bw, bh);
      cv::Mat roi = canvas(box & cv::Rect(0, 0, canvas.cols, canvas.rows));
      cv::addWeighted(roi, 0.2, cv::Mat(roi.size(), roi.type(), colors::white), 0.8, 0, roi);
      cv::rectangle(canvas, box, Color(200, 200, 200), 1);
      for (size_t i = 0; i < labeled.size(); ++i) {
        const Item* it = labeled[i];
        int cy = by + 4 + rowh * (int)i + rowh / 2;
        if (it->kind == Kind::Line || it->kind == Kind::HLine || it->kind == Kind::VLine ||
            it->kind == Kind::Ellipse || it->kind == Kind::Polygon || it->kind == Kind::Circle) {
          std::string line = it->style.line;
          if (line.empty() && !it->style.marker) line = "-";
          draw_polyline(canvas, {{double(bx + 8), double(cy)}, {double(bx + 36), double(cy)}}, it->style.color,
                        it->style.linewidth, line);
          if (it->style.marker)
            draw_marker(canvas, {double(bx + 22), double(cy)}, it->style.marker, it->style.markersize,
                        it->style.color, it->style.filled);
        }
        put_text(canvas, it->style.label_, {bx + 42, cy + 5}, 0.42, colors::black);
      }
    }
  }
}


// ----------------------------------------------------------------------------
// Figure
// ----------------------------------------------------------------------------

Figure::Figure(int nrows, int ncols, int width, int height)
    : nrows_(nrows), ncols_(ncols), width_(width), height_(height), axes_(nrows * ncols), cells_(nrows * ncols) {
  if (width_ <= 0) width_ = std::min(1800, 500 * ncols);
  if (height_ <= 0) height_ = std::min(1400, 400 * nrows);
}

Axes& Figure::span(int row, int col, int rowspan, int colspan) {
  for (int r = row; r < row + rowspan && r < nrows_; ++r)
    for (int c = col; c < col + colspan && c < ncols_; ++c) cells_[r * ncols_ + c].hidden = true;
  Cell& cell = cells_[row * ncols_ + col];
  cell.hidden = false;
  cell.rowspan = rowspan;
  cell.colspan = colspan;
  return ax(row, col);
}

cv::Mat Figure::render() const {
  int top = suptitle_.empty() ? 0 : 34;
  cv::Mat canvas(height_ + top, width_, CV_8UC3, colors::white);
  int cw = width_ / ncols_, ch = height_ / nrows_;
  for (int r = 0; r < nrows_; ++r)
    for (int c = 0; c < ncols_; ++c) {
      const Cell& cell = cells_[r * ncols_ + c];
      if (cell.hidden) continue;
      axes_[r * ncols_ + c].render(canvas, cv::Rect(c * cw, top + r * ch, cw * cell.colspan, ch * cell.rowspan));
    }
  if (!suptitle_.empty()) put_text_centered(canvas, suptitle_, {width_ / 2, 16}, 0.6, colors::black);
  return canvas;
}

void write_image(const cv::Mat& img, const std::string& path) {
  cv::imwrite(path, img);
  std::cout << "[saved] " << path << std::endl;
  const char* show = std::getenv("PORA_SHOW");
  if (show && std::string(show) == "1") {
    cv::imshow(path, img);
    cv::waitKey(0);
    cv::destroyWindow(path);
  }
}

}  // namespace pora::plt
