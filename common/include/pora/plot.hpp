// A small matplotlib-like plotting API rendered with OpenCV. It replaces the
// `matplotlib.pyplot` calls of the notebooks: figures are saved as PNG files
// into the exercise output directory (see pora::output_path) and, if the
// environment variable PORA_SHOW=1 is set, also shown in a window.
//
//   pora::plt::Figure fig(1, 2);                       // like plt.subplots(1, 2)
//   fig.ax(0).plot(t, theta, "b-", "theta");           // fmt strings like matplotlib
//   fig.ax(0).set_xlabel("t");
//   fig.ax(1).plot(x, y, pora::plt::fmt("r--").lw(2).alpha(0.5).label("ref"));
//   fig.save("result.png");
//
// Text is drawn with OpenCV's Hershey fonts, so only ASCII is supported.
#pragma once

#include "pora/core.hpp"

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pora::plt {

using Color = cv::Scalar;  // BGR, 0-255

// matplotlib "tab10" colours
namespace colors {
const Color tab_blue(180, 119, 31), tab_orange(14, 127, 255), tab_green(44, 160, 44), tab_red(40, 39, 214),
    tab_purple(189, 103, 148), tab_brown(75, 86, 140), tab_pink(194, 119, 227), tab_gray(127, 127, 127),
    tab_olive(34, 189, 188), tab_cyan(207, 190, 23);
const Color black(0, 0, 0), white(255, 255, 255), red(0, 0, 255), green(0, 128, 0), blue(255, 0, 0),
    cyan(191, 191, 0), magenta(191, 0, 191), yellow(0, 191, 191), purple(128, 0, 128), orange(0, 165, 255),
    gray(128, 128, 128);
}  // namespace colors

// i-th colour of the default matplotlib colour cycle
Color cycle_color(int i);

// Line / marker style, built either from a matplotlib format string with
// fmt("r--") or by chaining setters: fmt("-").c(colors::tab_red).lw(2).
struct Style {
  bool has_color = false;
  Color color = colors::tab_blue;
  std::string line = "-";  // "-", "--", ":", "-." or "" (no line)
  char marker = 0;         // 0 (none), '.', 'o', '+', 'x', '*', '^', 's'
  double linewidth = 1.5;
  double markersize = 5.0;
  double alpha_ = 1.0;
  bool filled = false;  // for markers and shapes
  std::string label_;

  Style& c(const Color& col) { color = col; has_color = true; return *this; }
  Style& lw(double w) { linewidth = w; return *this; }
  Style& ms(double s) { markersize = s; return *this; }
  Style& alpha(double a) { alpha_ = a; return *this; }
  Style& label(const std::string& l) { label_ = l; return *this; }
  Style& fill(bool f = true) { filled = f; return *this; }
  Style& ls(const std::string& l) { line = l; return *this; }
  Style& mk(char m) { marker = m; return *this; }
};

// Parse a matplotlib format string such as "r--", "b.", "go", "k-", "--*".
Style fmt(const std::string& format = "-");

// Colour maps for heat maps / point colouring.
enum class ColorMap { Viridis, Inferno, Blues, Gray, Jet };
Color colormap(ColorMap map, double value01);

class Axes {
 public:
  // Line plot of y versus x (any container with size() and operator[]).
  template <class X, class Y>
  void plot(const X& x, const Y& y, const Style& style) {
    plot_impl(to_vec(x), to_vec(y), style);
  }
  template <class X, class Y, class = std::enable_if_t<!std::is_convertible_v<const Y&, std::string>>>
  void plot(const X& x, const Y& y, const std::string& format = "-", const std::string& label = "") {
    Style s = fmt(format);
    s.label_ = label;
    plot_impl(to_vec(x), to_vec(y), s);
  }
  // Plot y against its index, like plt.plot(y).
  template <class Y>
  void plot(const Y& y, const Style& style) {
    std::vector<double> yy = to_vec(y), xx(yy.size());
    for (size_t i = 0; i < xx.size(); ++i) xx[i] = static_cast<double>(i);
    plot_impl(xx, yy, style);
  }
  template <class Y>
  void plot(const Y& y, const std::string& format = "-", const std::string& label = "") {
    Style s = fmt(format);
    s.label_ = label;
    plot(y, s);
  }
  // Scatter plot (markers only). Optional per-point colours.
  template <class X, class Y>
  void scatter(const X& x, const Y& y, Style style = fmt("o"), const std::vector<Color>& point_colors = {}) {
    style.line = "";
    if (!style.marker) style.marker = 'o';
    style.filled = true;
    plot_impl(to_vec(x), to_vec(y), style, point_colors);
  }

  void axhline(double y, const Style& style = fmt("k--"));
  void axvline(double x, const Style& style = fmt("k--"));
  // Ellipse centred at (cx, cy) with full width/height and rotation angle in degrees
  // (same arguments as matplotlib.patches.Ellipse).
  void ellipse(double cx, double cy, double width, double height, double angle_deg, const Style& style = fmt("r-"));
  void circle(double cx, double cy, double radius, const Style& style = fmt("k-"));
  void rectangle(double x0, double y0, double x1, double y1, const Style& style = fmt("k-"));
  void polygon(const std::vector<double>& xs, const std::vector<double>& ys, const Style& style = fmt("k-"));
  // Arrows from (x, y) with direction (u, v) scaled by `scale` (data units), like plt.quiver.
  template <class A, class B, class C, class D>
  void quiver(const A& x, const B& y, const C& u, const D& v, double scale = 0.3, const Style& style = fmt("k-")) {
    quiver_impl(to_vec(x), to_vec(y), to_vec(u), to_vec(v), scale, style);
  }
  void text(double x, double y, const std::string& s, const Color& color = colors::black, double font_scale = 0.5);

  // Heat map of a matrix; data(row, col) is drawn at (x=col, y=row). With
  // origin_lower=true row 0 is at the bottom (like imshow(..., origin='lower')).
  void imshow(const Eigen::MatrixXd& data, bool origin_lower = true, ColorMap map = ColorMap::Viridis,
              const std::string& colorbar_label = "", bool annotate = false);
  // Show an image (grayscale or BGR). Axes are in pixel coordinates (y down).
  void image(const cv::Mat& img, double alpha = 1.0);
  // 3D scatter plot projected with the matplotlib view convention (azim/elev in deg).
  void scatter3d(const Eigen::MatrixXd& points, double azim, double elev, ColorMap map, double point_size = 1.5);

  void set_title(const std::string& t) { title_ = t; }
  void set_xlabel(const std::string& l) { xlabel_ = l; }
  void set_ylabel(const std::string& l) { ylabel_ = l; }
  void set_xlim(double lo, double hi) { xlim_ = {lo, hi}; has_xlim_ = true; }
  void set_ylim(double lo, double hi) { ylim_ = {lo, hi}; has_ylim_ = true; }
  void axis(double x0, double x1, double y0, double y1) { set_xlim(x0, x1); set_ylim(y0, y1); }
  void axis_equal(bool equal = true) { equal_ = equal; }
  void grid(bool on = true) { grid_ = on; }
  // loc: "best"/"upper right", "upper left", "lower left", "lower right"
  void legend(const std::string& loc = "best") { legend_ = true; legend_loc_ = loc; }
  void axis_off() { axis_off_ = true; }
  void hide_yticks() { hide_yticks_ = true; }
  void set_xticklabels(const std::vector<std::string>& labels) { xticklabels_ = labels; }
  void set_yticklabels(const std::vector<std::string>& labels) { yticklabels_ = labels; }

  // Render into `canvas` inside `area` (used by Figure).
  void render(cv::Mat& canvas, const cv::Rect& area) const;

 private:
  enum class Kind { Line, HLine, VLine, Ellipse, Circle, Polygon, Quiver, Text, Heatmap, Image, Points3D };
  struct Item {
    Kind kind;
    std::vector<double> x, y, u, v;
    std::vector<Color> point_colors;
    Style style;
    std::string text;
    double a = 0, b = 0, c = 0, d = 0, e = 0;
    Eigen::MatrixXd data;
    cv::Mat img;
    bool flag = false;
    ColorMap map = ColorMap::Viridis;
  };

  template <class C>
  static std::vector<double> to_vec(const C& c) {
    std::vector<double> out(static_cast<size_t>(c.size()));
    for (size_t i = 0; i < out.size(); ++i) out[i] = static_cast<double>(c[i]);
    return out;
  }
  void plot_impl(const std::vector<double>& x, const std::vector<double>& y, Style style,
                 const std::vector<Color>& point_colors = {});
  void quiver_impl(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& u,
                   const std::vector<double>& v, double scale, const Style& style);
  Style next_style(Style s);

  std::vector<Item> items_;
  int color_index_ = 0;
  std::string title_, xlabel_, ylabel_, legend_loc_ = "best", colorbar_label_;
  std::pair<double, double> xlim_{0, 1}, ylim_{0, 1};
  bool has_xlim_ = false, has_ylim_ = false, equal_ = false, grid_ = false, legend_ = false;
  bool axis_off_ = false, hide_yticks_ = false, image_axes_ = false, has_colorbar_ = false;
  double cbar_min_ = 0, cbar_max_ = 1;
  ColorMap cbar_map_ = ColorMap::Viridis;
  std::vector<std::string> xticklabels_, yticklabels_;
};

class Figure {
 public:
  // Grid of nrows x ncols axes on a width x height pixel canvas.
  explicit Figure(int nrows = 1, int ncols = 1, int width = 0, int height = 0);
  Axes& ax(int i = 0) { return axes_.at(i); }
  Axes& ax(int row, int col) { return axes_.at(row * ncols_ + col); }
  // Let the axes at (row, col) span several grid cells (like plt.subplot2grid); the
  // other covered cells are hidden. Returns that axes.
  Axes& span(int row, int col, int rowspan, int colspan);
  int size() const { return static_cast<int>(axes_.size()); }
  void suptitle(const std::string& t) { suptitle_ = t; }
  cv::Mat render() const;
  // Save as PNG inside this exercise's output directory (or at an explicit path
  // if `filename` contains a '/'). Shows the figure when PORA_SHOW=1.
  inline void save(const std::string& filename) const;

 private:
  int nrows_, ncols_, width_, height_;
  std::vector<Axes> axes_;
  struct Cell {
    int rowspan = 1, colspan = 1;
    bool hidden = false;
  };
  std::vector<Cell> cells_;
  std::string suptitle_;
};

// Write an image to `path` (and show it when PORA_SHOW=1).
void write_image(const cv::Mat& img, const std::string& path);

// File name -> path inside this exercise's output directory (paths containing '/' are kept).
inline std::string resolve_output(const std::string& filename) {
  return filename.find('/') == std::string::npos ? pora::output_path(filename) : filename;
}

// Write an image (e.g. a cv::Mat result) into the output directory, and show it when PORA_SHOW=1.
inline void save_image(const cv::Mat& img, const std::string& filename) {
  write_image(img, resolve_output(filename));
}

inline void Figure::save(const std::string& filename) const { write_image(render(), resolve_output(filename)); }

}  // namespace pora::plt
