// C++ port of ch03/exercises/cartpole_utils.py
#include "cartpole_utils.hpp"

#include <pora/core.hpp>
#include <pora/numerics.hpp>
#include <pora/plot.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <cmath>
#include <cstdio>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

VectorXd cartpole_dynamics(const VectorXd& s, const VectorXd& u, const CartPole& cartpole) {
  const double mp = cartpole.mp, mc = cartpole.mc, L = cartpole.L, g = cartpole.g;
  const double th = s[1], dx = s[2], dth = s[3];
  const double sinth = std::sin(th), costh = std::cos(th);
  const double h = mc + mp * (sinth * sinth);
  VectorXd ds(4);
  ds << dx,
        dth,
        (mp * sinth * (L * (dth * dth) + g * costh) + u[0]) / h,
        -((mc + mp) * g * sinth + mp * L * (dth * dth) * sinth * costh + u[0] * costh) / (h * L);
  return ds;
}

std::pair<MatrixXd, MatrixXd> simulate_cartpole(const CartPole& cartpole, const VectorXd& t, const VectorXd& s0,
                                                const CartpoleController& controller, bool closed_loop) {
  const int T = static_cast<int>(t.size());
  MatrixXd s = MatrixXd::Zero(T, 4);
  MatrixXd u = MatrixXd::Zero(T - 1, 1);
  s.row(0) = s0.transpose();
  for (int k = 0; k < T - 1; ++k) {
    u.row(k) = controller.compute_control(k, s.row(k).transpose(), closed_loop).transpose();
    VectorXd uk = u.row(k).transpose();
    VectorXd tk(2);
    tk << t[k], t[k + 1];
    MatrixXd sol = pora::odeint([&](const VectorXd& ss, double) { return cartpole_dynamics(ss, uk, cartpole); },
                                s.row(k).transpose(), tk);
    s.row(k + 1) = sol.row(1);
  }
  return {s, u};
}

void animate_cartpole(const VectorXd& t, const VectorXd& x, const VectorXd& theta, const std::string& name) {
  // Geometry
  const double cart_width = 2., cart_height = 1., wheel_radius = 0.3, wheel_sep = 1., pole_length = 5.,
               mass_radius = 0.25;

  // Figure and axis
  const double x_min = x.minCoeff() - 1.1 * pole_length, x_max = x.maxCoeff() + 1.1 * pole_length;
  const double y_min = -pole_length, y_max = 1.1 * (wheel_radius + cart_height + pole_length);
  const int W = 800;
  const double scale = W / (x_max - x_min);
  const int H = std::max(200, static_cast<int>(std::lround((y_max - y_min) * scale)));
  auto P = [&](double wx, double wy) {
    return cv::Point(static_cast<int>(std::lround((wx - x_min) * scale)),
                     static_cast<int>(std::lround((y_max - wy) * scale)));
  };
  const double dt = t[1] - t[0];
  const std::string path = pora::output_path(name + ".mp4");
  cv::VideoWriter video(path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 1.0 / dt, cv::Size(W, H));
  if (!video.isOpened()) {
    std::cerr << "Could not open video writer for " << path << std::endl;
    return;
  }
  const cv::Scalar tab_blue(180, 119, 31), tab_orange(14, 127, 255), black(0, 0, 0);
  std::vector<cv::Point> trace;
  for (int k = 0; k < t.size(); ++k) {
    cv::Mat frame(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::line(frame, P(x_min, 0.), P(x_max, 0.), black, 1, cv::LINE_AA);

    // Geometry
    const double cx = x[k] - cart_width / 2, cy = wheel_radius;
    const cv::Point2d pole_start(x[k], wheel_radius + cart_height);
    const cv::Point2d pole_end = pole_start + pole_length * cv::Point2d(std::sin(theta[k]), -std::cos(theta[k]));

    // Cart
    cv::rectangle(frame, P(cx, cy + cart_height), P(cx + cart_width, cy), tab_blue, cv::FILLED, cv::LINE_AA);
    cv::rectangle(frame, P(cx, cy + cart_height), P(cx + cart_width, cy), black, 1, cv::LINE_AA);

    // Wheels
    cv::circle(frame, P(x[k] - wheel_sep / 2, wheel_radius), static_cast<int>(wheel_radius * scale), black,
               cv::FILLED, cv::LINE_AA);
    cv::circle(frame, P(x[k] + wheel_sep / 2, wheel_radius), static_cast<int>(wheel_radius * scale), black,
               cv::FILLED, cv::LINE_AA);

    // Pendulum (and the dashed trace of the mass)
    trace.push_back(P(pole_end.x, pole_end.y));
    for (size_t i = 1; i < trace.size(); i += 2) cv::line(frame, trace[i - 1], trace[i], tab_orange, 2, cv::LINE_AA);
    cv::line(frame, P(pole_start.x, pole_start.y), P(pole_end.x, pole_end.y), black, 3, cv::LINE_AA);
    cv::circle(frame, P(pole_end.x, pole_end.y), static_cast<int>(mass_radius * scale), black, cv::FILLED,
               cv::LINE_AA);

    // Time-stamp
    char buf[32];
    std::snprintf(buf, sizeof(buf), "t = %.1f s", t[k]);
    cv::putText(frame, buf, {static_cast<int>(0.1 * W), static_cast<int>(0.1 * H) + 10}, cv::FONT_HERSHEY_SIMPLEX,
                0.6, black, 1, cv::LINE_AA);
    video.write(frame);
  }
  video.release();
  std::cout << "[saved] " << path << std::endl;
}

void plot_state_and_control_history(const MatrixXd& s, const MatrixXd& u, const VectorXd& t, const MatrixXd& s_ref,
                                    const std::string& name) {
  const int n = static_cast<int>(s.cols());
  const int m = static_cast<int>(u.cols());
  const int N_u = static_cast<int>(u.rows());
  pora::plt::Figure fig(1, n + m, 1800, 320);
  const char* labels_s[] = {"x(t)", "theta(t)", "xdot(t)", "thetadot(t)"};
  const char* labels_u[] = {"u(t)"};
  for (int i = 0; i < n; ++i) {
    fig.ax(i).plot(t, s.col(i), "-");
    fig.ax(i).plot(t, s_ref.col(i), "--");
    fig.ax(i).set_xlabel("t");
    fig.ax(i).set_ylabel(labels_s[i]);
  }
  for (int i = 0; i < m; ++i) {
    fig.ax(n + i).plot(t.head(N_u), u.col(i));
    fig.ax(n + i).set_xlabel("t");
    fig.ax(n + i).set_ylabel(labels_u[i]);
  }
  fig.save(name + ".png");
}
