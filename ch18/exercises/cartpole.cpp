// C++ port of ch18/exercises/cartpole.py
#include "cartpole.hpp"

#include <pora/core.hpp>
#include <pora/plot.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <numeric>

// -----------------------------------------------------------------------------
// CartPoleEnv (gymnasium CartPole-v1)
// -----------------------------------------------------------------------------

CartPoleEnv::CartPoleEnv() = default;

State CartPoleEnv::reset(long seed) {
  if (seed >= 0) np_random_.seed(static_cast<unsigned long long>(seed));
  std::uniform_real_distribution<double> dist(-0.05, 0.05);
  for (auto& s : state_) s = dist(np_random_);
  steps_beyond_terminated_ = -1;
  elapsed_steps_ = 0;
  return State(state_.begin(), state_.end());
}

StepResult CartPoleEnv::step(int action) {
  double x = state_[0], x_dot = state_[1], theta = state_[2], theta_dot = state_[3];
  const double force = action == 1 ? force_mag : -force_mag;
  const double costheta = std::cos(theta), sintheta = std::sin(theta);
  const double temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
  const double thetaacc =
      (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass));
  const double xacc = temp - polemass_length * thetaacc * costheta / total_mass;

  // euler integration
  x = x + tau * x_dot;
  x_dot = x_dot + tau * xacc;
  theta = theta + tau * theta_dot;
  theta_dot = theta_dot + tau * thetaacc;
  state_ = {x, x_dot, theta, theta_dot};

  const bool terminated =
      x < -x_threshold || x > x_threshold || theta < -theta_threshold_radians || theta > theta_threshold_radians;
  double reward;
  if (!terminated) {
    reward = 1.0;
  } else if (steps_beyond_terminated_ < 0) {
    steps_beyond_terminated_ = 0;
    reward = 1.0;
  } else {
    ++steps_beyond_terminated_;
    reward = 0.0;
  }
  ++elapsed_steps_;
  const bool truncated = elapsed_steps_ >= max_episode_steps;  // TimeLimit wrapper
  return {State(state_.begin(), state_.end()), reward, terminated, truncated};
}

cv::Mat CartPoleEnv::render() const {
  const int screen_width = 600, screen_height = 400;
  const double world_width = x_threshold * 2;
  const double scale = screen_width / world_width;
  const double polewidth = 10.0, polelen = scale * (2 * length), cartwidth = 50.0, cartheight = 30.0;

  cv::Mat surf(screen_height, screen_width, CV_8UC3, cv::Scalar(255, 255, 255));
  // gymnasium draws with y pointing up and flips the image at the end
  auto P = [&](double px, double py) { return cv::Point(static_cast<int>(std::lround(px)), screen_height - 1 - static_cast<int>(std::lround(py))); };

  const double l = -cartwidth / 2, r = cartwidth / 2, t = cartheight / 2, b = -cartheight / 2;
  const double axleoffset = cartheight / 4.0;
  const double cartx = state_[0] * scale + screen_width / 2.0;  // MIDDLE OF CART
  const double carty = 100;                                     // TOP OF CART
  std::vector<cv::Point> cart = {P(l + cartx, b + carty), P(l + cartx, t + carty), P(r + cartx, t + carty),
                                 P(r + cartx, b + carty)};
  cv::fillConvexPoly(surf, cart, cv::Scalar(0, 0, 0), cv::LINE_AA);

  const double pl = -polewidth / 2, pr = polewidth / 2, pt = polelen - polewidth / 2, pb = -polewidth / 2;
  std::vector<cv::Point> pole;
  const double a = -state_[2];
  for (auto [cx, cy] : std::vector<std::pair<double, double>>{{pl, pb}, {pl, pt}, {pr, pt}, {pr, pb}}) {
    double rx = cx * std::cos(a) - cy * std::sin(a), ry = cx * std::sin(a) + cy * std::cos(a);
    pole.push_back(P(rx + cartx, ry + carty + axleoffset));
  }
  cv::fillConvexPoly(surf, pole, cv::Scalar(101, 152, 202), cv::LINE_AA);  // RGB (202, 152, 101)
  cv::circle(surf, P(cartx, carty + axleoffset), static_cast<int>(polewidth / 2), cv::Scalar(203, 132, 129),
             cv::FILLED, cv::LINE_AA);  // RGB (129, 132, 203)
  cv::line(surf, P(0, carty), P(screen_width, carty), cv::Scalar(0, 0, 0), 1);
  return surf;
}

// -----------------------------------------------------------------------------
// Agent / Cartpole
// -----------------------------------------------------------------------------

Agent::Agent(int state_dim, int action_dim, const std::string& name, bool use_gpu)
    : state_dim(state_dim),
      action_dim(action_dim),
      device(use_gpu && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      name(name) {}

void Agent::plot_rewards(const std::vector<double>& reward_history, const std::string& filename,
                         const std::string& ylabel) {
  pora::plt::Figure fig(1, 1, 700, 480);
  fig.ax().plot(reward_history);
  fig.ax().set_xlabel("Episode");
  if (!ylabel.empty()) fig.ax().set_ylabel(ylabel);
  fig.ax().set_title("Learning Curve");
  fig.save(filename);
}

Cartpole::Cartpole() {
  // Create environment
  env.reset(0);
  state_dim = env.observation_dim;
  action_dim = 2;
}

void Cartpole::print_environment_info() const {
  std::printf("Observation space: Box([-4.8, -inf, -0.419, -inf], [4.8, inf, 0.419, inf], (4,), float32)\n");
  std::printf("Action space: Discrete(2)\n");
}

void Cartpole::simulate(Agent& agent, int num_episodes, bool visualize, int max_steps) {
  env.reset(0);
  std::vector<double> rewards;  // for episode cumulative reward
  const std::string video_filename = visualize ? pora::output_path(agent.name + "_cartpole_sim.mp4") : "";
  cv::VideoWriter video;
  for (int episode = 0; episode < num_episodes; ++episode) {
    progress(episode, num_episodes, "simulate");
    double episode_reward = 0;
    State obs = env.reset();
    bool terminated = false, truncated = false;
    int t = 0;
    while (!terminated && !truncated) {
      // Render video at specified interval of episodes
      if (episode == 0 && !video_filename.empty()) {
        // For episode 0 and t=0 start a video to store data
        if (t == 0) video.open(video_filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 50, cv::Size(600, 400));
        video.write(env.render());
      }

      // Evalute policy from the observation, apply action in the environment
      int action = agent.policy(obs);
      StepResult res = env.step(action);
      obs = res.observation;
      terminated = res.terminated;
      truncated = res.truncated;
      episode_reward += res.reward;

      truncated = (max_steps >= 0 && t >= max_steps) ? true : truncated;
      ++t;
    }
    rewards.push_back(episode_reward);
  }
  progress(num_episodes, num_episodes, "simulate");

  const double mean = std::accumulate(rewards.begin(), rewards.end(), 0.0) / rewards.size();
  double var = 0;
  for (double r : rewards) var += (r - mean) * (r - mean);
  std::printf("--- Reward Statistics ---\n");
  std::printf("Average: %g\n", mean);
  std::printf("Standard deviation: %g\n", std::sqrt(var / rewards.size()));
  std::printf("Minimum: %g\n", *std::min_element(rewards.begin(), rewards.end()));
  std::printf("Maximum: %g\n", *std::max_element(rewards.begin(), rewards.end()));

  if (!video_filename.empty()) {
    video.release();
    std::cout << "[saved] " << video_filename << std::endl;
  }
}

std::vector<torch::Tensor> clone_parameters(const torch::nn::Module& module) {
  std::vector<torch::Tensor> out;
  for (const auto& p : module.parameters()) out.push_back(p.detach().clone());
  return out;
}

void load_parameters(torch::nn::Module& module, const std::vector<torch::Tensor>& params) {
  torch::NoGradGuard no_grad;
  auto current = module.parameters();
  for (size_t i = 0; i < current.size(); ++i) current[i].copy_(params[i]);
}

void progress(int i, int n, const std::string& what) {
  const int width = 30;
  int filled = n > 0 ? width * i / n : width;
  std::fprintf(stderr, "\r%s %3d%% |%s%s| %d/%d", what.c_str(), n > 0 ? 100 * i / n : 100,
               std::string(filled, '#').c_str(), std::string(width - filled, ' ').c_str(), i, n);
  if (i >= n) std::fprintf(stderr, "\n");
}
