// C++ port of ch18/exercises/a2c_utils.py
//
// The Python version uses Gymnasium's "LunarLanderContinuous-v3" environment (Box2D physics).
// `LunarLanderEnv` below is a C++ port of that environment on top of the Box2D C++ library
// (same terrain generation, lander/legs, engines, reward shaping, termination, 1000 step time
// limit and 600x400 rendering).
#pragma once

#include <torch/torch.h>

#include <opencv2/core.hpp>

#include <memory>
#include <random>
#include <string>
#include <vector>

class b2World;
class b2Body;
class LunarLanderContactDetector;

struct LunarLanderStep {
  std::vector<float> observation;
  double reward;
  bool terminated;
  bool truncated;
};

/** C++ port of gymnasium's LunarLanderContinuous-v3 (with the TimeLimit wrapper). */
class LunarLanderEnv {
 public:
  LunarLanderEnv();
  ~LunarLanderEnv();

  /** Reset the episode; seeds the environment's random generator when seed >= 0. */
  std::vector<float> reset(long seed = -1);
  /** Apply an action [main engine, lateral engine], each clipped to [-1, 1]. */
  LunarLanderStep step(const std::vector<float>& action);
  /** Render the current state as a 600x400 BGR image. */
  cv::Mat render() const;

  const int observation_dim = 8;
  const int action_dim = 2;
  const int max_episode_steps = 1000;

  // Used by the contact listener
  bool game_over = false;
  b2Body* lander = nullptr;
  b2Body* legs[2] = {nullptr, nullptr};
  bool leg_contact[2] = {false, false};

 private:
  void destroy();

  std::unique_ptr<b2World> world_;
  std::unique_ptr<LunarLanderContactDetector> contact_listener_;
  b2Body* moon_ = nullptr;
  std::vector<std::vector<std::pair<float, float>>> sky_polys_;
  double helipad_x1_ = 0, helipad_x2_ = 0, helipad_y_ = 0;
  bool has_prev_shaping_ = false;
  double prev_shaping_ = 0;
  int elapsed_steps_ = 0;
  double last_m_power_ = 0, last_s_power_ = 0, last_s_dir_ = 0;
  std::mt19937_64 np_random_{0};
};

class LunarLander {
 public:
  explicit LunarLander(const std::string& video_file_directory = "lunar_lander_outputs");

  std::string video_file_directory;
  LunarLanderEnv env;
  int state_dim;
  int action_dim;
};

/**
 * Multivariate normal distribution with diagonal covariance.
 */
struct MultivariateNormalDiag {
  torch::Tensor mean;  // mean of the distribution
  torch::Tensor std;   // standard deviation

  /** Generate a random sample from the multivariate normal distribution. */
  torch::Tensor sample() const { return mean + std * torch::randn_like(mean); }

  /** Compute the log of the probability density function evaluate at x (summed over the last axis). */
  torch::Tensor log_prob(const torch::Tensor& x) const {
    const double log_sqrt_2pi = 0.5 * std::log(2 * M_PI);
    torch::Tensor z = (x - mean) / std;
    return (-0.5 * z * z - torch::log(std) - log_sqrt_2pi).sum(-1);
  }
};
