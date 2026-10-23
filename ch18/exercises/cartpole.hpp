// C++ port of ch18/exercises/cartpole.py
//
// The Python version uses the Gymnasium "CartPole-v1" environment. Gymnasium has no C++ API,
// so `CartPoleEnv` below re-implements CartPole-v1 (same dynamics, termination, reward, 500
// step time limit and 600x400 rendering) with the same reset()/step()/render() interface.
#pragma once

#include <torch/torch.h>

#include <opencv2/core.hpp>

#include <random>
#include <string>
#include <vector>

using State = std::vector<float>;  // observation [x, x_dot, theta, theta_dot]

struct StepResult {
  State observation;
  double reward;
  bool terminated;
  bool truncated;
};

/** C++ re-implementation of gymnasium's CartPole-v1 (with the TimeLimit wrapper). */
class CartPoleEnv {
 public:
  CartPoleEnv();

  /** Reset the episode; seeds the environment's random generator when seed >= 0. */
  State reset(long seed = -1);
  /** Apply an action (0: push left, 1: push right). */
  StepResult step(int action);
  /** Render the current state as a 600x400 BGR image. */
  cv::Mat render() const;

  const int observation_dim = 4;
  const int num_actions = 2;
  const int max_episode_steps = 500;

 private:
  const double gravity = 9.8, masscart = 1.0, masspole = 0.1, total_mass = 1.1, length = 0.5,
               polemass_length = 0.05, force_mag = 10.0, tau = 0.02;
  const double theta_threshold_radians = 12 * 2 * 3.14159265358979323846 / 360, x_threshold = 2.4;
  std::vector<double> state_{0, 0, 0, 0};
  int steps_beyond_terminated_ = -1;
  int elapsed_steps_ = 0;
  std::mt19937_64 np_random_{0};
};

class Agent {
 public:
  Agent(int state_dim, int action_dim, const std::string& name = "agent", bool use_gpu = false);
  virtual ~Agent() = default;

  /** Evaluate the (deterministic or sampled) policy at the given state. */
  virtual int policy(const State& state) = 0;

  /** Plot a learning curve and save it as `<filename>`. */
  static void plot_rewards(const std::vector<double>& reward_history, const std::string& filename,
                           const std::string& ylabel = "Total Reward");

  int state_dim;
  int action_dim;
  double gamma = 0.99;  // discount factor
  double learning_rate = 0.001;
  int batch_size = 128;
  torch::Device device;
  std::string name;
};

class Cartpole {
 public:
  Cartpole();

  void print_environment_info() const;

  /**
   * Rollout a particular policy in the cartpole environment.
   *
   * Args:
   *     agent: agent with a policy
   *     num_episodes: number of episodes to run
   *     visualize: whether to save a video of the first episode
   *     max_steps: max number of steps in episode (-1 for no limit)
   */
  void simulate(Agent& agent, int num_episodes, bool visualize = false, int max_steps = -1);

  CartPoleEnv env;
  int state_dim;
  int action_dim;
};

/**
 * Basic agent whose policy is to simply move in the direction the pole is leaning.
 */
class Basic : public Agent {
 public:
  Basic(int state_dim, int action_dim) : Agent(state_dim, action_dim, "basic") {}

  /**
   * Implement a simple controller that moves the cart in the direction the pole
   * is leaning.
   */
  int policy(const State& state) override {
    const float theta = state[2];  // extract pole angle from environment
    const int left = 0, right = 1;
    return theta < 0 ? left : right;
  }
};

// Utilities shared by the deep RL agents --------------------------------------

// Observation as a (1, state_dim) float tensor (the counterpart of
// torch.tensor(state, dtype=torch.float32).unsqueeze(0)).
inline torch::Tensor to_tensor(const State& s) {
  return torch::tensor(std::vector<float>(s.begin(), s.end()), torch::kFloat32).unsqueeze(0);
}

// Copy of all parameters of a module (the counterpart of copy.deepcopy(module.state_dict())).
std::vector<torch::Tensor> clone_parameters(const torch::nn::Module& module);
// Restore parameters saved with clone_parameters (the counterpart of module.load_state_dict()).
void load_parameters(torch::nn::Module& module, const std::vector<torch::Tensor>& params);

// Text progress bar for training loops (the counterpart of tqdm).
void progress(int i, int n, const std::string& what = "");
