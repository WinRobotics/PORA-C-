// C++ port of ch18/exercises/q_learning.py (class declaration)
#pragma once

#include "cartpole.hpp"

#include <torch/torch.h>

#include <deque>
#include <random>
#include <tuple>
#include <vector>

struct Transition {
  torch::Tensor state;       // (1, state_dim)
  int action;
  torch::Tensor reward;      // (1,)
  torch::Tensor next_state;  // (1, state_dim), or an undefined tensor if terminal (Python: None)
};

class QLearning : public Agent {
 public:
  /**
   * Args:
   *     state_dim: state dimension for the environment
   *     action_dim: action dimension for the environment
   *     hidden_dim: dimension of the hidden layers of the policy network
   *     buffer_length: max number of samples to store in replay buffer
   *     eps_decay: increase to slow the rate of epsilon decrease for
   *                epsilon-greedy exploration
   *     use_gpu: whether to use GPU
   */
  QLearning(int state_dim, int action_dim, int hidden_dim = 24, int buffer_length = 1000, int eps_decay = 10000,
            bool use_gpu = false);

  /** Reset model, buffer, and iteration count. */
  void reset();

  /**
   * Compute epsilon threshold for eps-greedy exploration that decreases
   * as the iteration count increases.
   */
  double eps_threshold();

  /**
   * Construct a Multi-Layer Perceptron that takes as input the state vector
   * and outputs the Q-value vector Q(s, a) for the state. This network has
   * 3 layers (2 hidden layers) with ReLU activation.
   *
   * Returns:
   *     Network model.
   */
  torch::nn::Sequential build_network();

  /**
   * Evaluate policy at the given state.
   *
   * Args:
   *     state: state to evaluate policy
   *
   * Returns:
   *     Optimal action to take from the state.
   */
  int policy(const State& state) override;

  /**
   * Evaluate policy at the given state. This will be used during
   * the training process in train(). Uses an eps-greedy
   * approach.
   *
   * Args:
   *     state: state to evaluate policy
   *
   * Returns:
   *     Action to take from the state, as an int64 tensor of shape (1, 1).
   */
  torch::Tensor policy_train(const State& state);

  /**
   * Compute target Q-value, reward if terminal (next_state is undefined) or
   * reward + gamma * max_a Q(next_state, a) otherwise.
   */
  torch::Tensor compute_target(const torch::Tensor& reward, const torch::Tensor& next_state);

  /** Add a transition to the buffer (next_state is ignored when `terminal` is true). */
  void add_to_buffer(const State& state, int action, double reward, const State& next_state, bool terminal);

  /**
   * Sample replay buffer.
   *
   * Returns:
   *     States, actions, and targets for the minibatch.
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sample_buffer();

  /**
   * Train action-value network using deep Q-learning approach described in Algorithm 1
   * from "Playing Atari with Deep Reinforcement Learning".
   *
   * Args:
   *     cartpole: cart-pole environment to train in
   *     num_episodes: number of episodes to run
   */
  void train(Cartpole& cartpole, int num_episodes = 100);

  int hidden_dim;
  torch::nn::Sequential policy_network{nullptr};
  std::vector<torch::Tensor> initial_state_dict;
  int buffer_length;
  std::deque<Transition> buffer;  // replay buffer with the `buffer_length` most recent transitions
  long iteration = 0;
  int eps_decay;
  std::mt19937_64 random{0};  // the counterpart of Python's `random` module
};
