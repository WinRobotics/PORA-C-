// C++ port of ch18/exercises/reinforce.py (class declaration)
#pragma once

#include "cartpole.hpp"

#include <torch/torch.h>

#include <memory>
#include <utility>
#include <vector>

class Reinforce : public Agent {
 public:
  /**
   * Args:
   *     state_dim: state dimension for the environment
   *     action_dim: action dimension for the environment
   *     hidden_dim: dimension of the hidden layers of the policy network
   */
  Reinforce(int state_dim, int action_dim, int hidden_dim = 24);

  /** Reset model. */
  void reset();

  /**
   * Construct a Multi-Layer Perceptron that takes as input the state vector
   * and outputs a vector of action probabilities for the state. This network has
   * 2 layers with ReLU activation and softmax output.
   *
   * Returns:
   *     Network model.
   */
  torch::nn::Sequential build_network();

  /**
   * Evaluate learned stochastic policy.
   *
   * Args:
   *     state: state to evaluate policy
   *
   * Returns:
   *     Action to take from the state.
   */
  int policy(const State& state) override;

  /**
   * Evaluate policy at the given state. This will be used during
   * the training process in train().
   *
   * Args:
   *     state: state to evaluate policy
   *
   * Returns:
   *     Action sampled from the policy distribution, log probability
   *     of the sampled action.
   */
  std::pair<int, torch::Tensor> policy_train(const State& state);

  /**
   * Sample an episode from the environment.
   *
   * Args:
   *     cartpole: cartpole environment
   *
   * Returns:
   *     List of rewards at each timestep.
   *     List of log probabilities of the actions taken at each timestep.
   */
  std::pair<std::vector<double>, std::vector<torch::Tensor>> sample_episode(Cartpole& cartpole);

  /**
   * Train policy network using the REINFORCE algorithm.
   *
   * Args:
   *     cartpole: cartpole environment
   *     num_episodes: number of episodes to run
   *     use_causality: whether to use the "causality trick" when defining loss function
   *     use_baseline: whether to use a baseline when defining the loss function
   */
  void train(Cartpole& cartpole, int num_episodes = 1000, bool use_causality = false, bool use_baseline = true);

  int hidden_dim;
  torch::nn::Sequential policy_network{nullptr};
  std::vector<torch::Tensor> initial_state_dict;
  std::unique_ptr<torch::optim::Adam> optimizer;
};
