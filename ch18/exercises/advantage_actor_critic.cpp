// =============================================================================
// Exercise 3: Advantage Actor-Critic (A2C)
// (C++ port of ch18/exercises/advantage_actor_critic.ipynb)
//
// NOTE: The Python version uses JAX + Equinox + Optax. In C++ we use LibTorch (the C++ API of
// PyTorch): torch::nn modules for the model, autograd for the gradients and torch::optim::Adam
// for the optimizer. The lunar lander environment is the C++ Box2D port in a2c_utils.cpp.
// =============================================================================
#include "a2c_utils.hpp"

#include <pora/core.hpp>
#include <pora/plot.hpp>

#include <Eigen/Dense>
#include <torch/torch.h>

#include <opencv2/videoio.hpp>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using Eigen::VectorXd;

// The Lunar Lander environment is a playground for RL defined in the Gymnasium library
// (https://gymnasium.farama.org/environments/box2d/lunar_lander/). In this exercise, we will use
// the advantage actor-critic (A2C) method to learn a policy to control the lunar lander in an
// optimal way.

// -----------------------------------------------------------------------------
// ### Exercise 3.1: Define the Policy
// In the first part of this exercise, implement the `PolicyImpl::forward` method to define the
// model that we will train to control the lunar lander. This model will output a
// `MultivariateNormalDiag` object that defines a multivariate normal distribution over the
// action space (see the definition in `a2c_utils.hpp`), and also an estimate of the value
// function for the input.
// -----------------------------------------------------------------------------
struct PolicyImpl : torch::nn::Module {
  // Define a model to represent the actor and critic.
  std::vector<torch::nn::Linear> trunk_layers;
  torch::nn::Linear action_mean_head{nullptr};
  torch::nn::Linear action_std_head{nullptr};
  torch::nn::Linear value_head{nullptr};

  /**
   * Args:
   *     state_dim: state dimension
   *     action_dim: action dimension
   */
  PolicyImpl(int state_dim, int action_dim) {
    // Embedding layers.
    trunk_layers = {register_module("trunk0", torch::nn::Linear(state_dim, 128)),
                    register_module("trunk1", torch::nn::Linear(128, 128))};

    // Actor's layers.
    action_mean_head = register_module("action_mean_head", torch::nn::Linear(128, action_dim));
    action_std_head = register_module("action_std_head", torch::nn::Linear(128, action_dim));

    // Critic's layers.
    value_head = register_module("value_head", torch::nn::Linear(128, 1));
  }

  /**
   * Evaluate the policy.
   *
   * Args:
   *     x: input to evaluate, shape (state_dim,) or a batch of shape (batch, state_dim)
   *
   * Returns:
   *     A pair with the first element being a MultivariateNormalDiag representing
   *     the policy for the input and the second element being the value function
   *     estimate for the input.
   */
  std::pair<MultivariateNormalDiag, torch::Tensor> forward(torch::Tensor x) {
    // ##### YOUR CODE STARTS HERE #####
    // Create a model with two linear embedding layers with ReLU activations
    // that feed into the action_mean head, standard deviation head, and value head.
    // Hint: use torch::softplus for the standard deviation head.
    // Hint: make sure to use all of the Policy member variables defined above
    throw pora::NotImplementedError("Need to implement code here.");
    // ###### YOUR CODE ENDS HERE ######
  }
};
TORCH_MODULE(Policy);

// -----------------------------------------------------------------------------
// ### Exercise 3.2 + 3.3: Compute Episode Returns and Training Loss Function
// Implement the following functions:
// 1. `compute_returns`: a function to compute the discounted tail returns for an episode
// 2. `train_loss_for_epsiode`: a function to compute the training loss for an episode
// -----------------------------------------------------------------------------

/**
 * Compute the discounted returns from a sequence of rewards:
 *   rewards = [r_0, r_1, ..., r_{T-1}].
 *
 * Specifically, compute the list of discounted tail returns, [G_0, G_1, ..., G_{T-1}]
 * where:
 *   G_t = sum_{k=t}^{T-1} γ^{k-t} r_k
 *
 * Args:
 *     rewards: array of rewards, [r_0, r_1, ..., r_{T-1}]
 *     discount_factor: temporal discount factor for rewards
 *
 * Returns:
 *     Array of returns, [G_0, G_1, ..., G_{T-1}]
 */
VectorXd compute_returns(const std::vector<double>& rewards, double discount_factor) {
  // ##### YOUR CODE STARTS HERE #####
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
}

/**
 * Compute the loss function for the given episode data. Uses Monte Carlo estimates
 * of the value function via the `returns` input to compute the advantage function.
 *
 * Args:
 *     policy: policy model
 *     states: states for the episode, shape (max_steps, state_dim)
 *     actions: actions taken in the episode, shape (max_steps, action_dim)
 *     returns: discounted tail returns for the episode, shape (max_steps,)
 *     num_steps: number of steps in the episode
 *
 * Returns:
 *     The actor loss + critic loss.
 */
torch::Tensor train_loss_for_epsiode(Policy& policy, const torch::Tensor& states, const torch::Tensor& actions,
                                     const torch::Tensor& returns, int num_steps) {
  torch::Tensor mask = (torch::arange(states.size(0)) < num_steps).to(torch::kFloat32);
  // ##### YOUR CODE STARTS HERE #####
  // Hint: the policy accepts a batch of states directly (no need for the counterpart of jax.vmap)
  // Hint: when you compute the actor loss, use advantages.detach() (the counterpart of
  // jax.lax.stop_gradient(advantages)) to make sure the gradients of the critic model are not
  // computed in the actor loss.
  // Hint: use the `mask` variable on the computed advantages since the states,
  // actions, and returns are padded but only the first num_steps values are valid.
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
}

/**
 * Use the function `train_loss_for_epsiode` to update the model parameters.
 *
 * Args:
 *     optimizer: optimizer
 *     policy: policy model
 *     states: states for the episode
 *     actions: actions taken in the episode
 *     returns: discounted tail returns for the episode
 *     num_steps: number of steps in the episode
 */
void train_step_for_episode(torch::optim::Optimizer& optimizer, Policy& policy, const torch::Tensor& states,
                            const torch::Tensor& actions, const torch::Tensor& returns, int num_steps) {
  torch::Tensor loss = train_loss_for_epsiode(policy, states, actions, returns, num_steps);
  optimizer.zero_grad();
  loss.backward();
  optimizer.step();
}

int main() {
  return pora::run([] {
    // -------------------------------------------------------------------------
    // ### Exercise 3.4: Train Model
    // Run the code below to train the model. Note an interesting practical aspect of this
    // training process is that we "standardize" the returns computed by `compute_returns`
    // before passing them to the training step. This standardization occurs using an
    // exponential moving average that adapts over time to how well the model has already
    // learned to perform. This can help with overall training stability. Try removing this
    // standardization to see how it affects the training performance.
    // -------------------------------------------------------------------------

    // Training parameters.
    const double discount_factor = 0.99;  // Discount factor for computing tail returns.
    const double ema_factor = 0.99;       // Exponential moving average for standardizing returns.
    torch::manual_seed(0);                // Random seed for NN initialization, action sampling.
    const int max_steps = 600;            // Max number of steps in a given episode
    const int num_episodes = 300;         // Number of episodes to train over
    const int num_renders = 4;            // Render video of the lunar lander this many times
    const bool show_animation = false;

    // Define lunar lander environment
    LunarLander lunar_lander;

    // Define policy object
    Policy policy(lunar_lander.state_dim, lunar_lander.action_dim);
    torch::optim::Adam optimizer(policy->parameters(), torch::optim::AdamOptions(1e-3));

    // Train
    std::vector<double> episodic_rewards_ema;  // Exponential moving average of episodic rewards.
    double return_ema = 0;  // Exponential moving average of returns (i.e., critic targets).
    double return_emv = 0;  // Exponential moving variance of returns (i.e., critic targets).
    const int render_interval = num_episodes / num_renders;
    for (int i_episode = 0; i_episode <= num_episodes; ++i_episode) {
      std::string video_filename =
          (i_episode % render_interval == 0 && show_animation)
              ? lunar_lander.video_file_directory + "/lunar_lander_episode_" + std::to_string(i_episode) + ".mp4"
              : "";

      // Reset environment at the start of each episode, and clear accumulators.
      std::vector<float> state = lunar_lander.env.reset();
      double episodic_reward = 0;
      std::vector<torch::Tensor> states, actions;
      std::vector<double> rewards;
      cv::VideoWriter video;

      // Sample a trajectory (episode) according to our stochastic policy/environment dynamics.
      for (int t = 0; t < max_steps; ++t) {
        torch::Tensor s = torch::tensor(state);
        states.push_back(s);
        torch::Tensor action;
        {
          torch::NoGradGuard no_grad;
          auto [action_distribution, value] = policy->forward(s);
          action = action_distribution.sample();
        }
        std::vector<float> a(action.data_ptr<float>(), action.data_ptr<float>() + action.numel());
        LunarLanderStep res = lunar_lander.env.step(a);
        state = res.observation;
        bool done = res.terminated;
        episodic_reward += res.reward;
        actions.push_back(action);
        rewards.push_back(res.reward);

        // Render video at specified interval of episodes
        if (!video_filename.empty()) {
          if (t == 0) video.open(video_filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 50, cv::Size(600, 400));
          video.write(lunar_lander.env.render());
        }
        if (done) break;
      }

      if (!video_filename.empty()) {
        video.release();
        std::cout << "[saved] " << video_filename << std::endl;
      }

      // Compute (standardized) tail returns for the episode and update moving averages.
      VectorXd returns = compute_returns(rewards, discount_factor);
      const double returns_mean = returns.mean();
      if (i_episode == 0) {
        episodic_rewards_ema.push_back(episodic_reward);
        return_ema = returns_mean;
        return_emv = (returns.array() - returns_mean).square().mean();
      } else {
        episodic_rewards_ema.push_back(0.95 * episodic_rewards_ema.back() + (1 - 0.95) * episodic_reward);
        return_ema = ema_factor * return_ema + (1 - ema_factor) * returns_mean;
        return_emv = ema_factor * return_emv + (1 - ema_factor) * (returns.array() - return_ema).square().mean();
      }

      // Note: standardizing returns using moving population statistics is reminiscent of batch normalization.
      VectorXd standardized_returns = (returns.array() - return_ema) / (std::sqrt(return_emv) + 1e-6);

      // Run a train step based on the episode's data.
      const int num_steps = static_cast<int>(states.size());
      // As in the JAX version, we pad all arrays to the same shape (max_steps).
      torch::Tensor states_t = torch::zeros({max_steps, lunar_lander.state_dim});
      torch::Tensor actions_t = torch::zeros({max_steps, lunar_lander.action_dim});
      torch::Tensor returns_t = torch::zeros({max_steps});
      states_t.narrow(0, 0, num_steps).copy_(torch::stack(states));
      actions_t.narrow(0, 0, num_steps).copy_(torch::stack(actions));
      for (int t = 0; t < num_steps; ++t) returns_t[t] = standardized_returns[t];
      train_step_for_episode(optimizer, policy, states_t, actions_t, returns_t, num_steps);

      // Periodically log results.
      if (i_episode % 10 == 0)
        std::printf("Episode %d\tLast reward: %.2f\tMoving average reward: %.2f\n", i_episode, episodic_reward,
                    episodic_rewards_ema.back());
    }

    // Plot EMA rewards
    pora::plt::Figure fig(1, 1, 1000, 800);
    fig.ax().plot(episodic_rewards_ema);
    fig.ax().set_xlabel("Episode");
    fig.ax().set_ylabel("Total Reward (exponential moving average)");
    fig.ax().set_title("Total Reward per Episode");
    fig.save("episodic_rewards_ema.png");
  });
}
