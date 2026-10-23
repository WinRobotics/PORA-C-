// C++ port of ch18/exercises/reinforce.py
#include "reinforce.hpp"

#include <pora/core.hpp>

#include <numeric>

Reinforce::Reinforce(int state_dim, int action_dim, int hidden_dim)
    : Agent(state_dim, action_dim, "reinforce"), hidden_dim(hidden_dim) {
  policy_network = build_network();
  initial_state_dict = clone_parameters(*policy_network);
  optimizer = std::make_unique<torch::optim::Adam>(policy_network->parameters());
}

void Reinforce::reset() { load_parameters(*policy_network, initial_state_dict); }

torch::nn::Sequential Reinforce::build_network() {
  // ##### YOUR CODE STARTS HERE #####
  // Hint: use torch::nn::Sequential
  // Hint: use state_dim, hidden_dim and action_dim
  // Hint: use torch::nn::Softmax(torch::nn::SoftmaxOptions(1)) for the output
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
}

int Reinforce::policy(const State& state) {
  torch::NoGradGuard no_grad;
  torch::Tensor action_probs = policy_network->forward(to_tensor(state));
  // Sample from the categorical distribution defined by the action probabilities
  return static_cast<int>(torch::multinomial(action_probs, 1).item<int64_t>());
}

std::pair<int, torch::Tensor> Reinforce::policy_train(const State& state_in) {
  torch::Tensor state = to_tensor(state_in);

  // ##### YOUR CODE STARTS HERE #####
  // Hint: LibTorch has no torch.distributions.Categorical; sample the action with
  //       torch::multinomial(action_probs, 1) and compute its log probability with
  //       torch::log of the selected probability (keep the gradient!).
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
}

std::pair<std::vector<double>, std::vector<torch::Tensor>> Reinforce::sample_episode(Cartpole& cartpole) {
  std::vector<torch::Tensor> log_probs;
  std::vector<double> rewards;
  bool terminated = false, truncated = false;
  State state = cartpole.env.reset();
  while (!terminated && !truncated) {
    auto [action, log_prob] = policy_train(state);
    StepResult res = cartpole.env.step(action);
    state = res.observation;
    terminated = res.terminated;
    truncated = res.truncated;
    log_probs.push_back(log_prob);
    rewards.push_back(res.reward);
  }
  return {rewards, log_probs};
}

void Reinforce::train(Cartpole& cartpole, int num_episodes, bool use_causality, bool use_baseline) {
  reset();
  cartpole.env.reset(0);

  std::vector<double> reward_history;
  for (int episode = 0; episode < num_episodes; ++episode) {
    progress(episode, num_episodes, "train");
    // Sample cartpole episode and get list of rewards and log probabilities
    // of the actions taken
    auto [rewards, log_probs] = sample_episode(cartpole);
    reward_history.push_back(std::accumulate(rewards.begin(), rewards.end(), 0.0));

    std::vector<double> returns_list;
    // ##### YOUR CODE STARTS HERE #####
    // Compute `returns_list`, a list of discounted returns [R_0, R_1, ...]
    // where R_t = \sum_{t'=t}^{T-1} \gamma^(t'-t) r_t'
    // Hint: use gamma
    throw pora::NotImplementedError("Need to implement code here.");
    // ###### YOUR CODE ENDS HERE ######
    TORCH_CHECK(returns_list.size() == rewards.size());
    torch::Tensor returns = torch::tensor(returns_list, torch::kFloat32);

    std::vector<torch::Tensor> loss_list;
    if (use_causality) {
      if (use_baseline) {
        // ##### YOUR CODE STARTS HERE #####
        // Modify `returns` using a baseline that is the average of the previously
        // computed returns, and scale by standard deviation for normalization.
        throw pora::NotImplementedError("Need to implement code here.");
        // ###### YOUR CODE ENDS HERE ######
      }

      // ##### YOUR CODE STARTS HERE #####
      // Compute the list `loss_list` of each time step loss where instead of using
      // the full episode return we use the tail return only
      throw pora::NotImplementedError("Need to implement code here.");
      // ###### YOUR CODE ENDS HERE ######
    } else {
      // ##### YOUR CODE STARTS HERE #####
      // Compute the list `loss_list` of each time step loss using the vanilla
      // approach where the full episode return is used.
      throw pora::NotImplementedError("Need to implement code here.");
      // ###### YOUR CODE ENDS HERE ######
    }
    torch::Tensor loss = torch::stack(loss_list).sum();

    optimizer->zero_grad();
    loss.backward();
    optimizer->step();
  }
  progress(num_episodes, num_episodes, "train");

  // Visualize reward history
  plot_rewards(reward_history, name + "_rewards.png");
}
