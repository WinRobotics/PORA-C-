// C++ port of ch18/exercises/q_learning.py
#include "q_learning.hpp"

#include <pora/core.hpp>

#include <algorithm>
#include <cmath>
#include <iterator>

QLearning::QLearning(int state_dim, int action_dim, int hidden_dim, int buffer_length, int eps_decay, bool use_gpu)
    : Agent(state_dim, action_dim, "q_learning", use_gpu),
      hidden_dim(hidden_dim),
      buffer_length(buffer_length),
      eps_decay(eps_decay) {
  policy_network = build_network();
  policy_network->to(device);
  initial_state_dict = clone_parameters(*policy_network);
}

void QLearning::reset() {
  load_parameters(*policy_network, initial_state_dict);
  buffer.clear();
  iteration = 0;
}

double QLearning::eps_threshold() {
  const double eps_start = 0.9, eps_end = 0.05;
  iteration += 1;
  return eps_end + (eps_start - eps_end) * std::exp(-1.0 * iteration / eps_decay);
}

torch::nn::Sequential QLearning::build_network() {
  // ##### YOUR CODE STARTS HERE #####
  // Hint: use torch::nn::Sequential with torch::nn::Linear and torch::nn::ReLU
  // Hint: use state_dim, hidden_dim and action_dim
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
}

int QLearning::policy(const State& state) {
  torch::Tensor s = to_tensor(state).to(device);
  torch::NoGradGuard no_grad;
  return static_cast<int>(torch::argmax(policy_network->forward(s), 1).item<int64_t>());
}

torch::Tensor QLearning::policy_train(const State& state_in) {
  torch::Tensor state = to_tensor(state_in).to(device);
  // ##### YOUR CODE STARTS HERE #####
  // Hint: use std::uniform_real_distribution<double>(0, 1)(random) (the counterpart of
  //       random.random()) and eps_threshold() for eps-greedy
  // Hint: torch::randint, torch::argmax, and torch::NoGradGuard are useful
  // Hint: return an int64 tensor of shape (1, 1)
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
}

torch::Tensor QLearning::compute_target(const torch::Tensor& reward, const torch::Tensor& next_state) {
  // ##### YOUR CODE STARTS HERE #####
  // Hint: use torch::NoGradGuard and torch::max; next_state.defined() is false for terminal states
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
}

void QLearning::add_to_buffer(const State& state, int action, double reward, const State& next_state,
                              bool terminal) {
  Transition tr;
  tr.state = to_tensor(state);
  tr.action = action;
  tr.reward = torch::tensor({static_cast<float>(reward)});
  if (!terminal) tr.next_state = to_tensor(next_state);
  buffer.push_back(tr);
  if (static_cast<int>(buffer.size()) > buffer_length) buffer.pop_front();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> QLearning::sample_buffer() {
  TORCH_CHECK(static_cast<int>(buffer.size()) > batch_size);
  std::vector<Transition> samples;
  std::sample(buffer.begin(), buffer.end(), std::back_inserter(samples), batch_size, random);
  std::shuffle(samples.begin(), samples.end(), random);

  std::vector<torch::Tensor> states, targets;
  std::vector<int64_t> actions;
  for (int i = 0; i < batch_size; ++i) {
    torch::Tensor s = samples[i].state.to(device);
    int a = samples[i].action;
    torch::Tensor r = samples[i].reward.to(device);
    torch::Tensor sp = samples[i].next_state.defined() ? samples[i].next_state.to(device) : torch::Tensor();
    states.push_back(s);
    actions.push_back(a);
    targets.push_back(compute_target(r, sp));
  }

  return {torch::cat(states), torch::tensor(actions, torch::kInt64).to(device).unsqueeze(1),
          torch::cat(targets).unsqueeze(1)};
}

void QLearning::train(Cartpole& cartpole, int num_episodes) {
  reset();  // reset model params, replay buffer, epsilon-greedy status
  cartpole.env.reset(0);

  // Define optimizer and loss function
  torch::optim::AdamW optimizer(policy_network->parameters(), torch::optim::AdamWOptions(learning_rate));
  torch::nn::MSELoss loss_function;

  std::vector<double> reward_history;
  std::vector<double> avg_q_history;
  std::deque<Transition> eval_batch;
  for (int episode = 0; episode < num_episodes; ++episode) {
    progress(episode, num_episodes, "train");
    // Initialize episode by resetting the environment
    State state = cartpole.env.reset();
    double total_reward = 0;
    bool terminated = false, truncated = false;

    // Run episode
    while (!terminated && !truncated) {
      State next_state;
      double reward = 0;
      // ##### YOUR CODE STARTS HERE #####
      // Perform the following steps here:
      // 1) Evaluate the policy with epsilon greedy with policy_train()
      // 2) Use cartpole.env.step to compute next_state, reward, terminated, and truncated
      // 3) Add the transition to the buffer using add_to_buffer()
      throw pora::NotImplementedError("Need to implement code here.");
      // ###### YOUR CODE ENDS HERE ######

      state = next_state;
      total_reward += reward;

      // If the buffer is not yet big enough, skip
      if (static_cast<int>(buffer.size()) <= batch_size) {
        eval_batch = buffer;
        continue;
      }

      torch::Tensor loss;
      // ##### YOUR CODE STARTS HERE #####
      // Sample a mini-batch of transitions from the buffer using sample_buffer()
      // then compute the `loss` using loss_function defined above.
      // Hint: when evaluating policy_network->forward() on the minibatch of states,
      // you can use .gather(1, actions) to get the elements according to the sampled
      // actions.
      throw pora::NotImplementedError("Need to implement code here.");
      // ###### YOUR CODE ENDS HERE ######

      // Run a gradient descent step
      optimizer.zero_grad();
      loss.backward();
      torch::nn::utils::clip_grad_value_(policy_network->parameters(), 100);
      optimizer.step();
    }

    // Store reward for the episode
    reward_history.push_back(total_reward);

    // For training status visualization, we store the average of the max
    // Q-value for a mini batch of states, which we should see steadily
    // increasing over the episodes
    double value = 0;
    {
      torch::NoGradGuard no_grad;
      for (const auto& sample : eval_batch) {
        torch::Tensor s = sample.state.to(device);
        value += torch::max(policy_network->forward(s)).item<double>() / eval_batch.size();
      }
    }
    avg_q_history.push_back(value);
  }
  progress(num_episodes, num_episodes, "train");

  plot_rewards(avg_q_history, name + "_avg_max_q.png", "Avg Max Q");
  plot_rewards(reward_history, name + "_rewards.png");
}
