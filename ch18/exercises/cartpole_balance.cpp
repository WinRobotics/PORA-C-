// =============================================================================
// Exercise 2: Cart-pole Balance
// (C++ port of ch18/exercises/cartpole_balance.ipynb, using LibTorch)
// =============================================================================
#include "cartpole.hpp"
#include "q_learning.hpp"
#include "reinforce.hpp"

#include <pora/core.hpp>

#include <torch/torch.h>

int main() {
  return pora::run([] {
    torch::manual_seed(0);

    // In this exercise, we will use model-free reinforcement learning methods to stabilize a
    // cart-pole system. To start, run the code below to create a Cartpole system and
    // instantiate a basic agent whose policy is to simply move the cart in the direction the
    // pole is leaning. We shouldn't expect this naive policy to perform well. The code below
    // will simulate episodes of this policy and compute some reward statistics. We will use
    // this naive policy as a baseline to compare our other methods.
    Cartpole cartpole;
    cartpole.print_environment_info();
    Basic basic_agent(cartpole.state_dim, cartpole.action_dim);
    int num_episodes = 10;
    cartpole.simulate(basic_agent, num_episodes, true);

    // -------------------------------------------------------------------------
    // ### Exercise 2.1: Deep Q-learning
    // In the file `q_learning.cpp`, implement the functions `QLearning::build_network`,
    // `QLearning::policy_train`, `QLearning::compute_target`, and `QLearning::train` to train
    // a policy using deep Q-learning with experience replay. Follow the instructions in the
    // code comments. Then, run the code below to train a Q-learning agent and simulate to see
    // how well the policy performs.
    //
    // There should be two plots, one shows the average max_a Q(s,a) for a mini-batch of
    // states, which is one indicator that training is progressing. The other shows the total
    // reward for each episode in the training process. Notice that the total reward itself is
    // quite noisy from episode to episode.
    //
    // Try playing around in the code with different hyper parameters, such as the episode
    // length, epsilon-greedy parameters, or the replay buffer max length. How does the training
    // differ?
    // -------------------------------------------------------------------------
    {
      // Create Q-learning agent and train it
      int buffer_length = 1000;
      int eps_decay = 10000;
      QLearning q_learning_agent(cartpole.state_dim, cartpole.action_dim, 24, buffer_length, eps_decay);
      q_learning_agent.train(cartpole, 100);

      // Evaluate over some episodes and display
      cartpole.simulate(q_learning_agent, 10, true);
    }

    // -------------------------------------------------------------------------
    // ### Exercise 2.2: REINFORCE
    // Now you will implement the REINFORCE algorithm.
    // In the file `reinforce.cpp`, implement the functions `Reinforce::build_network`,
    // `Reinforce::policy_train`, and `Reinforce::train`. Follow the instructions in the code
    // comments. Then, run the code below to train a REINFORCE agent and simulate to see how
    // well the policy performs.
    // Note that in `Reinforce::train` you will implement three different methods for the
    // policy gradient:
    // 1. Episode reward.
    // 2. "Causality trick": tail rewards.
    // 3. "Causality trick" + baseline.
    // -------------------------------------------------------------------------
    {
      // Create REINFORCE agent and train it
      Reinforce reinforce_agent(cartpole.state_dim, cartpole.action_dim, 24);
      reinforce_agent.train(cartpole, 1000, true, true);

      // Evaluate over some episodes and display
      cartpole.simulate(reinforce_agent, 10, true);
    }
  });
}
