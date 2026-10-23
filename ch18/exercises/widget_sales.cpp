// =============================================================================
// Exercise 1: Q-learning Widget Sales
// (C++ port of ch18/exercises/widget_sales.ipynb)
// =============================================================================
#include <pora/core.hpp>
#include <pora/numerics.hpp>
#include <pora/plot.hpp>
#include <pora/random.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Below we define a simple `WidgetShop` class that defines the true shop dynamics and demand
// distributions, as well as the reward function.
struct WidgetShop {
  static constexpr int num_states = 6;
  static constexpr int num_actions = 3;
  static inline const std::vector<int> X = {0, 1, 2, 3, 4, 5};                // state space
  static inline const std::vector<int> U = {0, 2, 4};                         // action space
  static inline const std::vector<int> D = {0, 1, 2, 3, 4};                   // demand space
  static inline const std::vector<double> P = {0.1, 0.3, 0.3, 0.2, 0.1};    // daily demand distribution

  /** Maps an action in WidgetShop::U to an index. */
  static int action_idx(int u) { return u / 2; }

  /** Compute the next state given the current state, action, and demand. */
  static int step(int x, int u, int d) { return std::clamp(x + u - d, 0, 5); }

  /** Compute the reward given the current state, action, and demand. */
  static double reward(int x, int u, int d) {
    const double price = 1.2;
    const double cost_rent = 1.;
    const double cost_storage = 0.05 * x;
    const double cost_order = std::sqrt(u);
    return price * std::min(x + u, d) - cost_rent - cost_storage - cost_order;
  }

  /**
   * Simulate widget sales for a given policy.
   *
   * Args:
   *     policy: policy to simulate, callable as policy(x)
   *     T: number of time steps to simulate
   *     x0: initial inventory
   *
   * Returns:
   *     State trajectory, shape (T + 1,)
   *     Actions at each time step, shape (T,)
   *     Rewards at each time step, shape (T,)
   */
  static std::tuple<std::vector<int>, std::vector<int>, VectorXd> simulate(const std::function<int(int)>& policy,
                                                                           int T, int x0 = 5) {
    std::vector<int> x(T + 1, 0);  // states
    std::vector<int> u(T, 0);      // actions
    VectorXd r = VectorXd::Zero(T);  // rewards
    x[0] = x0;                     // initial state
    pora::Rng rng(0);              // for reproducibility
    for (int t = 0; t < T; ++t) {
      // Sample demand
      int d = D[rng.choice(P)];

      // Record action, reward, and next state
      u[t] = policy(x[t]);
      r[t] = reward(x[t], u[t], d);
      x[t + 1] = step(x[t], u[t], d);
    }
    return {x, u, r};
  }
};

// Random generator shared by the data generation and Q-learning (like the notebook's `rng`).
// NOTE: the notebook seeds it with 0 like the demand generator inside WidgetShop::simulate.
// With C++'s generators, two identically seeded streams would make the random actions
// correlated with the sampled demand (biasing the logged data), so a different seed is used.
pora::Rng rng(1);

// -----------------------------------------------------------------------------
// ### Exercise 1.1: Model-free Q-learning
// In this first exercise, you will use a dataset of past widget sales to learn tabulated
// Q-values without the use of a model of the environment. Specifically, in the code below
// implement the function `q_learning` to implement a tabulated Q-learning algorithm given the
// input data.
// -----------------------------------------------------------------------------

/**
 * Tabulated Q-learning for the widget shop, given logged data.
 *
 * Args:
 *     x_log: log of state values, shape (T + 1,)
 *     u_log: log of actions, shape (T,)
 *     r_log: log of rewards, shape (T,)
 *     gamma: discount factor
 *     alpha: learning rate
 *     num_epochs: number of epochs of learning
 *
 * Returns:
 *     Q-values after each epoch, num_epochs + 1 arrays of shape (num_states, num_actions)
 */
std::vector<MatrixXd> q_learning(const std::vector<int>& x_log, const std::vector<int>& u_log, const VectorXd& r_log,
                                 double gamma, double alpha, int num_epochs) {
  // Initialize action values to zero
  MatrixXd Q = MatrixXd::Zero(WidgetShop::num_states, WidgetShop::num_actions);
  std::vector<MatrixXd> Q_epoch(num_epochs + 1, MatrixXd::Zero(WidgetShop::num_states, WidgetShop::num_actions));
  const int T = static_cast<int>(u_log.size());

  for (int k = 1; k <= num_epochs; ++k) {
    // Shuffle transition tuple indices
    std::vector<int> shuffled_indices = rng.permutation(T);

    // ##### YOUR CODE STARTS HERE #####
    // Hint: You can use WidgetShop::action_idx() to convert the action to a tabular index
    // Do a Q-update for each transition tuple
    throw pora::NotImplementedError("Need to implement code here.");
    // ###### YOUR CODE ENDS HERE ######

    // Record Q-values for this epoch
    Q_epoch[k] = Q;
  }
  return Q_epoch;
}

// -----------------------------------------------------------------------------
// ### Exercise 1.2: Action-Value Iteration
// We will now leverage the model of the widget shop supply and demand to compute the Q-value
// using value iteration. Finish the function `action_value_iteration` below to compute the
// Q-values using value iteration. Then, run the code below to produce plots comparing the
// Q-values from value iteration and the values computed above using the model-free data
// approach.
// -----------------------------------------------------------------------------

/**
 * Action-value iteration for the widget shop, directly using the model from WidgetShop.
 *
 * Args:
 *     gamma: discount factor
 *     eps: convergence threshold
 *     max_iters: max number of iterations
 *
 * Returns:
 *     Optimal action-value function, shape (num_states, num_actions)
 */
MatrixXd action_value_iteration(double gamma, double eps = 1e-4, int max_iters = 500) {
  bool converged = false;
  MatrixXd Q = MatrixXd::Zero(WidgetShop::num_states, WidgetShop::num_actions);
  MatrixXd Q_prev = MatrixXd::Constant(Q.rows(), Q.cols(), pora::kInf);
  for (int k = 0; k < max_iters; ++k) {
    // ##### YOUR CODE STARTS HERE #####
    throw pora::NotImplementedError("Need to implement code here.");
    // ###### YOUR CODE ENDS HERE ######

    if ((Q - Q_prev).cwiseAbs().maxCoeff() < eps) {
      converged = true;
      std::printf("Value iteration converged after %d iterations.\n", k);
      break;
    } else {
      Q_prev = Q;
    }
  }

  if (!converged) throw std::runtime_error("Value iteration did not converge!");

  return Q;
}

int main() {
  return pora::run([] {
    // Run the code below to compare the Q-values from Q-learning and value iteration. What do you
    // notice about the difference, and why might this occur?

    // Generate historical data with a uniformly random policy
    int T = 3 * 365;
    auto random_policy = [](int) { return WidgetShop::U[rng.integers(0, WidgetShop::num_actions)]; };
    auto [x_log, u_log, r_log] = WidgetShop::simulate(random_policy, T);

    // Q-learning
    const double gamma = 0.95;  // discount factor
    const double alpha = 1e-2;  // learning rate
    const int num_epochs = 5 * static_cast<int>(1 / alpha);  // number of epochs
    std::vector<MatrixXd> Q_ql_epoch = q_learning(x_log, u_log, r_log, gamma, alpha, num_epochs);
    MatrixXd Q_ql = Q_ql_epoch.back();

    // Value iteration
    MatrixXd Q_vi = action_value_iteration(gamma);

    // Plot Q-values for each epoch
    {
      pora::plt::Figure fig(2, WidgetShop::num_states / 2, 1500, 750);
      for (int i = 0; i < WidgetShop::num_states; ++i) {
        auto& ax = fig.ax(i);
        for (int j = 0; j < WidgetShop::num_actions; ++j) {
          std::vector<double> q(num_epochs + 1);
          for (int k = 0; k <= num_epochs; ++k) q[k] = Q_ql_epoch[k](i, j);
          pora::plt::Color c = pora::plt::cycle_color(j);
          ax.plot(q, pora::plt::fmt("-").c(c).label("u = " + std::to_string(WidgetShop::U[j])));
          ax.axhline(Q_vi(i, j), pora::plt::fmt("--").c(c));
        }
        ax.legend("lower right");
        ax.set_title("x = " + std::to_string(WidgetShop::X[i]));
        if (i >= WidgetShop::num_states / 2) ax.set_xlabel("epoch");
        if (i % (WidgetShop::num_states / 2) == 0) ax.set_ylabel("Q(x,u)");
      }
      fig.save("q_values.png");
    }

    // -------------------------------------------------------------------------
    // ### Exercise 1.3: Simulate Q-learning and Value Iteration Optimal Policies
    // Write code below to compute:
    // 1. `pi_ql(x)` and `pi_vi(x)`: lambda functions, using the corresponding Q-value arrays
    //    computed above.
    // 2. `r_ql` and `r_vi`: arrays of rewards from simulateing the shop with the above
    //    policies for 5 years.
    // 3. `profit_ql` and `profit_vi`: arrays of cumulative profits for each day over 5 years
    //    (can use `pora::cumsum`)
    //
    // What do you notice about the cumulative profits from the two methods? Why might there be
    // a difference between the non-discount and discounted cumulative profits? Try playing
    // around with the discount factor to see how this affects the results.
    // -------------------------------------------------------------------------

    // Simulate optimal policies for 5 years
    T = 5 * 365;

    std::function<int(int)> pi_ql, pi_vi;
    VectorXd r_ql, r_vi, profit_ql, profit_vi;
    // ##### YOUR CODE STARTS HERE #####
    // Hint: use Q_ql and Q_vi from above
    throw pora::NotImplementedError("Need to implement code here.");
    // ###### YOUR CODE ENDS HERE ######

    std::string s_ql, s_vi;
    for (int x : WidgetShop::X) {
      s_ql += std::to_string(pi_ql(x)) + " ";
      s_vi += std::to_string(pi_vi(x)) + " ";
    }
    std::printf("Optimal policy (Q-learning):      [%s]\n", s_ql.c_str());
    std::printf("Optimal policy (value iteration): [%s]\n", s_vi.c_str());

    // Plot results
    {
      pora::plt::Figure fig(1, 1, 700, 500);
      fig.ax().plot(profit_ql, "-", "Q-learning");
      fig.ax().plot(profit_vi, "-", "value iteration");
      fig.ax().legend("lower right");
      fig.ax().set_xlabel("day t");
      fig.ax().set_ylabel("cumulative profit sum_{k<=t} r_k");
      fig.save("cumulative_profit.png");
    }

    // Repeat with discounted profits
    VectorXd discounts(T);
    for (int t = 0; t < T; ++t) discounts[t] = std::pow(gamma, t);
    VectorXd profit_ql_discounted = pora::cumsum(discounts.cwiseProduct(r_ql));
    VectorXd profit_vi_discounted = pora::cumsum(discounts.cwiseProduct(r_vi));
    {
      pora::plt::Figure fig(1, 1, 700, 500);
      fig.ax().plot(profit_ql_discounted, "-", "Q-learning");
      fig.ax().plot(profit_vi_discounted, "-", "value iteration");
      fig.ax().legend("lower right");
      fig.ax().set_xlabel("day t");
      fig.ax().set_ylabel("cumulative discounted profit sum_{k<=t} gamma^k r_k");
      fig.save("cumulative_discounted_profit.png");
    }
  });
}
