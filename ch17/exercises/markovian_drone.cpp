// =============================================================================
// Exercise 3: Markovian Drone
// (C++ port of ch17/exercises/markovian_drone.ipynb)
// =============================================================================
#include <pora/core.hpp>
#include <pora/plot.hpp>
#include <pora/random.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <vector>

using Eigen::MatrixXd;
using Eigen::MatrixXi;

// Minimal dense arrays of rank 3 and 5 (stand-ins for the NumPy arrays of the notebook).
struct Array3 {
  Array3(int d0 = 0, int d1 = 0, int d2 = 0, double value = 0.0)
      : d0(d0), d1(d1), d2(d2), data(static_cast<size_t>(d0) * d1 * d2, value) {}
  double& operator()(int i, int j, int k) { return data[(static_cast<size_t>(i) * d1 + j) * d2 + k]; }
  double operator()(int i, int j, int k) const { return data[(static_cast<size_t>(i) * d1 + j) * d2 + k]; }
  int d0, d1, d2;
  std::vector<double> data;
};

struct Array5 {
  Array5(int d0, int d1, int d2, int d3, int d4, double value = 0.0)
      : d0(d0), d1(d1), d2(d2), d3(d3), d4(d4), data(static_cast<size_t>(d0) * d1 * d2 * d3 * d4, value) {}
  size_t index(int i, int j, int k, int l, int m) const {
    return (((static_cast<size_t>(i) * d1 + j) * d2 + k) * d3 + l) * d4 + m;
  }
  double& operator()(int i, int j, int k, int l, int m) { return data[index(i, j, k, l, m)]; }
  double operator()(int i, int j, int k, int l, int m) const { return data[index(i, j, k, l, m)]; }
  int d0, d1, d2, d3, d4;
  std::vector<double> data;
};

// -----------------------------------------------------------------------------
// ### Exercise 3.1: Dynamic Programming
// Implement the remaining functions of `DroneMDP` to compute the optimal policy via dynamic
// programming. Specifically, implement the functions:
// 1. `compute_candidate_cost_to_go`: computes the candidate expected cost-to-go,
//    E_w[g_t(x_t, u_t) + J^*_{t+1}(f_t(x_t, u_t, w_t))]
// 2. `dynamic_programming`: performs dynamic programming to compute the optimal cost-to-go
//    values
// 3. `optimal_policy_sequence`: extracts the optimal policy sequence from the optimal
//    cost-to-go
// -----------------------------------------------------------------------------
class DroneMDP {
 public:
  DroneMDP() : P(20, 20, 5, 20, 20), G_t(20, 20, 5, 1.0) {
    n = 20;              // grid size
    x_goal << 19, 9;     // goal location

    // Storm params
    x_eye << 15, 15;     // storm center
    sigma = 10;          // storm strength standard deviation

    // Control input set: 0 = `up`, 1 = `down`, 2 = `left`, 3 = `right`, 4 = `land`
    m = 5;  // number of control inputs

    // Enumerate the transition probabilities for each state and control, from
    // each possible current state
    // NOTE: `p(x_next | x, u) = P(x[0], x[1], u, x_next[0], x_next[1])`
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        // Start by identifying the set of all possible next states
        auto clip = [&](int v) { return std::min(std::max(v, 0), n - 1); };  // clip to grid
        const int X_next[5][2] = {
            {clip(i), clip(j + 1)},  // right
            {clip(i), clip(j - 1)},  // left
            {clip(i - 1), clip(j)},  // up
            {clip(i + 1), clip(j)},  // down
            {clip(i), clip(j)},      // land
        };

        // If the drone is "caught" by the storm, it may move in any given
        // direction with probability `w*0.25` unless it lands
        const double d2 = (i - x_eye[0]) * (i - x_eye[0]) + (j - x_eye[1]) * (j - x_eye[1]);
        const double w = std::exp(-d2 / (2 * (sigma * sigma)));
        for (int u = 0; u < 4; ++u)
          for (int k = 0; k < 4; ++k) P(i, j, u, X_next[k][0], X_next[k][1]) += w * 0.25;

        // Otherwise, the drone moves in the chosen direction with
        // probability `1 - w`
        for (int u = 0; u < 4; ++u) P(i, j, u, X_next[u][0], X_next[u][1]) += 1 - w;

        // If the drone lands, it stays deterministically
        P(i, j, 4, i, j) = 1;
      }
    }

    // Define stage cost matrix, g_t(x,u) = G_t(x[0], x[1], u)
    for (int u = 0; u < m; ++u) G_t(x_goal[0], x_goal[1], u) = 0.;

    // Define terminal cost matrix, g_T(x) = G_T(x[0], x[1])
    G_T = 100 * MatrixXd::Ones(n, n);
    G_T(x_goal[0], x_goal[1]) = 0.;
  }

  /**
   * Simulate the drone MDP given a starting position and a policy.
   *
   * Args:
   *     x_start: starting position, shape (2,)
   *     pi: List of length T of arrays of shape (n, n), for the policies {π_0, ..., π_{T-1}}
   *
   * Returns:
   *     Trajectory array, shape (T, 2)
   */
  MatrixXi simulate(const Eigen::Vector2i& x_start, const std::vector<MatrixXi>& pi) const {
    const int T = static_cast<int>(pi.size());
    pora::Rng rng(0);  // seed RNG for reproducibility
    MatrixXi x = MatrixXi::Zero(T, 2);
    x.row(0) = x_start.transpose();
    for (int t = 0; t < T - 1; ++t) {
      const MatrixXi& pi_t = pi[t];
      int u = pi_t(x(t, 0), x(t, 1));
      std::vector<double> P_next(n * n);
      for (int k = 0; k < n; ++k)
        for (int l = 0; l < n; ++l) P_next[k * n + l] = P(x(t, 0), x(t, 1), u, k, l);
      int flat_index = rng.choice(P_next);
      x(t + 1, 0) = flat_index / n;
      x(t + 1, 1) = flat_index % n;
    }
    return x;
  }

  /**
   * Compute array of candidate cost-to-go values:
   *
   * E_w[g_t(x_t, u_t) + J^*_{t+1}(f_t(x_t, u_t, w_t))],
   *
   * for all x in the statespace.
   *
   * Args:
   *     J: array of the optimal tail cost-to-go values, J^*_{t+1}, for each
   *        state in the grid, shape (n, n)
   *
   * Returns:
   *     Candidate cost-to-go values, shape (n, n, m)
   */
  Array3 compute_candidate_cost_to_go(const MatrixXd& J) const {
    // ##### YOUR CODE STARTS HERE #####
    // Hint: Use G_t and P
    throw pora::NotImplementedError("Need to implement code here.");
    // ###### YOUR CODE ENDS HERE ######
  }

  /**
   * Use dynamic programming to compute the sequence of optimal cost-to-go arrays J^*_t
   * for t = 0, ..., T.
   *
   * Args:
   *     T: horizon length
   *
   * Returns:
   *     Optimal cost-to-go values for each time step, T + 1 arrays of shape (n, n).
   */
  std::vector<MatrixXd> dynamic_programming(int T) const {
    std::vector<MatrixXd> J(T + 1, MatrixXd::Zero(n, n));
    // ##### YOUR CODE STARTS HERE #####
    throw pora::NotImplementedError("Need to implement code here.");
    // ###### YOUR CODE ENDS HERE ######
    return J;
  }

  /**
   * Compute the optimal policy sequence given the optimal cost-to-go
   * sequence J* = {J*_0, ..., J*_{T}}.
   *
   * Args:
   *     J_star: the optimal cost-to-go values at each time step, T + 1 arrays of shape (n, n)
   *
   * Returns:
   *     List where index t is an array of shape (n, n) of the optimal action indices
   */
  std::vector<MatrixXi> optimal_policy_sequence(const std::vector<MatrixXd>& J_star) const {
    // ##### YOUR CODE STARTS HERE #####
    throw pora::NotImplementedError("Need to implement code here.");
    // ###### YOUR CODE ENDS HERE ######
  }

  int n, m;
  Eigen::Vector2i x_goal, x_eye;
  double sigma;
  Array5 P;     // transition probabilities, shape (n, n, m, n, n)
  Array3 G_t;   // stage cost, shape (n, n, m)
  MatrixXd G_T; // terminal cost, shape (n, n)
};

int main() {
  return pora::run([] {
    // -------------------------------------------------------------------------
    // ### Exercise 3.2: Simulate and Plot
    // Run the code below to compute the optimal policy, simulate it, and plot the results.
    // -------------------------------------------------------------------------

    // Compute the optimal cost-to-go sequence and policy via dynamic programming
    const int T = 100;
    DroneMDP mdp;
    std::vector<MatrixXd> J_star = mdp.dynamic_programming(T);
    std::vector<MatrixXi> pi_star = mdp.optimal_policy_sequence(J_star);

    // Simulate the drone from a starting position
    Eigen::Vector2i x_start(0, 19);
    MatrixXi x = mdp.simulate(x_start, pi_star);

    // Plot the optimal cost-to-go values as a heat map
    {
      pora::plt::Figure fig(1, 1, 700, 600);
      fig.ax().imshow(J_star[0].transpose(), true, pora::plt::ColorMap::Viridis, "V*(x)");
      fig.ax().set_xlabel("x_1");
      fig.ax().set_ylabel("x_2");
      fig.save("optimal_cost_to_go.png");
    }

    // Plot the optimal policy as a heat map, and overlay the simulated trajectory
    {
      pora::plt::Figure fig(1, 1, 700, 600);
      auto& ax = fig.ax();
      ax.imshow(pi_star[0].cast<double>().transpose(), true, pora::plt::ColorMap::Viridis, "pi*(x)");
      ax.plot(std::vector<double>{double(mdp.x_goal[0])}, std::vector<double>{double(mdp.x_goal[1])},
              pora::plt::fmt("*").c(pora::plt::colors::black).ms(14));
      ax.plot(std::vector<double>{double(mdp.x_eye[0])}, std::vector<double>{double(mdp.x_eye[1])},
              pora::plt::fmt("^").c(pora::plt::colors::tab_orange).ms(12));
      ax.plot(std::vector<double>{double(x_start[0])}, std::vector<double>{double(x_start[1])},
              pora::plt::fmt("o").c(pora::plt::colors::tab_green).ms(12));
      ax.plot(x.col(0).cast<double>(), x.col(1).cast<double>(), pora::plt::fmt("-").c(pora::plt::colors::tab_red));
      ax.set_xlabel("x_1");
      ax.set_ylabel("x_2");
      fig.save("optimal_policy.png");
    }
  });
}
