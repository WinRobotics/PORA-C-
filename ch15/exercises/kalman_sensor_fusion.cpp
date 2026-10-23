// =============================================================================
// Exercise 2: Kalman Sensor Fusion
// (C++ port of ch15/exercises/kalman_sensor_fusion.ipynb)
//
// Consider longitudinal car motion with state x = [p, v, a] where p is the position, v is the
// velocity, and a is the acceleration, and the control is the jerk, j. The kinematics for this
// system are:
//     \dot{p} = v, \quad \dot{v} = a, \quad \dot{a} = j,
// which we can discretize exactly with sampling time T as:
//     p_{t+1} = p_t + v_t T + \frac{1}{2}a_tT^2 + \frac{1}{6}jT^3,
//     v_{t+1} = v_t + a_t T + \frac{1}{2}jT^2,
//     a_{t+1} = a_t + j T.
// We also assume there are three sensors we could have on the vehicle:
// 1. An IMU that gives a measurement of acceleration a.
// 2. A GNSS sensor that gives a direct measurement of vehicle position, p.
// 3. A lidar sensor that gives a direct measurement of vehicle position, p, via range to a
//    known object.
// =============================================================================
#include "utils.hpp"

#include <pora/core.hpp>
#include <pora/numerics.hpp>
#include <pora/random.hpp>

#include <Eigen/Dense>

#include <cstdio>
#include <tuple>
#include <utility>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Define process and measurement noises
const MatrixXd Q = VectorXd((VectorXd(3) << 0.0, 0.01, 0.0).finished()).asDiagonal();  // process noise
const double std_imu = 0.25;
const double std_gnss = 5.0;
const double std_lidar = 5.0;

std::pair<VectorXd, MatrixXd> kalman_filter_update(const MatrixXd& A, const MatrixXd& B, const MatrixXd& C,
                                                   const MatrixXd& Q, const MatrixXd& R, VectorXd mean, MatrixXd cov,
                                                   const VectorXd& u, const VectorXd& z);

/**
 * Simulate the linear system and KF estimation. Compute RSME
 * to estimate error of the filter across time and over a batch of runs.
 *
 * Args:
 *     x0: initial state of the system, shape (n,)
 *     u: controls to apply at each time step, shape (m, N)
 *     A: dynamics matrix, shape (n, n)
 *     B: control matrix, shape (n, m)
 *     C: measurement matrix, shape (o, n)
 *     Q: process noise covariance, shape (n, n)
 *     R: measurement noise covariance, shape (o, o)
 *     mean0: initial KF mean, shape (n,)
 *     cov0: initial KF covariance, shape (n, n)
 *     num_sims: number of simulations to batch for RMSE calc
 *     bias: bias to the measurement, or an empty vector to not add a bias
 *
 * Returns:
 *     Actual state trajectory from the last batch run, shape (n, N + 1).
 *     KF mean estimate trajectory from the last batch, shape (n, N + 1).
 *     Average RMSE over the batch of runs.
 */
std::tuple<MatrixXd, MatrixXd, double> simulate(const VectorXd& x0, const MatrixXd& u, const MatrixXd& A,
                                                const MatrixXd& B, const MatrixXd& C, const MatrixXd& Q,
                                                const MatrixXd& R, const VectorXd& mean0, const MatrixXd& cov0,
                                                int num_sims, const VectorXd& bias = VectorXd()) {
  double total_pos_rmse = 0;
  MatrixXd x, mean;
  for (int run = 0; run < num_sims; ++run) {
    const int n = static_cast<int>(x0.size());
    const int o = static_cast<int>(C.rows());
    const int N = static_cast<int>(u.cols());
    x = MatrixXd::Zero(n, N + 1);
    x.col(0) = x0;
    mean = MatrixXd::Zero(n, N + 1);
    mean.col(0) = mean0;
    MatrixXd cov = cov0;
    pora::Rng rng(run);  // for determinism
    for (int i = 0; i < N; ++i) {
      // Simulate dynamics
      VectorXd eps = rng.multivariate_normal(VectorXd::Zero(Q.rows()), Q);
      x.col(i + 1) = A * x.col(i) + B * u.col(i) + eps;

      // Simulate measurement
      VectorXd delta = rng.multivariate_normal(VectorXd::Zero(R.rows()), R);
      VectorXd b = bias.size() == 0 ? VectorXd::Zero(o) : bias;
      VectorXd z = C * x.col(i) + b + delta;

      // Estimation
      auto [mean_next, cov_next] = kalman_filter_update(A, B, C, Q, R, mean.col(i), cov, u.col(i), z);
      mean.col(i + 1) = mean_next;
      cov = cov_next;
    }

    // Position error metric
    total_pos_rmse += rmse(x.row(0).transpose(), mean.row(0).transpose());
  }
  return {x, mean, total_pos_rmse / num_sims};
}

/**
 * Kalman filter update step.
 *
 * Args:
 *     A: dynamics matrix, shape (n, n)
 *     B: control matrix, shape (n, m)
 *     C: measurement matrix, shape (o, n)
 *     Q: process noise covariance, shape (n, n)
 *     R: measurement noise covariance, shape (o, o)
 *     mean: mean estimate, shape (n,)
 *     cov: estimate covariance, shape (n, n)
 *     u: control input, shape (m,)
 *     z: measurement, shape (o,)
 *
 * Returns:
 *     Posterior mean and covariance.
 */
std::pair<VectorXd, MatrixXd> kalman_filter_update(const MatrixXd& A, const MatrixXd& B, const MatrixXd& C,
                                                   const MatrixXd& Q, const MatrixXd& R, VectorXd mean, MatrixXd cov,
                                                   const VectorXd& u, const VectorXd& z) {
  // ##### YOUR CODE STARTS HERE #####
  // NOTE: Implement Kalman Filtering Predict and Update steps.
  // Write resulting means to mean.
  // Write resulting covariances to cov.
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
  return {mean, cov};
}

int main() {
  return pora::run([] {
    // -------------------------------------------------------------------------
    // ### Exercise 2.1 Implement Kalman Filter
    // First, define matrices A and B for the car's dynamics model described above in discrete
    // time. Use the variable `T` as the sampling time. Then, implement the function
    // `kalman_filter_update` to perform a single step of a Kalman filter update.
    // -------------------------------------------------------------------------
    const double T = 0.01;  // sampling time
    MatrixXd A, B;
    // ##### YOUR CODE STARTS HERE #####
    // Longitudinal car dynamics model (discrete time)
    throw pora::NotImplementedError("Need to implement code here.");
    // ###### YOUR CODE ENDS HERE ######

    // -------------------------------------------------------------------------
    // ### Exercise 2.2: KF Sensor Fusion for Autonomous Car
    // First, run the code below to simulate the system with a provided open-loop control input
    // and see what the estimates look like with no sensors to correct the estimate. We will
    // compute the average RMSE error of the position estimate, and us this to compare across
    // the different solutions later.
    // -------------------------------------------------------------------------

    // Form an interesting jerk input sequence
    std::vector<std::pair<int, double>> segments = {{100, 1}, {150, -1}, {50, 1}, {100, 0},
                                                    {75, -1}, {100, 1},  {100, -0.25}, {100, 0}};
    std::vector<double> jerk;
    for (auto [len, val] : segments) jerk.insert(jerk.end(), len, val);
    MatrixXd u = Eigen::Map<Eigen::RowVectorXd>(jerk.data(), static_cast<int>(jerk.size()));

    // Assume we don't know the exact initial condition
    VectorXd x0(3);
    x0 << 10, 5, 0.1;

    // First run a case where we have no measurements to see the error
    MatrixXd C = MatrixXd::Zero(1, 3);
    MatrixXd R = MatrixXd::Identity(1, 1);

    // Simulate
    const int num_sims = 10;
    VectorXd mean0 = VectorXd::Zero(3);
    MatrixXd cov0 = VectorXd((VectorXd(3) << 100.0, 10.0, 1.0).finished()).asDiagonal();
    auto [x, mean, p_error_none] = simulate(x0, u, A, B, C, Q, R, mean0, cov0, num_sims);
    std::printf("State is observable with sensor config: %s\n", observable(A, C) ? "True" : "False");

    // Compute RMSE for position
    std::printf("Position RMSE: %.3f\n", p_error_none);

    // Plot to see accuracy
    plot_car_estimation(x, mean, T, "no_sensors.png");

    // Now, implement the C and R matrices below to define a setup that has only an IMU sensor.
    // Run the code to see how the position error estimate is impacted. Is this what you would
    // expect? Use the variable `std_imu` for the IMU sensor noise standard deviation.

    // ##### YOUR CODE STARTS HERE #####
    // Define measurement model for IMU only
    throw pora::NotImplementedError("Need to implement code here.");
    // ###### YOUR CODE ENDS HERE ######

    // Simulate
    double p_error_imu;
    std::tie(x, mean, p_error_imu) = simulate(x0, u, A, B, C, Q, R, mean0, cov0, num_sims);
    std::printf("State is observable with sensor config: %s\n", observable(A, C) ? "True" : "False");

    // Compute RMSE for position
    double improvement = 100 * (p_error_imu - p_error_none) / p_error_none;
    std::printf("IMU Position RMSE: %.3f\n", p_error_imu);
    std::printf("RMSE change compared to no sensors: %.2f%%\n", improvement);

    // Plot to see accuracy
    plot_car_estimation(x, mean, T, "imu.png");

    // Repeat the above exercise but now define C and R with an IMU + lidar sensor setup. Use the
    // variables `std_imu` and `std_lidar` for the IMU sensor noise standard deviation. The error
    // should significantly improve!

    // ##### YOUR CODE STARTS HERE #####
    // Define measurement model for IMU + lidar
    throw pora::NotImplementedError("Need to implement code here.");
    // ###### YOUR CODE ENDS HERE ######

    // Simulate
    double p_error_imu_lidar;
    std::tie(x, mean, p_error_imu_lidar) = simulate(x0, u, A, B, C, Q, R, mean0, cov0, num_sims);
    std::printf("State is observable with sensor config: %s\n", observable(A, C) ? "True" : "False");

    // Compute RMSE for position
    improvement = 100 * (p_error_imu_lidar - p_error_none) / p_error_none;
    std::printf("IMU + lidar Position RMSE: %.3f\n", p_error_imu_lidar);
    std::printf("RMSE change compared to no sensors: %.2f%%\n", improvement);

    // Plot to see accuracy
    plot_car_estimation(x, mean, T, "imu_lidar.png");

    // Repeat the above exercise but now define C and R with an IMU + lidar + GNSS sensor setup.
    // Use the variables `std_imu`, `std_lidar`, `std_gnss` for the IMU sensor noise standard
    // deviation. Note that here we assume the GNSS sensor has a constant bias error. How do you
    // think this will effect the position estimate?

    // ##### YOUR CODE STARTS HERE #####
    // Define measurement model for IMU + lidar + GNSS with bias
    throw pora::NotImplementedError("Need to implement code here.");
    // ###### YOUR CODE ENDS HERE ######

    // GNSS bias
    const double gnss_bias = 10.0;
    VectorXd bias(3);
    bias << 0, 0, gnss_bias;

    // Simulate
    double p_error_imu_lidar_gnss;
    std::tie(x, mean, p_error_imu_lidar_gnss) = simulate(x0, u, A, B, C, Q, R, mean0, cov0, num_sims, bias);
    std::printf("State is observable with sensor config: %s\n", observable(A, C) ? "True" : "False");

    // Compute RMSE for position
    improvement = 100 * (p_error_imu_lidar_gnss - p_error_none) / p_error_none;
    double improvement_lidar_only = 100 * (p_error_imu_lidar_gnss - p_error_imu_lidar) / p_error_imu_lidar;
    std::printf("IMU + lidar + GNSS Position RMSE: %.3f\n", p_error_imu_lidar_gnss);
    std::printf("RMSE change compared to no sensors: %.2f%%\n", improvement);
    std::printf("RMSE change compared to IMU + lidar: %.2f%%\n", improvement_lidar_only);

    // Plot to see accuracy
    plot_car_estimation(x, mean, T, "imu_lidar_gnss.png");

    // -------------------------------------------------------------------------
    // ### Exercise 2.3 Augmented State
    // Using the GNSS sensor above with bias has an impact on the ability to accurately estimate
    // the position. In this part of the exercise, implement an augmented state version of the
    // Kalman filter to estimate the GNSS bias. Specifically, define the variables `A_aug`,
    // `B_aug`, `C_aug`, `Q_aug`, and `R_aug`.
    // -------------------------------------------------------------------------
    MatrixXd A_aug, B_aug, C_aug, Q_aug, R_aug;
    // ##### YOUR CODE STARTS HERE #####
    // Define augmented state system
    throw pora::NotImplementedError("Need to implement code here.");
    // ###### YOUR CODE ENDS HERE ######

    // Simulate
    VectorXd x0_aug(4), mean0_aug(4);
    x0_aug << x0, gnss_bias;
    mean0_aug << mean0, 0;
    MatrixXd cov0_aug = pora::block_diag(cov0, 10 * MatrixXd::Identity(1, 1));
    double p_error_imu_lidar_aug_gnss;
    std::tie(x, mean, p_error_imu_lidar_aug_gnss) =
        simulate(x0_aug, u, A_aug, B_aug, C_aug, Q_aug, R_aug, mean0_aug, cov0_aug, num_sims, bias);
    std::printf("State + bias is observable with sensor config: %s\n", observable(A_aug, C_aug) ? "True" : "False");

    // Compute RMSE for position
    improvement = 100 * (p_error_imu_lidar_aug_gnss - p_error_none) / p_error_none;
    improvement_lidar_only = 100 * (p_error_imu_lidar_aug_gnss - p_error_imu_lidar) / p_error_imu_lidar;
    double improvement_lidar_bias_gnss =
        100 * (p_error_imu_lidar_aug_gnss - p_error_imu_lidar_gnss) / p_error_imu_lidar_gnss;
    std::printf("IMU + lidar + augmented GNSS Position RMSE: %.3f\n", p_error_imu_lidar_aug_gnss);
    std::printf("RMSE change compared to no sensors: %.2f%%\n", improvement);
    std::printf("RMSE change compared to IMU + lidar: %.2f%%\n", improvement_lidar_only);
    std::printf("RMSE change compared to IMU + lidar + (non-augmented) GNSS: %.2f%%\n", improvement_lidar_bias_gnss);

    // Plot to see accuracy
    plot_car_estimation(x, mean, T, "imu_lidar_augmented_gnss.png");
  });
}
