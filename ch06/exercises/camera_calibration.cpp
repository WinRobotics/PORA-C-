// C++ port of ch06/exercises/camera_calibration.py
#include "camera_calibration.hpp"

#include <pora/core.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#include <stdexcept>

using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

std::tuple<VectorXd, VectorXd, VectorXd> generate_chessboard_3D_world_coordinates() {
  VectorXd X_world, Y_world, Z_world;
  // ##### YOUR CODE STARTS HERE #####
  // Hint: Use the SQUARE_SIZE, NUM_CORNERS_X, and NUM_CORNERS_Y
  // Hint: Use two nested loops (the counterpart of np.meshgrid) to generate the X, Y coordinates
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######

  // Sanity check the order is correct
  if (X_world[0] > X_world[X_world.size() - 1] || Y_world[0] < Y_world[Y_world.size() - 1])
    throw std::invalid_argument("Check that order of the coordinates is correct");

  return {X_world, Y_world, Z_world};
}

std::pair<VectorXd, VectorXd> generate_chessboard_2D_pixel_coordinates(const std::string& img_path) {
  cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
  if (img.empty()) throw std::runtime_error("Could not read image " + img_path);

  VectorXd u_meas, v_meas;
  // ##### YOUR CODE STARTS HERE #####
  // NOTE: Ensure that points are ordered left-to-right, bottom-to-top
  // Hint: cv::findChessboardCorners is the C++ counterpart of cv2.findChessboardCorners
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######

  return {u_meas, v_meas};
}

Matrix3d compute_homography(const VectorXd& u, const VectorXd& v, const VectorXd& X, const VectorXd& Y) {
  // ##### YOUR CODE STARTS HERE #####
  // Hint: use Eigen::JacobiSVD<MatrixXd> svd(L, Eigen::ComputeFullV) (the counterpart of np.linalg.svd)
  // Hint: use Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(m.data()) to give M the
  //       proper dimensions (the counterpart of <array>.reshape(3, 3)).
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
}

Matrix3d compute_intrinsics(const std::vector<Matrix3d>& homographies) {
  auto v = [](const Matrix3d& M, int i, int j) {
    Eigen::Matrix<double, 6, 1> out;
    out << M(0, i) * M(0, j),
           M(0, i) * M(1, j) + M(1, i) * M(0, j),
           M(1, i) * M(1, j),
           M(2, i) * M(0, j) + M(0, i) * M(2, j),
           M(2, i) * M(1, j) + M(1, i) * M(2, j),
           M(2, i) * M(2, j);
    return out;
  };

  const int n = static_cast<int>(homographies.size());
  MatrixXd V(2 * n, 6);
  for (int k = 0; k < n; ++k) {
    V.row(k) = v(homographies[k], 0, 1).transpose();
    V.row(n + k) = (v(homographies[k], 0, 0) - v(homographies[k], 1, 1)).transpose();
  }

  // ##### YOUR CODE STARTS HERE #####
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
}

std::pair<Matrix3d, Vector3d> compute_extrinsics(const Matrix3d& K, const Matrix3d& H) {
  Matrix3d R;
  Vector3d t;
  // ##### YOUR CODE STARTS HERE #####
  // Hint: use K.inverse() to compute KinvM = inv(K) * M
  // Hint: use .norm() to compute the normalization constant for r1, r2, and t
  // Hint: use .cross() to compute r3
  // Hint: use Q << r1, r2, r3 (or Q.col(i) = ...) to combine r1, r2, and r3 into the matrix Q
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######

  return {R, t};
}

Vector3d transform_world_to_pixel(const Matrix3d& K, const Matrix3d& R, const Vector3d& t, const Vector3d& P) {
  // ##### YOUR CODE STARTS HERE #####
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
}
