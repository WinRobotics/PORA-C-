// =============================================================================
// Exercise 2: Computing Camera Intrinsics
// (C++ port of ch06/exercises/camera_intrinsics.ipynb; the functions to implement live in
// camera_calibration.cpp)
// =============================================================================
#include "camera_calibration.hpp"

#include <pora/core.hpp>

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using Eigen::Matrix3d;
using Eigen::VectorXd;

// Define paths.
const std::vector<int> IMG_TO_USE = {1, 4, 9, 13, 14, 15, 16, 17, 23};

int main() {
  return pora::run([] {
    std::vector<std::string> IMG_PATHS;
    for (int i : IMG_TO_USE) {
      char name[64];
      std::snprintf(name, sizeof(name), "chessboard/board-%02d.png", i);
      IMG_PATHS.push_back(pora::data_path(name));
    }

    // -------------------------------------------------------------------------
    // ### Exercise 2.1: Compute Intrinsics
    // NOTE: this exercise needs to be completed after the exercises described in
    // `camera_extrinsics.cpp`
    //
    // In the file, `camera_calibration.cpp`, implement the function `compute_intrinsics` to
    // compute the camera intrinsic matrix K given a list of homographies. Then, run the code
    // below to display the result.
    // -------------------------------------------------------------------------

    // Compute world coordinate points
    auto [X_w, Y_w, Z_w] = generate_chessboard_3D_world_coordinates();

    // For each image, compute the pixel coordinates of the corners
    std::vector<VectorXd> u_meas, v_meas;
    for (const auto& img_path : IMG_PATHS) {
      auto [u, v] = generate_chessboard_2D_pixel_coordinates(img_path);
      u_meas.push_back(u);
      v_meas.push_back(v);
    }

    // Compute homography for each image.
    std::vector<Matrix3d> homographies;
    for (size_t i = 0; i < u_meas.size(); ++i) homographies.push_back(compute_homography(u_meas[i], v_meas[i], X_w, Y_w));

    // Compute intrinsics from homographies.
    Matrix3d K = compute_intrinsics(homographies);
    std::cout << K << std::endl;
  });
}
