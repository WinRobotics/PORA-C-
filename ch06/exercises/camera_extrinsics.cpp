// =============================================================================
// Exercise 1: Computing Camera Extrinsics
// (C++ port of ch06/exercises/camera_extrinsics.ipynb; the functions to implement live in
// camera_calibration.cpp)
// =============================================================================
#include "camera_calibration.hpp"

#include <pora/core.hpp>
#include <pora/plot.hpp>

#include <opencv2/imgcodecs.hpp>

#include <cstdio>
#include <string>
#include <vector>

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::VectorXd;

// Define paths for the set of images to use for this exercise
const std::vector<int> IMG_TO_USE = {1, 4, 9, 13, 14, 15, 16, 17, 23};

std::vector<std::string> img_paths() {
  std::vector<std::string> paths;
  for (int i : IMG_TO_USE) {
    char name[64];
    std::snprintf(name, sizeof(name), "chessboard/board-%02d.png", i);
    paths.push_back(pora::data_path(name));
  }
  return paths;
}

/**
 * Transforms 2D points in world coordinates (assuming Z_w=0) to points in pixel
 * coordinates and projects them onto the image, and plots the result.
 *
 * Args:
 *     img_path: filepath to load chessboard image
 *     X_w: (N,) array of points along x-axis in world coordinates
 *     Y_w: (N,) array of points along y-axis in world coordinates
 *     K: (3, 3) camera intrinsic matrix.
 *     filename: name of the figure to save
 */
void plot_image_and_points(const std::string& img_path, const VectorXd& X_w, const VectorXd& Y_w, const Matrix3d& K,
                           const std::string& filename) {
  cv::Mat image = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
  pora::plt::Figure fig(1, 1, 700, 560);
  auto& ax = fig.ax();
  ax.image(image, 0.2);
  ax.set_xlabel("u");
  ax.set_ylabel("v");

  // Plot measured corners
  auto [u_meas, v_meas] = generate_chessboard_2D_pixel_coordinates(img_path);
  ax.plot(u_meas, v_meas, "r+");

  // Compute homography and camera extrinsics
  Matrix3d M = compute_homography(u_meas, v_meas, X_w, Y_w);
  auto [R, t] = compute_extrinsics(K, M);

  // Compute and plot world points projected onto pixels
  VectorXd u_proj = VectorXd::Zero(u_meas.size());
  VectorXd v_proj = VectorXd::Zero(v_meas.size());
  for (int i = 0; i < X_w.size(); ++i) {
    Vector3d P_world(X_w[i], Y_w[i], 0.);
    Vector3d p_proj = transform_world_to_pixel(K, R, t, P_world);
    u_proj[i] = p_proj[0] / p_proj[2];
    v_proj[i] = p_proj[1] / p_proj[2];
  }
  ax.plot(u_proj, v_proj, "b.");
  fig.save(filename);
}

int main() {
  return pora::run([] {
    const std::vector<std::string> IMG_PATHS = img_paths();

    // Pre-computed camera intrinsic matrix
    Matrix3d K;
    K << 471.87661974, 8.08064642, 289.24694223,
         0., 435.6080468, 219.34418427,
         0., 0., 1.;

    // -------------------------------------------------------------------------
    // ### Exercise 1.1: Compute Chessboard World and Pixel Coordinates
    // Before we can compute a homography for an image we need to have a set of point
    // correspondences. In the list `IMG_PATHS` we have a collection of images of a chessboard
    // that we will use (see image below). First, complete the following implementations in the
    // file `camera_calibration.cpp`:
    // 1. Implement `generate_chessboard_3D_world_coordinates` to compute a set of world
    //    coordinate points for the chessboard corners.
    // 2. Implement `generate_chessboard_2D_pixel_coordinates` to compute the corresponding set
    //    of pixel frame coordinates for the same corners.
    // -------------------------------------------------------------------------
    {
      cv::Mat image = cv::imread(IMG_PATHS[0], cv::IMREAD_GRAYSCALE);
      pora::plt::Figure fig(1, 1, 700, 560);
      fig.ax().image(image);
      fig.save("chessboard.png");
    }

    // -------------------------------------------------------------------------
    // ### Exercise 1.2: Compute Homographies and Camera Extrinsics
    // Now that we can compute a set of corresponding points in the world coordinates and pixel
    // frame coordinates, we can compute the homography matrix and, given the camera intrinsics
    // `K` defined at the top of this file, we can compute the extrinsics. In the file
    // `camera_calibration.cpp`, implement the following functions:
    // 1. `compute_homography` to compute the homography for a given set of point correspondences
    // 2. `compute_extrinsics` to compute the extrinsics R and t
    // 3. `transform_world_to_pixel` to transform a point from the world coordinate frame into
    //    the pixel frame
    //
    // Then, run the code below to see the result of your computations to project the world
    // coordinates onto each image! (The notebook uses an interactive slider; here one figure
    // per image is saved.)
    // -------------------------------------------------------------------------
    auto [X_w, Y_w, Z_w] = generate_chessboard_3D_world_coordinates();
    for (size_t i = 0; i < IMG_PATHS.size(); ++i)
      plot_image_and_points(IMG_PATHS[i], X_w, Y_w, K, "extrinsics_" + std::to_string(i) + ".png");
  });
}
