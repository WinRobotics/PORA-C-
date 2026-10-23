// C++ port of ch06/exercises/camera_calibration.py (declarations)
#pragma once

#include <Eigen/Dense>

#include <string>
#include <tuple>
#include <utility>
#include <vector>

// Chessboard size constants
constexpr double SQUARE_SIZE = 0.0205;
constexpr int NUM_CORNERS_X = 9;
constexpr int NUM_CORNERS_Y = 7;

/**
 * Computes 3D points of chessboard cross-junctions in the world
 * coordinate frame. The order of the points in the output arrays
 * should be left-to-right, top-to-bottom.
 *
 * Returns:
 *     X_world: (NUM_CORNERS_X * NUM_CORNERS_Y,) array of world
 *              points along the x-axis.
 *     Y_world: (NUM_CORNERS_X * NUM_CORNERS_Y,) array of world
 *              points along the y-axis.
 *     Z_world: (NUM_CORNERS_X * NUM_CORNERS_Y,) array of world
 *              points along the z-axis.
 */
std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> generate_chessboard_3D_world_coordinates();

/**
 * Computes 2D pixel coordinates chessboard cross-junctions. The order
 * of the points in the output arrays should be left-to-right,
 * top-to-bottom.
 *
 * Args:
 *     img_path: path to image.
 *
 * Returns:
 *     u_meas: (NUM_CORNERS_X * NUM_CORNERS_Y,) array of pixel
 *             coordinates along the x-axis.
 *     v_meas: (NUM_CORNERS_X * NUM_CORNERS_Y,) array of pixel
 *             coordinates along the y-axis.
 */
std::pair<Eigen::VectorXd, Eigen::VectorXd> generate_chessboard_2D_pixel_coordinates(const std::string& img_path);

/**
 * Computes homography matrix, which is required to transform 3D world
 * coordinates to 2D pixel coordinates.
 *
 * Args:
 *     u: (N,) pixel x-coordinates.
 *     v: (N,) pixel y-coordinates.
 *     X: (N,) 3D world x-coordinates.
 *     Y: (N,) 3D world y-coordinates.
 *
 * Returns:
 *     Homography matrix, M, with shape (3, 3).
 */
Eigen::Matrix3d compute_homography(const Eigen::VectorXd& u, const Eigen::VectorXd& v, const Eigen::VectorXd& X,
                                   const Eigen::VectorXd& Y);

/**
 * Computes camera intrinsics from a list of homographies.
 *
 * Args:
 *     homographies: List of (3, 3) homography matrices, one per calibration image.
 *
 * Returns:
 *     Camera intrinsic matrix, K, with shape (3, 3).
 */
Eigen::Matrix3d compute_intrinsics(const std::vector<Eigen::Matrix3d>& homographies);

/**
 * Computes camera extrinsics from a homography and intrinsics.
 *
 * Args:
 *     K: (3, 3) camera intrinsic matrix.
 *     H: (3, 3) homography matrix.
 *
 * Returns:
 *     R: (3, 3) camera extrinsic rotation matrix.
 *     t: (3, 1) camera extrinsic translation vector.
 */
std::pair<Eigen::Matrix3d, Eigen::Vector3d> compute_extrinsics(const Eigen::Matrix3d& K, const Eigen::Matrix3d& H);

/**
 * Transforms 3D point in world coordinates to homogenous point in pixel
 * coordinates.
 *
 * Note: Recall the convention, where R, t are the rotation and
 * translation of the world frame with respect to the camera frame (not
 * vice versa!)
 *
 * Args:
 *     K: (3, 3) camera intrinsic matrix.
 *     R: (3, 3) camera extrinsic rotation matrix.
 *     t: (3, 1) camera extrinsic translation vector.
 *     P: (3,) 3D point in world coordinates.
 *
 * Returns:
 *     p: (3,) homogenous point in pixel coordinates.
 */
Eigen::Vector3d transform_world_to_pixel(const Eigen::Matrix3d& K, const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
                                         const Eigen::Vector3d& P);
