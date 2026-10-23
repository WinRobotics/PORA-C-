// Minimal ASCII PLY reader, the stand-in for `open3d.io.read_point_cloud`.
#pragma once

#include <Eigen/Dense>

#include <string>

namespace pora {

// Read the x, y, z vertex coordinates of an ASCII PLY file into an (N, 3) matrix.
Eigen::MatrixXd read_ply_points(const std::string& path);

}  // namespace pora
