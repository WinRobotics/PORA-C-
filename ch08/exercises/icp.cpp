// =============================================================================
// Exercise 2: ICP For Point Cloud Registration
// (C++ port of ch08/exercises/icp.ipynb)
//
// In this problem you will be implementing the ICP (Iterative Closest Point) algorithm to
// register two point clouds. Registration refers to aligning the point clouds to be coherent.
// In robotics, an example where this task is important is when a lidar scan from different
// parts of a room need to be joined together to create a full map of the environment.
// You can see some examples in http://redwood-data.org/indoor_lidar_rgbd/paper.pdf if you're
// interested.
//
// NOTE: The Python version uses Open3D and scikit-learn. In C++ we use the Point Cloud
// Library (PCL): voxel down-sampling, FPFH features, RANSAC registration, KD-trees and ICP.
// Point clouds are passed around as Eigen (N, 3) matrices.
// =============================================================================
#include <pora/core.hpp>
#include <pora/plot.hpp>
#include <pora/ply.hpp>

#include <Eigen/Dense>

#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/search/kdtree.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <tuple>
#include <utility>
#include <vector>

using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

using Cloud = pcl::PointCloud<pcl::PointXYZ>;
using Features = pcl::PointCloud<pcl::FPFHSignature33>;

// Conversions between (N, 3) Eigen matrices and PCL point clouds.
Cloud::Ptr to_pcl(const MatrixXd& points) {
  Cloud::Ptr cloud(new Cloud);
  cloud->resize(points.rows());
  for (int i = 0; i < points.rows(); ++i)
    (*cloud)[i] = pcl::PointXYZ(static_cast<float>(points(i, 0)), static_cast<float>(points(i, 1)),
                                static_cast<float>(points(i, 2)));
  return cloud;
}

MatrixXd from_pcl(const Cloud& cloud) {
  MatrixXd points(cloud.size(), 3);
  for (size_t i = 0; i < cloud.size(); ++i) points.row(i) << cloud[i].x, cloud[i].y, cloud[i].z;
  return points;
}

// Apply a homogeneous transformation to an (N, 3) point matrix (like pcd.transform(T)).
MatrixXd transform(const MatrixXd& points, const Matrix4d& T) {
  return ((points * T.topLeftCorner<3, 3>().transpose()).rowwise() + T.topRightCorner<3, 1>().transpose());
}

/**
 * Plot point clouds.
 *
 * Args:
 *     pcd_list: list of point cloud data objects
 *     filename: figure file name
 *     azim: azimuth angle (deg)
 *     elev: elevation angle (deg)
 *
 * (The notebook uses interactive sliders; here two views are saved side by side:
 *  the requested one and a 3/4 view.)
 */
void plot_point_clouds(const std::vector<MatrixXd>& pcd_list, const std::string& filename, double azim = -90,
                       double elev = 90) {
  pora::plt::Figure fig(1, 2, 1300, 650);
  const pora::plt::ColorMap color_fns[] = {pora::plt::ColorMap::Viridis, pora::plt::ColorMap::Inferno};
  const double views[2][2] = {{azim, elev}, {-60, 30}};
  for (int v = 0; v < 2; ++v) {
    for (size_t i = 0; i < pcd_list.size(); ++i)
      fig.ax(v).scatter3d(pcd_list[i], views[v][0], views[v][1], color_fns[i % 2], 2.0);
    char label[64];
    std::snprintf(label, sizeof(label), "azim = %g, elev = %g", views[v][0], views[v][1]);
    fig.ax(v).set_xlabel(label);
  }
  fig.save(filename);
}

/**
 * Extract features from point cloud data.
 *
 * Args:
 *     pcd: point cloud data
 *     voxel_size: voxel size
 *
 * Returns:
 *     Down sampled point cloud and the "FPFH" features.
 */
std::pair<Cloud::Ptr, Features::Ptr> preprocess_point_cloud(const Cloud::Ptr& pcd, double voxel_size) {
  Cloud::Ptr pcd_down_sampled(new Cloud);
  pcl::VoxelGrid<pcl::PointXYZ> voxel;
  voxel.setInputCloud(pcd);
  voxel.setLeafSize(voxel_size, voxel_size, voxel_size);
  voxel.filter(*pcd_down_sampled);

  double radius_normal = voxel_size * 2;
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  ne.setInputCloud(pcd_down_sampled);
  ne.setSearchMethod(tree);
  ne.setRadiusSearch(radius_normal);
  ne.compute(*normals);

  double radius_feature = voxel_size * 5;
  Features::Ptr pcd_fpfh(new Features);
  pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
  fpfh.setInputCloud(pcd_down_sampled);
  fpfh.setInputNormals(normals);
  fpfh.setSearchMethod(tree);
  fpfh.setRadiusSearch(radius_feature);
  fpfh.compute(*pcd_fpfh);
  return {pcd_down_sampled, pcd_fpfh};
}

/**
 * Run RANSAC algorithm to perform global registration on the source and target
 * point clouds.
 *
 * Args:
 *     source_down: down sampled source point cloud
 *     target_down: down sampled target point cloud
 *     source_fpfh: source point cloud FPFH features
 *     target_fpfh: target point cloud FPFH features
 *     voxel_size: voxel size
 *
 * Returns:
 *     Transformation matrix, T.
 */
Matrix4d execute_global_registration(const Cloud::Ptr& source_down, const Cloud::Ptr& target_down,
                                     const Features::Ptr& source_fpfh, const Features::Ptr& target_fpfh,
                                     double voxel_size) {
  double distance_threshold = voxel_size * 1.5;
  std::printf(":: RANSAC registration on downsampled point clouds.\n");
  std::printf("   Since the downsampling voxel size is %.3f,\n", voxel_size);
  std::printf("   we use a liberal distance threshold %.3f.\n", distance_threshold);
  pcl::SampleConsensusPrerejective<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> ransac;
  ransac.setInputSource(source_down);
  ransac.setSourceFeatures(source_fpfh);
  ransac.setInputTarget(target_down);
  ransac.setTargetFeatures(target_fpfh);
  ransac.setMaximumIterations(100000);
  ransac.setNumberOfSamples(3);
  ransac.setCorrespondenceRandomness(5);
  ransac.setSimilarityThreshold(0.9f);
  ransac.setMaxCorrespondenceDistance(distance_threshold);
  ransac.setInlierFraction(0.25f);
  Cloud aligned;
  ransac.align(aligned);
  return ransac.getFinalTransformation().cast<double>();
}

/**
 * Calculates the least-squares best-fit transform that maps corresponding
 * points A to B in m spatial dimensions.
 *
 * Args:
 *   A: (Na, m) array of corresponding points
 *   B: (Nb, m) array of corresponding points
 *
 * Returns:
 *   T: (m + 1, m + 1) homogeneous transformation matrix that maps A on to B
 *   R: (m, m) rotation matrix
 *   t: (m, 1) translation vector
 */
std::tuple<MatrixXd, MatrixXd, VectorXd> best_fit_transform(const MatrixXd& A, const MatrixXd& B) {
  // Get number of dimensions
  const int m = static_cast<int>(A.cols());

  // Translate points to their centroids
  VectorXd centroid_A = A.colwise().mean().transpose();
  VectorXd centroid_B = B.colwise().mean().transpose();
  MatrixXd AA = A.rowwise() - centroid_A.transpose();
  MatrixXd BB = B.rowwise() - centroid_B.transpose();

  // Rotation matrix
  MatrixXd H = AA.transpose() * BB;
  Eigen::JacobiSVD<MatrixXd> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  MatrixXd U = svd.matrixU();
  MatrixXd Vt = svd.matrixV().transpose();
  MatrixXd R = Vt.transpose() * U.transpose();

  // Special reflection case
  if (R.determinant() < 0) {
    Vt.row(m - 1) *= -1;
    R = Vt.transpose() * U.transpose();
  }

  // Translation
  VectorXd t = centroid_B - R * centroid_A;

  // Homogeneous transformation
  MatrixXd T = MatrixXd::Identity(m + 1, m + 1);
  T.topLeftCorner(m, m) = R;
  T.topRightCorner(m, 1) = t;

  return {T, R, t};
}

// -----------------------------------------------------------------------------
// ### Exercise 2.3: Basic ICP
// Now, you will implement a basic version of the ICP algorithm. The ICP algorithm has the
// following steps:
// 1. For every point in the source point cloud, find its nearest neighbor in the target point
//    cloud.
// 2. Find a matrix T that maps points in the source point cloud to the target point cloud
//    while minimizing the Euclidean distance between a point and its nearest neighbor (this
//    will be our **error** for each point).
// 3. Apply the transformation, T, to the source point cloud data.
// 4. Compute the average error for this transformation across all points in the source point
//    cloud data.
// 5. Repeat steps 1 - 4 for N iterations, or break if |e_t - e_{t-1}| < τ where e_i is the
//    average error in the i-th iteration and tau is the tolerance set as a hyperparameter.
//
// Below, implement the functions:
// 1. `nearest_neighbor`: finds the nearest neighbor for each point to a set of other points.
// 2. `icp`: function that runs the ICP algorithm.
// -----------------------------------------------------------------------------

/**
 * Find the nearest (Euclidean) neighbor in dst for each point in src
 *
 * Args:
 *     src: (Na, m) array of points
 *     dst: (Nb, m) array of points
 *
 * Returns:
 *     distances: Euclidean distances of the nearest neighbor
 *     indices: dst indices of the nearest neighbor
 */
std::pair<VectorXd, VectorXi> nearest_neighbor(const MatrixXd& src, const MatrixXd& dst, double radius = 0.01) {
  VectorXd distances(src.rows());
  VectorXi indices(src.rows());
  // ##### YOUR CODE STARTS HERE #####
  // Use the pcl::KdTreeFLANN<pcl::PointXYZ> class (the C++ counterpart of
  // sklearn.neighbors.NearestNeighbors) to compute the nearest neighbor of each point in the
  // source dataset to the target dataset.
  // Note that by using this class and its methods correctly (setInputCloud, nearestKSearch),
  // you should be able to get the distance between a point and its nearest neighbor,
  // as well as the indices of the nearest neighbor in the target point cloud.
  // Hint: to_pcl() converts an (N, 3) matrix into a PCL point cloud; nearestKSearch
  //       returns *squared* distances.
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
  return {distances, indices};
}

/**
 * The Iterative Closest Point method: finds best-fit transform that maps points A
 * on to points B.
 *
 * Args:
 *     A: (Na, m) array of source points
 *     B: (Nb, m) array of destination points
 *     init_pose: (m+1, m+1) homogeneous transformation (empty matrix for none)
 *     max_iterations: exit algorithm after max_iterations
 *     tolerance: convergence criteria
 *     knn_radius: k-nearest neighbors radius
 *
 * Returns:
 *     Homogeneous transformation that maps A onto B.
 */
MatrixXd icp(const MatrixXd& A, const MatrixXd& B, const MatrixXd& init_pose = MatrixXd(), int max_iterations = 20,
             double tolerance = 0.001, double knn_radius = 0.01) {
  // Get number of points and point dimension
  const int Na = static_cast<int>(A.rows()), m = static_cast<int>(A.cols());
  const int Nb = static_cast<int>(B.rows());

  // Make points homogeneous, copy them to maintain the originals
  MatrixXd src = MatrixXd::Ones(m + 1, Na);
  MatrixXd dst = MatrixXd::Ones(m + 1, Nb);
  src.topRows(m) = A.transpose();
  dst.topRows(m) = B.transpose();

  // Apply the initial pose estimation
  if (init_pose.size() > 0) src = init_pose * src;

  double prev_error = 0;

  // ##### YOUR CODE STARTS HERE #####
  // Write the loop for the ICP algorithm here
  // Hints:
  // - Use the functions `nearest_neighbor()` and `best_fit_transform()`
  // - src and dst matrices are defined above with shape (m+1, N), while the above
  //   two function expect matrices of shape (N,m)
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######

  // Calculate final transformation
  auto [T, R, t] = best_fit_transform(A, src.topRows(m).transpose());

  return T;
}

int main() {
  return pora::run([] {
    // -------------------------------------------------------------------------
    // ### Exercise 2.1: Load Point Cloud Data
    // In this exercise, we will be using the famous Stanford Bunny point cloud
    // (https://graphics.stanford.edu/data/3Dscanrep/). Run the code below to load the point
    // cloud and print out its shape.
    // -------------------------------------------------------------------------

    // Load full bunny point cloud
    MatrixXd full_bunny = pora::read_ply_points(pora::data_path("bun_zipper.ply"));

    // Load partial bunny point cloud
    // Note: `pcd` refers to "point cloud data"
    MatrixXd pcd = pora::read_ply_points(pora::data_path("bun045.ply"));

    // The 3D points representing the lidar data are the rows of the (N, 3) matrix:
    std::printf("(%ld, %ld)\n", static_cast<long>(pcd.rows()), static_cast<long>(pcd.cols()));

    // Run the code below to visualize the full bunny mesh model.
    plot_point_clouds({full_bunny}, "full_bunny.png");

    // Let us now visualize the partial point cloud data. You should be able to see that this
    // scan is not a complete scan of the bunny. We will be referring to this partial point
    // cloud data as the "source", and the full scan of the bunny as the "target".
    //
    // The goal therefore will be to match points in the source to the target. Mathematically,
    // we can say that our overarching goal is to find a matrix T, such that:
    //     P_{source} T = P_{target}
    // where P_i represents a matrix of the point cloud data
    plot_point_clouds({pcd}, "partial_bunny.png");

    // -------------------------------------------------------------------------
    // ### Exercise 2.2: Initialize Transformation Estimate with RANSAC
    // Before we can use the ICP algorithm, we need to generate an initial transformation that
    // can represent our best guess for registering the two point clouds. We will do this using
    // the RANSAC algorithm.
    //
    // Explore the code below to get an understanding of how this approach works. First, run
    // the code below to see the source and target point clouds displayed together without any
    // registration knowledge.
    // -------------------------------------------------------------------------
    plot_point_clouds({full_bunny, pcd}, "unregistered.png");

    // We can see that the two point clouds are clearly misaligned right now. So first, we will
    // extract some features of each point cloud. These features, called the "FPFH" features of
    // the point clouds are a vector of 33 values for each point in the point cloud that
    // represents some unique features of that point. Therefore, if you have N points in your
    // point cloud, your FPFH feature matrix for that point cloud will be of shape (N, 33).
    //
    // Since N can be really large for raw point cloud data, we will downsample it a bit so that
    // it is easier to experiment with.
    Cloud::Ptr source = to_pcl(pcd), target = to_pcl(full_bunny);
    double voxel_size = 0.01;
    auto [source_down, source_fpfh] = preprocess_point_cloud(source, voxel_size);
    auto [target_down, target_fpfh] = preprocess_point_cloud(target, voxel_size);

    // Run the code below to run the RANSAC algorithm using PCL. You can see that the function
    // returns a transformation matrix, which we can apply to our source point cloud data.
    Matrix4d T = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, 0.01);
    std::cout << "T shape: (4, 4)\n" << T << std::endl;

    // We can now apply this transformation matrix to our point cloud data to see how well the
    // point clouds are now registered.
    MatrixXd lidar_transformed = transform(pcd, T);
    plot_point_clouds({full_bunny, lidar_transformed}, "ransac_registration.png");

    // As you can see, they are much better aligned than before but need fine-tuning. We will
    // now use ICP to try to improve this alignment, a process also called "Local Refinement"

    // Now, run the code below to apply your ICP implementation to compute the new
    // transformation matrix `T_icp` and then visualize the results.

    // Convert to Eigen matrices
    MatrixXd source_np = from_pcl(*source_down);
    MatrixXd target_np = from_pcl(*target_down);

    // Run ICP to get transformation matrix
    Matrix4d T_icp = icp(source_np, target_np, T, 20, 1e-5, 1e-2);

    // Transform point clouds
    MatrixXd pcd_down_ICP = transform(source_np, T_icp);  // transformed downsampled pcd
    MatrixXd pcd_ICP = transform(pcd, T_icp);              // transformed complete size pcd

    // Display result
    plot_point_clouds({full_bunny, pcd_ICP}, "basic_icp.png");

    // If your implementation of ICP is correct, you should see that the point clouds roughly
    // align. The source point cloud can seem invisible, which means that it can be aligned to
    // be inside the target point cloud, which is acceptable at this stage. This happens because
    // we implemented a very basic version of ICP that is not robust to noise, outliers, etc.

    // This visualization should be slightly easier to view and you should see some points from
    // the source point cloud along the edges of the bunny
    plot_point_clouds({full_bunny, pcd_down_ICP}, "basic_icp_downsampled.png");

    // -------------------------------------------------------------------------
    // ### Exercise 2.4: Robust ICP Using PCL
    // We will now use PCL's functions to use a robust version of the ICP algorithm. You do not
    // have to write the ICP algorithm itself, but you will only be using library functions from
    // PCL.
    //
    // Follow this tutorial https://pcl.readthedocs.io/projects/tutorials/en/latest/iterative_closest_point.html
    // to see which functions to call and how to use them. You should be using the
    // pcl::IterativeClosestPoint class (point-to-point), set the max correspondence distance to
    // `threshold`, include a convergence criteria to set a max number of iterations
    // (setMaximumIterations; setTransformationEpsilon / setEuclideanFitnessEpsilon let it stop
    // early like Open3D does), and use the RANSAC transformation T as the initial guess
    // (`reg_p2p.align(output, T.cast<float>())`). Store the result in `reg_p2p_transformation`.
    // -------------------------------------------------------------------------
    double threshold = 0.01;
    int max_iterations = 3000;

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> reg_p2p;
    Matrix4d reg_p2p_transformation = Matrix4d::Identity();
    // ##### YOUR CODE STARTS HERE #####
    throw pora::NotImplementedError("Need to implement code here.");
    // ###### YOUR CODE ENDS HERE ######
    std::printf("ICP converged: %d, fitness score: %g\n", reg_p2p.hasConverged(), reg_p2p.getFitnessScore());
    std::cout << "Transformation is:\n" << reg_p2p_transformation << std::endl;

    MatrixXd pcd_robust_ICP = transform(pcd, reg_p2p_transformation);
    plot_point_clouds({full_bunny, pcd_robust_ICP}, "robust_icp.png");

    // This visualization should now look very well aligned. If done correctly, you should see
    // the source and target point clouds mixing with each other a bit.
  });
}
