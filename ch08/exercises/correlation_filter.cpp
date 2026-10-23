// =============================================================================
// Exercise 1: Correlation and Gaussian Smoothing Filters
// (C++ port of ch08/exercises/correlation_filter.ipynb)
// =============================================================================
#include <pora/core.hpp>
#include <pora/plot.hpp>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <stdexcept>

using Eigen::MatrixXd;

// Show a matrix as a grayscale image (auto-scaled like plt.imshow(..., cmap='gray')).
void show_gray(pora::plt::Axes& ax, const MatrixXd& img, const std::string& title) {
  cv::Mat m;
  cv::eigen2cv(img, m);
  ax.image(m);
  ax.set_title(title);
  ax.axis_off();
}

// -----------------------------------------------------------------------------
// ### Exercise 1.2: Correlation Filter
// Complete the `correlate_image` function below, which implements a correlation filter
// described by the equation:
//     I'(x,y) = \sum_{i=0}^{K-1} \sum_{j=0}^{L-1} F(x,y)I(x+i, y+j).
// -----------------------------------------------------------------------------

/**
 * Correlate the grayscale image with the filter.
 *
 * Args:
 *     image: (N, M) The grayscale image to filter.
 *     filter: (K, L) The filter to apply.
 *
 * Returns:
 *     filtered_image: (N-K+1, M-L+1) filtered image.
 */
MatrixXd correlate_image(const MatrixXd& image_in, const MatrixXd& filter) {
  MatrixXd image = image_in;  // copy

  MatrixXd filtered_image;
  // ##### YOUR CODE STARTS HERE #####
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######

  return filtered_image;
}

// -----------------------------------------------------------------------------
// ### Exercise 1.3: Gaussian Blur Filter
// Complete the `create_gaussian_filter` function below. Gaussian smoothing filters are
// similar to moving average filters, but instead of weighting all of the pixels evenly they
// are weighted by the Gaussian function:
//     G_\sigma(x,y) = \frac{1}{2\pi\sigma^2} \exp ( -\frac{x^2 + y^2}{2\sigma^2} ).
// where (x,y) are the pixel coordinates and σ is a tunable parameter.
// -----------------------------------------------------------------------------

/**
 * Construct a (3, 3) normalized Gaussian filter with the given standard
 * deviation parameter.
 *
 * Args:
 *     sigma: The standard deviation of the Gaussian filter.
 *
 * Returns:
 *     Normalized Gaussian filter with shape (3, 3).
 */
MatrixXd create_gaussian_filter(double sigma) {
  MatrixXd gaussian_filter(3, 3);
  // ##### YOUR CODE STARTS HERE #####
  // Hint: two nested loops over x, y in {-1, 0, 1} play the role of np.meshgrid() here.
  // Hint: Normalize filter so its values sum up to 1.
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######

  return gaussian_filter;
}

int main() {
  return pora::run([] {
    // Loading example image.
    cv::Mat image_cv = cv::imread(pora::data_path("camera_man_grayscale.jpeg"), cv::IMREAD_GRAYSCALE);
    if (image_cv.empty()) throw std::runtime_error("Could not read camera_man_grayscale.jpeg");
    MatrixXd image;
    cv::cv2eigen(image_cv, image);
    {
      pora::plt::Figure fig(1, 1, 520, 540);
      show_gray(fig.ax(), image, "Original Image");
      fig.save("original.png");
    }

    // Run the code below to test your implementation of `correlate_image`:
    // NOTE: You should not need to modify the code below.
    {
      // Example a) horizontal edge detection filter.
      MatrixXd horizontal_filter(3, 3);
      horizontal_filter << -1, -1, -1,
                            2, 2, 2,
                           -1, -1, -1;
      MatrixXd horizontal_filtered_image = correlate_image(image, horizontal_filter);

      // Example b) vertical edge detection filter.
      MatrixXd vertical_filter(3, 3);
      vertical_filter << -1, 2, -1,
                         -1, 2, -1,
                         -1, 2, -1;
      MatrixXd vertical_filtered_image = correlate_image(image, vertical_filter);

      // Plot results.
      pora::plt::Figure fig(1, 3, 1500, 520);
      show_gray(fig.ax(0), image, "Original Image");
      show_gray(fig.ax(1), horizontal_filtered_image, "Horizontally Filtered Image");
      show_gray(fig.ax(2), vertical_filtered_image, "Vertically Filtered Image");
      fig.save("edge_filters.png");
    }

    // Run the code below to test your implementation of `create_gaussian_filter`:
    // NOTE: You should not need to modify the code below.
    {
      // Example a) Gaussian filter with 0.5 standard deviation.
      MatrixXd gaussian_filter_1 = create_gaussian_filter(0.5);
      MatrixXd gaussian_filtered_image_1 = correlate_image(image, gaussian_filter_1);

      // Example b) Gaussian filter with 2.0 standard deviation.
      MatrixXd gaussian_filter_2 = create_gaussian_filter(2.0);
      MatrixXd gaussian_filtered_image_2 = correlate_image(image, gaussian_filter_2);

      // Plot results.
      pora::plt::Figure fig(1, 3, 1500, 520);
      show_gray(fig.ax(0), image, "Original Image");
      show_gray(fig.ax(1), gaussian_filtered_image_1, "Gaussian Blur sigma=0.5");
      show_gray(fig.ax(2), gaussian_filtered_image_2, "Gaussian Blur sigma=2.0");
      fig.save("gaussian_filters.png");
    }
  });
}
