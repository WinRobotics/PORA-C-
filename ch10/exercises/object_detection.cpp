// =============================================================================
// Exercise 1: Object Detection
// (C++ port of ch10/exercises/object_detection.ipynb)
//
// In this problem, we will explore some of the features that OpenCV's DNN module offers for
// visualizing images, performing object detection with pretrained CNN models, and extracting
// and plotting detected bounding boxes.
//
// NOTE: The Python version can use torchvision's FCOS-ResNet50 or YOLOv5 from torch.hub. In
// C++ we run the pretrained YOLOv5s network with OpenCV's DNN module, which needs the model
// in ONNX format: run ch10/exercises/export_yolov5_onnx.sh once to create
// ch10/exercises/yolov5s.onnx. (There is no C++ counterpart of the torchvision FCOS model
// here, so only the YOLOv5 part of Exercise 1.1 is ported.)
//
// Usage: ch10_object_detection [image path]   (default: ch10/exercises/images/airport.jpeg)
// The original notebook expects images/airport.jpeg, images/cars.jpeg or images/dog_park.jpeg,
// which are not part of the repository - use any photo you like.
// =============================================================================
#include <pora/core.hpp>
#include <pora/plot.hpp>

#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

// Bounding boxes are stored as cv::Vec4f in xyxy format: (x_min, y_min, x_max, y_max).
using Boxes = std::vector<cv::Vec4f>;
using Scores = std::vector<float>;
using Labels = std::vector<std::string>;

// The 80 COCO class names predicted by YOLOv5.
const std::vector<std::string> COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"};

/**
 * Plots an image (BGR cv::Mat) by saving it into the output directory.
 */
void show(const cv::Mat& img, const std::string& filename) {
  pora::plt::Figure fig(1, 1, 900, 700);
  fig.ax().image(img);
  fig.ax().axis_off();
  fig.save(filename);
}

// -----------------------------------------------------------------------------
// ### Exercise 1.1: Object detection with pretrained models
// In the first part of this problem, we will perform object detection using a pretrained
// model. By *pretrained*, we mean that the weights of the model have already undergone
// training, and will not be changing over the course of this problem. We will be using a
// pretrained YOLOv5 model (https://github.com/ultralytics/yolov5).
// Your task is to implement:
// 1. `yolo_v5`: evaluates an image using the YOLOv5 model.
//
// The function will take an image as input and output `boxes`, `scores`, and `labels` for each
// object detected in the image.
// -----------------------------------------------------------------------------

/**
 * Evaluate the YOLOv5 model with the given image.
 *
 * Args:
 *     image: image to evaluate (BGR, as loaded by cv::imread)
 *
 * Returns:
 *     boxes: box detections (xyxy, in pixel coordinates of `image`)
 *     scores: scores for each detection
 *     labels: labels for each detection
 */
cv::dnn::Net load_yolov5() {
  const std::string path = pora::data_path("yolov5s.onnx");
  if (!std::filesystem::exists(path))
    throw std::runtime_error("Missing " + path + "\nRun ch10/exercises/export_yolov5_onnx.sh to create it.");
  return cv::dnn::readNetFromONNX(path);
}

std::tuple<Boxes, Scores, Labels> yolo_v5(const cv::Mat& image) {
  // Load the YOLOv5s network exported to ONNX (see export_yolov5_onnx.sh)
  static cv::dnn::Net model = load_yolov5();

  // Evaluate the network: the input is a 640x640 RGB image scaled to [0, 1]
  const int input_size = 640;
  cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(input_size, input_size), cv::Scalar(),
                                        /*swapRB=*/true, /*crop=*/false);
  model.setInput(blob);
  cv::Mat output = model.forward();  // shape (1, 25200, 85)
  const int num_candidates = output.size[1], num_values = output.size[2];
  cv::Mat predictions(num_candidates, num_values, CV_32F, output.ptr<float>());  // (25200, 85)
  const float x_scale = static_cast<float>(image.cols) / input_size;
  const float y_scale = static_cast<float>(image.rows) / input_size;
  const float conf_threshold = 0.25f, nms_threshold = 0.45f;

  Boxes boxes;
  Scores scores;
  Labels labels;
  // ##### YOUR CODE STARTS HERE #####
  // Hints:
  // 1. Each row of `predictions` is [cx, cy, w, h, objectness, 80 class scores], where the box
  //    is given in the 640x640 input image (use x_scale / y_scale to map back to `image`).
  // 2. The confidence of a detection is objectness * (best class score); its label is the
  //    COCO_CLASSES entry of the best class (cv::minMaxLoc helps).
  // 3. Keep candidates with confidence >= conf_threshold, then run non-maximum suppression
  //    with cv::dnn::NMSBoxes(rects, confidences, conf_threshold, nms_threshold, keep).
  // 4. Return the boxes in xyxy format (x_min, y_min, x_max, y_max).
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######

  return {boxes, scores, labels};
}

// -----------------------------------------------------------------------------
// ### Exercise 1.2: Visualize Object Detections
// Implement the function `draw_result` which produces an image with an overlay of the bounding
// box detections, their labels, and their scores.
// -----------------------------------------------------------------------------

/**
 * Draw bounding box visualization over top of the raw image.
 *
 * Args:
 *     image: (H, W) BGR image with values 0-255.
 *     boxes: (N,) detected bounding boxes in xyxy format.
 *     labels: (N,) length list of string label names
 *     scores: (N,) confidence scores.
 *
 * Returns:
 *     img_with_bbox: (H, W) BGR image with bounding boxes drawn atop.
 */
cv::Mat draw_result(const cv::Mat& image, const Boxes& boxes, const Labels& labels, const Scores& scores) {
  cv::Mat img_with_bbox = image.clone();
  // ##### YOUR CODE STARTS HERE #####
  // Hints:
  // 1. This function is small, the solution is only a couple of lines.
  // 2. See the documentation of cv::rectangle and cv::putText:
  //    https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html
  // 3. You need to create label strings that combine class names and confidence scores
  // 4. Format strings like "<label>: <score with 2 decimals>" work well for labels
  //    (e.g. with std::snprintf(buf, sizeof(buf), "%s: %.2f", ...))
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######

  return img_with_bbox;
}

// -----------------------------------------------------------------------------
// ### Exercise 1.3: Filter Low Confidence Detections
// Implement the function `filter` which produces a filtered set of object detections based on
// the score.
// -----------------------------------------------------------------------------

/**
 * Filter the detections based on scores and a given threshold.
 *
 * Args:
 *     boxes: box detections
 *     labels: list of label strings
 *     scores: scores for each detection
 *     threshold: threshold for the score
 *
 * Returns:
 *     filtered boxes
 *     filtered labels
 *     filtered scores
 */
std::tuple<Boxes, Labels, Scores> filter(const Boxes& boxes, const Labels& labels, const Scores& scores,
                                         float threshold) {
  // ##### YOUR CODE STARTS HERE #####
  // Hints for threshold exploration:
  // 1. Keep detection i if scores[i] >= threshold
  // 2. Build the three output vectors with push_back in a single loop
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
}

int main(int argc, char** argv) {
  return pora::run([&] {
    // Run the code below with a selected image to evaluate.

    // NOTE: Select an image on which to perform object detection.
    std::string imagepath = argc > 1 ? argv[1] : pora::data_path("images/airport.jpeg");
    // std::string imagepath = pora::data_path("images/cars.jpeg");
    // std::string imagepath = pora::data_path("images/dog_park.jpeg");

    // Load image
    cv::Mat image = cv::imread(imagepath, cv::IMREAD_COLOR);
    if (image.empty())
      throw std::runtime_error("Could not read image " + imagepath +
                               "\nPass the path of a photo as argument: ch10_object_detection <image path>");
    show(image, "input.png");

    // Evaluate the model
    auto [boxes, scores, labels] = yolo_v5(image);
    std::printf("%zu detections\n", boxes.size());

    // Show overlayed image with detections
    show(draw_result(image, boxes, labels, scores), "detections.png");

    // Run the code below to show the filtered object detections.
    auto [f_boxes, f_labels, f_scores] = filter(boxes, labels, scores, 0.5f);
    show(draw_result(image, f_boxes, f_labels, f_scores), "detections_filtered.png");
  });
}
