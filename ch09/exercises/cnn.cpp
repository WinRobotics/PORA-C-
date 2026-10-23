// =============================================================================
// Exercise 1: Convolutional Neural Networks
// (C++ port of ch09/exercises/cnn.ipynb, using LibTorch - the C++ API of PyTorch)
//
// The CIFAR-10 binary dataset is expected in ch09/exercises/data/cifar-10-batches-bin
// (run ch09/exercises/download_cifar10.sh once to fetch it).
// =============================================================================
#include <pora/core.hpp>
#include <pora/plot.hpp>

#include <torch/torch.h>

#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

const std::vector<std::string> classes = {"plane", "car", "bird", "cat", "deer",
                                          "dog", "frog", "horse", "ship", "truck"};

// CIFAR-10 dataset reader (binary version), the counterpart of torchvision.datasets.CIFAR10.
// Images are normalized per RGB channel with mean 0.5 and std 0.5, like
// transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)).
class CIFAR10 : public torch::data::datasets::Dataset<CIFAR10> {
 public:
  CIFAR10(const std::string& root, bool train) {
    std::vector<std::string> files;
    if (train)
      for (int i = 1; i <= 5; ++i) files.push_back(root + "/data_batch_" + std::to_string(i) + ".bin");
    else
      files.push_back(root + "/test_batch.bin");
    const int record = 1 + 3 * 32 * 32;
    std::vector<uint8_t> buffer;
    for (const auto& f : files) {
      std::ifstream in(f, std::ios::binary);
      if (!in)
        throw std::runtime_error("Missing " + f + "\nRun ch09/exercises/download_cifar10.sh to download CIFAR-10.");
      buffer.insert(buffer.end(), std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    const int64_t n = static_cast<int64_t>(buffer.size() / record);
    auto raw = torch::from_blob(buffer.data(), {n, record}, torch::kUInt8).clone();
    targets_ = raw.select(1, 0).to(torch::kInt64);
    images_ = raw.narrow(1, 1, 3 * 32 * 32).reshape({n, 3, 32, 32}).to(torch::kFloat32).div(255.0).sub(0.5).div(0.5);
  }

  torch::data::Example<> get(size_t index) override { return {images_[index], targets_[index]}; }
  torch::optional<size_t> size() const override { return images_.size(0); }

 private:
  torch::Tensor images_, targets_;
};

// Convert a batch of normalized images (B, 3, H, W) into one BGR image grid
// (the counterpart of torchvision.utils.make_grid + imshow).
cv::Mat make_grid(const torch::Tensor& images, int scale = 4) {
  auto img = (images.detach().cpu() / 2 + 0.5).clamp(0, 1).mul(255).to(torch::kUInt8);  // unnormalize RGB channels
  const int64_t B = img.size(0), H = img.size(2), W = img.size(3);
  cv::Mat grid(H + 4, B * (W + 2) + 2, CV_8UC3, cv::Scalar(0, 0, 0));
  for (int64_t b = 0; b < B; ++b) {
    auto hwc = img[b].permute({1, 2, 0}).contiguous();
    cv::Mat rgb(H, W, CV_8UC3, hwc.data_ptr<uint8_t>()), bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    bgr.copyTo(grid(cv::Rect(2 + b * (W + 2), 2, W, H)));
  }
  cv::resize(grid, grid, {}, scale, scale, cv::INTER_NEAREST);
  return grid;
}

void show_images(const torch::Tensor& images, const std::string& title, const std::string& filename) {
  pora::plt::Figure fig(1, 1, 1100, 260);
  fig.ax().image(make_grid(images));
  fig.ax().axis_off();
  fig.ax().set_title(title);
  fig.save(filename);
}

// -----------------------------------------------------------------------------
// ### Exercise 1.2: Define Simple CNN Model and Train
// Implement the structure of the CNN in the class `SimpleCNN` below. Specifically:
// 1. In the `SimpleCNNImpl` constructor, define `conv_layers` and `fc_layers` for the
//    convolution and fully-connected layers of the network.
//    1. Your model should have two convolution layers in `conv_layers`, each with ReLU
//       activation after the convolution and a 2D max pooling to downsample. Use the parameter
//       values given below to define each feature.
//    2. For `fc_layers`, you should flatten the output of the convolution layers, then have
//       two linear layers separated by a ReLU activation. Use the parameter value given below
//       to define the output size of the first linear layer. You must determine the size of
//       the input for the first linear layer and the size of the final layer output.
// 2. Implement the remaining code in the training loop to train your model using the
//    `criterion` and `optimizer` provided.
// 3. Run the code provided to evaluate the accuracy of your model.
// -----------------------------------------------------------------------------
struct SimpleCNNImpl : torch::nn::Module {
  SimpleCNNImpl() {
    const int conv_kernel_size = 3;
    const int conv_padding = 1;
    const int max_pool_kernel_size = 2;
    const int max_pool_stride = 2;
    const int conv1_out_channels = 32;
    const int conv2_out_channels = 64;
    // ##### YOUR CODE STARTS HERE #####
    // Define conv_layers
    // Hint: use torch::nn::Sequential with torch::nn::Conv2d(torch::nn::Conv2dOptions(...)),
    //       torch::nn::ReLU() and torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(...))
    throw pora::NotImplementedError("Need to implement code here.");
    // ###### YOUR CODE ENDS HERE ######

    const int hidden_layer_size = 256;
    // ##### YOUR CODE STARTS HERE #####
    // Define fc_layers
    // Hint: use torch::nn::Sequential with torch::nn::Flatten() and torch::nn::Linear(in, out)
    throw pora::NotImplementedError("Need to implement code here.");
    // ###### YOUR CODE ENDS HERE ######

    register_module("conv_layers", conv_layers);
    register_module("fc_layers", fc_layers);
  }

  /**
   * Evaluate forward pass of the model.
   *
   * Args:
   *     x: model input, size (batch_size, 3, H, W)
   *
   * Returns:
   *     Model output, size (batch_size, 10)
   */
  torch::Tensor forward(torch::Tensor x) {
    x = conv_layers->forward(x);
    x = fc_layers->forward(x);
    return x;
  }

  torch::nn::Sequential conv_layers{nullptr};
  torch::nn::Sequential fc_layers{nullptr};
};
TORCH_MODULE(SimpleCNN);

int main() {
  return pora::run([] {
    torch::manual_seed(0);
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    // -------------------------------------------------------------------------
    // ### Exercise 1.1: Load the CIFAR-10
    // Run the code below to load the CIFAR-10 dataset (https://www.cs.toronto.edu/~kriz/cifar.html)
    // and visualize some of the images in the dataset. The images in this dataset are 32 x 32
    // color images of the following objects:
    // classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    // -------------------------------------------------------------------------

    // Define batch size for the train and test datasets
    const int batch_size = 64;

    // Load CIFAR-10 dataset and normalize the RGB channels
    const std::string root = pora::data_path("data/cifar-10-batches-bin");
    auto train_dataset = CIFAR10(root, true).map(torch::data::transforms::Stack<>());
    auto test_dataset = CIFAR10(root, false).map(torch::data::transforms::Stack<>());
    const size_t num_train_batches = (train_dataset.size().value() + batch_size - 1) / batch_size;

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), torch::data::DataLoaderOptions().batch_size(batch_size));
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), torch::data::DataLoaderOptions().batch_size(batch_size));

    // Get a few random training images
    auto first_batch = *train_loader->begin();
    torch::Tensor sample_images = first_batch.data, sample_labels = first_batch.target;

    // Visualize some example images from the dataset
    std::string names;
    for (int j = 0; j < 8; ++j) names += classes[sample_labels[j].item<int64_t>()] + "  ";
    show_images(sample_images.narrow(0, 0, 8), "Sample CIFAR-10 Images: " + names, "samples.png");

    // Now, run the code below to train your model. You should see the epoch loss decreasing.
    SimpleCNN model;
    model->to(device);

    // Define loss function and optimizer
    const double learning_rate = 0.001;
    torch::nn::CrossEntropyLoss criterion;
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

    // Train the model
    const int num_epochs = 5;
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
      model->train();
      double running_loss = 0.0;

      for (auto& batch : *train_loader) {
        torch::Tensor images = batch.data.to(device), labels = batch.target.to(device);

        torch::Tensor loss;
        // ##### YOUR CODE STARTS HERE #####
        throw pora::NotImplementedError("Need to implement code here.");
        // ###### YOUR CODE ENDS HERE ######

        running_loss += loss.item<double>();
      }

      std::printf("Epoch [%d/%d], Loss: %.4f\n", epoch + 1, num_epochs, running_loss / num_train_batches);
    }

    // Run the code below to estimate the accuracy of your trained model on the test dataset.

    // Evaluate model accuracy on test dataset
    model->eval();
    std::vector<int64_t> all_preds, all_labels;
    {
      torch::NoGradGuard no_grad;
      for (auto& batch : *test_loader) {
        torch::Tensor images = batch.data.to(device), labels = batch.target.to(device);
        torch::Tensor outputs = model->forward(images);
        torch::Tensor predictions = std::get<1>(torch::max(outputs, 1)).cpu();
        torch::Tensor l = labels.cpu();
        for (int64_t i = 0; i < predictions.size(0); ++i) {
          all_preds.push_back(predictions[i].item<int64_t>());
          all_labels.push_back(l[i].item<int64_t>());
        }
      }
    }

    int correct = 0;
    for (size_t i = 0; i < all_preds.size(); ++i) correct += (all_preds[i] == all_labels[i]);
    double accuracy = 100.0 * correct / all_preds.size();
    std::printf("\nTest Accuracy: %.2f%%\n", accuracy);

    // -------------------------------------------------------------------------
    // ### Exercise 1.3: Analyze Confusion Matrix
    // Run the provided code to plot the confusion matrix from your test above. Take a look at
    // the results, do you see any interesting patterns?
    // -------------------------------------------------------------------------

    // Display confusion matrix (rows: true label, columns: predicted label)
    Eigen::MatrixXd cm = Eigen::MatrixXd::Zero(10, 10);
    for (size_t i = 0; i < all_preds.size(); ++i) cm(all_labels[i], all_preds[i]) += 1;
    {
      pora::plt::Figure fig(1, 1, 800, 800);
      fig.ax().imshow(cm, false, pora::plt::ColorMap::Blues, "", true);
      fig.ax().set_xticklabels(classes);
      fig.ax().set_yticklabels(classes);
      fig.ax().set_xlabel("Predicted label");
      fig.ax().set_ylabel("True label");
      fig.ax().set_title("Confusion Matrix - CIFAR-10");
      fig.save("confusion_matrix.png");
    }

    // Finally, run the code provided below to revisit the sampled images from the dataset and
    // see how your model predicted the image's classes!

    // Display sampled images from above
    torch::Tensor predicted;
    {
      torch::NoGradGuard no_grad;
      torch::Tensor outputs = model->forward(sample_images.to(device));
      predicted = std::get<1>(torch::max(outputs, 1)).cpu();
    }
    std::string pred_names, true_names;
    for (int j = 0; j < 8; ++j) {
      pred_names += classes[predicted[j].item<int64_t>()] + "  ";
      true_names += classes[sample_labels[j].item<int64_t>()] + "  ";
    }
    std::printf("Predicted: %s\nTrue:      %s\n", pred_names.c_str(), true_names.c_str());
    show_images(sample_images.narrow(0, 0, 8), "Predicted: " + pred_names, "predictions.png");
  });
}
