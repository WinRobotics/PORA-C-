// Core helpers shared by every exercise: the NotImplementedError used to mark
// the TODO blocks, output paths, and a small `main` wrapper.
#pragma once

#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef PORA_OUTPUT_DIR
#define PORA_OUTPUT_DIR "outputs"
#endif
#ifndef PORA_EXERCISE_NAME
#define PORA_EXERCISE_NAME "exercise"
#endif
#ifndef PORA_EXERCISE_DIR
#define PORA_EXERCISE_DIR "."
#endif

namespace pora {

// C++ counterpart of Python's `raise NotImplementedError(...)`.
struct NotImplementedError : public std::logic_error {
  explicit NotImplementedError(const std::string& what) : std::logic_error(what) {}
};

// Directory that holds the exercise source file (and its data files), the
// equivalent of `os.getcwd()` inside the original notebooks.
inline std::filesystem::path exercise_dir() { return std::filesystem::path(PORA_EXERCISE_DIR); }

// Path of a data file that lives next to the exercise source.
inline std::string data_path(const std::string& relative) {
  return (exercise_dir() / relative).string();
}

// Path inside this exercise's output directory (created on demand). All
// figures, videos and CSV files are written there.
inline std::string output_path(const std::string& filename) {
  std::filesystem::path dir = std::filesystem::path(PORA_OUTPUT_DIR) / PORA_EXERCISE_NAME;
  std::filesystem::create_directories(dir);
  return (dir / filename).string();
}

// Runs the body of an exercise and turns exceptions into readable messages,
// e.g. "NotImplementedError: Need to implement code here."
inline int run(const std::function<void()>& body) {
  try {
    body();
  } catch (const NotImplementedError& e) {
    std::cerr << "NotImplementedError: " << e.what() << std::endl;
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}

}  // namespace pora
