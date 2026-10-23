#include "pora/ply.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace pora {

Eigen::MatrixXd read_ply_points(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Cannot open PLY file: " + path);

  struct Element {
    std::string name;
    long count = 0;
    std::vector<std::string> props;
  };
  std::vector<Element> elements;
  std::string line;
  bool ascii = false;
  while (std::getline(in, line)) {
    std::istringstream ss(line);
    std::string key;
    ss >> key;
    if (key == "format") {
      std::string f;
      ss >> f;
      ascii = (f == "ascii");
    } else if (key == "element") {
      Element e;
      ss >> e.name >> e.count;
      elements.push_back(e);
    } else if (key == "property" && !elements.empty()) {
      std::string type, name;
      ss >> type;
      if (type == "list") {
        std::string t1, t2;
        ss >> t1 >> t2;
      }
      ss >> name;
      elements.back().props.push_back(name);
    } else if (key == "end_header") {
      break;
    }
  }
  if (!ascii) throw std::runtime_error("Only ASCII PLY files are supported: " + path);

  Eigen::MatrixXd points;
  for (const auto& e : elements) {
    if (e.name != "vertex") {
      for (long i = 0; i < e.count; ++i) std::getline(in, line);  // skip
      continue;
    }
    int ix = -1, iy = -1, iz = -1;
    for (int k = 0; k < static_cast<int>(e.props.size()); ++k) {
      if (e.props[k] == "x") ix = k;
      if (e.props[k] == "y") iy = k;
      if (e.props[k] == "z") iz = k;
    }
    if (ix < 0 || iy < 0 || iz < 0) throw std::runtime_error("PLY vertex element has no x/y/z: " + path);
    points.resize(e.count, 3);
    std::vector<double> vals(e.props.size());
    for (long i = 0; i < e.count; ++i) {
      std::getline(in, line);
      std::istringstream ss(line);
      for (auto& v : vals) ss >> v;
      points(i, 0) = vals[ix];
      points(i, 1) = vals[iy];
      points(i, 2) = vals[iz];
    }
    break;
  }
  return points;
}

}  // namespace pora
