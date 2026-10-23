add_library(pora_common STATIC
  src/plot.cpp
  src/qp.cpp
  src/spline.cpp
  src/ply.cpp)
target_include_directories(pora_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(pora_common PUBLIC Eigen3::Eigen ${OpenCV_LIBS})
target_include_directories(pora_common PUBLIC ${OpenCV_INCLUDE_DIRS})
