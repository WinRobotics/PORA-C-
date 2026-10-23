cmake_minimum_required(VERSION 3.16)
project(pora_exercises_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Figures, videos and CSV files are written to <build>/outputs/<exercise>/
set(PORA_OUTPUT_DIR "${CMAKE_BINARY_DIR}/outputs" CACHE PATH "Where exercises write their results")

# ---------------------------------------------------------------------------
# Dependencies
#   required: Eigen3, OpenCV
#   optional: PCL     (ch08 ICP)
#             LibTorch (ch09 CNN, ch18 deep RL) -> pass -DCMAKE_PREFIX_PATH=/path/to/libtorch
#             Box2D    (ch18 A2C lunar lander, downloaded automatically when LibTorch is found)
# ---------------------------------------------------------------------------
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio calib3d dnn)

option(PORA_WITH_PCL "Build the exercises that need PCL" ON)
option(PORA_WITH_TORCH "Build the exercises that need LibTorch" ON)
option(PORA_WITH_BOX2D "Download Box2D and build the lunar lander A2C exercise" ON)

if(PORA_WITH_PCL)
  find_package(PCL 1.10 QUIET COMPONENTS common kdtree search features filters registration sample_consensus)
endif()
if(PORA_WITH_TORCH)
  find_package(Torch QUIET)
endif()

add_subdirectory(common)

# pora_add_exercise(<target> <source dir> SOURCES ... [LIBS ...])
function(pora_add_exercise target dir)
  cmake_parse_arguments(ARG "" "" "SOURCES;LIBS" ${ARGN})
  set(srcs)
  foreach(s ${ARG_SOURCES})
    list(APPEND srcs ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/${s})
  endforeach()
  add_executable(${target} ${srcs})
  target_link_libraries(${target} PRIVATE pora_common ${ARG_LIBS})
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
  target_compile_definitions(${target} PRIVATE
    PORA_EXERCISE_NAME="${target}"
    PORA_EXERCISE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/${dir}"
    PORA_OUTPUT_DIR="${PORA_OUTPUT_DIR}")
endfunction()

# ---- Chapter 1: Dynamics and simulation -----------------------------------
pora_add_exercise(ch01_simulation ch01/exercises SOURCES simulation.cpp)
pora_add_exercise(ch01_nonholonomic_wheeled_robot_dynamics ch01/exercises SOURCES nonholonomic_wheeled_robot_dynamics.cpp)
pora_add_exercise(ch01_double_integrator_dynamics ch01/exercises SOURCES double_integrator_dynamics.cpp)

# ---- Chapter 2: Optimal control --------------------------------------------
pora_add_exercise(ch02_unicycle_optimal_control ch02/exercises SOURCES unicycle_optimal_control.cpp)

# ---- Chapter 3: Feedback control -------------------------------------------
pora_add_exercise(ch03_pid_control ch03/exercises SOURCES pid_control.cpp)
pora_add_exercise(ch03_cartpole_lqr_control ch03/exercises SOURCES cartpole_lqr_control.cpp cartpole_utils.cpp)
pora_add_exercise(ch03_cartpole_iterative_lqr ch03/exercises SOURCES cartpole_iterative_lqr.cpp cartpole_utils.cpp)
pora_add_exercise(ch03_linear_mpc ch03/exercises SOURCES linear_mpc.cpp)
pora_add_exercise(ch03_unicycle_trajectory_tracking ch03/exercises SOURCES unicycle_trajectory_tracking.cpp unicycle_utils.cpp)

# ---- Chapter 4: Motion planning --------------------------------------------
pora_add_exercise(ch04_a_star ch04/exercises SOURCES a_star_main.cpp a_star.cpp utils.cpp)
pora_add_exercise(ch04_rrt ch04/exercises SOURCES rrt_main.cpp rrt.cpp utils.cpp)

# ---- Chapter 6: Cameras -----------------------------------------------------
pora_add_exercise(ch06_camera_extrinsics ch06/exercises SOURCES camera_extrinsics.cpp camera_calibration.cpp)
pora_add_exercise(ch06_camera_intrinsics ch06/exercises SOURCES camera_intrinsics.cpp camera_calibration.cpp)

# ---- Chapter 8: Image processing / point clouds ------------------------------
pora_add_exercise(ch08_correlation_filter ch08/exercises SOURCES correlation_filter.cpp)
if(PCL_FOUND)
  pora_add_exercise(ch08_icp ch08/exercises SOURCES icp.cpp LIBS ${PCL_LIBRARIES})
  target_include_directories(ch08_icp PRIVATE ${PCL_INCLUDE_DIRS})
  target_compile_definitions(ch08_icp PRIVATE ${PCL_DEFINITIONS})
else()
  message(STATUS "PCL not found: skipping ch08_icp")
endif()

# ---- Chapter 9 / 10: Deep learning for perception ---------------------------
if(Torch_FOUND)
  pora_add_exercise(ch09_cnn ch09/exercises SOURCES cnn.cpp LIBS ${TORCH_LIBRARIES})
else()
  message(STATUS "LibTorch not found: skipping ch09_cnn (set CMAKE_PREFIX_PATH=/path/to/libtorch)")
endif()
pora_add_exercise(ch10_object_detection ch10/exercises SOURCES object_detection.cpp)

# ---- Chapters 12-15: State estimation, localization, SLAM, sensor fusion ----
pora_add_exercise(ch12_kalman_filter ch12/exercises SOURCES kalman_filter.cpp)
pora_add_exercise(ch13_ekf_localization ch13/exercises SOURCES ekf_localization.cpp)
pora_add_exercise(ch13_particle_filter_localization ch13/exercises SOURCES particle_filter_localization.cpp)
pora_add_exercise(ch14_ekf_slam ch14/exercises SOURCES ekf_slam.cpp)
pora_add_exercise(ch15_kalman_sensor_fusion ch15/exercises SOURCES kalman_sensor_fusion.cpp)

# ---- Chapter 17 / 18: Decision making and reinforcement learning -------------
pora_add_exercise(ch17_markovian_drone ch17/exercises SOURCES markovian_drone.cpp)
pora_add_exercise(ch18_widget_sales ch18/exercises SOURCES widget_sales.cpp)
if(Torch_FOUND)
  pora_add_exercise(ch18_cartpole_balance ch18/exercises
    SOURCES cartpole_balance.cpp cartpole.cpp q_learning.cpp reinforce.cpp LIBS ${TORCH_LIBRARIES})
  if(PORA_WITH_BOX2D)
    include(FetchContent)
    set(BOX2D_BUILD_TESTBED OFF CACHE BOOL "" FORCE)
    set(BOX2D_BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
    set(BOX2D_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(box2d
      GIT_REPOSITORY https://github.com/erincatto/box2d.git
      GIT_TAG v2.4.1)
    FetchContent_MakeAvailable(box2d)
    pora_add_exercise(ch18_advantage_actor_critic ch18/exercises
      SOURCES advantage_actor_critic.cpp a2c_utils.cpp LIBS ${TORCH_LIBRARIES} box2d)
  endif()
else()
  message(STATUS "LibTorch not found: skipping ch18_cartpole_balance and ch18_advantage_actor_critic")
endif()
