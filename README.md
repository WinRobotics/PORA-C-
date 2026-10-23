# PORA Exercises in C++

Hands-on robotics exercises from *Principles of Robot Autonomy* (PORA), rewritten in modern
C++17. They cover dynamics, optimal control, planning, perception, state estimation, SLAM and
reinforcement learning.

This is a port of the course's Python/Jupyter exercise set. Every exercise keeps the original
explanations, equations, docstrings and hints. The parts you implement are marked the same way:

```cpp
VectorXd robot_dynamics(const VectorXd& x, const VectorXd& u) {
  // ##### YOUR CODE STARTS HERE #####
  throw pora::NotImplementedError("Need to implement code here.");
  // ###### YOUR CODE ENDS HERE ######
}
```

Replace the `throw` with your implementation, rebuild, and run the exercise. Each program
simulates, plots and animates the results the way the notebooks did.

## Exercises

| Chapter | Topic | Exercises (executable names) |
|---|---|---|
| 1 | Dynamics & simulation | `ch01_simulation`: numerical integration of a pendulum and a bicycle<br>`ch01_nonholonomic_wheeled_robot_dynamics`: unicycle model and rollouts<br>`ch01_double_integrator_dynamics`: batched rollouts |
| 2 | Optimal control | `ch02_unicycle_optimal_control`: necessary optimality conditions and single shooting |
| 3 | Feedback control | `ch03_pid_control`: PD control of an inverted pendulum<br>`ch03_cartpole_lqr_control`: LQR stabilization and tracking<br>`ch03_cartpole_iterative_lqr`: iLQR swing-up<br>`ch03_linear_mpc`: constrained MPC, terminal cost and feasibility<br>`ch03_unicycle_trajectory_tracking`: differential flatness tracking |
| 4 | Motion planning | `ch04_a_star`: A* on a grid and spline smoothing<br>`ch04_rrt`: RRT, shortcutting and RRT* |
| 6 | Cameras | `ch06_camera_extrinsics`: homographies and extrinsics<br>`ch06_camera_intrinsics`: Zhang's calibration |
| 8 | Image processing & point clouds | `ch08_correlation_filter`: correlation and Gaussian filters<br>`ch08_icp`: ICP point cloud registration |
| 9 | Deep learning | `ch09_cnn`: CNN classifier on CIFAR-10 |
| 10 | Object detection | `ch10_object_detection`: YOLOv5 detection, drawing and filtering |
| 12 | Kalman filtering | `ch12_kalman_filter`: landmark localization |
| 13 | Localization | `ch13_ekf_localization`: EKF localization<br>`ch13_particle_filter_localization`: particle filter localization |
| 14 | SLAM | `ch14_ekf_slam`: EKF SLAM |
| 15 | Sensor fusion | `ch15_kalman_sensor_fusion`: IMU/lidar/GNSS fusion and bias estimation |
| 17 | Decision making | `ch17_markovian_drone`: dynamic programming in a stochastic grid world |
| 18 | Reinforcement learning | `ch18_widget_sales`: tabular Q-learning vs. value iteration<br>`ch18_cartpole_balance`: deep Q-learning and REINFORCE<br>`ch18_advantage_actor_critic`: A2C on a lunar lander |

The sources live in `chNN/exercises/`. In the Python version, some exercises live in a
helper file instead of the notebook (`a_star.py`, `rrt.py`, `camera_calibration.py`,
`q_learning.py`, `reinforce.py`). They live in the matching `.cpp` file here.

## Getting started

### Requirements

| Dependency | Needed for |
|---|---|
| C++17 compiler, CMake ≥ 3.16 | everything |
| Eigen 3, OpenCV 4 | everything |
| PCL | `ch08_icp` |
| LibTorch | `ch09_cnn`, `ch18_cartpole_balance`, `ch18_advantage_actor_critic` |
| Box2D 2.4.1 | `ch18_advantage_actor_critic`; downloaded automatically by CMake |

On Ubuntu:

```bash
sudo apt install build-essential cmake libeigen3-dev libopencv-dev libpcl-dev

# Optional: LibTorch (CPU build) for the deep learning / deep RL exercises
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.5.1+cpu.zip -d $HOME
```

### Build

```bash
git clone https://github.com/WinRobotics/PORA-C-.git
cd PORA-C-
cmake -S . -B build -DCMAKE_PREFIX_PATH=$HOME/libtorch   # omit the prefix path without LibTorch
cmake --build build -j
```

If an optional dependency is missing, CMake skips the exercises that need it and prints a
message.

### Run

```bash
./build/bin/ch01_simulation
PORA_SHOW=1 ./build/bin/ch12_kalman_filter   # also open each figure in a window
PORA_SEED=3 ./build/bin/ch04_rrt             # another random outcome for unseeded exercises
```

- Figures (PNG), animations (MP4) and other results are written to
  `build/outputs/<exercise>/`.
- A program stops with `NotImplementedError: Need to implement code here.` at the first part
  you haven't implemented yet. Work through an exercise from top to bottom and rebuild only
  that exercise with `cmake --build build --target <exercise>`.

## Data

- **Included:** chessboard images (ch06), the camera-man image and the Stanford bunny point
  clouds (ch08), and a pretrained YOLOv5s model in ONNX format (ch10).
- **CIFAR-10 (ch09):** run `ch09/exercises/download_cifar10.sh` once (about 170 MB).
- **Photos for object detection (ch10):** pass any image on the command line:
  `./build/bin/ch10_object_detection my_photo.jpg`.
- **Regenerating the YOLOv5 model:** run `ch10/exercises/export_yolov5_onnx.sh` in a Python
  environment with PyTorch. OpenCV 4.5 needs an opset 12, simplified export.

## Repository layout

```
CMakeLists.txt          one executable per exercise
common/                 pora_common, the shared support library
  include/pora/
    core.hpp            NotImplementedError, data/output paths
    numerics.hpp        odeint, finite-difference Jacobian, DARE solver, linspace, interp
    random.hpp          NumPy-style random numbers (seedable via PORA_SEED)
    plot.hpp            matplotlib-style plotting rendered with OpenCV
    qp.hpp              convex QP solver (OSQP-style ADMM)
    spline.hpp          cubic smoothing splines
    ply.hpp             PLY point cloud reader
chNN/exercises/         exercise sources and their data files
```

## Python → C++ cheat sheet

| Python | C++ |
|---|---|
| `numpy` | Eigen (`VectorXd`, `MatrixXd`) |
| `matplotlib.pyplot` | `pora::plt::Figure`, `Axes::plot(x, y, "r--")`, `fmt("o").ms(4).label("...")` |
| `scipy.integrate.odeint` | `pora::odeint(f, x0, t)` |
| `jax.jacobian` | `pora::jacobian(f, x)` (central finite differences) |
| `np.random` | `pora::rng()`, `pora::Rng` |
| `cvxpy` | `pora::QP` (stack the variables into one vector, add `l <= A z <= u` constraints) |
| `scipy.linalg.solve_discrete_are` | `pora::solve_discrete_are` |
| `scipy.interpolate.make_splrep` / `splev` | `pora::make_splrep` / `pora::splev` |
| `shapely` boxes | `Box` / `box(...)` (`ch04/exercises/utils.hpp`) |
| `cv2` | OpenCV C++ API |
| `open3d`, `sklearn.neighbors` | PCL (`pcl::KdTreeFLANN`, `pcl::IterativeClosestPoint`) |
| `torch`, `jax` + `equinox` + `optax` | LibTorch (`torch::nn`, `torch::optim`) |
| `gymnasium` CartPole / LunarLander | C++ ports in `ch18/exercises/cartpole.cpp` and `a2c_utils.cpp` |

## Differences from the Python exercises

- Random numbers come from C++ generators, so noisy simulations won't reproduce the
  notebooks' exact numbers. The qualitative behaviour is the same.
- Jacobians use finite differences instead of automatic differentiation.
- **ch02:** the initial co-state guess is `(-1, -1, 1)`. With the standard Hamiltonian
  convention, the original `(1, 1, -1)` makes Newton–Raphson diverge.
- **ch10:** only YOLOv5 is used (there is no C++ FCOS model). Your implementation also decodes
  the raw network output and runs non-maximum suppression.
- **ch18 widget sales:** the logged data is generated with seed 1 instead of 0, so that the
  random actions aren't correlated with the simulated demand.
- Interactive sliders are replaced by saving one figure per setting or view.

## Credits

The exercise content (problem statements, explanations, provided code and data) comes from the
*Principles of Robot Autonomy* course exercises. This repository contains only the C++
translation and its supporting library.
