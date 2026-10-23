# PORA Exercises — C++ edition

C++17 port of the exercises in `../pora-exercises` (*Principles of Robot Autonomy*).
Every notebook / script under `chNN/exercises/` has a C++ counterpart with the **same
explanations, docstrings, hints and `YOUR CODE STARTS HERE` / `YOUR CODE ENDS HERE` blocks**.
Each block initially throws `pora::NotImplementedError("Need to implement code here.")`,
just like the Python `raise NotImplementedError(...)`.

The worked-example notebooks at the top of each chapter (`chNN/*.ipynb`, e.g. `lqr.ipynb`,
`prm_star.ipynb`) contain no exercises and are not ported.

## Build

Required: a C++17 compiler, CMake ≥ 3.16, Eigen 3, OpenCV 4 (with `dnn`, `calib3d`, `videoio`).
Optional: PCL (ch08 ICP), LibTorch (ch09 CNN and the ch18 deep-RL exercises; Box2D for the
lunar lander is downloaded automatically by CMake).

```bash
# Ubuntu
sudo apt install build-essential cmake libeigen3-dev libopencv-dev libpcl-dev

# LibTorch (CPU), only for ch09 / ch18 deep RL
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.5.1+cpu.zip -d $HOME

cmake -S . -B build -DCMAKE_PREFIX_PATH=$HOME/libtorch
cmake --build build -j
```

Exercises whose optional dependency is missing are skipped (CMake prints a message).

## Run

```bash
./build/bin/ch01_simulation
./build/bin/ch04_a_star
PORA_SHOW=1 ./build/bin/ch12_kalman_filter   # also open each figure in a window
PORA_SEED=3 ./build/bin/ch04_rrt             # different random outcome for unseeded exercises
```

- Figures (PNG), videos (MP4) and other results go to `build/outputs/<target>/`.
- A program stops at the first block that is still unimplemented, so you can work through an
  exercise top to bottom, re-building after each part (`cmake --build build --target <target>`).

## Exercise map

| Python                                           | C++ source (in `chNN/exercises/`)                                   | Target |
|--------------------------------------------------|---------------------------------------------------------------------|--------|
| ch01 `simulation.ipynb`                          | `simulation.cpp`                                                    | `ch01_simulation` |
| ch01 `nonholonomic_wheeled_robot_dynamics.ipynb` + `dynamics.py` | `nonholonomic_wheeled_robot_dynamics.cpp`, `dynamics.hpp` | `ch01_nonholonomic_wheeled_robot_dynamics` |
| ch01 `double_integrator_dynamics.ipynb`          | `double_integrator_dynamics.cpp`                                    | `ch01_double_integrator_dynamics` |
| ch02 `unicycle_optimal_control.ipynb`            | `unicycle_optimal_control.cpp`                                      | `ch02_unicycle_optimal_control` |
| ch03 `pid_control.ipynb`                         | `pid_control.cpp`                                                   | `ch03_pid_control` |
| ch03 `cartpole_lqr_control.ipynb` + `cartpole_utils.py` | `cartpole_lqr_control.cpp`, `cartpole_utils.{hpp,cpp}`       | `ch03_cartpole_lqr_control` |
| ch03 `cartpole_iterative_lqr.ipynb`              | `cartpole_iterative_lqr.cpp`                                        | `ch03_cartpole_iterative_lqr` |
| ch03 `linear_mpc.ipynb`                          | `linear_mpc.cpp`                                                    | `ch03_linear_mpc` |
| ch03 `unicycle_trajectory_tracking.ipynb` + `unicycle_utils.py` | `unicycle_trajectory_tracking.cpp`, `unicycle_utils.{hpp,cpp}` | `ch03_unicycle_trajectory_tracking` |
| ch04 `a_star.ipynb` + `a_star.py` + `utils.py`   | `a_star_main.cpp`, `a_star.{hpp,cpp}`, `utils.{hpp,cpp}`            | `ch04_a_star` |
| ch04 `rrt.ipynb` + `rrt.py`                      | `rrt_main.cpp`, `rrt.{hpp,cpp}`                                     | `ch04_rrt` |
| ch06 `camera_extrinsics.ipynb` + `camera_calibration.py` | `camera_extrinsics.cpp`, `camera_calibration.{hpp,cpp}`     | `ch06_camera_extrinsics` |
| ch06 `camera_intrinsics.ipynb`                   | `camera_intrinsics.cpp`                                             | `ch06_camera_intrinsics` |
| ch08 `correlation_filter.ipynb`                  | `correlation_filter.cpp`                                            | `ch08_correlation_filter` |
| ch08 `icp.ipynb`                                 | `icp.cpp` (PCL)                                                     | `ch08_icp` |
| ch09 `cnn.ipynb`                                 | `cnn.cpp` (LibTorch)                                                | `ch09_cnn` |
| ch10 `object_detection.ipynb`                    | `object_detection.cpp` (OpenCV DNN)                                 | `ch10_object_detection` |
| ch12 `kalman_filter.ipynb`                       | `kalman_filter.cpp`                                                 | `ch12_kalman_filter` |
| ch13 `ekf_localization.ipynb`                    | `ekf_localization.cpp`                                              | `ch13_ekf_localization` |
| ch13 `particle_filter_localization.ipynb`        | `particle_filter_localization.cpp`                                  | `ch13_particle_filter_localization` |
| ch14 `ekf_slam.ipynb`                            | `ekf_slam.cpp`                                                      | `ch14_ekf_slam` |
| ch15 `kalman_sensor_fusion.ipynb` + `utils.py`   | `kalman_sensor_fusion.cpp`, `utils.hpp`                             | `ch15_kalman_sensor_fusion` |
| ch17 `markovian_drone.ipynb`                     | `markovian_drone.cpp`                                               | `ch17_markovian_drone` |
| ch18 `widget_sales.ipynb`                        | `widget_sales.cpp`                                                  | `ch18_widget_sales` |
| ch18 `cartpole_balance.ipynb` + `cartpole.py`, `q_learning.py`, `reinforce.py` | `cartpole_balance.cpp`, `cartpole.{hpp,cpp}`, `q_learning.{hpp,cpp}`, `reinforce.{hpp,cpp}` (LibTorch) | `ch18_cartpole_balance` |
| ch18 `advantage_actor_critic.ipynb` + `a2c_utils.py` | `advantage_actor_critic.cpp`, `a2c_utils.{hpp,cpp}` (LibTorch, Box2D) | `ch18_advantage_actor_critic` |

## Python → C++ equivalents

| Python                                   | C++ (this repo)                                                              |
|------------------------------------------|------------------------------------------------------------------------------|
| `numpy` arrays                           | Eigen (`Eigen::VectorXd`, `Eigen::MatrixXd`, ...)                            |
| `matplotlib.pyplot`                      | `pora::plt` (`common/include/pora/plot.hpp`): `Figure`, `Axes::plot`, `fmt("r--")`, ... |
| `scipy.integrate.odeint`                 | `pora::odeint(f, x0, t)` (adaptive Dormand–Prince)                           |
| `jax.jacobian`                           | `pora::jacobian(f, x)` (central finite differences)                          |
| `np.random.*`                            | `pora::rng()` / `pora::Rng` (`random.hpp`)                                   |
| `cvxpy` (QP)                             | `pora::QP` (OSQP-style ADMM, `qp.hpp`)                                       |
| `scipy.linalg.solve_discrete_are`        | `pora::solve_discrete_are`                                                   |
| `scipy.interpolate.make_splrep` / `splev`| `pora::make_splrep` / `pora::splev` (cubic smoothing spline, `spline.hpp`)   |
| `shapely.box`                            | `Box` / `box(...)` in `ch04/exercises/utils.hpp`                             |
| `cv2`                                    | OpenCV C++                                                                   |
| `open3d`, `sklearn.neighbors`            | PCL (`pcl::KdTreeFLANN`, `pcl::IterativeClosestPoint`, ...) and `pora::read_ply_points` |
| `torch`, `torchvision`                   | LibTorch (`torch::nn`, `torch::optim`)                                       |
| `jax` + `equinox` + `optax` (A2C)        | LibTorch                                                                     |
| `gymnasium` CartPole-v1 / LunarLanderContinuous-v3 | C++ ports in `ch18/exercises/cartpole.cpp` / `a2c_utils.cpp` (Box2D)  |
| torchvision FCOS / torch.hub YOLOv5      | YOLOv5s through OpenCV DNN (`ch10/exercises/yolov5s.onnx`)                   |

## Data

- Chessboard images (ch06), camera-man image and bunny point clouds (ch08) are copied into
  the corresponding `exercises/` folders.
- **CIFAR-10 (ch09):** run `ch09/exercises/download_cifar10.sh` once.
- **YOLOv5 (ch10):** `ch10/exercises/yolov5s.onnx` was exported from the repository's
  `yolov5s.pt` (ONNX opset 12, simplified, so that OpenCV 4.5 can load it). To regenerate it,
  run `ch10/exercises/export_yolov5_onnx.sh` in a Python environment with PyTorch (e.g. the
  `pora` conda env). The notebook's sample photos (`images/airport.jpeg`, ...) are not part of
  the original repository; pass any photo: `./build/bin/ch10_object_detection my_photo.jpg`.

## Differences from the Python version

- Random numbers come from C++ generators, so noisy simulations do not reproduce the
  notebooks' exact numbers (the qualitative behaviour is the same).
- `pora::jacobian` uses finite differences instead of automatic differentiation.
- ch02: the initial co-state guess is `(-1, -1, 1)` instead of `(1, 1, -1)`. With the standard
  Hamiltonian convention, the notebook's guess makes Newton–Raphson diverge (it does so in the
  JAX version too). From the flipped guess it converges in about 6 iterations.
- ch10: only the YOLOv5 model is ported (there is no C++ FCOS). The YOLOv5 TODO also includes
  decoding the raw network output and running NMS, because there is no `Results` object.
- ch18 widget sales: the data-generation generator uses seed 1 instead of 0. Identically seeded
  C++ generators would correlate the random actions with the simulated demand.
- ch18 Q-learning / REINFORCE / A2C: actions are plain `int`s / tensors. `None` (terminal next
  state) is an undefined `torch::Tensor`. `Categorical` is replaced by `torch::multinomial` +
  `torch::log`.
- Interactive widgets (sliders) are replaced by saving one figure per setting/view.
