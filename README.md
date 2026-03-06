# mecanum_LTV_OCP

Model Predictive Controller for mecanum-wheeled FTC robots using a Linear Time-Varying (LTV) formulation. Heavy matrix math (discretization, condensing, Cholesky) is precomputed offline so the robot only runs a fast QP solve each control loop.

**Target**: FTC Control Hub (ARM Cortex-A53) via JNI, with host testing on macOS/Linux.

## Prerequisites

- **CMake** >= 3.16
- **C++17** compiler (Clang or GCC)
- **Git** (for the BLASFEO submodule)
- **Android NDK** (for robot cross-compilation)
- **JDK** (for host JNI builds / simulation)

Clone with submodules:

```bash
git clone --recursive <repo-url>
cd mecanum_LTV_OCP
```

If already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

## Quick Start

### 1. Build and test on host

```bash
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)
ctest --output-on-failure
```

This builds the static library `libmecanum_mpc.a` and runs all unit tests.

### 2. Build with the CLI precomputation tool

```bash
mkdir build && cd build
cmake -DMPC_BUILD_CLI=ON ..
cmake --build . -j$(nproc)
```

This additionally builds the `mpc_precompute` executable.

### 3. Build for Android (robot)

```bash
./scripts/build_android_cortex_a53.sh
```

Requires `ANDROID_NDK_HOME` or `ANDROID_NDK_ROOT` set. Produces `libmecanum_ltv_jni.so` in `build-android-a53/`.

### 4. Build host JNI (for simulation)

```bash
./scripts/build_host_jni.sh
```

Auto-detects JDK and CPU architecture. Produces `libmecanum_ltv_jni.dylib` (macOS) or `.so` (Linux) in `build-host-jni/`.

## Precomputing MPC Windows

The whole point of offline precomputation: run the expensive math on your dev machine, ship a `.bin` file to the robot, and load it instantly with `loadWindows()` instead of running `loadTrajectory()` (which calls `mpc_precompute_all()` on-device).

### Input files

You need two files:

**1. Trajectory JSON** (`trajopt.json`) -- output from mecanum_trajopt:

```json
{
  "robotParams": {
    "mass": 14.1,
    "inertia": 0.5,
    "wheel_radius": 0.048,
    "lx": 0.2,
    "ly": 0.2,
    "t_max": 2.56,
    "w_max": 45.07
  },
  "trajectories": [
    {
      "name": "Trajectory 1",
      "trajectory": {
        "times": [0.0, 0.01, 0.02, ...],
        "states": [
          [vx, vy, omega, px, py, theta],
          ...
        ],
        "controls": [
          [drive, strafe, turn],
          ...
        ],
        "totalTime": 3.5
      }
    }
  ]
}
```

**2. MPC config JSON** (`config.json`):

```json
{
  "N": 20,
  "dt": 0.02,
  "u_min": -1.0,
  "u_max": 1.0,
  "Q_diag": [300, 300, 300, 10, 10, 10],
  "R_diag": [0.005, 0.005, 0.005, 0.005],
  "Qf_diag": [0, 0, 0, 0, 0, 0],
  "damping_linear": 0.0,
  "damping_angular": 0.0
}
```

| Field | Description |
|-------|-------------|
| `N` | Prediction horizon (number of steps, max 30) |
| `dt` | Control timestep in seconds |
| `u_min`, `u_max` | Motor duty cycle bounds (typically -1 to 1) |
| `Q_diag` | State cost weights: [px, py, theta, vx, vy, omega] |
| `R_diag` | Control cost weights: [FL, FR, RL, RR] |
| `Qf_diag` | Terminal state cost weights |
| `damping_linear` | Linear viscous friction coefficient (optional, default 0) |
| `damping_angular` | Angular viscous friction coefficient (optional, default 0) |

### Running the precomputation

```bash
# Build the CLI tool
cd build
cmake -DMPC_BUILD_CLI=ON ..
cmake --build . --target mpc_precompute

# Precompute trajectory index 0
./mpc_precompute path/to/trajopt.json path/to/config.json output.bin

# Precompute a different trajectory index
./mpc_precompute path/to/trajopt.json path/to/config.json output.bin --traj-index 1
```

### Deploying to the robot

Push the `.bin` file to the robot's trajopt directory:

```bash
adb push output.bin /sdcard/FIRST/trajopt/
```

The naming convention expected by the auto opmodes is `{project}_{trajN}.bin`, e.g.:

```bash
adb push auto-1_traj0.bin /sdcard/FIRST/trajopt/
adb push turntest_traj0.bin /sdcard/FIRST/trajopt/
```

## Cross-Compiling for the Robot

### Android build (Cortex-A53)

```bash
# Set your NDK path
export ANDROID_NDK_HOME=/path/to/android-ndk

# Build
./scripts/build_android_cortex_a53.sh
```

Options:

```
--ndk PATH          Android NDK path (overrides env var)
--build-dir DIR     Output directory (default: build-android-a53)
--api-level LEVEL   Android API level (default: 24)
```

The script builds with `-O3 -mcpu=cortex-a53 -flto` and targets `arm64-v8a`. Output is `libmecanum_ltv_jni.so`.

### Deploying the .so

Copy the built `.so` into your FTC app's JNI libs:

```bash
cp build-android-a53/libmecanum_ltv_jni.so \
   /path/to/SigmaDecode/TeamCode/src/main/jniLibs/arm64-v8a/
```

Or use whatever native library deployment your FTC project uses (e.g., hashed library loading via Sinister).

## Running with Simulation

The host JNI build lets you run the MPC controller from a Java/Kotlin simulation environment on your dev machine.

### 1. Build the host JNI library

```bash
./scripts/build_host_jni.sh
```

This auto-detects your JDK and CPU, builds with optimized BLASFEO kernels, and runs the test suite.

### 2. Point your simulation to the library

The JNI library path needs to be on `java.library.path`. When running from your FTC project in SIM mode:

```bash
# macOS
export DYLD_LIBRARY_PATH=/path/to/mecanum_LTV_OCP/build-host-jni:$DYLD_LIBRARY_PATH

# Linux
export LD_LIBRARY_PATH=/path/to/mecanum_LTV_OCP/build-host-jni:$LD_LIBRARY_PATH
```

Or pass it as a JVM argument:

```bash
-Djava.library.path=/path/to/mecanum_LTV_OCP/build-host-jni
```

### 3. Kotlin usage (SigmaDecode)

**Fast path (precomputed .bin):**

```kotlin
LTVClient.fromPrecomputed("/sdcard/FIRST/trajopt/auto-1_traj0.bin").use { ltv ->
    // ltv.dt, ltv.numWindows() available immediately
    while (running) {
        val u = ltv.solve(state.mecanumState, elapsed)  // [FL, BL, BR, FR]
        applyMotorPowers(u)
    }
}
```

**Online precomputation path (slower load, no .bin needed):**

```kotlin
LTVClient(drivetrainParameters).use { ltv ->
    ltv.loadTrajectory(traj)  // runs mpc_precompute_all() — slow on robot
    while (running) {
        val u = ltv.solve(state.mecanumState, elapsed)
        applyMotorPowers(u)
    }
}
```

### 4. Rerun visualization (optional)

Build with Rerun support for visual debugging of trajectory tracking:

```bash
cd build
cmake -DMPC_ENABLE_RERUN=ON ..
cmake --build .
./test_trajopt_rerun path/to/trajopt.json [trajectory_index]
```

This opens the Rerun viewer showing reference vs. actual trajectory, control inputs, and tracking errors over time.

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `MPC_BUILD_CLI` | OFF | Build `mpc_precompute` CLI tool |
| `MPC_BUILD_JNI` | OFF | Build JNI shared library (host; automatic on Android) |
| `MPC_ENABLE_RERUN` | OFF | Build Rerun visualization tests |
| `MPC_USE_NEON` | OFF | ARM NEON kernels (currently disabled) |
| `MPC_USE_HPIPM` | OFF | HPIPM dense QP solver |
| `MPC_USE_QPOASES` | OFF | qpOASES QP solver |
| `BLASFEO_TARGET` | GENERIC | BLASFEO CPU target (e.g., `X64_INTEL_HASWELL`, `ARMV8A_ARM_CORTEX_A53`) |

## Project Structure

```
mecanum_LTV_OCP/
├── include/           # Public headers (mpc_types.h, mecanum_ltv.h, ...)
├── src/               # Implementation + JNI bridge + CLI tool
├── tests/             # Unit and integration tests
├── scripts/           # Build scripts for host JNI and Android
├── deps/blasfeo/      # BLAS library (git submodule)
└── docs/              # Detailed documentation (math, architecture, solvers)
```

## Binary Format (v2)

Precomputed `.bin` files use a simple format:

```
[MPCFileHeader]        magic=0x4D504351, version=2, n_windows, N, nx, nu, u_min, u_max, dt
[Window 0]             H, L, F, f_const, x_ref_0, lambda_max
[Window 1]             ...
...
```

Each window contains the condensed Hessian, its Cholesky factor, the gradient matrices, and the reference state -- everything the online solver needs. No recomputation required on load.

## Architecture Overview

**Offline** (dev machine or `loadTrajectory`):
1. Resample trajectory to uniform dt
2. Exact discretize all intervals (LTV linearization around reference heading)
3. Condense each sliding window into a dense QP
4. Cholesky-factorize each Hessian
5. Serialize to `.bin`

**Online** (robot control loop):
1. Load precomputed windows (`loadWindows` -- just `fread`)
2. Each tick: compute tracking error, form gradient, solve box-constrained QP via FISTA
3. Apply first control input to motors

The online solver uses warm-starting (shifted previous solution) and typically converges in 0-5 iterations.
