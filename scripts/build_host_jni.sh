#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<USAGE
Usage: ${0##*/} [--build-dir <dir>] [--java-home <path>] [--blasfeo-target <target>]

Options:
  --build-dir <dir>         Build directory. Defaults to build-host-jni.
  --java-home <path>        Path to JDK root. Defaults to JAVA_HOME env var.
  --blasfeo-target <target> BLASFEO architecture target. Defaults to auto-detect.
  --release                 Build in Release mode (default: RelWithDebInfo).
  -h, --help                Show this help message.

Builds the mecanum MPC solver and JNI shared library for the host machine,
suitable for running Java/JUnit tests without an Android device.
USAGE
}

BUILD_DIR="build-host-jni"
JAVA_HOME_ARG=""
BLASFEO_TARGET=""
BUILD_TYPE="RelWithDebInfo"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)    shift; BUILD_DIR="$1" ;;
        --build-dir=*)  BUILD_DIR="${1#*=}" ;;
        --java-home)    shift; JAVA_HOME_ARG="$1" ;;
        --java-home=*)  JAVA_HOME_ARG="${1#*=}" ;;
        --blasfeo-target)   shift; BLASFEO_TARGET="$1" ;;
        --blasfeo-target=*) BLASFEO_TARGET="${1#*=}" ;;
        --release)      BUILD_TYPE="Release" ;;
        -h|--help)      usage; exit 0 ;;
        *)              echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
    shift || true
done

# Resolve JAVA_HOME
if [[ -n "${JAVA_HOME_ARG}" ]]; then
    export JAVA_HOME="${JAVA_HOME_ARG}"
elif [[ -z "${JAVA_HOME:-}" ]] || [[ ! -d "${JAVA_HOME}/include" ]]; then
    # Auto-detect: find a JDK (not JRE) that has include/jni.h
    JAVA_HOME=""
    for candidate in /usr/lib/jvm/java-*-openjdk-*; do
        if [[ -f "${candidate}/include/jni.h" ]]; then
            export JAVA_HOME="${candidate}"
            break
        fi
    done
    if [[ -z "${JAVA_HOME}" ]] && command -v java >/dev/null 2>&1; then
        JAVA_BIN="$(readlink -f "$(command -v java)")"
        export JAVA_HOME="${JAVA_BIN%/bin/java}"
    fi
fi

if [[ -z "${JAVA_HOME:-}" ]]; then
    echo "JAVA_HOME not set and could not be auto-detected." >&2
    echo "Install a JDK or pass --java-home <path>." >&2
    exit 1
fi

if [[ ! -d "${JAVA_HOME}/include" ]]; then
    echo "JAVA_HOME=${JAVA_HOME} does not contain an include/ directory." >&2
    echo "Ensure JAVA_HOME points to a JDK (not a JRE)." >&2
    exit 1
fi

echo "Using JAVA_HOME=${JAVA_HOME}"

# Auto-detect BLASFEO target for host CPU
if [[ -z "${BLASFEO_TARGET}" ]]; then
    ARCH="$(uname -m)"
    case "${ARCH}" in
        x86_64|amd64)
            # Check for AVX2 support
            if grep -q avx2 /proc/cpuinfo 2>/dev/null; then
                BLASFEO_TARGET="X64_INTEL_HASWELL"
            elif grep -q avx /proc/cpuinfo 2>/dev/null; then
                BLASFEO_TARGET="X64_INTEL_SANDY_BRIDGE"
            elif grep -q sse3 /proc/cpuinfo 2>/dev/null; then
                BLASFEO_TARGET="X64_AMD_BULLDOZER"
            else
                BLASFEO_TARGET="X64_AUTOMATIC"
            fi
            ;;
        aarch64|arm64)
            BLASFEO_TARGET="ARMV8A_ARM_CORTEX_A57"
            ;;
        *)
            BLASFEO_TARGET="GENERIC"
            ;;
    esac
    echo "Auto-detected BLASFEO target: ${BLASFEO_TARGET}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "${BUILD_DIR}" != /* ]]; then
    BUILD_DIR="${REPO_ROOT}/${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"

if command -v nproc >/dev/null 2>&1; then
    PARALLEL_JOBS="$(nproc)"
else
    PARALLEL_JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
fi

cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DBLASFEO_TARGET="${BLASFEO_TARGET}" \
    -DMPC_BUILD_JNI=ON \
    -DMPC_USE_HPIPM=ON \
    -DCMAKE_C_FLAGS="-march=native" \
    -DCMAKE_CXX_FLAGS="-march=native" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build "${BUILD_DIR}" --parallel "${PARALLEL_JOBS}"

# Run CTest
echo ""
echo "Running tests..."
ctest --test-dir "${BUILD_DIR}" --output-on-failure

JNI_LIB="${BUILD_DIR}/libmecanum_ltv_jni.so"
if [[ "$(uname)" == "Darwin" ]]; then
    JNI_LIB="${BUILD_DIR}/libmecanum_ltv_jni.dylib"
fi

cat <<INFO

Build completed. Host JNI artifacts in:
  ${BUILD_DIR}

JNI library:
  ${JNI_LIB}

To use in Java tests, add to JVM args:
  -Djava.library.path=${BUILD_DIR}
INFO
