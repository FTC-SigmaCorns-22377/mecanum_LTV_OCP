cmake_minimum_required(VERSION 3.16)
project(mecanum_mpc LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(ExternalProject)

option(MPC_USE_NEON "Use custom NEON kernels for online linear algebra" OFF)
option(MPC_BUILD_JNI "Build JNI shared library (requires JDK)" OFF)
option(MPC_ENABLE_RERUN "Enable Rerun visualization for tests" OFF)
option(MPC_USE_HPIPM "Enable HPIPM OCP solver" OFF)

set(BLASFEO_TARGET "GENERIC" CACHE STRING "BLASFEO target architecture")

# ---------------------------------------------------------------------------
# External dependencies (built from source via ExternalProject)
# ---------------------------------------------------------------------------
set(EXTERNAL_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/external)

# Forward Android cross-compilation variables to ExternalProject calls
set(EXTERNAL_ANDROID_ARGS "")
foreach(var ANDROID_ABI ANDROID_PLATFORM ANDROID_STL ANDROID_ARM_NEON ANDROID_PIE)
    if(DEFINED ${var})
        list(APPEND EXTERNAL_ANDROID_ARGS "-D${var}=${${var}}")
    endif()
endforeach()

# Compose flags for external C projects
set(EXTERNAL_C_FLAGS "-ffunction-sections -fdata-sections -O3")
set(EXTERNAL_CXX_FLAGS "-ffunction-sections -fdata-sections -O3")
set(EXTERNAL_ASM_FLAGS "")

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    string(APPEND EXTERNAL_C_FLAGS " -DOS_LINUX")
    string(APPEND EXTERNAL_CXX_FLAGS " -DOS_LINUX")
    string(APPEND EXTERNAL_ASM_FLAGS " -DOS_LINUX")
endif()

# Append user-provided flags
if(CMAKE_C_FLAGS)
    string(APPEND EXTERNAL_C_FLAGS " ${CMAKE_C_FLAGS}")
endif()
if(CMAKE_CXX_FLAGS)
    string(APPEND EXTERNAL_CXX_FLAGS " ${CMAKE_CXX_FLAGS}")
endif()
if(CMAKE_ASM_FLAGS)
    string(APPEND EXTERNAL_ASM_FLAGS " ${CMAKE_ASM_FLAGS}")
endif()

# BLASFEO (always built — provides Fortran BLAS API for linear algebra)
ExternalProject_Add(blasfeo_ext
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/deps/blasfeo
    CMAKE_ARGS
        -DTARGET=${BLASFEO_TARGET}
        -DBLAS_API=ON
        -DFORTRAN_BLAS_API=ON
        -DBLASFEO_EXAMPLES=OFF
        -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_PREFIX}
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
        "-DCMAKE_C_FLAGS=${EXTERNAL_C_FLAGS}"
        "-DCMAKE_CXX_FLAGS=${EXTERNAL_CXX_FLAGS}"
        "-DCMAKE_ASM_FLAGS=${EXTERNAL_ASM_FLAGS}"
        -DBUILD_SHARED_LIBS=OFF
        ${EXTERNAL_ANDROID_ARGS}
    BUILD_BYPRODUCTS ${EXTERNAL_INSTALL_PREFIX}/lib/libblasfeo.a
)

# Pre-create include directories so CMake doesn't error at configure time
file(MAKE_DIRECTORY ${EXTERNAL_INSTALL_PREFIX}/include)
file(MAKE_DIRECTORY ${EXTERNAL_INSTALL_PREFIX}/include/blasfeo/include)

# Create imported target for BLASFEO
add_library(blasfeo STATIC IMPORTED)
set_target_properties(blasfeo PROPERTIES
    IMPORTED_LOCATION ${EXTERNAL_INSTALL_PREFIX}/lib/libblasfeo.a
    INTERFACE_INCLUDE_DIRECTORIES ${EXTERNAL_INSTALL_PREFIX}/include
)
add_dependencies(blasfeo blasfeo_ext)

if(MPC_USE_HPIPM)
    # HPIPM only knows AVX (x86) and GENERIC; ARM must use GENERIC
    if(CMAKE_SYSTEM_NAME STREQUAL "Android" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
        set(HPIPM_TARGET "GENERIC")
    else()
        set(HPIPM_TARGET "AVX")
    endif()

    # HPIPM (reuses the same BLASFEO install)
    ExternalProject_Add(hpipm_ext
        GIT_REPOSITORY https://github.com/giaf/hpipm.git
        GIT_TAG        master
        GIT_SHALLOW    TRUE
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_PREFIX}
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DBLASFEO_PATH=${EXTERNAL_INSTALL_PREFIX}
            -DBUILD_SHARED_LIBS=OFF
            -DTARGET=${HPIPM_TARGET}
            -DHPIPM_TESTING=OFF
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
            "-DCMAKE_C_FLAGS=${EXTERNAL_C_FLAGS}"
            "-DCMAKE_CXX_FLAGS=${EXTERNAL_CXX_FLAGS}"
            ${EXTERNAL_ANDROID_ARGS}
        DEPENDS blasfeo_ext
        BUILD_BYPRODUCTS ${EXTERNAL_INSTALL_PREFIX}/lib/libhpipm.a
    )

    file(MAKE_DIRECTORY ${EXTERNAL_INSTALL_PREFIX}/include/hpipm)

    add_library(hpipm STATIC IMPORTED)
    set_target_properties(hpipm PROPERTIES
        IMPORTED_LOCATION ${EXTERNAL_INSTALL_PREFIX}/lib/libhpipm.a
        INTERFACE_INCLUDE_DIRECTORIES "${EXTERNAL_INSTALL_PREFIX}/include;${EXTERNAL_INSTALL_PREFIX}/include/hpipm"
        INTERFACE_LINK_LIBRARIES blasfeo
    )
    add_dependencies(hpipm hpipm_ext)
endif()

# ---------------------------------------------------------------------------
# Main library
# ---------------------------------------------------------------------------
add_library(mecanum_mpc STATIC
    src/mecanum_model.cpp
    src/discretizer.cpp
    src/condensing.cpp
    src/cholesky.cpp
    src/box_qp_solver.cpp
    src/mpc_offline.cpp
    src/mpc_online.cpp
    src/blas_dispatch.cpp
    src/mpc_solver_dispatch.cpp
    src/mecanum_ltv.cpp
    src/heading_lookup_offline.cpp
    src/heading_lookup_online.cpp
    src/ipm_solver.cpp
    src/waypoint_solve.cpp
)
target_link_libraries(mecanum_mpc PUBLIC blasfeo m)
target_include_directories(mecanum_mpc PUBLIC include/)

if(MPC_USE_NEON)
    target_sources(mecanum_mpc PRIVATE src/neon_kernels.cpp)
    target_compile_definitions(mecanum_mpc PUBLIC MPC_USE_NEON=1)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        # AArch64: NEON always available
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
        target_compile_options(mecanum_mpc PRIVATE -mfpu=neon -mfloat-abi=hard)
    endif()
endif()

if(MPC_USE_HPIPM)
    target_sources(mecanum_mpc PRIVATE src/hpipm_ocp_solver.cpp)
    target_compile_definitions(mecanum_mpc PUBLIC MPC_USE_HPIPM=1)
    target_link_libraries(mecanum_mpc PUBLIC hpipm)
    target_include_directories(mecanum_mpc PUBLIC
        ${EXTERNAL_INSTALL_PREFIX}/include
        ${EXTERNAL_INSTALL_PREFIX}/include/hpipm
        ${EXTERNAL_INSTALL_PREFIX}/include/blasfeo/include
    )
endif()

# ---------------------------------------------------------------------------
# JNI shared library
# ---------------------------------------------------------------------------
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    add_library(mecanum_ltv_jni SHARED src/mecanum_ltv_jni.cpp)
    target_link_libraries(mecanum_ltv_jni PRIVATE mecanum_mpc)
    target_link_options(mecanum_ltv_jni PRIVATE -Wl,--gc-sections)
elseif(MPC_BUILD_JNI)
    find_package(JNI REQUIRED)
    add_library(mecanum_ltv_jni SHARED src/mecanum_ltv_jni.cpp)
    target_link_libraries(mecanum_ltv_jni PRIVATE mecanum_mpc JNI::JNI)
endif()

# ---------------------------------------------------------------------------
# Tests (skip when cross-compiling — can't run on host)
# ---------------------------------------------------------------------------
if(NOT CMAKE_CROSSCOMPILING)
    enable_testing()

    function(add_mpc_test test_name)
        add_executable(${test_name} tests/${test_name}.cpp)
        target_link_libraries(${test_name} PRIVATE mecanum_mpc)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endfunction()

    add_mpc_test(test_discretizer)
    add_mpc_test(test_condensing)
    add_mpc_test(test_cholesky)
    add_mpc_test(test_box_qp)
    add_mpc_test(test_mpc_roundtrip)
    add_mpc_test(test_timing)
    add_mpc_test(test_heading_lookup)
    if(MPC_USE_NEON)
        add_mpc_test(test_riccati_tracking)
    endif()
    add_mpc_test(test_ipm_solver)
    add_mpc_test(test_waypoint_tracking)

    # Rerun visualization tests (optional)
    if(MPC_ENABLE_RERUN)
        include(FetchContent)

        FetchContent_Declare(
            rerun_sdk
            URL https://github.com/rerun-io/rerun/releases/download/0.25.1/rerun_cpp_sdk.zip
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_Declare(
            nlohmann_json
            URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(rerun_sdk nlohmann_json)

        add_executable(test_trajopt_rerun tests/test_trajopt_rerun.cpp)
        target_link_libraries(test_trajopt_rerun PRIVATE mecanum_mpc rerun_sdk nlohmann_json::nlohmann_json)

        add_executable(test_solver_comparison_rerun tests/test_solver_comparison_rerun.cpp)
        target_link_libraries(test_solver_comparison_rerun PRIVATE mecanum_mpc rerun_sdk nlohmann_json::nlohmann_json)

        add_executable(test_heading_noise_comparison_rerun tests/test_heading_noise_comparison_rerun.cpp)
        target_link_libraries(test_heading_noise_comparison_rerun PRIVATE mecanum_mpc rerun_sdk nlohmann_json::nlohmann_json)
    endif()
endif()
