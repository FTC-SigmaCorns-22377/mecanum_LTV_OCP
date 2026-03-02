cmake_minimum_required(VERSION 3.16)
project(mecanum_mpc LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

option(MPC_USE_NEON "Use custom NEON kernels for online linear algebra" OFF)
option(MPC_ENABLE_RERUN "Enable Rerun visualization for tests" OFF)
option(MPC_USE_HPIPM "Enable HPIPM dense QP solver" OFF)
option(MPC_USE_QPOASES "Enable qpOASES QP solver" OFF)

# Find BLAS/LAPACK
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# ---------------------------------------------------------------------------
# External dependencies (built from source via ExternalProject)
# ---------------------------------------------------------------------------
set(EXTERNAL_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/external)

if(MPC_USE_HPIPM)
    include(ExternalProject)

    # BLASFEO (HPIPM dependency)
    ExternalProject_Add(blasfeo_ext
        GIT_REPOSITORY https://github.com/giaf/blasfeo.git
        GIT_TAG        master
        GIT_SHALLOW    TRUE
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_PREFIX}
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DCMAKE_C_FLAGS=-fPIC
            -DTARGET=GENERIC
            -DBUILD_SHARED_LIBS=OFF
        BUILD_BYPRODUCTS ${EXTERNAL_INSTALL_PREFIX}/lib/libblasfeo.a
    )

    # HPIPM
    ExternalProject_Add(hpipm_ext
        GIT_REPOSITORY https://github.com/giaf/hpipm.git
        GIT_TAG        master
        GIT_SHALLOW    TRUE
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_PREFIX}
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DBLASFEO_PATH=${EXTERNAL_INSTALL_PREFIX}
            -DBUILD_SHARED_LIBS=OFF
        DEPENDS blasfeo_ext
        BUILD_BYPRODUCTS ${EXTERNAL_INSTALL_PREFIX}/lib/libhpipm.a
    )

    # Pre-create include directories so CMake doesn't error at configure time
    file(MAKE_DIRECTORY ${EXTERNAL_INSTALL_PREFIX}/include)
    file(MAKE_DIRECTORY ${EXTERNAL_INSTALL_PREFIX}/include/hpipm)
    file(MAKE_DIRECTORY ${EXTERNAL_INSTALL_PREFIX}/include/blasfeo/include)

    # Create imported targets
    add_library(blasfeo STATIC IMPORTED)
    set_target_properties(blasfeo PROPERTIES
        IMPORTED_LOCATION ${EXTERNAL_INSTALL_PREFIX}/lib/libblasfeo.a
        INTERFACE_INCLUDE_DIRECTORIES ${EXTERNAL_INSTALL_PREFIX}/include
    )
    add_dependencies(blasfeo blasfeo_ext)

    add_library(hpipm STATIC IMPORTED)
    set_target_properties(hpipm PROPERTIES
        IMPORTED_LOCATION ${EXTERNAL_INSTALL_PREFIX}/lib/libhpipm.a
        INTERFACE_INCLUDE_DIRECTORIES "${EXTERNAL_INSTALL_PREFIX}/include;${EXTERNAL_INSTALL_PREFIX}/include/hpipm"
    )
    add_dependencies(hpipm hpipm_ext)
endif()

if(MPC_USE_QPOASES)
    include(ExternalProject)

    ExternalProject_Add(qpoases_ext
        GIT_REPOSITORY https://github.com/coin-or/qpOASES.git
        GIT_TAG        master
        GIT_SHALLOW    TRUE
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_PREFIX}
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DBUILD_SHARED_LIBS=OFF
            -DQPOASES_BUILD_EXAMPLES=OFF
        BUILD_BYPRODUCTS ${EXTERNAL_INSTALL_PREFIX}/lib/libqpOASES.a
    )

    file(MAKE_DIRECTORY ${EXTERNAL_INSTALL_PREFIX}/include)

    add_library(qpoases STATIC IMPORTED)
    set_target_properties(qpoases PROPERTIES
        IMPORTED_LOCATION ${EXTERNAL_INSTALL_PREFIX}/lib/libqpOASES.a
        INTERFACE_INCLUDE_DIRECTORIES ${EXTERNAL_INSTALL_PREFIX}/include
    )
    add_dependencies(qpoases qpoases_ext)
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
)
target_link_libraries(mecanum_mpc PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
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
    target_sources(mecanum_mpc PRIVATE src/hpipm_solver.cpp)
    target_compile_definitions(mecanum_mpc PUBLIC MPC_USE_HPIPM=1)
    target_link_libraries(mecanum_mpc PUBLIC hpipm blasfeo m)
    target_include_directories(mecanum_mpc PUBLIC
        ${EXTERNAL_INSTALL_PREFIX}/include
        ${EXTERNAL_INSTALL_PREFIX}/include/hpipm
        ${EXTERNAL_INSTALL_PREFIX}/include/blasfeo/include
    )
endif()

if(MPC_USE_QPOASES)
    target_sources(mecanum_mpc PRIVATE src/qpoases_solver.cpp)
    target_compile_definitions(mecanum_mpc PUBLIC MPC_USE_QPOASES=1)
    target_link_libraries(mecanum_mpc PUBLIC qpoases)
    target_include_directories(mecanum_mpc PUBLIC
        ${EXTERNAL_INSTALL_PREFIX}/include
    )
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
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
endif()
